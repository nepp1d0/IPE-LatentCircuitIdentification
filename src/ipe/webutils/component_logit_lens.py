"""Component Logit Lens for circuit visualization.

MLP outputs come from hook_mlp_out. Individual attention head outputs
are reconstructed from hook_z through W_O[head], without the shared b_O.
These residual-space vectors are decoded using final layer normalization
and unembedding.

For each component, it extracts the activation at the selected token
position and computes the top-k tokens with their corresponding softmax
probabilities.

Readouts use clean-prompt activations independently of the search's
patching contribution values.

The module also supports live circuit discovery by reusing cached
activations and storing previously computed readouts, avoiding
unnecessary recomputation as new nodes are discovered.

Note that these readouts represent isolated component outputs,
not the model's final predictions or direct logit attributions.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch


logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _diagnostic(message: str):
	"""Bound repeated diagnostics, including repeated live/final events."""
	logger.warning("Component Logit Lens: %s", message)


def _failure(node, stage, exc):
	identifier = node.get("id", (node.get("kind"), node.get("layer"), node.get("head")))
	_diagnostic(f"{identifier}: {stage}: {type(exc).__name__}: {exc}")


def _cache_get(cache, name: str):
	if cache is None:
		return None
	try:
		return cache[name]
	except KeyError:
		return None


def _attention_hook(model, layer: int):
	"""Validate hook_z capture and its per-head residual-space projection."""
	name = f"blocks.{layer}.attn.hook_z"
	hooks = getattr(model, "hook_dict", None)
	if hooks is not None and name not in hooks:
		raise ValueError(f"{name} is not registered on the model")
	w_o = getattr(model.blocks[layer].attn, "W_O", None)
	expected = (model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model)
	if not isinstance(w_o, torch.Tensor) or tuple(w_o.shape) != expected:
		raise ValueError(f"blocks.{layer}.attn.W_O must have shape {expected} [n_heads, d_head, d_model], got {getattr(w_o, 'shape', None)}")
	return name


def _prompt_data(model, clean_prompt):
	tokens = model.to_tokens(clean_prompt, prepend_bos=True)
	if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
		raise ValueError("Expected one non-empty, BOS-inclusive clean prompt")
	strings = model.to_str_tokens(tokens[0], prepend_bos=False)
	if len(strings) != tokens.shape[1]:
		raise ValueError("Clean prompt token IDs and strings have different lengths")
	return tokens, strings


def _decode_token(model, token_id: int) -> str:
	"""Return TransformerLens' readable token string, with a tokenizer fallback."""
	try:
		pieces = model.to_str_tokens(torch.tensor([token_id], dtype=torch.long), prepend_bos=False)
		if isinstance(pieces, str):
			return pieces
		return str(pieces[0])
	except Exception as exc:
		_diagnostic(f"token {token_id}: to_str_tokens failed: {type(exc).__name__}: {exc}")
		tokenizer = getattr(model, "tokenizer", None)
		if tokenizer is None:
			return str(token_id)
		try:
			return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
		except Exception as exc:
			_diagnostic(f"token {token_id}: tokenizer.decode failed: {type(exc).__name__}: {exc}")
			return str(token_id)


def _component_vector(model, node: dict, cache, position: int, batch_index: int, *, seq_len=None):
	"""Extract hook_mlp_out or reconstruct one head from hook_z @ W_O[head].

	Caches must come from the clean prompt in batch row 0, with BOS retained
	and right padding (the visualization/ExperimentManager convention).
	"""
	layer = int(node["layer"])
	if not 0 <= layer < int(model.cfg.n_layers):
		raise IndexError(f"Layer index {layer} is outside the model")
	d_model = int(model.cfg.d_model)
	if node["kind"] not in ("mlp", "attn"):
		raise ValueError(f"Unsupported component kind: {node['kind']}")
	head = int(node["head"]) if node["kind"] == "attn" else None
	if head is not None and not 0 <= head < model.cfg.n_heads:
		raise IndexError(f"Head index {head} is outside the model")
	if position < 0 or batch_index < 0 or (seq_len is not None and position >= seq_len):
		raise IndexError(f"Invalid batch/clean token position: {batch_index}/{position}")
	name = f"blocks.{layer}.hook_mlp_out" if head is None else _attention_hook(model, layer)
	width = d_model if head is None else model.cfg.d_head
	sources = cache if isinstance(cache, (tuple, list)) else (cache,)
	errors = []
	for source in sources:
		try:
			value = _cache_get(source, name)
			if value is None:
				continue
			if (not isinstance(value, torch.Tensor) or value.ndim != (3 if head is None else 4)
					or value.shape[-1] != width
					or (head is not None and value.shape[2] != model.cfg.n_heads)):
				raise ValueError(f"Incompatible activation shape {getattr(value, 'shape', None)}")
			if value.shape[0] <= batch_index or value.shape[1] < max(position + 1, seq_len or 0):
				raise ValueError(f"Activation shape {value.shape} does not cover batch {batch_index}, clean sequence {seq_len}, position {position}")
			if head is None:
				return value[batch_index, position]
			w_o = model.blocks[layer].attn.W_O
			z = value[batch_index, position, head].to(device=w_o.device, dtype=w_o.dtype)
			return z @ w_o[head]
		except Exception as exc:
			_failure(node, name, exc)
			errors.append(f"{name}: {type(exc).__name__}: {exc}")
	if errors:
		raise ValueError("No usable activation; " + "; ".join(errors))
	raise KeyError(f"Missing activations: {name}")


def _isolated_readouts(model, vectors, top_k: int, token_cache: dict[int, str]) -> list[list[dict] | None]:
	"""Decode component vectors in small batches and retain only their top-k."""
	vectors = torch.stack(vectors)
	if vectors.ndim != 2 or vectors.shape[-1] != int(model.cfg.d_model):
		raise ValueError(f"Expected component vectors [batch, d_model={model.cfg.d_model}]")
	if top_k < 1:
		raise ValueError("top_k must be positive")
	readouts = []
	for start in range(0, vectors.shape[0], 16):
		batch = vectors[start:start + 16]
		normalized = model.ln_final(batch)
		logits = model.unembed(normalized)
		probabilities = torch.softmax(logits.float(), dim=-1)
		finite = (torch.isfinite(batch).all(dim=-1)
			& torch.isfinite(normalized).all(dim=-1)
			& torch.isfinite(logits).all(dim=-1)
			& torch.isfinite(probabilities).all(dim=-1)).detach().cpu().tolist()
		k = min(top_k, probabilities.shape[-1])
		top_probabilities, top_ids = torch.topk(probabilities, k=k, dim=-1)
		ids_by_vector = top_ids.detach().cpu().tolist()
		probs_by_vector = top_probabilities.detach().cpu().tolist()
		for token_ids, token_probs, valid in zip(ids_by_vector, probs_by_vector, finite):
			if not valid:
				readouts.append(None)
				continue
			component_tokens = []
			for rank, (token_id, probability) in enumerate(zip(token_ids, token_probs), start=1):
				token_id = int(token_id)
				if token_id not in token_cache:
					token_cache[token_id] = _decode_token(model, token_id)
				component_tokens.append({
					"rank": rank,
					"token": token_cache[token_id],
					"probability": float(probability),
				})
			readouts.append(component_tokens)
	return readouts


def decode_circuit_components(
	model,
	graph: dict,
	clean_prompt: str,
	activation_cache=None,
	*,
	top_k: int = 5,
	allow_forward: bool = True,
	_prompt=None,
	_token_cache=None,
) -> dict:
	"""Attach isolated clean-component readouts to the existing graph nodes.

	Extract hook_mlp_out for MLPs or hook_z @ W_O[head] for attention heads,
	then apply final layer normalization, unembedding and float32 softmax.
	These readouts are independent of patching contributions and are not the
	model's final predictions. The shared attention bias b_O is excluded.

	Args:
        model (HookedTransformer):
            The transformer model used to extract and decode component activations.
        graph (dict):
            The circuit graph containing the nodes to decode. The existing node
            identifiers and connections are preserved.
        clean_prompt (str):
            The input prompt used to obtain the component activations.
        activation_cache (dict, optional):
            Matching clean-prompt activations with the displayed prompt in row 0,
            BOS retained and padding on the right. The caller is responsible
            for prompt identity, which plain cache dictionaries do not record.
            Missing activations can be retrieved through an additional forward
            pass. Defaults to None.
        top_k (int, optional):
            The number of highest-probability vocabulary tokens to return
            for each component. Defaults to 5.
        allow_forward (bool, optional):
            Whether to allow an additional model forward pass to retrieve
            missing activations. If False, only cached activations are used.
            Defaults to True.

    Returns:
        dict:
            The input circuit graph, updated in place with a
            'component_logit_lens' field for each MLP and attention head node.
            This field contains the top-k decoded tokens and their probabilities,
            or indicates that the readout is unavailable.
    """
	components = [n for n in graph.get("nodes", []) if n.get("kind") in ("mlp", "attn")]
	if not components:
		return graph

	try:
		prompt_tokens, token_strings = _prompt if _prompt is not None else _prompt_data(model, clean_prompt)
		seq_len = len(token_strings)
	except Exception as exc:
		_diagnostic(f"clean prompt tokenization: {type(exc).__name__}: {exc}")
		for node in components:
			node["component_logit_lens"] = {"available": False}
		return graph

	hook_names: set[str] = set()
	prepared = []
	for node in components:
		try:
			layer = int(node["layer"])
			if layer < 0 or layer >= int(model.cfg.n_layers):
				raise IndexError(f"Layer index {layer} is outside the model")
			position = seq_len - 1 if node.get("position") is None else int(node["position"])
			if not 0 <= position < seq_len:
				raise IndexError(f"Token position {position} is outside the clean prompt")
			if node["kind"] == "attn" and not 0 <= int(node["head"]) < model.cfg.n_heads:
				raise IndexError(f"Head index {node['head']} is outside the model")
			prepared.append((node, position))
		except Exception as exc:
			_failure(node, "component metadata", exc)
			node["component_logit_lens"] = {"available": False}
	vectors = {}
	for node, position in prepared:
		try:
			with torch.no_grad():
				vectors[id(node)] = _component_vector(model, node, activation_cache, position, 0, seq_len=seq_len).detach()
		except Exception as exc:
			if not allow_forward:
				_failure(node, "activation extraction", exc)
				continue
			layer = int(node["layer"])
			try:
				name = (f"blocks.{layer}.hook_mlp_out" if node["kind"] == "mlp" else
					_attention_hook(model, layer))
				hook_names.add(name)
			except Exception as capability_error:
				_failure(node, "attention capability", capability_error)

	captured_cache = None
	if hook_names and allow_forward:
		try:
			with torch.no_grad():
				_, captured_cache = model.run_with_cache(
					prompt_tokens,
					names_filter=lambda name: name in hook_names,
				)
		except Exception as exc:
			_diagnostic(f"capture {sorted(hook_names)}: {type(exc).__name__}: {exc}")
			captured_cache = None

	cache = (activation_cache, captured_cache)
	decodable = []
	for node, position in prepared:
		try:
			vector = vectors.get(id(node))
			if vector is None:
				with torch.no_grad():
					vector = _component_vector(model, node, cache, position, 0, seq_len=seq_len)
			decodable.append((node, position, vector.detach()))
		except Exception as exc:
			_failure(node, "activation extraction", exc)
			node["component_logit_lens"] = {"available": False}
	if decodable:
		try:
			with torch.no_grad():
				token_cache = {} if _token_cache is None else _token_cache
				readouts = _isolated_readouts(
					model, [entry[2] for entry in decodable], top_k, token_cache
				)
			for (node, position, _), top_tokens in zip(decodable, readouts):
				if top_tokens is None:
					_failure(node, "ln_final/unembed readout", ValueError("Non-finite activations, logits or probabilities"))
					node["component_logit_lens"] = {"available": False}
					continue
				node["component_logit_lens"] = {
					"available": True,
					"convention": "isolated_component_softmax",
					"position": position,
					"position_token": token_strings[position],
					"top_tokens": top_tokens,
				}
		except Exception as exc:
			for node, _, _ in decodable:
				_failure(node, "ln_final/unembed readout", exc)
				node["component_logit_lens"] = {"available": False}
	return graph


class LiveComponentLogitLens:
	"""Job-local activation preparation and on-demand top-5 memoization.

	Retain clean hook_mlp_out and hook_z activations; heads use W_O[head].
	Prepare before installing search hooks (especially ACDC's patching hooks).
	Live enrichment never runs a forward pass or resets hooks. Only discovered
	components are unembedded, once per component/position; repeated events and
	the completed graph reuse the same isolated-component softmax readouts.
	"""

	def __init__(self, model, clean_prompt: str, activation_cache=None):
		self.model = model
		self.clean_prompt = clean_prompt
		self.activations = {}
		self.readouts = {}
		self.token_cache = {}
		self._final_retried = set()
		self.prompt = None
		try:
			self.prompt = _prompt_data(model, clean_prompt)
		except Exception as exc:
			_diagnostic(f"clean prompt tokenization: {type(exc).__name__}: {exc}")
			return
		seq_len = len(self.prompt[1])

		def retain(name, source, layer, kind):
			try:
				value = _cache_get(source, name)
				if value is None:
					return False
				node = {"kind": kind, "layer": layer, "head": 0}
				with torch.no_grad():
					_component_vector(model, node, {name: value}, seq_len - 1, 0, seq_len=seq_len)
				self.activations[name] = value[:1, :seq_len].detach().clone()
				return True
			except Exception as exc:
				_diagnostic(f"preparation {name}: {type(exc).__name__}: {exc}")
				return False

		missing = set()
		for layer in range(model.cfg.n_layers):
			mlp_name = f"blocks.{layer}.hook_mlp_out"
			if not retain(mlp_name, activation_cache, layer, "mlp"):
				missing.add(mlp_name)
			try:
				z_name = _attention_hook(model, layer)
				if not retain(z_name, activation_cache, layer, "attn"):
					missing.add(z_name)
			except Exception as exc:
				_diagnostic(f"layer {layer} attention capability: {type(exc).__name__}: {exc}")
		if missing:
			try:
				with torch.no_grad():
					_, cache = model.run_with_cache(
						self.prompt[0],
						names_filter=lambda name: name in missing,
					)
				for name in missing:
					layer = int(name.split(".")[1])
					retain(name, cache, layer, "mlp" if name.endswith("hook_mlp_out") else "attn")
			except Exception as exc:
				_diagnostic(f"preparation capture {sorted(missing)}: {type(exc).__name__}: {exc}")

	def _key(self, node):
		position = node.get("position")
		if position is None and self.prompt is not None:
			position = len(self.prompt[1]) - 1
		return (node.get("kind"), node.get("layer"), node.get("head"), position)

	def enrich(self, graph: dict, *, final: bool = False) -> dict:
		pending = {}
		for node in graph.get("nodes", []):
			if node.get("kind") not in ("mlp", "attn"):
				continue
			key = self._key(node)
			cached = self.readouts.get(key)
			if cached is None or (final and not cached.get("available") and key not in self._final_retried):
				pending.setdefault(key, node)
		if pending:
			if self.prompt is None:
				for node in pending.values():
					node["component_logit_lens"] = {"available": False}
			else:
				decode_circuit_components(
					self.model, {"nodes": list(pending.values())}, self.clean_prompt,
					activation_cache=self.activations, allow_forward=final,
					_prompt=self.prompt, _token_cache=self.token_cache,
				)
			if final:
				self._final_retried.update(pending)
			for key, node in pending.items():
				self.readouts[key] = node["component_logit_lens"]
		for node in graph.get("nodes", []):
			key = self._key(node)
			if key in self.readouts:
				node["component_logit_lens"] = self.readouts[key]
		return graph
