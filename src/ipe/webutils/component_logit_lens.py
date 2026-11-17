"""Component Logit Lens for circuit visualization.

MLP outputs are obtained from hook_mlp_out, while attention head outputs
are reconstructed from hook_z using W_O[head], without the shared b_O bias.
These residual-space vectors are then decoded using the final layer
normalization and unembedding.

For each component, the activation at the selected token position is
used to compute the softmax probabilities and extract the top-k tokens
with the highest probabilities.

All readouts are computed using clean-prompt activations, independently
of the patching contribution scores used during circuit discovery.

The resulting readouts represent isolated component outputs, rather than
the model's final predictions or direct logit attributions.

During live circuit discovery, cached activations and previously computed
readouts are reused to avoid unnecessary computations.
"""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, cast
import torch

if TYPE_CHECKING:
	from transformer_lens import ActivationCache, HookedTransformer
	from transformer_lens.components.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _diagnostic(message: str) -> None:
	"""Log a warning message, avoiding repeated logs for the
    same message.

    Args:
        message (str):
            The diagnostic text to log with the 'Component Logit Lens' prefix.
    """
	logger.warning("Component Logit Lens: %s", message)


def _failure(node: dict, stage: str, exc: Exception) -> None:
	"""Log an error related to a circuit component,
    including its identifier and the operation that failed.

    Args:
        node (dict):
            The component where the error occurred.
        stage (str):
            The operation in which the error occurred.
        exc (Exception):
            The exception that caused the error.
    """
	identifier = node.get("id", (node.get("kind"), node.get("layer"), node.get("head")))
	_diagnostic(f"{identifier}: {stage}: {type(exc).__name__}: {exc}")


def _cache_get(cache: dict | ActivationCache | None, name: str) -> object | None:
	"""Retrieve an activation from the cache, if it exists, using its hook name.

    Args:
        cache (dict or ActivationCache or None):
            The cache containing the activations.
        name (str):
            The hook name used to retrieve the activation.

    Returns:
        object or None:
            The cached activation, or None if the cache is missing
            or the hook name is not found.
    """
	if cache is None:
		return None
	try:
		return cache[name]
	except KeyError:
		return None


def _attention_hook(model: HookedTransformer, layer: int) -> str:
	"""Identify the attention hook for the specified layer, checking
    its availability and the shape of W_O needed to reconstruct
    the attention head outputs.

    Args:
        model (HookedTransformer):
            The transformer model.
        layer (int):
            The index of the attention layer.

    Returns:
        str:
            The attention hook name ('blocks.{layer}.attn.hook_z').

    Raises:
        ValueError:
            If the hook is not registered in the model
            or W_O is not a tensor of shape (n_heads, d_head, d_model).
    """
	name = f"blocks.{layer}.attn.hook_z"
	hooks = getattr(model, "hook_dict", None)
	if hooks is not None and name not in hooks:
		raise ValueError(f"{name} is not registered on the model")
	w_o = getattr(model.blocks[layer].attn, "W_O", None)
	expected = (model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model)
	if not isinstance(w_o, torch.Tensor) or tuple(w_o.shape) != expected:
		raise ValueError(f"blocks.{layer}.attn.W_O must have shape {expected} [n_heads, d_head, d_model], got {getattr(w_o, 'shape', None)}")
	return name


def _prompt_data(model: HookedTransformer, clean_prompt: str) -> tuple[torch.Tensor, list[str]]:
	"""Tokenize the clean prompt to obtain both token IDs and
    their corresponding readable strings.

    Args:
        model (HookedTransformer):
            The transformer model.
        clean_prompt (str):
            The clean prompt to tokenize, including the BOS token.

    Returns:
        tuple[torch.Tensor, list[str]]:
            The token IDs of shape (1, seq_len) and their
            corresponding readable strings.

    Raises:
        ValueError:
            If the tokenized prompt is empty, has an invalid shape,
            or the number of token IDs and strings does not match.
    """
	tokens = model.to_tokens(clean_prompt, prepend_bos=True)
	if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
		raise ValueError("Expected one non-empty, BOS-inclusive clean prompt")
	strings = cast(list[str], model.to_str_tokens(tokens[0], prepend_bos=False))
	if len(strings) != tokens.shape[1]:
		raise ValueError("Clean prompt token IDs and strings have different lengths")
	return tokens, strings


def _decode_token(model: HookedTransformer, token_id: int) -> str:
	"""Convert a vocabulary token ID into a readable string,
    using alternative methods if the conversion fails.

    Args:
        model (HookedTransformer):
            The transformer model.
        token_id (int):
            The vocabulary token ID to convert to a string.

    Returns:
        str:
            The decoded token string, or the token ID as a string
            if the conversion fails.
    """
	try:
		pieces = model.to_str_tokens(torch.tensor([token_id], dtype=torch.long), prepend_bos=False)
		pieces = cast(list[str], pieces)
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


def _component_vector(
	model: HookedTransformer,
	node: dict,
	cache: dict | ActivationCache | tuple | list | None,
	position: int,
	batch_index: int,
	*,
	seq_len: int | None = None,
) -> torch.Tensor:
	"""Extract an MLP output or reconstruct the attention head output
    in the residual space.

    Args:
        model (HookedTransformer):
            The transformer model.
        node (dict):
            The component to inspect, identified by its kind and layer
            and by its head index for attention components.
        cache (dict or ActivationCache or tuple or list or None):
            The cache containing the activations or multiple caches to search.
        position (int):
            The token position to extract, including BOS.
        batch_index (int):
            The batch index containing the clean prompt.
        seq_len (int, optional):
            The clean prompt length, including BOS. Defaults to None.

    Returns:
        torch.Tensor:
            The component output vector in residual space, with
            shape (d_model,).

    Raises:
        IndexError:
            If a layer, head, or token position is outside the valid range.
        ValueError:
            If the component information or cached activations are invalid.
        KeyError:
            If required component information or activations are missing.
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
			block = cast("TransformerBlock", model.blocks[layer])
			w_o = cast(torch.Tensor, block.attn.W_O)
			z = value[batch_index, position, head].to(device=w_o.device, dtype=w_o.dtype)
			return z @ w_o[head]
		except Exception as exc:
			_failure(node, name, exc)
			errors.append(f"{name}: {type(exc).__name__}: {exc}")
	if errors:
		raise ValueError("No usable activation; " + "; ".join(errors))
	raise KeyError(f"Missing activations: {name}")


def _isolated_readouts(
	model: HookedTransformer,
	vectors: list[torch.Tensor],
	top_k: int,
	token_cache: dict[int, str],
) -> list[list[dict] | None]:
	"""Decode isolated component vectors to obtain the top-k tokens
    with their corresponding softmax probabilities.

    Args:
        model (HookedTransformer):
            The transformer model.
        vectors (list[torch.Tensor]):
            The component vectors in residual space, each of shape (d_model,).
        top_k (int):
            The number of highest-probability tokens to return per vector,
            limited to the vocabulary size.
        token_cache (dict[int, str]):
            A cache storing previously decoded token IDs and their strings,
            to avoid repeated conversions.

    Returns:
        list[list[dict] | None]:
            For each vector, a list of dictionaries containing the token
            rank, decoded string and probability, ordered by decreasing
            probability. Returns None for a vector if non-finite values
            are encountered.

    Raises:
        ValueError:
            If the vectors have an invalid shape or top_k is not positive.
        RuntimeError:
            If the input vectors cannot be stacked into a single tensor.
    """
	stacked_vectors = torch.stack(vectors)
	if (stacked_vectors.ndim != 2 or stacked_vectors.shape[-1] != int(model.cfg.d_model)):
		raise ValueError(f"Expected component vectors [batch, d_model={model.cfg.d_model}]")
	if top_k < 1:
		raise ValueError("top_k must be positive")
	readouts = []
	for start in range(0, stacked_vectors.shape[0], 16):
		batch = stacked_vectors[start:start + 16]
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
	model: HookedTransformer,
	graph: dict,
	clean_prompt: str,
	activation_cache: dict | ActivationCache | tuple | list | None = None,
	*,
	top_k: int = 5,
	allow_forward: bool = True,
	_prompt: tuple[torch.Tensor, list[str]] | None = None,
	_token_cache: dict[int, str] | None = None,
) -> dict:
	"""Compute the Component Logit Lens readouts for the MLP and attention
    components in a circuit graph.

    For each component, extract its output from the clean-prompt activations,
    using cached values when available or collecting missing activations
    through an additional forward pass if allowed. The resulting vectors
    are then decoded to obtain the top-k tokens and their probabilities,
    which are stored in the corresponding graph nodes.

    The readouts are computed independently of the patching contribution
    scores and do not represent the model's final predictions.

    Args:
        model (HookedTransformer):
            The transformer model.
        graph (dict):
            The circuit graph containing the components to decode.
            Each component requires a kind and layer, as well as a head
            index for attention components. If no position is specified,
            the last clean-prompt token is used.
        clean_prompt (str):
            The clean prompt used to obtain component activations.
        activation_cache (dict or ActivationCache or tuple or list, optional):
            Previously computed clean-prompt activations, or multiple
            caches to search. Defaults to None.
        top_k (int, optional):
            The number of highest-probability tokens to return per vector,
            limited to the vocabulary size. Defaults to 5.
        allow_forward (bool, optional):
            Whether to allow an additional forward pass to retrieve
            missing activations. Defaults to True.
        _prompt (tuple[torch.Tensor, list[str]], optional):
            Previously computed token IDs and strings, used to avoid
            repeated tokenization. Defaults to None.
        _token_cache (dict[int, str], optional):
            Previously decoded token strings, reused to avoid repeated
            conversions. Defaults to None.

    Returns:
        dict:
            The same graph, updated with a 'component_logit_lens' field
            for each MLP and attention node. Available readouts contain
            the selected position, its token, and the top-k tokens with
            their ranks and probabilities. Unavailable readouts are
            marked with 'available': False.
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
	"""Manage Component Logit Lens readouts during live circuit discovery.

    Prepare and store clean-prompt activations, reusing previously
    computed readouts as new components are discovered. Missing
    activations can be retrieved during initialization or final
    enrichment, avoiding additional forward passes during live updates.
    """

	def __init__(
		self,
		model: HookedTransformer,
		clean_prompt: str,
		activation_cache: dict | ActivationCache | None = None,
		*,
		top_k: int = 5,
	) -> None:
		"""Prepare and store clean-prompt activations for subsequent
        component decoding.

        Args:
            model (HookedTransformer):
                The transformer model.
            clean_prompt (str):
                The clean prompt used to obtain component activations.
            activation_cache (dict or ActivationCache, optional):
                Previously computed clean-prompt activations. Defaults to None.
            top_k (int, optional):
                The number of highest-probability tokens to return per
                component, limited to the vocabulary size. Defaults to 5.
        """
		self.model = model
		self.clean_prompt = clean_prompt
		self.top_k = top_k
		self.activations: dict[str, torch.Tensor] = {}
		self.readouts: dict[tuple, dict] = {}
		self.token_cache: dict[int, str] = {}
		self._final_retried: set[tuple] = set()
		self.prompt = None
		try:
			self.prompt = _prompt_data(model, clean_prompt)
		except Exception as exc:
			_diagnostic(f"clean prompt tokenization: {type(exc).__name__}: {exc}")
			return
		seq_len = len(self.prompt[1])

		def retain(name: str, source: dict | ActivationCache | None, layer: int, kind: str) -> bool:
			"""Retrieve and validate an activation, storing an independent
            copy for later use.

            Args:
                name (str):
                    The hook name of the activation to retrieve.
                source (dict or ActivationCache or None):
                    The cache containing the activation.
                layer (int):
                    The index of the component's layer.
                kind (str):
                    The component kind, either 'mlp' or 'attn'.

            Returns:
                bool:
                    True if the activation is successfully stored,
                    False if it is missing or invalid.
            """
			try:
				value = _cache_get(source, name)
				if value is None:
					return False
				node = {"kind": kind, "layer": layer, "head": 0}
				with torch.no_grad():
					_component_vector(model, node, {name: value}, seq_len - 1, 0, seq_len=seq_len)
				tensor_value = cast(torch.Tensor, value)
				self.activations[name] = tensor_value[:1, :seq_len].detach().clone()
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

	def _key(self, node: dict) -> tuple:
		"""Identify a component by its kind, layer, head and position, in order to enable
        reuse of previously computed readouts.

        Args:
            node (dict):
                The component to identify, using its kind, layer, head and token position.

        Returns:
            tuple:
                The (kind, layer, head, position) tuple, using the last
                clean-prompt token if no position is specified and
                prompt data is available.
        """
		position = node.get("position")
		if position is None and self.prompt is not None:
			position = len(self.prompt[1]) - 1
		return (node.get("kind"), node.get("layer"), node.get("head"), position)

	def enrich(self, graph: dict, *, final: bool = False) -> dict:
		"""Update the circuit graph with Component Logit Lens readouts,
        reusing cached results or decoding new components when needed.

        Args:
            graph (dict):
                The circuit graph containing the components to decode.
            final (bool, optional):
                Whether to allow an additional forward pass to retrieve
                missing activations and retry unavailable readouts.
                Defaults to False.

        Returns:
            dict:
                The same graph, updated with a 'component_logit_lens'
                field for each MLP and attention node.

        Notes:
            During live updates, only stored activations are used.
            Final enrichment should run after search and patching hooks
            have been removed.
        """
		pending: dict[tuple, dict] = {}
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
					top_k=self.top_k,
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
