"""Component Logit Lens for clean-prompt circuit visualization.

MLP outputs are obtained from hook_mlp_out, while attention head outputs
are reconstructed from hook_z using W_O[head], without the shared b_O bias.
These residual-space vectors are decoded through the final layer
normalization, unembedding and softmax to obtain the top-k tokens.

The resulting readouts represent isolated component outputs, rather than
the model's final predictions or patching contribution scores.

During circuit discovery, clean activations and previously computed
readouts are reused to avoid unnecessary computations.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, cast, Any

import torch

if TYPE_CHECKING:
	from transformer_lens import ActivationCache, HookedTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _diagnostic(message: str) -> None:
	"""Log a warning message, avoiding repeated messages.

    Args:
        message (str):
            The diagnostic text to log with the 'Component Logit Lens' prefix.
    """
	logger.warning("Component Logit Lens: %s", message)


def _decode_token(model: HookedTransformer, token_id: int) -> str:
	"""Convert a vocabulary token ID into a readable string.

    Args:
        model (HookedTransformer):
            The transformer model.
        token_id (int):
            The vocabulary token ID to convert to a string.

    Returns:
        str:
            The decoded token string, or the token ID as a string
            if the model has no tokenizer.
    """
	if hasattr(model, "tokenizer") and model.tokenizer is None:
		return str(token_id)
	tokens = cast(list[str], model.to_str_tokens(torch.tensor([token_id]), prepend_bos=False))
	return tokens[0]

def _component_vector(
    model: HookedTransformer,
    node: dict[str, Any],
    cache: dict[str, torch.Tensor] | ActivationCache | None,
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
        cache (dict or ActivationCache or None):
            The cache containing the clean prompt activations.
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
            If the component kind is not supported.
        KeyError:
            If required component information or activations are missing.
    """
    layer = int(node["layer"])
    kind = node["kind"]

    if not 0 <= layer < model.cfg.n_layers:
        raise IndexError(f"Invalid layer: {layer}")

    if position < 0 or batch_index < 0 or (seq_len is not None and position >= seq_len):
        raise IndexError("Invalid batch or token position")

    if kind == "mlp":
        name = f"blocks.{layer}.hook_mlp_out"

    elif kind == "attn":
        head = int(node["head"])
        name = f"blocks.{layer}.attn.hook_z"

        if not 0 <= head < model.cfg.n_heads:
            raise IndexError(f"Invalid attention head: {head}")

    else:
        raise ValueError(f"Unsupported node kind: {kind}")

    if cache is None:
        raise KeyError(f"Missing activation cache for {name}")
    value = cache[name]

    if kind == "mlp":
        return value[batch_index, position]

    z = value[batch_index, position, head]
    w_o = model.blocks[layer].attn.W_O[head]

    return z.to(w_o) @ w_o


def _isolated_readouts(
	model: HookedTransformer,
	vectors: list[torch.Tensor],
	top_k: int,
	token_cache: dict[int, str],
) -> list[list[dict] | None]:
	"""Decode isolated component vectors into top-k vocabulary tokens.

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
            If top_k is not positive.
    """
	if top_k < 1:
		raise ValueError("top_k must be positive")
	readouts = []
	for batch in torch.stack(vectors).split(16):
		normalized = model.ln_final(batch)
		logits = model.unembed(normalized)
		probabilities = torch.softmax(logits.float(), dim=-1)
		top_probs, top_ids = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
		finite = (
			torch.isfinite(logits).all(dim=-1)
			& torch.isfinite(probabilities).all(dim=-1)
		).tolist()

		for ids, probs, valid in zip(top_ids.tolist(), top_probs.tolist(), finite):
			if not valid:
				readouts.append(None)
				continue
			for token_id in ids:
				if token_id not in token_cache:
					token_cache[token_id] = _decode_token(model, token_id)

			component_tokens = []

			for rank, (token_id, probability) in enumerate(zip(ids, probs), start=1):
				component_tokens.append({
					"rank": rank,
					"token": token_cache[token_id],
					"probability": probability,
				})
			readouts.append(component_tokens)

	return readouts


@torch.no_grad()
def decode_circuit_components(
    model: HookedTransformer,
    graph: dict,
    clean_prompt: str,
    activation_cache: dict[str, torch.Tensor] | ActivationCache,
    *,
    top_k: int = 5,
    _prompt: tuple[torch.Tensor, list[str]] | None = None,
    _token_cache: dict[int, str] | None = None,
) -> dict:
    """Decode prepared clean activations for components in a circuit graph.
	
	For each MLP or attention component, extract its residual-space
	output from the supplied cache and compute the top-k vocabulary
	tokens and their softmax probabilities.
	
	The readouts represent isolated component outputs, independently
	of patching contribution scores and final model predictions.
	
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
        activation_cache (dict or ActivationCache):
            Prepared clean-prompt activations, with BOS-inclusive positions.
        top_k (int, optional):
            The number of highest-probability tokens to return per vector,
            limited to the vocabulary size. Defaults to 5.
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

    nodes = [
        node for node in graph.get("nodes", [])
        if node.get("kind") in ("mlp", "attn")
    ]

    if not nodes:
        return graph

    if _prompt is None:
        tokens = model.to_tokens(clean_prompt, prepend_bos=True)
        strings = cast(
            list[str],
            model.to_str_tokens(tokens[0], prepend_bos=False)
        )
    else:
        strings = _prompt[1]

    positions = [
        len(strings) - 1 if node.get("position") is None
        else int(node["position"])
        for node in nodes
    ]

    available: list[tuple[int, torch.Tensor]] = []
    for i, node in enumerate(nodes):
        node["component_logit_lens"] = {"available": False}
        try:
            vector = _component_vector(
                model, node, activation_cache, positions[i], 0,
                seq_len=len(strings)
            )
        except Exception as exc:
            _diagnostic(
                f"Component {node.get('id', i)}: activation extraction: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            available.append((i, vector))

    if not available:
        return graph

    try:
        readouts = _isolated_readouts(
            model,
            [vector for _, vector in available],
            top_k,
            {} if _token_cache is None else _token_cache,
        )
    except Exception as exc:
        component_ids = [nodes[i].get("id", i) for i, _ in available]
        _diagnostic(f"Component decoding {component_ids}: {type(exc).__name__}: {exc}")
        return graph

    for (i, _), top_tokens in zip(available, readouts):
        if top_tokens is None:
            continue

        nodes[i]["component_logit_lens"] = {
            "available": True,
            "convention": "isolated_component_softmax",
            "position": positions[i],
            "position_token": strings[positions[i]],
            "top_tokens": top_tokens,
        }

    return graph


class LiveComponentLogitLens:
	"""Prepare clean activations once and memoize readouts for one search job."""

	@torch.no_grad()
	def __init__(
		self,
		model: HookedTransformer,
		clean_prompt: str,
		activation_cache: dict | ActivationCache | None = None,
		*,
		top_k: int = 5,
	) -> None:
		"""Prepare and store clean-prompt activations for component decoding.
		Reuse the supplied cache when available, collecting missing
		activations with at most one additional forward pass.

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
		tokens = model.to_tokens(clean_prompt, prepend_bos=True)
		strings = cast(list[str], model.to_str_tokens(tokens[0], prepend_bos=False))
		self.prompt = tokens, strings
		self.activations: dict[str, torch.Tensor] = {}
		self.readouts: dict[tuple, dict] = {}
		self.token_cache: dict[int, str] = {}
		seq_len = len(strings)

		missing: dict[str, tuple[int, ...]] = {}
		hooks = getattr(model, "hook_dict", None)

		for layer in range(model.cfg.n_layers):
			for suffix, shape in (
				("hook_mlp_out", (model.cfg.d_model,)),
				("attn.hook_z", (model.cfg.n_heads, model.cfg.d_head)),
			):
				name = f"blocks.{layer}.{suffix}"
				if hooks is not None and name not in hooks:
					_diagnostic(f"{name} is not registered on the model")
					continue
				if suffix == "attn.hook_z":
					w_o = getattr(model.blocks[layer].attn, "W_O", None)
					if not isinstance(w_o, torch.Tensor) or w_o.shape != (*shape, model.cfg.d_model):
						_diagnostic(
							f"blocks.{layer}.attn.W_O must have shape {(*shape, model.cfg.d_model)}, "
							f"got {getattr(w_o, 'shape', None)}"
						)
						continue
				missing[name] = shape

		cache = activation_cache
		for collect_missing in (False, True):
			if not missing:
				break
			if collect_missing:
				names = set(missing)
				try:
					_, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
				except Exception as exc:
					_diagnostic(f"preparation capture: {type(exc).__name__}: {exc}")
					break
			for name, shape in list(missing.items()):
				try:
					value = cache[name] if cache is not None else None
				except KeyError:
					continue
				except Exception as exc:
					_diagnostic(f"preparation {name}: {type(exc).__name__}: {exc}")
					continue
				if (
					isinstance(value, torch.Tensor)
					and value.ndim == len(shape) + 2
					and value.shape[2:] == shape
					and value.shape[0] > 0
					and value.shape[1] >= seq_len
				):
					self.activations[name] = value[:1, :seq_len].detach().clone()
					del missing[name]

	def enrich(self, graph: dict) -> dict:
		"""Update the circuit graph with Component Logit Lens readouts,
        reusing cached results or decoding new components when needed.

        Args:
            graph (dict):
                The circuit graph containing the components to decode.

        Returns:
            dict:
                The same graph, updated with a 'component_logit_lens'
                field for each MLP and attention node.

        Notes:
            During live updates, only stored activations are used.
            Final enrichment should run after search and patching hooks
            have been removed.
        """
		occurrences: dict[tuple, list[dict]] = {}
		for node in graph.get("nodes", []):
			if node.get("kind") not in ("mlp", "attn"):
				continue
			position = len(self.prompt[1]) - 1 if node.get("position") is None else node["position"]
			key = (node["kind"], node["layer"], node.get("head"), position)
			occurrences.setdefault(key, []).append(node)

		pending = {
			key: nodes[0] for key, nodes in occurrences.items()
			if key not in self.readouts
		}

		if pending:
			decode_circuit_components(
				self.model, {"nodes": list(pending.values())}, self.clean_prompt,
				self.activations, top_k=self.top_k,
				_prompt=self.prompt, _token_cache=self.token_cache,
			)
			self.readouts.update((key, node["component_logit_lens"]) for key, node in pending.items())
		for key, nodes in occurrences.items():
			for node in nodes:
				node["component_logit_lens"] = self.readouts[key]
		return graph
