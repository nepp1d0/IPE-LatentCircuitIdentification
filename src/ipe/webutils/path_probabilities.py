"""Compare reference and patched final-output probabilities for isolated IPE paths.

Readouts use the final residuals already available during path evaluation,
without another transformer forward pass. Only the first prompt is decoded;
these probabilities are separate from the batch-averaged search metric.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import torch

from ipe.nodes import EMBED_Node, FINAL_Node
from ipe.webutils.component_logit_lens import _decode_token
from ipe.webutils.serialization import grid_id

if TYPE_CHECKING:
	from transformer_lens import HookedTransformer
	from ipe.nodes import Node

logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 5


@torch.no_grad()
def final_probabilities(model: HookedTransformer, residual: torch.Tensor) -> torch.Tensor:
	"""Decode a final residual vector using the model's output head.

    Apply final normalization when configured, unembedding and optional
    output logit soft-capping, followed by a full-vocabulary softmax.
    The input is a final residual, not an isolated component output.

    Args:
        model (HookedTransformer):
            The transformer model.
        residual (torch.Tensor):
            The final residual vector at one prediction position,
            with shape (d_model,).

    Returns:
        torch.Tensor:
            Float32 token probabilities with shape (d_vocab,).

    Raises:
        ValueError:
            If the residual or probabilities are not finite one-dimensional
            tensors.
    """
	if residual.ndim != 1 or not torch.isfinite(residual).all():
		raise ValueError("Expected a finite final residual vector")
	normalized = model.ln_final(residual) if model.cfg.normalization_type is not None else residual
	logits = model.unembed(normalized)
	cap = getattr(model.cfg, "output_logits_soft_cap", 0.0)
	if cap > 0:
		logits = cap * torch.tanh(logits / cap)
	probabilities = torch.softmax(logits.float(), dim=-1)
	if probabilities.ndim != 1 or not torch.isfinite(probabilities).all():
		raise ValueError("Invalid final output probabilities")
	return probabilities


def select_changed_tokens(reference: torch.Tensor, patched: torch.Tensor, *, top_k: int = DEFAULT_TOP_K) -> list[int]:
	"""Select the tokens with the largest absolute probability changes.

    Args:
        reference (torch.Tensor):
            Reference token probabilities with shape (d_vocab,).
        patched (torch.Tensor):
            Patched token probabilities for the same vocabulary.
        top_k (int, optional):
            The number of tokens to select, limited to the vocabulary size.
            Defaults to 5.

    Returns:
        list[int]:
            Selected token IDs, ordered by decreasing reference probability.
            Ties in selection and ordering use the smaller token ID.

    Raises:
        ValueError:
            If top_k is not positive, the vectors are empty or have invalid
            shapes, or a probability is non-finite or outside [0, 1].
    """
	if top_k < 1 or reference.ndim != 1 or reference.shape != patched.shape or not reference.numel():
		raise ValueError("Expected matching vocabulary vectors and positive top_k")
	for probabilities in (reference, patched):
		if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
			raise ValueError("Invalid probabilities")
	k = min(top_k, reference.numel())
	changes = (patched - reference).abs()
	boundary = torch.topk(changes, k).values[-1]
	greater = torch.where(changes > boundary)[0]
	equal = torch.where(changes == boundary)[0][:k - greater.numel()]
	selected = torch.cat((greater, equal)).tolist()
	values = reference[selected].tolist()
	return [token_id for _, token_id in sorted(zip(values, selected), key=lambda item: (-item[0], item[1]))]


class PathProbabilityCollector:
	"""Collect final-output probability comparisons during one circuit search.

    Decode a shared reference distribution once and store selected token
    comparisons on the original embedding leaves. Patched residuals and
    full patched distributions are not retained.
    """

	def __init__(self, model: HookedTransformer, root: Node, *, context_id: str, method: str, mode: str, reference_prompt: str,
			 target: str | None = None, counterfactual: str | None = None, top_k: int = DEFAULT_TOP_K) -> None:
		"""Prepare the reference probabilities and context for path comparisons.

        Args:
            model (HookedTransformer):
                The transformer model.
            root (Node):
                The search root.
            context_id (str):
                The identifier linking comparisons to this search context.
            method (str):
                The search method, either 'tree' or 'path'. The observer must
                be used with isolated path evaluations, not joint scoring.
            mode (str):
                The patching direction, 'noising' or 'denoising'.
            reference_prompt (str):
                The first prompt in the cached reference batch: the clean
                prompt for noising or counterfactual prompt for denoising.
            target (str, optional):
                The target token string, recorded only if it maps to a single
                token. It does not affect token selection. Defaults to None.
            counterfactual (str, optional):
                The counterfactual token string, recorded only if it maps to
                a single token. It does not affect selection. Defaults to None.
            top_k (int, optional):
                The number of tokens to retain per path, limited to
                the vocabulary size. Defaults to 5.
        """
		self.model = model
		self.root = root
		self.token_cache: dict[int, str] = {}
		self.reference: torch.Tensor | None = None
		self.context: dict = {
			"id": context_id, "method": method, "mode": mode, "top_k": top_k,
			"reference_label": "Counterfactual" if mode == "denoising" else "Clean",
			"reference_prompt": reference_prompt, "prompt_index": 0,
			"target_token_id": None, "counterfactual_token_id": None,
			"available": False,
		}
		try:
			if method not in ("tree", "path") or not isinstance(root, FINAL_Node):
				raise ValueError("No supported isolated final-output intervention")
			if top_k < 1:
				raise ValueError("top_k must be positive")
			tokens = model.to_str_tokens(reference_prompt, prepend_bos=True)
			position = len(tokens) - 1
			residual = root.msg_cache[root.input_name]
			# Unaligned/padded batches cannot be labelled as this prompt's prediction.
			if residual.ndim != 3 or residual.shape[1] != len(tokens) or root.position not in (None, position):
				raise ValueError("Reference prompt is not aligned with the cached prediction position")
			self.context.update({"position": position, "position_token": tokens[position],
				"patch_type": root.patch_type})
			# Optional roles are never inferred from decoded display strings.
			for key, token in (("target_token_id", target), ("counterfactual_token_id", counterfactual)):
				if token:
					try:
						self.context[key] = int(model.to_single_token(token))
					except (ValueError, AssertionError):
						pass
			self.reference = final_probabilities(model, residual[0, position])
			self.context["available"] = True
		except Exception as exc:
			self.context["reason"] = str(exc)
			logger.warning("Path probabilities unavailable: %s", exc)

	@torch.no_grad()
	def observe(self, path: list[Node], patched_residual: torch.Tensor) -> None:
		"""Store a path's selected reference and patched token probabilities.

        Reuse the residual supplied by evaluate_path without re-propagating
        the path or changing its contribution score.

        Args:
            path (list[Node]):
               The sequence of nodes representing the path to be evaluated.
            patched_residual (torch.Tensor):
                The residual passed to the search metric, with shape
                (batch, seq_len, d_model). Only the first prompt's final
                prediction position is decoded.
        """
		if not path or not isinstance(path[0], EMBED_Node):
			return
		payload = {"available": False, "context_id": self.context["id"],
			"node_ids": [grid_id(node) for node in path]}
		path[0].path_probability_comparison = payload
		if self.reference is None:
			payload["reason"] = self.context.get("reason", "Reference unavailable")
			return
		try:
			if path[-1] is not self.root or patched_residual.shape != self.root.msg_cache[self.root.input_name].shape:
				raise ValueError("Mismatched final-output intervention")
			position = self.context["position"]
			patched = final_probabilities(self.model, patched_residual[0, position])
			selected = select_changed_tokens(self.reference, patched, top_k=self.context["top_k"])
			reference_values = self.reference[selected].tolist()
			patched_values = patched[selected].tolist()
			tokens = []
			for token_id, reference, changed in zip(selected, reference_values, patched_values):
				if token_id not in self.token_cache:
					self.token_cache[token_id] = _decode_token(self.model, token_id)
				tokens.append({"token_id": token_id, "token": self.token_cache[token_id],
					"reference": reference, "patched": changed, "delta": changed - reference})
			payload.update({"available": True, "convention": "isolated_path_final_output_softmax",
				"position": position, "vocab_size": self.reference.numel(), "tokens": tokens})
		except Exception as exc:
			payload["reason"] = str(exc)
			logger.warning("Path probability readout failed: %s", exc)
