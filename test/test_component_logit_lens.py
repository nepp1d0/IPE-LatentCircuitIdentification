"""Component readouts, live enrichment, and probability transport without downloads."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from ipe.webutils.component_logit_lens import (
	_component_vector, _diagnostic, decode_circuit_components, LiveComponentLogitLens,
)


PROMPT_TOKENS = ["<BOS>", "The", " capital", " of", " Italy", " is"]


class FakeModel:
	def __init__(self):
		self.cfg = SimpleNamespace(
			n_layers=1, n_heads=2, d_head=3, d_model=3, use_attn_result=False, device="cpu"
		)
		self.projection = torch.tensor([
			[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
			[1.0, 2.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 0.0],
			[1.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 9.0],
		])
		self.z = torch.zeros(1, len(PROMPT_TOKENS), 2, 3)
		self.z[0, 4, 0] = torch.tensor([1.0, 0.0, 0.0])
		self.z[0, 4, 1] = torch.tensor([0.0, 0.0, 1.0])
		self.run_count = 0
		self.blocks = [SimpleNamespace(attn=SimpleNamespace(W_O=torch.eye(3).repeat(2, 1, 1)))]

	def to_tokens(self, prompt, prepend_bos=True):
		return torch.arange(len(PROMPT_TOKENS)).unsqueeze(0)

	def to_str_tokens(self, value, prepend_bos=True):
		if isinstance(value, str):
			return PROMPT_TOKENS
		if isinstance(value, torch.Tensor) and value.numel() == 1:
			return [f"token-{int(value.reshape(-1)[0])}"]
		return PROMPT_TOKENS

	def reset_hooks(self):
		pass

	def run_with_cache(self, tokens, names_filter):
		self.run_count += 1
		assert not names_filter("blocks.0.attn.hook_result")
		cache = {}
		if names_filter("blocks.0.attn.hook_z"):
			cache["blocks.0.attn.hook_z"] = self.z
		return torch.zeros(1, len(PROMPT_TOKENS), 10), cache

	def ln_final(self, vector):
		return vector

	def unembed(self, vector):
		return vector @ self.projection


def test_decodes_mlp_positions_and_individual_attention_heads():
	model = FakeModel()
	mlp = torch.zeros(1, len(PROMPT_TOKENS), 3)
	mlp[0, 0] = torch.tensor([1.0, 0.0, 0.0])
	mlp[0, 4] = torch.tensor([1.0, 0.0, 0.0])
	mlp[0, 5] = torch.tensor([0.0, 1.0, 0.0])
	graph = {"nodes": [
		{"id": "mlp:0:0", "kind": "mlp", "layer": 0, "position": 0},
		{"id": "mlp:0:4", "kind": "mlp", "layer": 0, "position": 4},
		{"id": "mlp:0:5", "kind": "mlp", "layer": 0, "position": 5},
		{"id": "attn:0:0:4", "kind": "attn", "layer": 0, "head": 0, "position": 4},
		{"id": "attn:0:1:4", "kind": "attn", "layer": 0, "head": 1, "position": 4},
	], "edges": [{"source": "mlp:0:0", "target": "mlp:0:4", "contribution": 0.25}], "metadata": "preserved"}
	original_ids = [node["id"] for node in graph["nodes"]]

	decoded = decode_circuit_components(
		model, graph, "The capital of Italy is",
		activation_cache={"blocks.0.hook_mlp_out": mlp},
	)
	by_id = {node["id"]: node["component_logit_lens"] for node in decoded["nodes"]}
	assert decoded is graph
	assert [node["id"] for node in decoded["nodes"]] == original_ids
	assert decoded["edges"] == [{"source": "mlp:0:0", "target": "mlp:0:4", "contribution": 0.25}]
	assert decoded["metadata"] == "preserved"
	assert by_id["mlp:0:0"]["position_token"] == "<BOS>"

	assert by_id["mlp:0:4"]["position"] == 4
	assert by_id["mlp:0:4"]["position_token"] == " Italy"
	assert by_id["mlp:0:4"]["top_tokens"][0]["token"] == "token-0"
	assert by_id["mlp:0:5"]["top_tokens"][0]["token"] == "token-2"
	assert by_id["attn:0:0:4"]["top_tokens"][0]["token"] == "token-0"
	assert by_id["attn:0:1:4"]["top_tokens"][0]["token"] == "token-9"
	for result in by_id.values():
		assert result["available"]
		assert len(result["top_tokens"]) == 5
		assert all(0.0 <= item["probability"] <= 1.0 for item in result["top_tokens"])
	assert model.run_count == 1  # the shared, filtered per-head z pass
	assert model.cfg.use_attn_result is False  # the shared model setting is unchanged


def test_captures_z_and_reconstructs_a_head():
	model = FakeModel()
	model.z[0, 4, 0] = torch.tensor([0.0, 1.0, 0.0])
	graph = {"nodes": [
		{"id": "attn:0:0:4", "kind": "attn", "layer": 0, "head": 0, "position": 4},
	], "edges": []}

	decode_circuit_components(model, graph, "The capital of Italy is")

	readout = graph["nodes"][0]["component_logit_lens"]
	assert readout["available"]
	assert readout["top_tokens"][0]["token"] == "token-2"
	assert model.run_count == 1
	assert model.cfg.use_attn_result is False


def test_reuses_cached_z_at_the_requested_head_and_position():
	model = FakeModel()
	model.z[0, 3, 1] = torch.tensor([0.0, 0.0, 1.0])
	graph = {"nodes": [
		{"id": "attn:0:1:3", "kind": "attn", "layer": 0, "head": 1, "position": 3},
	], "edges": []}

	decode_circuit_components(
		model, graph, "The capital of Italy is",
		activation_cache={"blocks.0.attn.hook_z": model.z}, allow_forward=False,
	)

	readout = graph["nodes"][0]["component_logit_lens"]
	assert readout["available"]
	assert readout["position"] == 3
	assert readout["top_tokens"][0]["token"] == "token-9"
	assert model.run_count == 0


def test_reuses_existing_component_activations_without_another_forward_pass():
	model = FakeModel()
	mlp = torch.zeros(1, len(PROMPT_TOKENS), 3)
	mlp[0, 2] = torch.tensor([0.0, 0.0, 1.0])
	graph = {"nodes": [
		{"id": "mlp:0:2", "kind": "mlp", "layer": 0, "position": 2},
	], "edges": []}

	decode_circuit_components(
		model, graph, "The capital of Italy is",
		activation_cache={"blocks.0.hook_mlp_out": mlp},
	)

	assert graph["nodes"][0]["component_logit_lens"]["available"]
	assert model.run_count == 0


def test_live_events_and_final_graph_reuse_readouts(monkeypatch):
	model = FakeModel()
	mlp = torch.ones(1, len(PROMPT_TOKENS), 3)
	# Reuse MLP output, capture missing head z once before the search.
	lens = LiveComponentLogitLens(
		model, "The capital of Italy is", {"blocks.0.hook_mlp_out": mlp}
	)
	assert model.run_count == 1
	decoded_batches = []
	original_unembed = model.unembed

	def counted_unembed(vectors):
		decoded_batches.append(vectors.shape[0])
		return original_unembed(vectors)

	monkeypatch.setattr(model, "unembed", counted_unembed)
	# Once search hooks are active, neither forwards nor resets are permitted.
	def forbidden(*args, **kwargs):
		raise AssertionError("Live enrichment must not execute or reset the model")

	monkeypatch.setattr(model, "run_with_cache", forbidden)
	monkeypatch.setattr(model, "reset_hooks", forbidden)
	nodes = [
		{"id": "mlp:0:4", "kind": "mlp", "layer": 0, "position": 4},
		{"id": "attn:0:1:4", "kind": "attn", "layer": 0, "head": 1, "position": 4},
	]
	# Duplicate admissions, path events and completed graphs share the same keys.
	lens.enrich({"nodes": nodes + [dict(nodes[0])]})
	assert decoded_batches == [2]
	assert all(n["component_logit_lens"]["available"] for n in nodes)
	repeated = {"nodes": [dict(n) for n in nodes]}
	lens.enrich(repeated)
	lens.enrich(repeated, final=True)
	assert decoded_batches == [2]
	assert repeated["nodes"][0]["component_logit_lens"] is nodes[0]["component_logit_lens"]


def test_live_acdc_snapshots_use_last_position_and_are_job_local():
	model = FakeModel()
	cache = {
		"blocks.0.hook_mlp_out": torch.ones(1, len(PROMPT_TOKENS), 3),
		"blocks.0.attn.hook_z": model.z,
	}
	first = LiveComponentLogitLens(model, "The capital of Italy is", cache)
	second = LiveComponentLogitLens(model, "Another prompt", cache)
	graph = {"nodes": [
		{"id": "attn:0:1:*", "kind": "attn", "layer": 0, "head": 1, "position": None},
	]}
	first.enrich(graph)
	assert graph["nodes"][0]["component_logit_lens"]["position"] == 5
	assert len(first.readouts) == 1
	assert second.readouts == {}
	assert model.run_count == 0


def test_missing_live_activations_do_not_forward_and_final_can_retry(monkeypatch):
	model = FakeModel()
	lens = LiveComponentLogitLens(model, "The capital of Italy is")
	lens.activations.clear()  # Simulate failed preparation.
	graph = {"nodes": [
		{"id": "attn:0:1:4", "kind": "attn", "layer": 0, "head": 1, "position": 4},
	]}
	passes = model.run_count
	lens.enrich(graph)
	lens.enrich(graph)
	assert model.run_count == passes
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	lens.enrich(graph, final=True)
	assert graph["nodes"][0]["component_logit_lens"]["available"]
	assert model.run_count == passes + 1


@pytest.mark.parametrize("shape", [(1, 6, 3), (1, 6, 1, 3), (1, 2, 2, 3), (0, 6, 2, 3), (1, 6, 2, 7)])
def test_invalid_z_is_replaced_by_valid_z_from_capture(shape):
	model = FakeModel()
	graph = {"nodes": [{"id": "attn:0:1:4", "kind": "attn", "layer": 0, "head": 1, "position": 4}]}
	decode_circuit_components(
		model, graph, "The capital of Italy is",
		activation_cache={"blocks.0.attn.hook_z": torch.zeros(shape)},
	)
	assert graph["nodes"][0]["component_logit_lens"]["top_tokens"][0]["token"] == "token-9"
	assert model.run_count == 1


def test_projects_with_the_correct_head_weights_dtype_and_residual_width():
	model = FakeModel()
	model.cfg.d_head = 2
	w_o = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
	model.blocks[0].attn.W_O = w_o
	model.blocks[0].attn.b_O = torch.full((3,), 100.)
	z = torch.zeros(2, 6, 2, 2, dtype=torch.float64)
	z[1, 4, 1] = torch.tensor([2., 3.])
	vector = _component_vector(model, {"kind": "attn", "layer": 0, "head": 1},
		{"blocks.0.attn.hook_z": z}, 4, 1, seq_len=6)
	torch.testing.assert_close(vector, torch.tensor([44., 49., 54.]))
	assert vector.shape == (model.cfg.d_model,)
	assert vector.dtype == w_o.dtype
	assert model.run_count == 0


@pytest.mark.parametrize("field,value", [("layer", -1), ("layer", 1), ("head", -1), ("head", 2)])
def test_rejects_invalid_layer_and_head(field, value):
	model = FakeModel()
	node = {"kind": "attn", "layer": 0, "head": 0, field: value}
	with pytest.raises(IndexError):
		_component_vector(model, node, {"blocks.0.attn.hook_z": model.z}, 4, 0, seq_len=6)
	graph = {"nodes": [dict(node, position=4)]}
	decode_circuit_components(model, graph, "prompt")
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	assert model.run_count == 0


@pytest.mark.parametrize("position,batch", [(-1, 0), (6, 0), (4, -1), (4, 1)])
def test_rejects_invalid_token_and_batch_indices(position, batch):
	model = FakeModel()
	with pytest.raises((IndexError, ValueError)):
		_component_vector(model, {"kind": "attn", "layer": 0, "head": 0},
			{"blocks.0.attn.hook_z": model.z}, position, batch, seq_len=6)


@pytest.mark.parametrize("shape", [(6, 3), (2, 3, 4), (1, 3, 3), (2, 4, 3), None])
def test_incompatible_projection_is_unavailable_without_capture(shape, caplog):
	model = FakeModel()
	model.blocks[0].attn.W_O = None if shape is None else torch.zeros(shape)
	graph = {"nodes": [{"id": "bad-projection", "kind": "attn", "layer": 0, "head": 0, "position": 4}]}
	decode_circuit_components(model, graph, "prompt", {"blocks.0.attn.hook_z": model.z})
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	assert "blocks.0.attn.W_O must have shape" in caplog.text
	assert model.run_count == 0


@pytest.mark.parametrize("activation", [None, "not a tensor", torch.zeros(1, 5, 2, 3)])
def test_missing_or_invalid_z_never_uses_result_as_a_substitute(activation):
	model = FakeModel()
	cache = {"blocks.0.attn.hook_z": activation, "blocks.0.attn.hook_result": torch.ones(1, 6, 2, 3)}
	graph = {"nodes": [{"kind": "attn", "layer": 0, "head": 0, "position": 4}]}
	decode_circuit_components(model, graph, "prompt", cache, allow_forward=False)
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	assert model.run_count == 0


def test_live_reuses_tokenization_strings_and_owns_only_clean_row(monkeypatch):
	model = FakeModel()
	counts = {"tokens": 0, "strings": 0}
	original_tokens = model.to_tokens
	original_strings = model.to_str_tokens
	def tokens(*args, **kwargs):
		counts["tokens"] += 1
		return original_tokens(*args, **kwargs)
	def strings(value, **kwargs):
		if isinstance(value, torch.Tensor) and value.numel() == 1:
			counts["strings"] += 1
		return original_strings(value, **kwargs)
	monkeypatch.setattr(model, "to_tokens", tokens)
	monkeypatch.setattr(model, "to_str_tokens", strings)
	mlp = torch.zeros(3, 8, 3)
	mlp[0, :6, 0] = 1
	mlp[1:, :, 2] = 1  # other prompts must not affect the displayed clean row
	lens = LiveComponentLogitLens(model, "The capital of Italy is", {
		"blocks.0.hook_mlp_out": mlp, "blocks.0.attn.hook_z": model.z,
		"blocks.0.attn.hook_result": torch.ones(3, 8, 2, 3),
	})
	assert set(lens.activations) == {"blocks.0.hook_mlp_out", "blocks.0.attn.hook_z"}
	assert lens.activations["blocks.0.hook_mlp_out"].shape == (1, 6, 3)
	mlp.zero_()  # the search may mutate its cache after initialization
	for position in (4, 5, None):
		graph = {"nodes": [{"kind": "mlp", "layer": 0, "position": position}]}
		lens.enrich(graph)
		assert graph["nodes"][0]["component_logit_lens"]["top_tokens"][0]["token"] == "token-0"
	assert counts == {"tokens": 1, "strings": 5}
	assert len(lens.readouts) == 2  # None aliases the final BOS-inclusive position
	assert model.run_count == 0


def test_diagnostics_distinguish_missing_shape_and_unexpected_errors(caplog):
	_diagnostic.cache_clear()
	model = FakeModel()
	node = {"id": "diagnostic-head", "kind": "attn", "layer": 0, "head": 1, "position": 4}
	graph = {"nodes": [node]}
	decode_circuit_components(model, graph, "prompt", allow_forward=False)
	assert "Missing activations" in caplog.text
	assert "blocks.0.attn.hook_z" in caplog.text
	cache = {"blocks.0.attn.hook_z": torch.zeros(1, 6, 1, 3)}
	decode_circuit_components(model, graph, "prompt", cache, allow_forward=False)
	assert "Incompatible activation shape" in caplog.text
	class BrokenCache:
		def __getitem__(self, name):
			raise RuntimeError("broken cache backend")
	decode_circuit_components(model, graph, "prompt", BrokenCache(), allow_forward=False)
	assert "RuntimeError: broken cache backend" in caplog.text
	assert "diagnostic-head" in caplog.text
	count = len(caplog.records)
	decode_circuit_components(model, graph, "prompt", BrokenCache(), allow_forward=False)
	assert len(caplog.records) == count
	assert node["component_logit_lens"] == {"available": False}


def test_failed_final_retry_is_memoized(monkeypatch):
	model = FakeModel()
	lens = LiveComponentLogitLens(model, "prompt")
	lens.activations.clear()
	def broken_forward(*args, **kwargs):
		model.run_count += 1
		raise RuntimeError("capture failed")
	monkeypatch.setattr(model, "run_with_cache", broken_forward)
	graph = {"nodes": [{"kind": "attn", "layer": 0, "head": 0, "position": 4}]}
	lens.enrich(graph)
	before = model.run_count
	lens.enrich(graph, final=True)
	lens.enrich(graph, final=True)
	assert model.run_count == before + 1
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}


def test_missing_z_hook_is_unavailable_without_changing_config(caplog):
	model = FakeModel()
	model.cfg.use_attn_result = True
	model.hook_dict = {"blocks.0.hook_mlp_out": object()}
	lens = LiveComponentLogitLens(model, "prompt", {"blocks.0.hook_mlp_out": torch.ones(1, 6, 3)})
	graph = {"nodes": [{"kind": "attn", "layer": 0, "head": 1, "position": 4}]}
	lens.enrich(graph)
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	assert "blocks.0.attn.hook_z is not registered" in caplog.text
	assert model.cfg.use_attn_result is True
	assert model.run_count == 0


def test_decoding_failure_logs_component_and_keeps_graph_usable(monkeypatch, caplog):
	model = FakeModel()
	def broken_unembed(vectors):
		raise RuntimeError("unembedding failed")
	monkeypatch.setattr(model, "unembed", broken_unembed)
	node = {"id": "failed-readout", "kind": "attn", "layer": 0, "head": 0, "position": 4}
	graph = {"nodes": [node], "edges": [{"source": "a", "target": "b"}]}
	assert decode_circuit_components(model, graph, "prompt", {"blocks.0.attn.hook_z": model.z}) is graph
	assert node["component_logit_lens"] == {"available": False}
	assert graph["edges"] == [{"source": "a", "target": "b"}]
	assert "failed-readout: ln_final/unembed readout: RuntimeError: unembedding failed" in caplog.text


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage", ["activation", "ln_final", "logits", "probabilities"])
def test_nonfinite_readouts_are_unavailable_without_affecting_valid_rows(monkeypatch, stage, invalid):
	model = FakeModel()
	mlp = torch.zeros(1, len(PROMPT_TOKENS), 3)
	mlp[0, :, 0] = 1
	if stage == "activation":
		mlp[0, 0, 0] = invalid
	else:
		owner, name = {
			"ln_final": (model, "ln_final"),
			"logits": (model, "unembed"),
			"probabilities": (torch, "softmax"),
		}[stage]
		original = getattr(owner, name)
		def nonfinite(*args, **kwargs):
			result = original(*args, **kwargs).clone()
			# Include -Inf logits, which can otherwise yield finite softmax values.
			result[0, -1] = invalid
			return result
		monkeypatch.setattr(owner, name, nonfinite)
	graph = {"nodes": [
		{"kind": "mlp", "layer": 0, "position": position} for position in (0, 1)
	]}
	decode_circuit_components(model, graph, "prompt", {"blocks.0.hook_mlp_out": mlp})
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}
	valid = graph["nodes"][1]["component_logit_lens"]
	assert valid["available"]
	expected_probs, expected_ids = model.projection[0].softmax(dim=-1).topk(5)
	assert [item["token"] for item in valid["top_tokens"]] == [f"token-{i}" for i in expected_ids.tolist()]
	assert [item["probability"] for item in valid["top_tokens"]] == expected_probs.tolist()
	assert model.run_count == 0


@pytest.mark.parametrize("kv_heads", [None, 1])
def test_transformer_lens_projection_and_scoped_hook_safety(kv_heads):
	transformer_lens = pytest.importorskip("transformer_lens")
	model = transformer_lens.HookedTransformer(transformer_lens.HookedTransformerConfig(
		n_layers=1, n_heads=2, n_key_value_heads=kv_heads, d_model=10, d_head=4,
		d_mlp=16, n_ctx=8, d_vocab=16, act_fn="relu", device="cpu", seed=42,
	))
	model.eval()
	tokens = torch.tensor([[0, 1, 2]])
	names = lambda name: name.endswith(("attn.hook_z", "hook_attn_out"))
	with torch.no_grad():
		_, cache = model.run_with_cache(tokens, names_filter=names)
	z = cache["blocks.0.attn.hook_z"]
	w_o = model.blocks[0].attn.W_O
	# Independent elementwise expression for the per-head projection.
	expected = (z.unsqueeze(-1) * w_o).sum(-2)
	torch.testing.assert_close(expected.sum(2) + model.blocks[0].attn.b_O, cache["blocks.0.hook_attn_out"])
	for head in range(2):
		node = {"kind": "attn", "layer": 0, "head": head}
		actual = _component_vector(model, node, cache, 2, 0, seq_len=3)
		torch.testing.assert_close(actual, expected[0, 2, head])
	# Let the real run_with_cache install/remove its own hooks inside an outer
	# search-like context. decode must preserve both this hook and its context.
	calls = []
	hook = model.hook_dict["blocks.0.attn.hook_z"]
	with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", lambda value, hook: calls.append(value.shape))]):
		hook.ctx["search-owned"] = True
		graph = {"nodes": [{"kind": "attn", "layer": 0, "head": 1, "position": 2}]}
		decode_circuit_components(model, graph, "unused", _prompt=(tokens, ["BOS", "one", "two"]))
		assert calls == [torch.Size([1, 3, 2, 4])]
		assert len(hook.fwd_hooks) == 1
		assert hook.ctx["search-owned"] is True
		assert graph["nodes"][0]["component_logit_lens"]["available"]
	assert not hook.fwd_hooks
	assert model.cfg.use_attn_result is False


# Clean-prompt decoder contract and isolation between saved/live results.

def _make_readout_contract():
	"""Use distinct batch rows, nontrivial normalization, and a shared-bias sentinel."""
	model = SimpleNamespace(cfg=SimpleNamespace(n_layers=1, n_heads=1, d_head=2, d_model=2))
	model.to_tokens = Mock(return_value=torch.tensor([[0, 1, 2]]))
	model.to_str_tokens = lambda ids, **kwargs: [f" token{int(i)}" for i in ids]
	model.blocks = [SimpleNamespace(attn=SimpleNamespace(W_O=torch.eye(2).unsqueeze(0), b_O=torch.tensor([999., 999.])))]
	model.ln_final = Mock(side_effect=lambda vector: vector / 2)
	weights = torch.tensor([[1., 2., -1., 0., 3., -2.], [0., -1., 2., 3., -2., 1.]])
	model.unembed = Mock(side_effect=lambda vector: vector @ weights)
	model.run_with_cache = Mock(side_effect=AssertionError("No forward allowed"))
	activation = torch.tensor([[[1., 0.], [0., 1.], [2., -1.]], [[99., 99.], [99., 99.], [99., 99.]]])
	cache = {"blocks.0.hook_mlp_out": activation, "blocks.0.attn.hook_z": activation.unsqueeze(2)}
	return model, cache, weights


@pytest.mark.parametrize("method", ["tree", "path", "acdc"])
def test_decoder_uses_first_clean_row_last_position_and_normalized_softmax(method):
	model, cache, weights = _make_readout_contract()
	nodes = [
		dict(id="mlp:0:*", kind="mlp", layer=0, position=None),
		dict(id="attn:0:0:*", kind="attn", layer=0, head=0, position=None, kv_position=0, variant="KV"),
		dict(id="mlp:0:1", kind="mlp", layer=0, position=1), dict(id="embed:*", kind="embed"),
	]
	graph = {"nodes": nodes, "edges": [], "method": method}
	assert decode_circuit_components(model, graph, "first clean prompt", cache, allow_forward=False) is graph
	expected, ids = torch.softmax((cache["blocks.0.hook_mlp_out"][0, 2] / 2 @ weights).float(), -1).topk(5)
	lens = nodes[0]["component_logit_lens"]
	assert lens == nodes[1]["component_logit_lens"]  # no shared attention bias; KV position is not readout position
	assert lens["position"] == 2 and lens["position_token"] == " token2"
	assert [t["probability"] for t in lens["top_tokens"]] == pytest.approx(expected.tolist())
	assert [t["token"] for t in lens["top_tokens"]] == [f" token{i}" for i in ids.tolist()]
	assert nodes[2]["component_logit_lens"]["position"] == 1
	assert "component_logit_lens" not in nodes[3]
	model.to_tokens.assert_called_with("first clean prompt", prepend_bos=True)
	model.run_with_cache.assert_not_called()


def test_live_readouts_share_q_kv_payloads_but_remain_job_local():
	model, cache, _ = _make_readout_contract()
	lens = LiveComponentLogitLens(model, "first clean prompt", cache)
	def graph(variant):
		return {"nodes": [dict(id="attn:0:0:*", kind="attn", layer=0, head=0, position=None, variant=variant, kv_position=0)], "edges": []}
	q, kv = graph("Q"), graph("KV")
	lens.enrich(q)
	count = model.unembed.call_count
	lens.enrich(kv)
	assert q["nodes"][0]["component_logit_lens"] is kv["nodes"][0]["component_logit_lens"]
	assert model.unembed.call_count == count
	model2, cache2, _ = _make_readout_contract()
	cache2["blocks.0.attn.hook_z"] = -cache2["blocks.0.attn.hook_z"]
	other = graph("Q")
	LiveComponentLogitLens(model2, "other saved prompt", cache2).enrich(other)
	assert other["nodes"][0]["component_logit_lens"] != q["nodes"][0]["component_logit_lens"]
	model.run_with_cache.assert_not_called()


def test_nonfinite_last_token_readout_is_unavailable():
	model, cache, _ = _make_readout_contract()
	cache["blocks.0.hook_mlp_out"][0, 2, 0] = float("nan")
	graph = {"nodes": [dict(id="mlp:0:*", kind="mlp", layer=0, position=None)]}
	decode_circuit_components(model, graph, "prompt", cache, allow_forward=False)
	assert graph["nodes"][0]["component_logit_lens"] == {"available": False}


# JSON transport: invalid component probabilities must not reach the browser.

@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_server_scrubs_invalid_component_readouts_without_mutating_payload(invalid):
	from visualization.server import Job, _scrub

	def node(probability):
		return {"component_logit_lens": {
			"available": True, "top_tokens": [{"token": "test", "probability": probability}],
		}}

	payload = {"graph": {"nodes": [node(invalid), node(0.0), node(0.25)]}, "metric": float("nan")}
	clean = _scrub(payload)
	assert clean["graph"]["nodes"][0]["component_logit_lens"] == {"available": False}
	assert clean["graph"]["nodes"][1:] == payload["graph"]["nodes"][1:]
	assert clean["metric"] is None
	assert payload["graph"]["nodes"][0]["component_logit_lens"]["available"]
	assert json.loads(json.dumps(clean, allow_nan=False)) == clean
	assert _scrub(clean) == clean
	job = Job("test", {})
	job.push(payload)
	assert job.events == [clean]
