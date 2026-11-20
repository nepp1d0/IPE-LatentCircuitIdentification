"""Final-output path readouts: no pretrained weights or network needed."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ipe import graph_search
from ipe.nodes import EMBED_Node, FINAL_Node, MLP_Node
from ipe.paths import evaluate_path
from ipe.webutils.path_probabilities import (
	PathProbabilityCollector, final_probabilities, select_changed_tokens,
)
from ipe.webutils.serialization import graph_from_paths, graph_from_tree


@pytest.mark.parametrize("normalization", [None, "LN", "RMS"])
@pytest.mark.parametrize("cap", [0.0, 0.5])
def test_final_probabilities_match_actual_model_logits(normalization, cap):
	model = HookedTransformer(HookedTransformerConfig(
		n_layers=1, d_model=8, d_head=4, n_heads=2, d_mlp=16, n_ctx=8,
		d_vocab=11, act_fn="relu", normalization_type=normalization,
		output_logits_soft_cap=cap, device="cpu", seed=42,
	))
	with torch.no_grad():
		logits, cache = model.run_with_cache(torch.tensor([[0, 3, 4]]))
	residual = cache["blocks.0.hook_resid_post"][0, -1]
	actual = final_probabilities(model, residual)
	assert actual.dtype == torch.float32
	assert torch.allclose(actual, logits[0, -1].float().softmax(-1), atol=1e-7)
	assert actual.sum().item() == pytest.approx(1)


def test_selection_uses_all_ids_both_signs_then_reference_order_not_change_order():
	# IDs 0/7 are the target/counterfactual and the most probable reference
	# tokens, but have zero change. IDs 1..5 must win without forcing roles in.
	reference = torch.tensor([.40, .01, .10, .02, .08, .04, .05, .30])
	patched = torch.tensor([.40, .07, .04, .06, .03, .05, .05, .30])
	selected = select_changed_tokens(reference, patched)
	assert selected == [2, 4, 5, 3, 1]  # NOT descending |delta|
	assert len(selected) == len(set(selected)) == 5
	assert 0 not in selected and 7 not in selected
	assert any((patched - reference)[selected] > 0)
	assert any((patched - reference)[selected] < 0)


def test_selection_and_reference_order_ties_resolve_to_lowest_exact_token_id():
	reference = torch.full((9,), 1 / 9)
	assert select_changed_tokens(reference, reference) == [0, 1, 2, 3, 4]
	assert select_changed_tokens(reference, reference, top_k=8) == list(range(8))
	assert select_changed_tokens(reference[:3], reference[:3]) == [0, 1, 2]
	# Exactly representable ties at the selection boundary.
	ref = torch.tensor([.125] * 8)
	patched = torch.tensor([.25, 0, .25, 0, .25, 0, .125, .125])
	assert select_changed_tokens(ref, patched) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_are_rejected(invalid):
	with pytest.raises(ValueError):
		select_changed_tokens(torch.tensor([.1, invalid]), torch.tensor([.1, .2]))


class Head:
	def __init__(self):
		self.cfg = SimpleNamespace(n_layers=1, n_heads=1, normalization_type="LN", output_logits_soft_cap=0)
		self.ln_final = Mock(side_effect=lambda x: x)
		self.unembed = Mock(side_effect=lambda x: x)
		self.to_single_token = lambda token: {" target": 0, " wrong": 7}[token]
		self.strings = ["<|endoftext|>", " token", "\t", "\n", "🧠", ' <script>"&', "other", " wrong"]
		self.to_str_tokens = lambda value, **kwargs: ["BOS", " last"] if isinstance(value, str) else [self.strings[int(value[0])]]


def setup(mode="noising", method="path", positional=True, top_k=5):
	model = Head()
	reference = torch.tensor([.40, .01, .10, .02, .08, .04, .05, .30]).log().repeat(2, 2, 1)
	patched = torch.tensor([.40, .07, .04, .06, .03, .05, .05, .30]).log().repeat(2, 2, 1)
	root = FINAL_Node(model, 0, position=1 if positional else None,
		msg_cache={"blocks.0.hook_resid_post": reference}, patch_type="counterfactual")
	leaf = EMBED_Node(model, position=1 if positional else None, parent=root)
	leaf.forward = Mock(return_value=reference - patched)
	collector = PathProbabilityCollector(model, root, context_id="job-1", method=method,
		mode=mode, reference_prompt="prompt", target=" target", counterfactual=" wrong", top_k=top_k)
	return model, root, leaf, collector, patched


@pytest.mark.parametrize("mode,label", [("noising", "Clean"), ("denoising", "Counterfactual")])
@pytest.mark.parametrize("positional", [False, True])
def test_exact_scored_final_residual_is_reused_and_only_first_prompt_decoded(mode, label, positional):
	model, root, leaf, collector, patched = setup(mode, positional=positional)
	metric = Mock(return_value=torch.tensor(-123.0))
	# Prompt 2 deliberately differs: it must not silently get averaged into chart.
	patched[1, -1] = torch.arange(8).float()
	leaf.forward.return_value = root.msg_cache[root.input_name] - patched
	assert evaluate_path([leaf, root], metric, on_evaluated=collector.observe) == -123
	assert leaf.forward.call_count == 1
	assert model.unembed.call_count == 2  # one shared reference + one patched head
	assert metric.call_count == 1
	assert torch.equal(metric.call_args.kwargs["corrupted_resid"][0, -1], patched[0, -1])
	payload = leaf.path_probability_comparison
	assert payload["available"] is True
	assert collector.context["reference_label"] == label
	assert collector.context["target_token_id"] == 0
	assert collector.context["counterfactual_token_id"] == 7
	assert collector.context["prompt_index"] == 0
	assert collector.context["position"] == 1
	expected = patched[0, -1].softmax(-1)
	for t in payload["tokens"]:
		assert t["patched"] == pytest.approx(expected[t["token_id"]].item())
		assert t["token"] == model.strings[t["token_id"]]
		assert t["delta"] == t["patched"] - t["reference"]
	assert len(payload["tokens"]) == 5
	assert all(t["token_id"] not in (0, 7) for t in payload["tokens"])
	assert not any(isinstance(value, torch.Tensor) for value in payload.values())
	json.dumps(payload, allow_nan=False)


SEARCHES = [graph_search.PathMessagePatching, graph_search.PathMessagePatching_LimitedLevelWidth,
	graph_search.TreeMessagePatching, graph_search.TreeMessagePatching_LimitedLevelWidth]


@pytest.mark.parametrize("search", SEARCHES)
def test_all_web_search_strategies_transport_original_path_comparisons_without_reevaluation(search, monkeypatch):
	method = "tree" if search.__name__.startswith("Tree") else "path"
	model, root, leaf, collector, _ = setup(method=method)
	root.get_expansion_candidates = lambda *args, **kwargs: [leaf]
	monkeypatch.setattr(graph_search, "tqdm", lambda values, **kwargs: values)
	metric = lambda corrupted_resid: torch.tensor(1.0)
	result = search(model, metric, root, on_path_evaluated=collector.observe)
	assert leaf.forward.call_count == 1
	graph = graph_from_tree(result, joint_scoring=False) if method == "tree" else graph_from_paths(result)
	record = graph["path_provenance"]["paths"][0]
	data = record["probability_comparison"]
	assert record["score"] == 1.0 and record["score_kind"] == "isolated_path_effect"
	assert data["path_id"] == record["id"]
	assert data["occurrence_ids"] == record["occurrence_ids"]
	assert data["node_ids"] == record["node_ids"]
	assert data["context_id"] == collector.context["id"]
	assert model.unembed.call_count == 2
	json.dumps(graph, allow_nan=False)


def test_incomplete_joint_and_mismatched_interventions_are_not_fabricated():
	model, root, leaf, collector, patched = setup()
	collector.observe([MLP_Node(model, 0), root], patched)
	assert not hasattr(root, "path_probability_comparison")
	other = FINAL_Node(model, 0)
	collector.observe([leaf, other], patched)
	assert leaf.path_probability_comparison["available"] is False
	# Joint scoring must never call the isolated observer.
	observer = Mock()
	root.get_expansion_candidates = lambda *args, **kwargs: []
	graph_search.TreeMessagePatching(model, lambda **kw: torch.tensor(0.), root,
		joint_scoring=True, on_path_evaluated=observer)
	observer.assert_not_called()
	leaf.path_probability_comparison["node_ids"] = ["wrong"]
	assert "probability_comparison" not in graph_from_paths([(1., [leaf, root])])["path_provenance"]["paths"][0]


def test_unavailable_invalid_and_padded_reference_states_are_explicit():
	model, root, leaf, collector, patched = setup()
	collector.observe([leaf, root], torch.full_like(patched, float("nan")))
	assert leaf.path_probability_comparison["available"] is False
	assert "tokens" not in leaf.path_probability_comparison
	root.msg_cache[root.input_name] = patched[:, :1]
	collector = PathProbabilityCollector(model, root, context_id="new", method="path", mode="noising", reference_prompt="prompt")
	assert collector.context["available"] is False
	collector.observe([leaf, root], patched)
	assert leaf.path_probability_comparison["available"] is False
	assert leaf.path_probability_comparison["context_id"] == "new"


def test_k_is_configurable_and_reference_distribution_is_not_recomputed():
	model, root, leaf, collector, patched = setup(top_k=7)
	collector.observe([leaf, root], patched)
	collector.observe([leaf, root], patched + 1)
	assert len(leaf.path_probability_comparison["tokens"]) == 7
	assert model.unembed.call_count == 3
	assert collector.context["top_k"] == 7


@pytest.fixture(scope="module")
def tiny_model():
	"""Real transformer and output head, with a deterministic in-memory tokenizer."""
	model = HookedTransformer(HookedTransformerConfig(
		n_layers=1, d_model=8, d_head=4, n_heads=2, d_mlp=16, n_ctx=8,
		d_vocab=11, act_fn="relu", normalization_type="LN", device="cpu", seed=42,
	))
	model.eval()
	prompts = {"clean": [0, 1, 2], "counterfactual": [0, 3, 4]}
	strings = ["<BOS>", " clean", " last", " corrupt", " end", " target", " wrong", "\t", "\n", "🧠", "<special>"]

	def tokenize(value, **kwargs):
		if isinstance(value, torch.Tensor):
			return value
		if isinstance(value, str):
			value = [value]
		return torch.tensor([prompts[p] for p in value])

	def string_tokens(value, **kwargs):
		if isinstance(value, str):
			return [strings[t] for t in prompts[value]] if value in prompts else [value]
		return [strings[int(t)] for t in value.flatten()]

	model.to_tokens = tokenize
	model.to_str_tokens = string_tokens
	model.to_single_token = lambda token: strings.index(token)
	model.tokenizer = SimpleNamespace(padding_side="right")
	return model


@pytest.mark.parametrize("method", ["tree", "path"])
@pytest.mark.parametrize("strategy", ["threshold", "topk"])
@pytest.mark.parametrize("mode,metric", [
	("noising", name) for name in ["logit_difference", "kl_divergence", "indirect_effect",
		"target_logit_percentage", "target_probability_percentage"]
] + [("denoising", name) for name in ["logit_difference", "indirect_effect"]])
def test_real_search_worker_metrics_directions_and_serialization(tiny_model, monkeypatch, method, strategy, mode, metric):
	from visualization import server

	monkeypatch.setattr(server, "get_model", lambda name: tiny_model)
	request = server.SearchRequest(prompts=["clean"], targets=[" target"],
		cf_prompts=["counterfactual"], cf_targets=[" wrong"], method=method,
		strategy=strategy, mode=mode, metric=metric, min_contribution=0, max_width=20)
	job = server.Job(f"{method}-{strategy}-{mode}-{metric}", request.model_dump())
	server._run_search(job, request)
	assert job.status == "complete", job.error
	context = job.result["meta"]["probability_context"]
	assert context["available"] is True
	assert context["reference_label"] == ("Counterfactual" if mode == "denoising" else "Clean")
	prompt = "counterfactual" if mode == "denoising" else "clean"
	assert context["reference_prompt"] == prompt
	with torch.no_grad():
		expected_reference = tiny_model(prompt)[0, -1].softmax(-1)
	records = [p for p in job.result["graph"]["path_provenance"]["paths"] if p["complete"]]
	assert records
	for record in records:
		data = record["probability_comparison"]
		assert data["available"] is True
		assert data["context_id"] == job.id
		assert data["path_id"] == record["id"]
		assert data["occurrence_ids"] == record["occurrence_ids"]
		assert len(data["tokens"]) == 5
		assert len({t["token_id"] for t in data["tokens"]}) == 5
		assert [t["reference"] for t in data["tokens"]] == sorted((t["reference"] for t in data["tokens"]), reverse=True)
		for token in data["tokens"]:
			assert token["reference"] == pytest.approx(expected_reference[token["token_id"]].item(), abs=1e-7)
	assert next(e for e in job.events if e["event"] == "result")["graph"] == job.result["graph"]
	json.dumps(job.result, allow_nan=False)


def test_acdc_worker_reports_no_isolated_path_interventions(tiny_model, monkeypatch):
	from visualization import server

	monkeypatch.setattr(server, "get_model", lambda name: tiny_model)
	monkeypatch.setattr(server.acdc_bridge, "run_acdc", lambda *a, **kw: (
		{"nodes": [], "edges": []}, {"threshold": .1, "ablation": "zero", "metric": "kl_div",
			"runtime": 0., "acdc_edges": 0, "initial_acdc_edges": 1, "final_metric": 0.}))
	request = server.SearchRequest(prompts=["clean"], method="acdc")
	job = server.Job("acdc", request.model_dump())
	server._run_search(job, request)
	assert job.status == "complete", job.error
	assert job.result["meta"]["probability_context"]["available"] is False
	assert "isolated path" in job.result["meta"]["probability_context"]["reason"]
	assert "path_provenance" not in job.result["graph"]


@pytest.mark.parametrize("method", ["tree", "path"])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("has_cf", [False, True])
def test_zero_and_counterfactual_patching_with_all_position_search(tiny_model, monkeypatch, method, positional, has_cf):
	from visualization import server

	monkeypatch.setattr(server, "get_model", lambda name: tiny_model)
	request = server.SearchRequest(prompts=["clean"], targets=[" target"],
		cf_prompts=["counterfactual"] if has_cf else [], cf_targets=[" wrong"] if has_cf else [],
		method=method, positional=positional, metric="kl_divergence", min_contribution=0)
	job = server.Job("patch-configuration", request.model_dump())
	server._run_search(job, request)
	assert job.status == "complete", job.error
	context = job.result["meta"]["probability_context"]
	assert context["available"] is True
	assert context["patch_type"] == ("counterfactual" if has_cf else "zero")
	assert context["reference_label"] == "Clean"
	assert context["position"] == 2
	assert all(p["probability_comparison"]["available"] for p in
		job.result["graph"]["path_provenance"]["paths"] if p["complete"])
