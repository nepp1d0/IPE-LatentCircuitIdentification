"""Path provenance and serialization contracts without loading a model."""

import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ipe import graph_search
from ipe.nodes import ATTN_Node, EMBED_Node, FINAL_Node, MLP_Node
from ipe.webutils.serialization import graph_from_paths, graph_from_tree, grid_id


@pytest.fixture
def model():
    return SimpleNamespace(cfg=SimpleNamespace(n_layers=3, n_heads=2))


def root(model):
    return FINAL_Node(model, layer=2, position=2)


def head(model, kv):
    return ATTN_Node(model, layer=0, head=0, position=2, keyvalue_position=kv,
                     patch_query=False, patch_key=True, patch_value=True)


def pairs(path):
    ids = path["node_ids"]
    return set(zip(ids[:-1], ids[1:]))


def test_original_paths_do_not_become_cross_products_at_merged_heads(model):
    final = root(model)
    a, b = head(model, 0), head(model, 1)
    paths = [(0.2, [EMBED_Node(model, position=0), a, MLP_Node(model, 1, position=2), final]),
             (0.9, [EMBED_Node(model, position=1), b, MLP_Node(model, 2, position=2), final])]
    graph = graph_from_paths(paths)
    provenance = graph["path_provenance"]
    assert provenance["order"] == "result"  # no new ranking from merged scores
    assert [p["score"] for p in provenance["paths"]] == [0.2, 0.9]
    assert [p["node_ids"] for p in provenance["paths"]] == [
        [grid_id(n) for n in path] for _, path in paths]
    assert len(provenance["paths"]) == 2  # not the four routes of the merged DAG
    assert all(p["complete"] for p in provenance["paths"])
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    pa, pb = provenance["paths"]
    oa, ob = (occurrences[p["occurrence_ids"][1]] for p in (pa, pb))
    assert oa["id"] != ob["id"]
    assert oa["node_id"] == ob["node_id"] == grid_id(a)
    assert (oa["kv_position"], ob["kv_position"]) == (0, 1)
    assert oa["variant"] == "KV" and oa["patch_query"] is False
    assert pa["occurrence_ids"][-1] == pb["occurrence_ids"][-1]  # actual shared root
    assert ("embed:0", "attn:0:0:2") in pairs(pa)
    assert ("attn:0:0:2", "mlp:2:2") not in pairs(pa)
    json.dumps(graph, allow_nan=False)


def test_shared_edges_and_existing_aggregation_are_preserved(model):
    final = root(model)
    a, b = head(model, 0), head(model, 1)
    a.contribution, b.contribution = 0.1, -0.8
    paths = [(0.1, [EMBED_Node(model, position=0), a, final]),
             (-0.8, [EMBED_Node(model, position=1), b, final])]
    graph = graph_from_paths(paths)
    shared = next(e for e in graph["edges"] if e["target"] == grid_id(final))
    assert shared == {"source": grid_id(a), "target": grid_id(final),
                      "count": 2, "contribution": -0.8, "complete": True}
    merged = next(n for n in graph["nodes"] if n["id"] == grid_id(a))
    assert merged["merged"] == 2
    assert merged["contribution"] == -0.8
    assert merged["kv_positions"] == [0, 1]
    assert all((shared["source"], shared["target"]) in pairs(p)
               for p in graph["path_provenance"]["paths"])


def test_all_paths_retained_in_discovery_order_not_score_order(model):
    final = root(model)
    discovered = []
    for i in range(12):
        leaf = EMBED_Node(model, position=i)
        leaf.discovery_index = i
        discovered.append((float(i), [leaf, final]))
    graph = graph_from_paths(list(reversed(discovered)))
    provenance = graph["path_provenance"]
    assert provenance["order"] == "discovery"
    assert len(provenance["paths"]) == 12
    assert [p["score"] for p in provenance["paths"]] == list(range(12))
    assert len({p["id"] for p in provenance["paths"]}) == 12
    assert graph["nodes"][0]["id"] == "embed:11"  # DAG iteration order unchanged


def test_tree_uses_original_branches_and_separates_pruned_leaves(model):
    final = root(model)
    a, b = head(model, 0), head(model, 1)
    early, late = EMBED_Node(model, position=0), EMBED_Node(model, position=1)
    pruned = MLP_Node(model, layer=0, position=1)
    # The two heads merge on the display, but have different original children.
    final.children = [a, b]
    a.children = [late, pruned]
    b.children = [early]
    early.discovery_index, pruned.discovery_index, late.discovery_index = 2, 3, 4
    graph = graph_from_tree(final)
    provenance = graph["path_provenance"]
    assert provenance["version"] == 1
    assert provenance["source"] == "tree"
    assert provenance["order"] == "discovery"
    assert [p["complete"] for p in provenance["paths"]] == [True, False, True]
    assert [p["node_ids"][0] for p in provenance["paths"]] == ["embed:0", "mlp:0:1", "embed:1"]
    assert len(provenance["paths"]) == 3
    assert next(n for n in graph["nodes"] if n["id"] == grid_id(pruned))["complete"] is False
    assert next(n for n in graph["nodes"] if n["id"] == grid_id(a))["complete"] is True
    assert next(e for e in graph["edges"] if e["source"] == grid_id(pruned))["complete"] is False
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    assert len(provenance["occurrences"]) == len(occurrences) == 6
    assert [p["node_ids"] for p in provenance["paths"]] == [
        [grid_id(n) for n in path] for path in ([early, b, final], [pruned, a, final], [late, a, final])]
    for path in provenance["paths"]:
        assert [occurrences[id_]["node_id"] for id_ in path["occurrence_ids"]] == path["node_ids"]
    assert len({p["occurrence_ids"][-1] for p in provenance["paths"]}) == 1
    assert provenance == graph_from_tree(final)["path_provenance"]


def test_legacy_tree_order_is_stable_and_not_contribution_ranked(model):
    final = root(model)
    a, b = EMBED_Node(model, position=0), EMBED_Node(model, position=1)
    a.contribution, b.contribution = 0.01, 999
    final.children = [b, a]
    first = graph_from_tree(final)["path_provenance"]
    final.children.reverse()
    assert first == graph_from_tree(final)["path_provenance"]
    assert first["order"] == "tree"
    assert first["paths"][0]["node_ids"][0] == "embed:0"


@pytest.mark.parametrize("method", ["tree", "path"])
@pytest.mark.parametrize("score", [1.25, -0.75, 0.0, None, float("nan"), float("inf"), -float("inf")])
def test_original_complete_scores_preserve_sign_zero_and_missing_values(model, method, score):
    final = root(model)
    leaf = EMBED_Node(model, position=0, parent=final)
    final.children = [leaf]
    # Any recomputation would fail: serialization must only read stored records.
    final.forward = leaf.forward = lambda *args, **kwargs: pytest.fail("score recomputed")
    if method == "tree":
        if score is not None:
            leaf.contribution = score
        graph = graph_from_tree(final, joint_scoring=False)
    else:
        # Path tuples, not even the original leaf's admission field, are authoritative.
        leaf.contribution = 99.0
        graph = graph_from_paths([(score, [leaf, final])])
    record = graph["path_provenance"]["paths"][0]
    expected = score if score is not None and math.isfinite(score) else None
    assert record["score"] == expected
    assert record["complete"] is True
    assert record["score_kind"] == "isolated_path_effect"
    assert record["node_ids"] == [grid_id(leaf), grid_id(final)]


@pytest.mark.parametrize("mode,kind", [(None, "unknown"), (True, "joint_candidate_effect"),
                                      (False, "isolated_path_effect")])
def test_tree_scoring_mode_is_explicit_not_inferred_from_discovery_order(model, mode, kind):
    final = root(model)
    leaf = EMBED_Node(model, position=0, parent=final)
    leaf.contribution, leaf.discovery_index = 0.25, 0
    final.children = [leaf]
    baseline = graph_from_tree(final)
    kwargs = {} if mode is None else {"joint_scoring": mode}
    graph = graph_from_tree(final, **kwargs)
    record = graph["path_provenance"]["paths"][0]
    assert record["score"] == 0.25
    assert record["score_kind"] == kind
    assert record["complete"] is True
    assert {key: graph[key] for key in ("nodes", "edges")} == {
        key: baseline[key] for key in ("nodes", "edges")}


def test_tree_scores_stay_with_exact_occurrences_not_merged_embeddings_or_edges(model):
    final = root(model)
    q = ATTN_Node(model, layer=0, head=0, position=2,
                  patch_query=True, patch_key=False, patch_value=False)
    kv = head(model, 2)
    left, right = MLP_Node(model, 1, position=2), MLP_Node(model, 2, position=2)
    early, late = EMBED_Node(model, position=2), EMBED_Node(model, position=2)
    pruned = MLP_Node(model, 0, position=0)
    final.children, left.children, right.children = [left, right, pruned], [q], [kv]
    q.children, kv.children = [early], [late]
    for parent, child in [(final, left), (final, right), (final, pruned),
                          (left, q), (right, kv), (q, early), (kv, late)]:
        child.parent = parent
    early.contribution, late.contribution = 0.25, -0.75
    q.contribution, kv.contribution, pruned.contribution = 90.0, -100.0, 8.0
    early.discovery_index, pruned.discovery_index, late.discovery_index = 0, 1, 2
    graph = graph_from_tree(final, joint_scoring=False)
    provenance = graph["path_provenance"]
    a, incomplete, b = provenance["paths"]
    assert [p["score"] for p in provenance["paths"]] == [0.25, None, -0.75]
    assert incomplete["complete"] is False and incomplete["score_kind"] is None
    assert a["node_ids"][0] == b["node_ids"][0] == grid_id(early)
    assert a["occurrence_ids"][0] != b["occurrence_ids"][0]
    assert a["node_ids"] == [grid_id(n) for n in [early, q, left, final]]
    assert b["node_ids"] == [grid_id(n) for n in [late, kv, right, final]]
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    for record, leaf, variant in [(a, early, "Q"), (b, late, "KV")]:
        assert occurrences[record["occurrence_ids"][0]]["contribution"] == leaf.contribution
        assert occurrences[record["occurrence_ids"][1]]["variant"] == variant
    assert occurrences[incomplete["occurrence_ids"][0]]["contribution"] == 8.0
    assert next(n for n in graph["nodes"] if n["id"] == grid_id(early))["contribution"] == -0.75
    assert next(n for n in graph["nodes"] if n["id"] == grid_id(q))["contribution"] == -100.0


@pytest.mark.parametrize("search", [graph_search.PathMessagePatching,
                                    graph_search.PathMessagePatching_LimitedLevelWidth,
                                    graph_search.TreeMessagePatching,
                                    graph_search.TreeMessagePatching_LimitedLevelWidth])
def test_search_records_order_before_results_are_sorted(model, monkeypatch, search):
    final = root(model)
    low, high = EMBED_Node(model, position=0), EMBED_Node(model, position=1)
    low.parent = high.parent = final
    final.get_expansion_candidates = lambda *args, **kwargs: [low, high]
    monkeypatch.setattr(graph_search, "evaluate_path", lambda path, metric: torch.tensor(
        9.0 if path[0] is high else 1.0))
    monkeypatch.setattr(graph_search, "tqdm", lambda iterable, **kwargs: iterable)
    admissions = []
    result = search(model, None, final, on_event=admissions.append)
    if isinstance(result, list):
        assert result[0][1][0] is high  # existing score-sorted search API unchanged
        graph = graph_from_paths(result)
        actual = [grid_id(e["path"][0]) for e in admissions if e["event"] == "path_complete"]
    else:
        graph = graph_from_tree(result, joint_scoring=False)
        actual = [grid_id(e["node"]) for e in admissions if e["event"] == "admit"]
    assert [p["node_ids"][0] for p in graph["path_provenance"]["paths"]] == actual
    for record in graph["path_provenance"]["paths"]:
        assert record["score"] == (9.0 if record["node_ids"][0] == grid_id(high) else 1.0)
        assert record["score_kind"] == "isolated_path_effect"


@pytest.mark.parametrize("method", ["path", "tree"])
@pytest.mark.parametrize("strategy", ["threshold", "topk"])
@pytest.mark.parametrize("score", [1.25, -0.75, 0.0, None, float("nan"), float("inf"), -float("inf")])
def test_completed_payload_keeps_provenance_through_enrichment_and_scrubbing(
        model, monkeypatch, method, strategy, score):
    from visualization import server

    final = root(model)
    leaf = EMBED_Node(model, position=0)
    leaf.contribution = score
    final.children = [leaf]
    model.to_str_tokens = lambda *args, **kwargs: ["BOS", "a", "b"]
    experiment = SimpleNamespace(root=final, metric=None, cache={})
    monkeypatch.setattr(server, "get_model", lambda name: model)
    monkeypatch.setattr(server, "ExperimentManager", lambda **kwargs: experiment)
    monkeypatch.setattr(server, "evaluate_tree", lambda *args: 999.0)

    class Lens:
        def __init__(self, *args):
            pass

        def enrich(self, graph, **kwargs):
            for node in graph["nodes"]:
                node["component_logit_lens"] = {"available": False}

    monkeypatch.setattr(server, "LiveComponentLogitLens", Lens)
    for name in ("PathMessagePatching", "PathMessagePatching_LimitedLevelWidth"):
        monkeypatch.setattr(server, name, lambda *args, **kwargs: [(score, [leaf, final])])

    def tree_search(*args, **kwargs):
        assert kwargs["joint_scoring"] is False
        return final

    for name in ("TreeMessagePatching", "TreeMessagePatching_LimitedLevelWidth"):
        monkeypatch.setattr(server, name, tree_search)
    request = server.SearchRequest(prompts=["a b"], targets=[" b"], method=method, strategy=strategy)
    job = server.Job("test", request.model_dump())
    server._run_search(job, request)
    assert job.status == "complete", job.error
    provenance = job.result["graph"]["path_provenance"]
    assert provenance["source"] == method
    assert provenance["paths"][0]["node_ids"] == [grid_id(leaf), grid_id(final)]
    expected = score if score is not None and math.isfinite(score) else None
    assert provenance["paths"][0]["score"] == expected
    assert provenance["paths"][0]["score_kind"] == "isolated_path_effect"
    assert job.result["meta"]["joint_tree_contribution"] == (999.0 if method == "tree" else None)
    assert next(e for e in job.events if e["event"] == "result")["graph"]["path_provenance"] == provenance
    assert all(n["component_logit_lens"] == {"available": False} for n in job.result["graph"]["nodes"])
    assert "component_logit_lens" not in json.dumps(provenance)
    json.dumps(job.result, allow_nan=False)


def test_root_only_tree_is_incomplete_and_unscored(model):
    final = root(model)
    final.contribution = 9.0  # A root contribution is not a complete-path score.
    graph = graph_from_tree(final, joint_scoring=False)
    provenance = graph["path_provenance"]
    assert len(provenance["paths"]) == len(provenance["occurrences"]) == 1
    record = provenance["paths"][0]
    occurrence = provenance["occurrences"][0]
    assert record["node_ids"] == [grid_id(final)]
    assert record["occurrence_ids"] == [occurrence["id"]]
    assert record["complete"] is False
    assert record["score"] is None and record["score_kind"] is None
    assert occurrence["node_id"] == grid_id(final)
    assert occurrence["contribution"] == 9.0
    assert graph["nodes"][0]["complete"] is False
    assert graph["edges"] == []
    json.dumps(graph, allow_nan=False)


def test_duplicate_display_routes_preserve_q_k_v_occurrences_and_scores(model):
    final = root(model)
    flags = [(True, False, False), (False, True, False), (False, False, True)]
    paths = []
    for (query, key, value), score in zip(flags, [-0.25, 0.0, 0.6]):
        attention = ATTN_Node(model, layer=1, head=0, position=2, keyvalue_position=1,
                              patch_query=query, patch_key=key, patch_value=value)
        paths.append((score, [EMBED_Node(model, position=2), attention, final]))
    graph = graph_from_paths(paths)
    provenance = graph["path_provenance"]
    records = provenance["paths"]
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    assert len(records) == len({p["id"] for p in records}) == 3
    assert len(provenance["occurrences"]) == len(occurrences) == 7
    assert [p["score"] for p in records] == [-0.25, 0.0, 0.6]
    assert all(p["complete"] and p["score_kind"] == "isolated_path_effect" for p in records)
    assert [p["node_ids"] for p in records] == [[grid_id(n) for n in path] for _, path in paths]
    assert len({tuple(p["node_ids"]) for p in records}) == 1
    assert len({p["occurrence_ids"][1] for p in records}) == 3
    assert len({p["occurrence_ids"][-1] for p in records}) == 1
    attention_occurrences = [occurrences[p["occurrence_ids"][1]] for p in records]
    assert [n["variant"] for n in attention_occurrences] == ["Q", "K", "V"]
    assert [tuple(n[flag] for flag in ("patch_query", "patch_key", "patch_value"))
            for n in attention_occurrences] == flags
    for path in records:
        assert [occurrences[id_]["node_id"] for id_ in path["occurrence_ids"]] == path["node_ids"]
    assert len(graph["nodes"]) == 3  # Only the display geometry is merged.
    assert len(graph["edges"]) == 2
    assert next(n for n in graph["nodes"] if n["kind"] == "attn")["variants"] == ["Q", "K", "V"]


def test_tree_diamond_preserves_original_paths_through_merged_heads(model):
    final = root(model)
    originals = []
    for upper_layer, lower_position in [(1, 0), (0, 1)]:
        upper = MLP_Node(model, upper_layer, position=2, parent=final)
        final.children.add(upper)
        shared = ATTN_Node(model, layer=0, head=0, position=2, parent=upper,
                           patch_query=True, patch_key=False, patch_value=False)
        upper.children.add(shared)
        leaf = EMBED_Node(model, position=lower_position, parent=shared)
        leaf.contribution = -float(lower_position)
        shared.children.add(leaf)
        originals.append([grid_id(n) for n in (leaf, shared, upper, final)])
    graph = graph_from_tree(final, joint_scoring=False)
    provenance = graph["path_provenance"]
    records = provenance["paths"]
    # The merged DAG admits four possible routes; only the two originals exist.
    assert len(records) == 2
    assert sorted(p["node_ids"] for p in records) == sorted(originals)
    merged_heads = [n for n in graph["nodes"] if n["kind"] == "attn"]
    assert len(merged_heads) == 1 and merged_heads[0]["merged"] == 2
    assert len({p["occurrence_ids"][1] for p in records}) == 2
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    for path in records:
        assert [occurrences[id_]["node_id"] for id_ in path["occurrence_ids"]] == path["node_ids"]


@pytest.mark.parametrize("search,kwargs,expected", [
    (graph_search.TreeMessagePatching, {"min_contribution": 0}, [-0.7, 0.0, 0.4]),
    (graph_search.TreeMessagePatching_LimitedLevelWidth, {"max_width": 3}, [-0.7, 0.4]),
], ids=["threshold", "limited_width"])
def test_tree_search_preserves_signed_scores_without_reevaluation(model, monkeypatch, search, kwargs, expected):
    final = root(model)
    candidates = [EMBED_Node(model, position=i, parent=final) for i in range(3)]
    monkeypatch.setattr(final, "get_expansion_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(graph_search, "tqdm", lambda iterable, **kwargs: iterable)
    scores = [-0.7, 0.4, 0.0]
    evaluate = Mock(side_effect=lambda path, metric: torch.tensor(scores[path[0].position], dtype=torch.float64))
    monkeypatch.setattr(graph_search, "evaluate_path", evaluate)
    search(model, None, final, joint_scoring=False, **kwargs)
    before = evaluate.call_count
    graph = graph_from_tree(final, joint_scoring=False)
    records = graph["path_provenance"]["paths"]
    # Threshold search retains zero at threshold 0; limited width excludes it.
    assert sorted(p["score"] for p in records) == expected
    assert evaluate.call_count == before == 3
    assert all(p["complete"] and p["score_kind"] == "isolated_path_effect" for p in records)


def test_shared_tree_trunk_occurrences_are_deduplicated_and_references_resolve(model):
    model.cfg.n_layers = 101
    final = FINAL_Node(model, layer=100, position=2)
    trunk = [final]
    parent = final
    for layer in range(99, -1, -1):
        child = MLP_Node(model, layer, position=2, parent=parent)
        parent.children.add(child)
        trunk.append(child)
        parent = child
    leaves = [EMBED_Node(model, position=i, parent=parent) for i in range(1000)]
    for leaf in leaves:
        leaf.contribution = -0.1
    parent.children.update(leaves)
    provenance = graph_from_tree(final, joint_scoring=False)["path_provenance"]
    records = provenance["paths"]
    occurrences = {n["id"]: n for n in provenance["occurrences"]}
    assert len(provenance["occurrences"]) == len(occurrences) == 1101
    assert len(records) == len({p["id"] for p in records}) == 1000
    assert {p["node_ids"][0] for p in records} == {grid_id(n) for n in leaves}
    shared_ids = records[0]["occurrence_ids"][1:]
    assert len(shared_ids) == len(set(shared_ids)) == 101
    expected_trunk = [grid_id(n) for n in reversed(trunk)]
    for path in records:
        assert path["complete"] is True
        assert path["score"] == -0.1 and path["score_kind"] == "isolated_path_effect"
        assert len(path["node_ids"]) == len(path["occurrence_ids"]) == 102
        assert path["node_ids"][1:] == expected_trunk
        assert path["occurrence_ids"][1:] == shared_ids
        assert [occurrences[id_]["node_id"] for id_ in path["occurrence_ids"]] == path["node_ids"]
    leaf_ids = {p["occurrence_ids"][0] for p in records}
    assert len(leaf_ids) == 1000 and leaf_ids.isdisjoint(shared_ids)
    assert set(occurrences) == leaf_ids | set(shared_ids)
