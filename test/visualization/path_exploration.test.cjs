"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const P = require("../../visualization/static/path_exploration.js");
const { fixture } = require("./path_fixtures.cjs");

describe("Original path indexing and provenance validation", () => {
	test("indexes keep all original paths, shared edges, and only consecutive pairs", () => {
		const graph = fixture();
		const before = JSON.stringify(graph);
		const index = P.buildIndex(graph, "path");
		assert.equal(index.paths.size, 7);
		assert.deepEqual(index.byNode.get("a"), Array.from({ length: 7 }, (_, i) => `path-${i + 1}`));
		assert.deepEqual(index.byEdge.get("a->b"), ["path-1", "path-3", "path-4", "path-6"]);
		assert.equal(index.byEdge.has("e0->f"), false);
		assert.equal(index.paths.get("path-1").edgeKeys.has("a->c"), false);
		assert.equal(JSON.stringify(graph), before);
	});

	test("index preserves component sequence and original attention variants for sidebar summaries", () => {
		const graph = fixture();
		graph.path_provenance.occurrences = [{ id: "query-occurrence", node_id: "a", kind: "attn",
			layer: 0, head: 0, position: 1, variant: "Q", kv_position: null },
		{ id: "kv-occurrence", node_id: "a", kind: "attn",
			layer: 0, head: 0, position: 1, variant: "KV", kv_position: 0 }];
		graph.path_provenance.paths[0].occurrence_ids = ["e", "query-occurrence", "b", "f"];
		graph.path_provenance.paths[2].occurrence_ids = ["e", "kv-occurrence", "b", "f"];
		const index = P.buildIndex(graph, "path");
		assert.equal(index.paths.get("path-1").components[1].variant, "Q");
		assert.equal(index.paths.get("path-3").components[1].variant, "KV");
		assert.equal(index.paths.get("path-3").components[1].kv_position, 0);
		assert.deepEqual(index.paths.get("path-1").components.map((n) => n.node_id || n.id), ["e0", "a", "b", "f"]);
	});

	test("index preserves original path scores, including zero, without using merged node contributions", () => {
		const graph = fixture();
		graph.path_provenance.paths[0].score = 0;
		graph.path_provenance.paths[1].score = -0.123456789;
		const index = P.buildIndex(graph, "path");
		assert.equal(index.paths.get("path-1").score, 0);
		assert.equal(index.paths.get("path-2").score, -0.123456789);
		assert.equal(index.paths.get("path-3").score, null);
	});

	test("scores remain attached to original identities and non-finite values never become display scores", () => {
		for (const method of ["tree", "path"]) {
			const graph = fixture(method);
			const scores = [1.25, -0.75, 0, undefined, NaN, Infinity, -Infinity];
			graph.path_provenance.paths.forEach((path, i) => {
				path.score = scores[i];
				path.score_kind = "isolated_path_effect";
			});
			const index = P.buildIndex(graph, method);
			for (const [i, score] of scores.entries()) {
				const path = index.paths.get(`${method}-${i + 1}`);
				assert.equal(path.score, Number.isFinite(score) ? score : null);
				assert.equal(path.scoreKind, "isolated_path_effect");
			}
			assert.deepEqual(index.paths.get(`${method}-1`).sequence, index.paths.get(`${method}-3`).sequence);
			assert.notEqual(index.paths.get(`${method}-1`).score, index.paths.get(`${method}-3`).score);
		}
	});

	test("Tree excludes incomplete branches, including when they share complete display nodes", () => {
		const graph = fixture("tree");
		graph.path_provenance.paths.push({ id: "pruned", complete: false, node_ids: ["spare", "a", "f"] });
		const index = P.buildIndex(graph, "tree");
		assert.equal(index.paths.size, 7);
		assert.equal(index.byNode.has("spare"), false);
		assert.equal(index.incompleteByNode.get("a"), 1);
		const explorer = new P.Explorer(index);
		explorer.open("spare");
		assert.equal(explorer.active, true);
		assert.equal(explorer.view().paths.length, 0);
	});

	test("ACDC and missing provenance cannot enable speculative route exploration", () => {
		for (const [graph, method] of [[fixture("acdc"), "acdc"], [{ nodes: [], edges: [] }, "path"],
			[fixture("path"), "tree"]]) {
			const explorer = new P.Explorer(P.buildIndex(graph, method));
			assert.equal(explorer.open("a"), false);
			assert.equal(explorer.active, false);
		}
	});

	test("invalid consecutive pairs are rejected rather than repaired by graph traversal", () => {
		const graph = fixture();
		graph.path_provenance.paths.push({ id: "invalid", complete: true, node_ids: ["e1", "b", "f"] });
		assert.equal(P.buildIndex(graph, "path").paths.has("invalid"), false);
	});

	test("repeated display identities do not invent self-edges or duplicate membership", () => {
		const graph = fixture();
		graph.path_provenance.paths[0].node_ids = ["e0", "a", "a", "b", "f"];
		const index = P.buildIndex(graph, "path");
		assert.equal(index.byNode.get("a").filter((id) => id === "path-1").length, 1);
		assert.equal(index.paths.get("path-1").edgeKeys.has("a->a"), false);
	});
});

describe("Explorer selection, sorting and stable lanes", () => {
	test("selects the first original path and exposes every alternative without a five-path limit", () => {
		const explorer = new P.Explorer(P.buildIndex(fixture(), "path"));
		explorer.open("a");
		assert.equal(explorer.eligible.length, 7);
		assert.equal(explorer.selectedId, "path-1");
		assert.equal(explorer.view().paths.length, 7);
		assert.equal(explorer.view().paths[0].offset, 0);
		assert.equal(explorer.view().paths.filter((p) => p.selected).length, 1);
		assert.equal(explorer.select("not-eligible"), false);
		assert.equal(explorer.select(null), false); // exactly one selection until clear
		const before = explorer.view();
		assert.equal(explorer.view(), before); // hover reuses cached memberships
		explorer.select("path-2");
		assert.deepEqual([...explorer.view().focusNodes], ["e1", "a", "c", "f"]);
		assert.deepEqual(explorer.view().paths.map((p) => p.offset), before.paths.map((p) => p.offset));
		assert.equal(explorer.select("path-7"), true);
		explorer.open("b");
		assert.equal(explorer.nodeId, "b");
		assert.equal(explorer.selectedId, "path-1");
		assert.equal(explorer.eligible.length, 4);
		explorer.clear();
		assert.equal(explorer.active, false);
		assert.equal(explorer.view().paths.length, 0);
	});

	test("many identical display routes keep separate original identities and bounded alternative offsets", () => {
		const graph = fixture();
		graph.path_provenance.paths = Array.from({ length: 24 }, (_, i) => ({
			id: `path-${i + 1}`, complete: true, node_ids: ["e0", "a", "b", "f"],
		}));
		const explorer = new P.Explorer(P.buildIndex(graph, "path"));
		explorer.open("a");
		assert.equal(explorer.view().paths.length, 24);
		assert.equal(new Set(explorer.view().paths.map((p) => p.offset)).size, 24);
		assert.equal(explorer.view().edges.get("a->b").length, 24);
		assert.ok(explorer.view().paths.every((p) => Math.abs(p.offset) <= 24));
		assert.equal(explorer.select("path-24"), true);
		assert.notEqual(explorer.view().paths[23].offset, 0);
	});

	test("signed and absolute list sorting keep discovery identities, lane slots, ties and selection fixed", () => {
		const graph = fixture();
		[0, -2, 1, 1, null, NaN, 2].forEach((score, i) => { graph.path_provenance.paths[i].score = score; });
		const explorer = new P.Explorer(P.buildIndex(graph, "path"));
		explorer.open("a");
		explorer.select("path-4");
		const before = explorer.view();
		const discovery = explorer.listedPaths().map((p) => p.id);
		assert.deepEqual(discovery, graph.path_provenance.paths.map((p) => p.id));
		assert.equal(explorer.setSortOrder("signed"), true);
		assert.deepEqual(explorer.listedPaths().map((p) => p.id), ["path-7", "path-3", "path-4", "path-1", "path-2", "path-5", "path-6"]);
		assert.equal(explorer.setSortOrder("absolute"), true);
		assert.deepEqual(explorer.listedPaths().map((p) => p.id), ["path-2", "path-7", "path-3", "path-4", "path-1", "path-5", "path-6"]);
		assert.equal(explorer.selectedId, "path-4");
		assert.equal(explorer.view(), before);
		assert.equal(explorer.setSortOrder("invalid"), false);
		explorer.setSortOrder("discovery");
		assert.deepEqual(explorer.listedPaths().map((p) => p.id), discovery);
		assert.deepEqual(graph.path_provenance.paths.map((p) => p.id), discovery);
	});

	test("curve offsets remain perpendicular and separated at consistent node ports", () => {
		for (const targetX of [80, 420, -360]) {
			const source = { cx: 80, top: 400 };
			const target = { cx: targetX, bottom: 80 };
			const center = P.parallelPoints(source, target, 0);
			const left = P.parallelPoints(source, target, -3.5);
			const right = P.parallelPoints(source, target, 3.5);
			for (let i = 0; i < center.length; i++) {
				assert.ok(Math.abs(Math.hypot(left[i].x - right[i].x, left[i].y - right[i].y) - 7) < 1e-9);
				assert.ok(Math.abs((left[i].x + right[i].x) / 2 - center[i].x) < 1e-9);
			}
			assert.deepEqual(right[0], { x: source.cx + 3.5, y: source.top });
			assert.deepEqual(right.at(-1), { x: target.cx + 3.5, y: target.bottom });
			const next = P.parallelPoints({ cx: targetX, top: 58 }, { cx: 200, bottom: 20 }, 3.5);
			assert.equal(next[0].x, right.at(-1).x); // both incident curves use the same lane x
			assert.match(P.parallelPath(source, target, 3.5), /^M .* L /);
		}
		const source = { cx: 0, top: 100 }, target = { cx: 400, bottom: 0 };
		const center = P.parallelPoints(source, target, 0);
		const offset = P.parallelPoints(source, target, 7);
		const mid = Math.floor(center.length / 2);
		assert.ok(Math.abs(offset[mid].y - center[mid].y) > 5); // not just a translated cubic
	});
});
