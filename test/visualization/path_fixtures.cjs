"use strict";

// Return fresh graph data so tests can mutate their fixtures independently.
function fixture(method = "path") {
	const node = (id, kind, position, layer, head) => ({ id, kind, position, layer, head,
		contribution: id === "spare" ? 0.01 : 0.5, complete: true, merged: 1,
		component_logit_lens: { available: true, top_tokens: [{ token: " <token>", probability: 0.4 }] } });
	const sequences = [
		["e0", "a", "b", "f"], ["e1", "a", "c", "f"], ["e0", "a", "b", "f"],
		["e1", "a", "b", "f"], ["e0", "a", "c", "f"], ["e1", "a", "b", "f"],
		["e0", "a", "c", "f"],
	];
	const edgeKeys = new Set(sequences.flatMap((ids) => ids.slice(1).map((id, i) => `${ids[i]}->${id}`)));
	edgeKeys.add("e0->f"); // shortcut/induced edge, not part of ANY original path
	edgeKeys.add("spare->f");
	return {
		nodes: [node("e0", "embed", 0), node("e1", "embed", 1), node("a", "attn", 1, 0, 0),
			node("b", "mlp", 1, 0), node("c", "mlp", 1, 1), node("f", "final", 1),
			node("spare", "mlp", 0, 1), node("second-head", "attn", 1, 0, 1)],
		edges: [...edgeKeys].map((key) => {
			const [source, target] = key.split("->");
			return { source, target, count: 2, contribution: 0.5, complete: true };
		}),
		path_provenance: { version: 1, source: method, order: "discovery",
			paths: sequences.map((node_ids, i) => ({ id: `${method}-${i + 1}`, complete: true, node_ids })) },
	};
}

function occurrenceFixture() {
	const graph = fixture();
	const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
	graph.path_provenance.occurrences = [];
	for (const path of graph.path_provenance.paths) {
		path.occurrence_ids = path.node_ids.map((id, i) => {
			const n = nodes.get(id);
			const occurrence = { ...n, id: `${path.id}-occ-${i}`, node_id: id, patch_type: "zero",
				...(n.kind === "attn" ? { variant: "Q", kv_position: null } : {}) };
			graph.path_provenance.occurrences.push(occurrence);
			return occurrence.id;
		});
	}
	return graph;
}

module.exports = { fixture, occurrenceFixture };
