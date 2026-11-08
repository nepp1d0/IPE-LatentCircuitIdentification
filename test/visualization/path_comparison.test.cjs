"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const P = require("../../visualization/static/path_exploration.js");

function component(kind, layer, extra = {}) {
	return { kind, layer, position: 5, patch_type: "zero", ...extra };
}
const final = component("final", 11);
const embed = component("embed", null);
const mlp = (layer) => component("mlp", layer);
const attn = (layer, head, extra = {}) => component("attn", layer,
	{ head, variant: "Q", kv_position: null, ...extra });
function path(components, id = "example") {
	return { id, components: [...components].reverse(), sequence: [...components].reverse()
		.map((n) => JSON.stringify([n.kind, n.layer, n.head ?? null, n.position])) };
}

describe("Path alignment and structural differences", () => {

test("single component substitutions are distinguished from intervention changes", () => {
	const a = path([final, attn(5, 5), mlp(0), embed]);
	const b = path([final, attn(5, 9), mlp(0), embed]);
	const diff = P.comparePaths(a, b);
	assert.deepEqual(diff.differences.map((d) => d.type), ["component"]);
	assert.match(diff.differences[0].label, /A5H5.* → A5H9/);
	assert.equal(diff.sameGeometry, false);
	assert.equal(diff.patchingOnly, false);
});

test("same displayed route explicitly retains Q/KV and K/V-position differences", () => {
	const a = path([final, attn(8, 6), mlp(0), embed]);
	const b = path([final, attn(8, 6, { variant: "KV", kv_position: 0 }), mlp(0), embed]);
	const diff = P.comparePaths(a, b);
	assert.equal(diff.sameGeometry, true);
	assert.equal(diff.patchingOnly, true);
	assert.equal(diff.metadataIncomplete, false);
	assert.deepEqual(diff.differences.map((d) => d.type), ["patching", "kv_position"]);
	assert.equal(diff.differences[0].label, "A8H6: Q → KV");
	assert.match(diff.differences[1].label, /K\/V position: unspecified → 0/);
});

test("K/V positions and token positions are compared separately, including position zero", () => {
	const a = path([final, attn(8, 6, { variant: "KV", kv_position: 0 }), embed]);
	const b = path([final, attn(8, 6, { variant: "KV", kv_position: 1, position: 6 }), embed]);
	const diff = P.comparePaths(a, b);
	assert.deepEqual(diff.differences.map((d) => d.type), ["position", "kv_position"]);
	assert.match(diff.differences[1].label, /0 → 1/);
	assert.equal(diff.sameGeometry, false);
});

test("different-length paths align shared components instead of comparing array indices", () => {
	const base = [final, attn(8, 6), mlp(7), mlp(0), embed];
	const longer = [final, attn(8, 6), mlp(7), attn(3, 2), mlp(0), embed];
	const added = P.comparePaths(path(base), path(longer)).differences;
	assert.equal(added.length, 1);
	assert.equal(added[0].type, "added");
	assert.match(added[0].label, /Added A3H2/);
	const removed = P.comparePaths(path(longer), path(base)).differences;
	assert.equal(removed.length, 1);
	assert.equal(removed[0].type, "removed");
});

test("alignment retains a shared middle component when a removal and insertion shift its index", () => {
	const a = path([final, mlp(9), mlp(7), embed]);
	const b = path([final, mlp(7), mlp(3), embed]);
	const diff = P.comparePaths(a, b).differences;
	assert.deepEqual(diff.map((d) => d.type), ["removed", "added"]);
	assert.match(diff[0].label, /MLP9/);
	assert.match(diff[1].label, /MLP3/);
});

test("larger unmatched runs are explicit removals/additions, not arbitrary component pairings", () => {
	const a = path([final, attn(8, 0), mlp(7), embed]);
	const b = path([final, mlp(6), attn(3, 2), embed]);
	const diff = P.comparePaths(a, b).differences;
	assert.deepEqual(diff.map((d) => d.type), ["removed", "removed", "added", "added"]);
});

test("repeated original occurrences are not collapsed, with exact metadata breaking alignment ties", () => {
	const q = attn(8, 6), kv = attn(8, 6, { variant: "KV", kv_position: 0 });
	const a = path([final, q, kv, embed]);
	const b = path([final, kv, embed]);
	const diff = P.comparePaths(a, b).differences;
	assert.equal(diff.length, 1);
	assert.equal(diff[0].type, "removed");
	assert.match(diff[0].label, /\[Q\]/);
	assert.equal(P.comparePaths(a, b).sameGeometry, true);
});

});

describe("Intervention metadata and comparison semantics", () => {

test("flags are compared canonically, including separate K/V and combined streams", () => {
	const left = attn(2, 1, { variant: undefined, patch_query: true, patch_key: false, patch_value: false });
	const right = attn(2, 1, { variant: undefined, patch_query: false, patch_key: true, patch_value: true });
	assert.equal(P.patchVariant(left), "Q");
	assert.equal(P.patchVariant(right), "KV");
	assert.equal(P.comparePaths(path([left]), path([right])).differences[0].label, "A2H1: Q → KV");
	assert.equal(P.patchVariant({ variant: "VQK" }), "QKV");
	assert.equal(P.patchVariant({ variant: "K" }), "K");
	assert.equal(P.patchVariant({ variant: "V" }), "V");
});

test("missing occurrence metadata is not reconstructed from merged variants or KV positions", () => {
	const unknown = attn(8, 6, { variant: undefined, kv_position: undefined,
		variants: ["Q", "KV"], kv_positions: [0, 1] });
	const diff = P.comparePaths(path([attn(8, 6)]), path([unknown]));
	assert.equal(diff.sameGeometry, true);
	assert.equal(diff.metadataIncomplete, true);
	assert.equal(diff.differences.length, 0);
	assert.equal(P.patchVariant(unknown), null);
	const missingPatchType = P.comparePaths(path([mlp(0)]), path([{ ...mlp(0), patch_type: null }]));
	assert.equal(missingPatchType.metadataIncomplete, true);
	assert.equal(missingPatchType.differences.length, 0);
});

test("object IDs and contribution scores do not become structural or scientific significance claims", () => {
	const a = path([final, { ...attn(8, 6), id: "occurrence-1", contribution: 100 }, embed], "path-1");
	const b = path([final, { ...attn(8, 6), id: "occurrence-99", contribution: -200 }, embed], "path-2");
	a.score = 0;
	b.score = -0.0123456789;
	const original = JSON.stringify([a, b]);
	const diff = P.comparePaths(a, b);
	assert.equal(diff.differences.length, 0);
	assert.equal(diff.metadataIncomplete, false);
	assert.equal(diff.sameGeometry, true);
	assert.equal(JSON.stringify([a, b]), original);
});

test("patch type differences are interventions, not inferred changes in path scores", () => {
	const a = path([mlp(0)]), b = path([{ ...mlp(0), patch_type: "counterfactual" }]);
	const diff = P.comparePaths(a, b);
	assert.equal(diff.patchingOnly, true);
	assert.equal(diff.differences[0].type, "patch_type");
});

});
