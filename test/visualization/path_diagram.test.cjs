"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const D = require("../../visualization/static/path_diagram.js");
const P = require("../../visualization/static/path_exploration.js");

const attention = (extra = {}) => ({ kind: "attn", layer: 11, head: 10, position: 14,
	patch_query: true, patch_key: false, patch_value: false, kv_position: null, ...extra });
const rows = (node, meta = {}, original = true) => D.metadataRows(node, { variant: P.patchVariant, meta, original });
const labels = (node, meta, original) => rows(node, meta, original).map((r) => r.label);

describe("Original occurrence and patching metadata", () => {

test("Q-only metadata omits a non-applicable K/V source, rather than calling it unspecified", () => {
	assert.deepEqual(labels(attention()), ["Query/output position: 14", "Patch Q"]);
	assert.deepEqual(labels(attention({ kv_position: undefined })), ["Query/output position: 14", "Patch Q"]);
});

test("positional KV metadata preserves both query/output and original source positions", () => {
	assert.deepEqual(labels(attention({ patch_query: false, patch_key: true, patch_value: true, kv_position: 3 })),
		["Query/output position: 14", "Patch KV · Source position: 3"]);
});

test("non-positional and unrestricted source scopes are explicit, never unavailable", () => {
	const meta = { tokens: ["must not pick this token"], positional: false };
	assert.deepEqual(rows(attention({ position: null }), meta),
		[{ label: "Query/output positions: all eligible" }, { label: "Patch Q" }]);
	assert.deepEqual(rows(attention({ position: null, patch_query: false, patch_key: true, patch_value: true }), meta),
		[{ label: "Query/output positions: all eligible" }, { label: "Patch KV · Source positions: all eligible" }]);
	assert.deepEqual(labels(attention({ patch_query: false, patch_value: true })),
		["Query/output position: 14", "Patch V · Source positions: all eligible"]);
});

test("every custom Q/K/V flag combination is represented without forcing Q versus KV", () => {
	for (let bits = 0; bits < 8; bits++) {
		const flags = { patch_query: Boolean(bits & 4), patch_key: Boolean(bits & 2), patch_value: Boolean(bits & 1) };
		const patch = [flags.patch_query && "Q", flags.patch_key && "K", flags.patch_value && "V"].filter(Boolean).join("") || "none";
		const result = labels(attention({ ...flags, variant: "wrong", kv_position: 3 }));
		const source = bits & 3 ? "Source position" : "Read source position";
		assert.equal(result[1], `Patch ${patch} · ${source}: 3`);
	}
	assert.equal(labels(attention({ patch_query: false }))[1], "Patch none");
});

test("an explicit read restriction on custom Q-only occurrences is not discarded or called a K/V patch", () => {
	assert.deepEqual(labels(attention({ kv_position: 0 })),
		["Query/output position: 14", "Patch Q · Read source position: 0"]);
});

test("missing original metadata is distinct from explicit null/all-position scope", () => {
	const node = { kind: "attn" };
	assert.deepEqual(labels(node), ["Query/output position: Unavailable", "Patch: Unavailable · Source position: Unavailable"]);
	assert.deepEqual(labels({ ...node, position: null, kv_position: null }),
		["Query/output positions: all eligible", "Patch: Unavailable · Source positions: all eligible"]);
	const grid = attention({ position: 0, kv_position: 1, patch_key: true, patch_value: true, variants: ["Q", "KV"], kv_positions: [0, 1] });
	assert.deepEqual(labels(grid, { tokens: ["wrong grid label"] }, false), labels(node));
});

test("original variants can fill absent flags but cannot contradict a supplied flag", () => {
	const legacy = { kind: "attn", position: 14, kv_position: 3, variant: "KV" };
	assert.equal(labels(legacy)[1], "Patch KV · Source position: 3");
	assert.equal(labels({ ...legacy, patch_query: true })[1], "Patch: Unavailable · Source position: 3");
	assert.equal(labels({ ...legacy, variant: undefined, patch_query: true })[1], "Patch: Unavailable · Source position: 3");
});

});

describe("Clean token labels and diagram text wrapping", () => {

test("zero-based clean token labels retain BOS, spaces and query/source distinctions even during denoising", () => {
	const meta = { mode: "denoising", tokens: ["<BOS>", " and", " to"], cf_tokens: ["other", "wrong", "answer"] };
	const result = rows(attention({ position: 2, kv_position: 0, patch_query: false, patch_key: true, patch_value: true }), meta);
	assert.equal(result[0].label, "Query/output position: 2");
	assert.equal(result[0].token, ' ("␠to")');
	assert.equal(result[1].label, "Patch KV · Source position: 0");
	assert.equal(result[1].token, ' ("<BOS>")');
	assert.equal(rows({ kind: "embed", position: 1 }, meta)[0].token, ' ("␠and")');
});

test("unresolvable tokens leave numerical coordinates intact without guessed labels", () => {
	assert.deepEqual(rows({ kind: "mlp", position: 1 }, null), [{ label: "Token position: 1", token: null }]);
	for (const tokens of [undefined, [], ["BOS"], ["BOS", null], { 1: "not a token array" }]) {
		assert.deepEqual(rows({ kind: "mlp", position: 1 }, { tokens }), [{ label: "Token position: 1", token: null }]);
	}
	for (const position of [undefined, -1, 1.5, "1", NaN, Infinity]) {
		assert.equal(labels({ kind: "mlp", position }, { tokens: ["BOS", "bad guess"] })[0], "Token position: Unavailable");
	}
});

test("metadata wrapping bounds long tokens, keeps Unicode intact and preserves secondary styling", () => {
	const segments = [{ text: "Patch KV · Source position: 0", secondary: false },
		{ text: ' ("␠' + "🧠<unsafe>".repeat(25) + '")', secondary: true }];
	const before = JSON.stringify(segments);
	const wrapped = D.wrapSegments(segments, 32, (s) => Array.from(s).length);
	assert.ok(wrapped.length > 2);
	assert.ok(wrapped.every((line) => Array.from(line.map((s) => s.text).join("")).length <= 32));
	assert.equal(wrapped.flat().map((s) => s.text).join("").replace(/\s/g, ""), segments.map((s) => s.text).join("").replace(/\s/g, ""));
	assert.ok(wrapped.flat().filter((s) => s.text.includes("🧠")).every((s) => s.secondary));
	assert.equal(JSON.stringify(segments), before);
});

});
