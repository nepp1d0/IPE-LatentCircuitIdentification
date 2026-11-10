"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const D = require("../../visualization/static/component_decoding.js");

const meta = { tokens: ["BOS", " first", " last"], mode: "denoising", cf_tokens: ["wrong"] };
function fixture(extra = {}) {
	const node = { id: "attn:0:1:2", kind: "attn", layer: 0, head: 1, position: 2, ...extra };
	node.component_logit_lens = { available: true, convention: "isolated_component_softmax", position: node.position ?? 2,
		position_token: meta.tokens[node.position ?? 2],
		top_tokens: [0.2, 0.1, 0.03, 1e-10, 0].map((probability, i) => ({ token: ` token${i}`, rank: i + 1, probability })) };
	return { node, occurrence: { ...node, node_id: node.id, id: "occurrence-1", kv_position: 0, variant: "KV" } };
}

test("query position, individual head and full-vocabulary probabilities are preserved without mutation", () => {
	const { node, occurrence } = fixture();
	const original = JSON.stringify(node);
	for (const kv_position of [0, 1, null, undefined]) {
		const result = D.readoutFor({ ...occurrence, kv_position }, node, meta);
		assert.equal(result.available, true);
		assert.equal(result.position, 2);
		assert.equal(result.positionToken, " last");
		assert.equal(result.fallback, false);
		assert.deepEqual(result.topTokens, node.component_logit_lens.top_tokens);
		assert.equal(result.topTokens[0].probability, 0.2); // not 0.2 / Top-5 sum
	}
	assert.equal(JSON.stringify(node), original);
	for (const [key, value] of [["head", 0], ["layer", 1], ["position", 0], ["kind", "mlp"], ["node_id", "other"]]) {
		assert.equal(D.readoutFor({ ...occurrence, [key]: value }, node, meta).available, false, key);
	}
});

test("explicit all-position components use the last clean token, while position zero remains BOS", () => {
	for (const kind of ["attn", "mlp"]) {
		for (const position of [null, 0]) {
			const { node, occurrence } = fixture({ kind, position });
			const result = D.readoutFor(occurrence, node, { ...meta, positional: position !== null });
			assert.equal(result.available, true);
			assert.equal(result.position, position === null ? 2 : 0);
			assert.equal(result.fallback, position === null);
		}
	}
});

test("missing, mismatched and invalid positions or conventions never guess a cached decoding", () => {
	for (const position of [undefined, -1, 1.5, "2", NaN, Infinity, 999]) {
		const { node, occurrence } = fixture({ position });
		assert.equal(D.readoutFor(occurrence, node, meta).available, false, String(position));
	}
	const { node, occurrence } = fixture();
	for (const badMeta of [null, {}, { tokens: [] }, { tokens: ["BOS", " first", "other prompt"] }]) {
		assert.equal(D.readoutFor(occurrence, node, badMeta).available, false);
	}
	assert.equal(D.readoutFor(occurrence, null, meta).available, false);
	for (const patch of [{ available: false }, { convention: undefined }, { convention: "path_message" },
		{ position: 0 }, { position: "2" }, { position_token: "other prompt" }]) {
		assert.equal(D.readoutFor(occurrence, { ...node, component_logit_lens: { ...node.component_logit_lens, ...patch } }, meta).available, false);
	}
	assert.equal(D.readoutFor(occurrence, { ...node, component_logit_lens: undefined }, meta).available, false);
});

test("malformed readouts are unavailable; valid zero and tiny probabilities remain numeric", () => {
	const { node, occurrence } = fixture();
	for (const top_tokens of [null, [], [null], [3], [{ probability: 0.1 }],
		...[null, undefined, NaN, Infinity, -0.1, 1.1, "0.2"].map((probability) => [{ token: "bad", probability }]),
		[{ token: "a", probability: 0.8 }, { token: "b", probability: 0.8 }]]) {
		assert.equal(D.readoutFor(occurrence, { ...node, component_logit_lens: { ...node.component_logit_lens, top_tokens } }, meta).available, false);
	}
	assert.equal(D.formatProbability(0), "0.0000");
	assert.equal(D.formatProbability(0.189), "0.1890");
	assert.equal(D.formatProbability(1), "1.0000");
	assert.equal(D.formatProbability(1e-10), "1.000e-10");
});

test("longer saved readouts are ordered and limited to nine without changing probabilities", () => {
	const { node, occurrence } = fixture();
	node.component_logit_lens.top_tokens = Array.from({ length: 12 }, (_, i) => ({ token: String(i), probability: i / 100 }));
	const result = D.readoutFor(occurrence, node, meta);
	assert.deepEqual(result.topTokens.map((t) => t.probability), [0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]);
});

test("probability axis uses readable upward-rounded ticks, not a Top-K-normalized domain", () => {
	const scale = D.probabilityScale(0.189);
	assert.equal(scale.maximum, 0.2);
	assert.deepEqual(scale.ticks.map((t) => t.label), ["0", "0.05", "0.10", "0.15", "0.20"]);
	for (const maximum of [0, Number.MIN_VALUE, 1e-45, 1e-10, 0.00011, 0.00189, 0.189, 0.2, 0.20000000000001, 0.3, 0.99, 1]) {
		for (const intervals of [1, 2, 4]) {
			const axis = D.probabilityScale(maximum, intervals);
			assert.ok(Number.isFinite(axis.maximum) && axis.maximum >= maximum && axis.maximum > 0);
			assert.equal(axis.ticks[0].value, 0);
			assert.equal(axis.ticks[0].label, "0");
			assert.equal(axis.ticks.at(-1).value, axis.maximum);
			assert.equal(new Set(axis.ticks.map((t) => t.label)).size, axis.ticks.length);
			assert.ok(axis.ticks.every((tick) => Number.isFinite(tick.value) && !tick.label.includes("%")));
		}
	}
});

test("chart margins, row spacing and tick density adapt to the available width", () => {
	const { node } = fixture();
	for (const width of [160, 180, 200, 300, 480]) {
		const g = D.chartGeometry(width, node.component_logit_lens.top_tokens);
		assert.ok(g.left >= 52);
		assert.ok(g.plotWidth > 0 && g.left + g.plotWidth < width);
		assert.ok(g.scale.maximum >= 0.2);
		assert.ok(g.scale.ticks.length >= 2);
		assert.ok(g.height > node.component_logit_lens.top_tokens.length * 14);
	}
	assert.ok(D.chartGeometry(480, node.component_logit_lens.top_tokens).scale.ticks.length
		> D.chartGeometry(180, node.component_logit_lens.top_tokens).scale.ticks.length);
});

test("narrow charts keep useful bar width and reserve space only for labels that need it", () => {
	for (const maximum of [0, Number.MIN_VALUE, 1e-45, 1e-10, 0.00011, 0.00189, 0.189, 0.25, 0.99, 1]) {
		const tokens = [1, 0.7, 0.55, 0.3, 0.1, 1e-10, 0].map((f) => ({ probability: maximum * f }));
		for (const width of [160, 180, 200, 300, 480]) {
			const g = D.chartGeometry(width, tokens);
			assert.ok(g.plotWidth > 0 && g.scale.maximum >= maximum);
			for (const item of tokens) {
				const barWidth = item.probability / g.scale.maximum * g.plotWidth;
				const valueWidth = D.formatProbability(item.probability).length * 6.7;
				assert.ok(barWidth >= valueWidth + 10 || g.left + barWidth + 4 + valueWidth <= width - 1.99,
					`${maximum} at ${width}px: value fits inside or beside its bar`);
			}
		}
	}
	const g = D.chartGeometry(181, [{ probability: 0.25 }, { probability: 0.125 }, { probability: 1e-10 }]);
	assert.ok(g.plotWidth > 100, "tiny probabilities' long value labels must not consume the plot's right half");
	assert.ok(g.scale.ticks.length >= 3, "a compact panel retains intermediate ticks when they fit");
	assert.equal(g.scale.maximum, D.chartGeometry(480, [{ probability: 0.25 }]).scale.maximum);
});
