"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const V = require("../../visualization/static/path_presentation.js");
const P = require("../../visualization/static/path_exploration.js");
const { fixture } = require("./path_fixtures.cjs");

describe("Compact score formatting", () => {

test("compact scores preserve signs, zero, negative zero and unavailable values", () => {
	for (const [score, expected] of [[1.25, "+1.25"], [-2.5, "−2.5"], [0, "+0"], [-0, "−0"],
		[0.123456789, "+0.123"], [-0.00123456789, "−0.00123"], [1.43051e-6, "+1.43 × 10⁻⁶"],
		[-1.43051e-6, "−1.43 × 10⁻⁶"], [1.23456e8, "+1.23 × 10⁸"]]) {
		assert.equal(V.formatScore(score), expected);
	}
	for (const score of [null, undefined, NaN, Infinity, -Infinity, "0.1"]) assert.equal(V.formatScore(score), "Unavailable");
});

test("extreme and rounding-boundary nonzero scores never become zero", () => {
	for (const score of [Number.MIN_VALUE, -Number.MIN_VALUE, 1e-300, -1e-300, 9.999e-4, 0.001, 999999.9, Number.MAX_VALUE]) {
		const formatted = V.formatScore(score);
		assert.doesNotMatch(formatted, /^[+−]0(?:\.0+)?$/);
		assert.equal(formatted.startsWith("−"), score < 0);
		assert.doesNotMatch(formatted, /Infinity|NaN/);
	}
	assert.equal(V.formatScore(Number.MIN_VALUE), "+4.94 × 10⁻³²⁴");
});

});

describe("Saved metrics and score interpretation", () => {

test("list context states the saved metric, units, direction and scoring semantics once", () => {
	const paths = [{ scoreKind: "isolated_path_effect" }];
	assert.equal(V.scoreContext(paths, { metric: "logit_difference", mode: "noising" }), "Effect on logit difference · Noising · logit units");
	assert.equal(V.scoreContext(paths, { metric: "kl_divergence", mode: "denoising" }), "Effect on KL divergence · Denoising · nats");
	assert.match(V.scoreContext(paths, { metric: "target_probability_percentage", mode: "noising" }), /relative target-probability change.*relative change × 100/);
	assert.match(V.scoreContext([{ scoreKind: "joint_candidate_effect" }], { metric: "logit_difference" }), /^In-context scores.*Direction unavailable/);
	assert.match(V.scoreContext([...paths, { scoreKind: "unknown" }]), /^Original scores.*metric unavailable.*Direction unavailable/);
	assert.equal(V.scoreNote(paths[0]), "");
	assert.equal(V.scoreNote({ scoreKind: "joint_candidate_effect" }), "In-context admission score");
	assert.equal(V.scoreNote({}), "Scoring mode unknown");
});

test("path effect formatting is signed, readable, metric-aware and retains the exact value in its tooltip", () => {
	const meta = { metric: "logit_difference", mode: "noising" };
	const show = (score) => P.scorePresentation({ score, scoreKind: "isolated_path_effect" }, meta);
	assert.equal(show(1.25).text, "Path effect · Logit Difference · +1.25 logit units");
	assert.equal(show(-0.123456789).text, "Path effect · Logit Difference · −0.123457 logit units");
	assert.equal(show(0).text, "Path effect · Logit Difference · +0 logit units");
	assert.match(show(1e-12).text, /\+1\.00000e-12 logit units/);
	assert.match(show(-0.123456789).tooltip, /Original value: -0\.123456789/);
	assert.match(show(1).tooltip, /Positive scores indicate that perturbing this path decreases the clean target's logit advantage over the counterfactual target\. Negative scores indicate the opposite\./);
	for (const score of [null, undefined, NaN, Infinity, -Infinity]) {
		assert.equal(show(score).text, "Path effect · Logit Difference · unavailable");
		assert.match(show(score).tooltip, /not substituted/);
	}
});

test("tooltips adapt to direction and metric without generalizing logit-margin or probability semantics", () => {
	const path = { score: 1, scoreKind: "isolated_path_effect" };
	const show = (metric, mode) => P.scorePresentation(path, { metric, mode });
	assert.match(show("logit_difference", "denoising").tooltip, /patching clean activations.*increases the clean target's logit advantage/);
	assert.doesNotMatch(show("logit_difference", "denoising").tooltip, /perturbing this path decreases/);
	assert.match(show("target_logit_percentage", "noising").text, /Target Logit Percentage · \+1 relative %/);
	assert.match(show("target_logit_percentage", "noising").tooltip, /absolute clean target logit/);
	assert.match(show("target_probability_percentage", "noising").tooltip, /relative changes × 100, not probabilities/);
	assert.match(show("indirect_effect", "noising").tooltip, /noising direction/);
	assert.match(show("indirect_effect", "denoising").tooltip, /counterfactual baseline/);
	assert.match(show("kl_divergence", "noising").text, /KL Divergence · \+1 nats/);
	assert.match(show("kl_divergence", "noising").tooltip, /KL\(patched \|\| clean\).*not necessarily worse target performance/);
	for (const [metric, mode] of [["custom", "noising"], ["target_logit_percentage", "denoising"], ["logit_difference", undefined]]) {
		assert.match(show(metric, mode).tooltip, /Directional interpretation is unavailable/);
		assert.doesNotMatch(show(metric, mode).tooltip, /Positive scores indicate/);
	}
	assert.match(P.scorePresentation(path).text, /Metric unavailable/);
});

test("joint and unknown legacy scores are never labeled as isolated path effects", () => {
	const meta = { metric: "logit_difference", mode: "noising" };
	for (const [kind, label] of [["joint_candidate_effect", "In-context admission score"], ["unknown", "Original score (mode unknown)"], [undefined, "Original score (mode unknown)"]]) {
		const presentation = P.scorePresentation({ score: 2, scoreKind: kind }, meta);
		assert.ok(presentation.text.startsWith(label));
		assert.doesNotMatch(presentation.tooltip, /Positive scores indicate/);
		assert.match(presentation.tooltip, /not.*isolated path effect/i);
	}
	const graph = fixture("tree");
	graph.path_provenance.paths[0].score = 2;
	assert.equal(P.buildIndex(graph, "tree").paths.get("tree-1").scoreKind, "unknown");
});

test("compact values do not alter exact scores, ranking or score interpretation", () => {
	const path = { score: 1.43051e-6, scoreKind: "isolated_path_effect" };
	const before = JSON.stringify(path);
	V.formatScore(path.score);
	assert.match(P.scorePresentation(path, { metric: "logit_difference", mode: "denoising" }).tooltip, /Original value: 0.00000143051/);
	assert.equal(JSON.stringify(path), before);
});

});

describe("Concise comparison labels", () => {

test("difference labels are shortened without changing comparison results or source indices", () => {
	const diffs = [
		{ type: "added", label: "Added A0H1@10[KV:4]" },
		{ type: "position", label: "EMB token position: 10 → 4" },
		{ type: "patching", label: "A0H1: Q → KV" },
		{ type: "kv_position", label: "A0H1 K/V position: unspecified → 0" },
		{ type: "position", label: "A0H1 token position: unspecified → 0" },
		{ type: "patch_type", label: "A0H1 patch type: zero → counterfactual" },
		{ type: "component", label: "A0H1@10[QK:0] → A0H2@10[V:4]" },
	];
	const original = JSON.stringify(diffs);
	assert.deepEqual(diffs.map(V.differenceLabel), ["Added A0H1@10 (KV:4)", "EMB: 10 → 4", "A0H1: Q → KV",
		"A0H1 source: all eligible → 0", "A0H1: all → 0", "A0H1 patch: zero → counterfactual", "A0H1@10 (QK:0) → A0H2@10 (V:4)"]);
	assert.equal(JSON.stringify(diffs), original);
});

});

describe("Component probability formatting", () => {
	test("invalid probabilities display N/A while finite values retain fixed or scientific notation", () => {
		const source = fs.readFileSync(require.resolve("../../visualization/static/app.js"), "utf8");
		const formatter = source.match(/^function fmtProbability\(probability\) \{.*?^\}/ms);
		assert.ok(formatter, "the application exposes its probability formatter");
		const fmtProbability = vm.runInNewContext(`(${formatter[0]})`);
		assert.deepEqual([
			null, undefined, NaN, Infinity, -Infinity, "NaN", "Infinity", 0, 0.25, 0.000001,
		].map(fmtProbability), [...Array(7).fill("N/A"), "0.00000", "0.25000", "1.000e-6"]);
	});
});
