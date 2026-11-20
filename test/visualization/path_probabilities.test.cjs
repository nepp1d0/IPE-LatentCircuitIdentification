"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const P = require("../../visualization/static/path_probabilities.js");

function fixture(mode = "noising") {
	const context = { available: true, id: "job-1", method: "path", mode,
		reference_label: mode === "denoising" ? "Counterfactual" : "Clean", top_k: 5,
		position: 2, prompt_index: 0, reference_prompt: "prompt", target_token_id: 17, counterfactual_token_id: 42 };
	const tokens = [17, 42, 11, 7, 8].map((token_id, i) => ({ token_id,
		token: [" target", " wrong", "<|endoftext|>", "\t\n\0", "🧠".repeat(50)][i],
		reference: [.3, .2, .1, 1e-10, 0][i], patched: [.2, .25, .03, 1e-8, 0][i] }));
	tokens.forEach((t) => { t.delta = t.patched - t.reference; });
	const data = { available: true, context_id: "job-1", path_id: "path-1", position: 2,
		convention: "isolated_path_final_output_softmax", vocab_size: 100, tokens,
		node_ids: ["e", "f"], occurrence_ids: ["o1", "o2"] };
	return { path: { id: "path-1", sequence: ["e", "f"], occurrenceIds: ["o1", "o2"], probabilityComparison: data },
		meta: { method: "path", mode, probability_context: context } };
}

test("paired exact IDs and backend reference ordering survive every metric without top-token reselection", () => {
	for (const mode of ["noising", "denoising"]) for (const metric of ["logit_difference", "kl_divergence", "target_probability_percentage", "other"]) {
		const { path, meta } = fixture(mode), original = JSON.stringify(path);
		const result = P.readoutFor(path, { ...meta, metric });
		assert.equal(result.available, true);
		assert.deepEqual(result.tokens.map((t) => t.token_id), [17, 42, 11, 7, 8]);
		assert.equal(JSON.stringify(path), original);
	}
});

test("roles use token IDs only, never decoded strings, and do not force targets into selection", () => {
	const { path, meta } = fixture(), context = meta.probability_context;
	assert.deepEqual(path.probabilityComparison.tokens.map((t) => P.roleFor(t.token_id, context)), ["target", "counterfactual", "other", "other", "other"]);
	assert.equal(P.colorFor("target"), "var(--accent)");
	assert.equal(P.colorFor("counterfactual"), "var(--probability-cf)");
	assert.equal(P.colorFor("other"), "var(--ink-muted)");
	context.target_token_id = 99; context.counterfactual_token_id = 98;
	const result = P.readoutFor(path, meta);
	assert.equal(result.tokens.length, 5);
	assert.ok(result.tokens.every((t) => P.roleFor(t.token_id, context) === "other"));
	context.target_token_id = context.counterfactual_token_id = 17;
	assert.equal(P.roleFor(17, context), "target");
});

test("stale contexts, methods, paths, occurrence variants and invalid probability pairs are unavailable", () => {
	for (const mutate of [
		(p, m) => { m.method = "acdc"; }, (p, m) => { m.mode = "denoising"; },
		(p, m) => { m.probability_context.id = "another job"; },
		(p, m) => { m.probability_context.position = 1; },
		(p, m) => { m.probability_context.reference_label = "Reference"; },
		(p) => { p.id = "path-2"; }, (p) => { p.sequence[0] = "other"; },
		(p) => { p.occurrenceIds[0] = "different-kv-variant"; },
		(p) => { p.probabilityComparison.convention = "component_logit_lens"; },
		(p) => { p.probabilityComparison.tokens.pop(); },
		(p) => { p.probabilityComparison.tokens[0].token_id = 42; },
		(p) => { p.probabilityComparison.tokens.reverse(); },
		(p) => { p.probabilityComparison.tokens[0].delta = 123; },
		(p) => { p.probabilityComparison.tokens[0].reference = NaN; },
		(p) => { p.probabilityComparison.tokens[0].patched = null; },
		(p) => { p.probabilityComparison.tokens[0].patched = -0.1; },
		(p) => { p.probabilityComparison.tokens[0].patched = 1.1; },
	]) {
		const { path, meta } = fixture(); mutate(path, meta);
		assert.equal(P.readoutFor(path, meta).available, false, mutate.toString());
	}
	assert.equal(P.readoutFor(null).available, false);
});

test("shared zero-based vertical scale, unmodified heights and collision-free visible labels at narrow widths", () => {
	for (const width of [220, 240, 260, 280, 300, 440, 680]) for (const maximum of [0, 1e-45, 1e-10, .001, .2, 1]) {
		const tokens = [1, .99, .5, .01, 0].map((factor) => ({ reference: maximum * factor, patched: maximum * factor * .99 }));
		const g = P.chartGeometry(width, tokens);
		assert.equal(g.bars.length, 10);
		assert.equal(g.ticks[0], 0);
		assert.ok(g.domain >= maximum && g.domain <= 1);
		assert.deepEqual(g.ticks, [0, .25, .5, .75, 1].map((fraction) => fraction * g.domain));
		assert.equal(g.plotHeight, 84);
		for (const [i, bar] of g.bars.entries()) {
			assert.equal(bar.condition, i % 2 ? "patched" : "reference");
			assert.equal(bar.height, bar.token[bar.condition] / g.domain * g.plotHeight);
			assert.ok(bar.x >= g.left && bar.x + g.barWidth <= width - g.right);
			const l = bar.label;
			assert.equal(l.x, bar.x + g.barWidth / 2, "values stay centered even for long scientific notation");
			assert.ok(l.y >= 12 && l.y < g.baseline);
			assert.ok(l.x - l.width / 2 >= 0 && l.x + l.width / 2 <= width);
			for (const earlier of g.bars.slice(0, i)) {
				const other = earlier.label;
				assert.ok(Math.abs(other.x - l.x) >= (other.width + l.width) / 2 + 2 || Math.abs(other.y - l.y) >= 13);
			}
		}
	}
});

test("result-wide scale includes both conditions, ignores unavailable readouts, and has five actual-probability ticks", () => {
	for (const mode of ["noising", "denoising"]) {
		const { path, meta } = fixture(mode);
		const other = structuredClone(path);
		other.id = other.probabilityComparison.path_id = "offscreen-path";
		other.probabilityComparison.tokens.forEach((t, i) => {
			t.reference = i ? 0 : .4; t.patched = i ? 0 : .61; t.delta = t.patched - t.reference;
		});
		other.probabilityComparison.tokens.sort((a, b) => b.reference - a.reference || a.token_id - b.token_id);
		const stale = structuredClone(other);
		stale.probabilityComparison.context_id = "stale";
		stale.probabilityComparison.tokens[0].patched = 1;
		const before = JSON.stringify({ path, other, stale, meta });
		const scale = P.scaleFor([path, other, stale], meta);
		assert.equal(scale.maximum, .8);
		assert.deepEqual(scale.ticks.map(P.formatPercent), ["0%", "20%", "40%", "60%", "80%"]);
		for (const p of [path, other]) {
			const g = P.chartGeometry(400, p.probabilityComparison.tokens, scale);
			assert.equal(g.domain, .8); assert.deepEqual(g.ticks, scale.ticks);
		}
		assert.equal(JSON.stringify({ path, other, stale, meta }), before);
		other.probabilityComparison.tokens[0].reference = .85;
		other.probabilityComparison.tokens[0].delta = .61 - .85;
		assert.equal(P.scaleFor([path, other], meta).maximum, 1, "clean values also determine the common maximum");
		other.probabilityComparison.tokens[0].reference = .4;
		other.probabilityComparison.tokens[0].patched = 1;
		other.probabilityComparison.tokens[0].delta = .6;
		assert.equal(P.scaleFor([path, other], meta).maximum, 1, "axis never exceeds 100%");
	}
	assert.deepEqual(P.scaleFor([], {}), { maximum: 1, ticks: [0, .25, .5, .75, 1] });
});

test("uncrowded values sit immediately above their own bars with no horizontal clamping", () => {
	const tokens = [.28, .22, .16, .12, .07].map((reference) => ({ reference, patched: reference * .7 }));
	const g = P.chartGeometry(680, tokens);
	for (const bar of g.bars) {
		assert.equal(bar.label.x, bar.x + g.barWidth / 2);
		assert.ok(Math.abs(g.baseline - bar.height - (bar.label.y + bar.label.descent) - 3) < 1e-9);
	}
});

test("percent precision preserves tiny probabilities, whitespace and exact token identity", () => {
	assert.equal(P.formatPercent(0), "0%");
	assert.equal(P.formatPercent(.1874), "18.7%");
	assert.equal(P.formatPercent(1), "100%");
	assert.equal(P.formatPercent(1e-10), "1.0e-8%");
	assert.equal(P.formatDelta(-1e-10), "−1.0e-8 pp");
	assert.equal(P.visibleToken(" a\t\n\0"), "␠a⇥↵\\u0000");
	assert.equal(P.visibleToken("<|endoftext|>"), "<|endoftext|>");
});

class Element {
	constructor(tag, doc) { this.tag = tag; this.ownerDocument = doc; this.children = []; this.attrs = {}; this.style = {}; }
	setAttribute(k, v) { this.attrs[k] = String(v); }
	appendChild(n) { this.children.push(n); return n; }
	replaceChildren() { this.children = []; }
	get textContent() { return (this.text || "") + this.children.map((n) => n.textContent).join(""); }
	set textContent(text) { this.text = text; }
	all(tag) { return this.children.flatMap((n) => [...(n.tag === tag ? [n] : []), ...n.all(tag)]); }
}

test("SVG solids/hatches share role colors with unique IDs; labels and both values render with no interaction", () => {
	const observers = [];
	const doc = { createElement: (tag) => new Element(tag, doc), createElementNS: (_, tag) => new Element(tag, doc),
		defaultView: { ResizeObserver: class {
			constructor(draw) { this.draw = draw; observers.push(this); }
			observe(node) { this.node = node; }
			disconnect() { this.disconnected = true; }
		} } };
	const a = new Element("section", doc), b = new Element("section", doc);
	const { path, meta } = fixture();
	P.render(a, path, meta); P.render(b, path, meta);
	const groups = a.all("g");
	assert.equal(groups.length, 10);
	assert.equal(a.all("div").filter((n) => n.attrs.class === "path-probability-token").length, 5);
	assert.equal(a.all("text").filter((n) => n.attrs.class === "path-probability-value").length, 10);
	for (let i = 0; i < groups.length; i += 2) {
		const solid = groups[i], patched = groups[i + 1];
		assert.equal(solid.attrs["data-token-id"], patched.attrs["data-token-id"]);
		assert.equal(solid.attrs["data-role"], patched.attrs["data-role"]);
		assert.equal(solid.children[0].attrs.fill, P.colorFor(solid.attrs["data-role"]));
		const patternId = patched.children[0].attrs.fill.slice(5, -1);
		const pattern = a.all("pattern").find((n) => n.attrs.id === patternId);
		assert.equal(pattern.children[0].attrs.fill, solid.children[0].attrs.fill);
		assert.equal(pattern.children[1].attrs["stroke-width"], "1.7");
	}
	const ids = [...a.all("pattern"), ...b.all("pattern")].map((n) => n.attrs.id);
	assert.equal(new Set(ids).size, 6);
	assert.match(a.textContent, /CleanPatchedTargetCounterfactualOther/);
	assert.equal(a.all("div").filter((n) => n.attrs.class === "path-probability-heading").length, 1);
	assert.match(a.textContent, /Final prediction probabilitiesTop 5 by \|Δp\| · sorted by clean P/);
	assert.match(a.textContent, /␠target/);
	const tokenCells = a.all("div").filter((n) => n.attrs.class === "path-probability-token");
	assert.match(tokenCells[0].attrs.title, /Token #17: " target"/);
	assert.equal(a.all("small").filter((n) => n.attrs.class === "path-probability-token-id").length, 0);
	assert.match(a.textContent, /30%/);
	assert.equal(a.all("svg")[0].attrs["data-axis-min"], "0");
	const ticks = a.all("text").filter((n) => n.attrs.class === "path-probability-tick");
	const grid = a.all("line").filter((n) => n.attrs.class === "path-probability-gridline");
	assert.equal(ticks.length, 5); assert.equal(grid.length, 5);
	for (const [i, line] of grid.entries()) {
		assert.equal(Number(line.attrs.y1), Number(ticks[i].attrs.y) - 3);
		assert.equal(line.attrs.x1, grid[0].attrs.x1); assert.equal(line.attrs.x2, grid[0].attrs.x2);
	}
	assert.equal(a.all("details").length, 0);
	const firstObserver = observers[0];
	firstObserver.node.clientWidth = 600; firstObserver.draw();
	assert.equal(a.all("svg")[0].attrs.width, "600");
	P.render(a, path, { method: "acdc" });
	assert.equal(firstObserver.disconnected, true);
	assert.equal(a.all("svg").length, 0);
	assert.match(a.textContent, /unavailable/);
	P.dispose(b);
	assert.equal(observers[1].disconnected, true);
});

test("shared legend contains only color/fill items; role IDs and missing-role details stay supplementary", () => {
	const doc = { createElement: (tag) => new Element(tag, doc) };
	const container = new Element("section", doc);
	const { path, meta } = fixture();
	P.renderGuide(container, meta, [path]);
	assert.equal(container.hidden, false);
	assert.equal(container.textContent, "CleanPatchedTargetCounterfactualOther");
	const groups = container.all("div").filter((n) => n.attrs.class === "path-probability-legend-group");
	assert.deepEqual(groups.map((n) => n.textContent), ["CleanPatched", "TargetCounterfactualOther"]);
	assert.deepEqual(groups.map((n) => [n.attrs.role, n.attrs["aria-label"]]),
		[["group", "Probability conditions: fill styles"], ["group", "Token roles: bar colors"]]);
	assert.match(container.all("span").find((n) => n.attrs.title?.startsWith("Target")).attrs.title, /configured token #17/);
	meta.probability_context.target_token_id = null;
	P.renderGuide(container, meta, [path]);
	assert.equal(container.textContent, "CleanPatchedTargetCounterfactualOther");
	assert.match(container.all("span").find((n) => n.attrs.title?.startsWith("Target")).attrs.title, /not recorded/);
	meta.probability_context.reference_label = undefined;
	P.renderGuide(container, meta, [path]);
	assert.equal(container.hidden, true, "malformed metadata cannot crash the shared guide or label stale data");
	P.renderGuide(container, null, []);
	assert.equal(container.hidden, true);
	assert.equal(container.children.length, 0);
});

function whitespaceFixture() {
	const { path, meta } = fixture();
	path.probabilityComparison.vocab_size = 50257;
	Object.assign(path.probabilityComparison.tokens[0], { token_id: 5335, token: " Mary" });
	Object.assign(path.probabilityComparison.tokens[1], { token_id: 1757, token: " John" });
	Object.assign(meta.probability_context, { target_token_id: 24119, counterfactual_token_id: 7554 });
	const params = { model: "gpt2-small", method: "path", mode: meta.mode,
		prompts: [meta.probability_context.reference_prompt], targets: ["Mary"], cf_targets: ["John"] };
	return { path, meta, params };
}

test("unambiguous GPT-2 leading-space fallback changes visual roles only, preserving selected IDs and probabilities", () => {
	const { path, meta, params } = whitespaceFixture();
	const original = JSON.stringify({ path, meta, params });
	const readout = P.readoutFor(path, meta);
	const roles = P.visualRolesFor(readout.tokens, readout.context, P.roleLabelsFor(meta, params));
	assert.deepEqual(readout.tokens.map((t) => roles.get(t.token_id).role), ["target", "counterfactual", "other", "other", "other"]);
	assert.equal(roles.get(5335).match, "leading-space");
	assert.equal(roles.get(1757).sourceId, 7554);
	assert.equal(JSON.stringify({ path, meta, params }), original);
	// Another prompt/task uses the same path, not hardcoded token names or IDs.
	Object.assign(readout.tokens[0], { token_id: 14862, token: " Alice" });
	Object.assign(readout.tokens[1], { token_id: 5811, token: " Bob" });
	Object.assign(readout.context, { target_token_id: 44484, counterfactual_token_id: 18861 });
	params.targets = ["Alice"]; params.cf_targets = ["Bob"];
	const renamed = P.visualRolesFor(readout.tokens, readout.context, P.roleLabelsFor(meta, params));
	assert.equal(renamed.get(14862).role, "target");
	assert.equal(renamed.get(5811).role, "counterfactual");
});

test("exact IDs take precedence over whitespace variants and cannot be overwritten by another role", () => {
	const { path, meta, params } = whitespaceFixture();
	const tokens = path.probabilityComparison.tokens, context = meta.probability_context;
	Object.assign(tokens[2], { token_id: 24119, token: "Mary" });
	let roles = P.visualRolesFor(tokens, context, P.roleLabelsFor(meta, params));
	assert.equal(roles.get(24119).role, "target");
	assert.equal(roles.get(24119).match, "exact");
	assert.equal(roles.get(5335).role, "other");
	context.target_token_id = 999; context.counterfactual_token_id = 5335;
	roles = P.visualRolesFor(tokens, context, P.roleLabelsFor(meta, params));
	assert.equal(roles.get(5335).role, "counterfactual");
	assert.equal(roles.get(5335).match, "exact");
});

test("ambiguous normalized labels or colliding roles remain neutral rather than guessing", () => {
	const { path, meta, params } = whitespaceFixture();
	const tokens = path.probabilityComparison.tokens;
	Object.assign(tokens[2], { token_id: 999, token: "  Mary" });
	let roles = P.visualRolesFor(tokens, meta.probability_context, P.roleLabelsFor(meta, params));
	assert.equal(roles.get(5335).role, "other");
	assert.equal(roles.get(999).role, "other");
	assert.equal(roles.get(1757).role, "counterfactual");
	params.cf_targets = [" Mary"];
	roles = P.visualRolesFor(tokens, meta.probability_context, P.roleLabelsFor(meta, params));
	assert.ok([...roles.values()].every((match) => match.role === "other"));
});

test("fallback requires matching saved request context, GPT-2 model, and leading spaces only", () => {
	for (const mutate of [
		(p) => { p.model = "pythia-160m"; }, (p) => { p.mode = "denoising"; },
		(p) => { p.method = "tree"; }, (p) => { p.prompts = ["another prompt"]; },
		(p) => { p.targets = ["mary"]; }, (p) => { p.targets = ["Mary "]; },
		(p) => { p.targets = ["Mar"]; }, (p) => { p.targets = ["Mary Jane"]; },
		(p) => { p.targets = ["\tMary"]; }, (p) => { p.targets = ["<|endoftext|>"]; },
	]) {
		const { path, meta, params } = whitespaceFixture(); mutate(params);
		const roles = P.visualRolesFor(path.probabilityComparison.tokens, meta.probability_context, P.roleLabelsFor(meta, params));
		assert.equal(roles.get(5335).role, "other", mutate.toString());
	}
	const { path, meta } = whitespaceFixture();
	const roles = P.visualRolesFor(path.probabilityComparison.tokens, meta.probability_context, P.roleLabelsFor(meta, null));
	assert.equal(roles.get(5335).role, "other", "no saved labels means no fallback");
});

test("denoising uses the saved counterfactual prediction context without swapping target/counterfactual roles", () => {
	const { path, meta, params } = whitespaceFixture();
	meta.mode = meta.probability_context.mode = params.mode = "denoising";
	meta.probability_context.reference_label = "Counterfactual";
	params.cf_prompts = [meta.probability_context.reference_prompt];
	params.prompts = ["different clean prompt"];
	const readout = P.readoutFor(path, meta);
	assert.equal(readout.available, true);
	const roles = P.visualRolesFor(readout.tokens, readout.context, P.roleLabelsFor(meta, params));
	assert.equal(roles.get(5335).role, "target");
	assert.equal(roles.get(1757).role, "counterfactual");
});

test("renderer preserves fallback bar colors and supplementary token metadata without inline token text coloring", () => {
	const doc = { createElement: (tag) => new Element(tag, doc), createElementNS: (_, tag) => new Element(tag, doc) };
	const container = new Element("section", doc);
	const { path, meta, params } = whitespaceFixture();
	const original = JSON.stringify(path);
	P.render(container, path, meta, { sharedGuide: true, params });
	const bars = container.all("g");
	assert.deepEqual(bars.map((n) => n.attrs["data-role"]), ["target", "target", "counterfactual", "counterfactual", ...Array(6).fill("other")]);
	for (let i = 0; i < 4; i += 2) {
		const solid = bars[i].children[0], patched = bars[i + 1].children[0];
		const pattern = container.all("pattern").find((n) => n.attrs.id === patched.attrs.fill.slice(5, -1));
		assert.equal(pattern.children[0].attrs.fill, solid.attrs.fill);
		assert.equal(bars[i].attrs["data-role-match"], "leading-space");
	}
	const cells = container.all("div").filter((n) => n.attrs.class === "path-probability-token");
	assert.equal(cells[0].attrs["data-role"], "target");
	assert.match(cells[0].attrs.title, /visual-only leading-space match.*#24119/);
	assert.ok(container.all("span").filter((n) => n.attrs.class === "path-probability-token-text").every((n) => !n.style.color));
	assert.equal(container.all("small").filter((n) => n.attrs.class === "path-probability-delta").length, 5);
	assert.equal(JSON.stringify(path), original);
});

test("GPT-2 prefixed tokens use exact numeric IDs, not identical-looking names or current form targets", () => {
	const { path, meta } = fixture();
	const tokens = path.probabilityComparison.tokens;
	path.probabilityComparison.vocab_size = 50257;
	Object.assign(tokens[0], { token_id: 5335, token: " Mary" });
	Object.assign(tokens[1], { token_id: 1757, token: " John" });
	Object.assign(tokens[2], { token_id: 24119, token: "Mary" });
	Object.assign(tokens[3], { token_id: 7554, token: "John" });
	const context = meta.probability_context;
	Object.assign(context, { target_token_id: 5335, counterfactual_token_id: 1757 });
	assert.deepEqual(P.readoutFor(path, meta).tokens.map((t) => P.roleFor(t.token_id, context)),
		["target", "counterfactual", "other", "other", "other"]);
	// These are distinct GPT-2 tokens, not alternative spellings of the same ID.
	Object.assign(context, { target_token_id: 24119, counterfactual_token_id: 7554 });
	assert.deepEqual(P.readoutFor(path, meta).tokens.map((t) => P.roleFor(t.token_id, context)),
		["other", "other", "target", "counterfactual", "other"]);
	// A different task uses the same role-ID pipeline, without hardcoded names.
	Object.assign(tokens[0], { token_id: 14862, token: " Alice" });
	Object.assign(tokens[1], { token_id: 5811, token: " Bob" });
	Object.assign(context, { target_token_id: 14862, counterfactual_token_id: 5811 });
	assert.equal(P.roleFor(tokens[0].token_id, context), "target");
	assert.equal(P.roleFor(tokens[1].token_id, context), "counterfactual");
});
