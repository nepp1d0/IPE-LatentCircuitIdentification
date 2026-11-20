"use strict";

/* Role = color; condition = fill. Payloads belong to original paths, never grid nodes. */
const PathProbabilities = (() => {
	const SVGNS = "http://www.w3.org/2000/svg";
	const probabilityScale = (typeof module !== "undefined" && module.exports
		? require("./component_decoding.js") : ComponentDecoding).probabilityScale;
	const observers = new WeakMap();
	let chartId = 0;
	const same = (a, b) => Array.isArray(a) && Array.isArray(b)
		&& a.length === b.length && a.every((value, i) => value === b[i]);
	const roleFor = (id, context) => id === context.target_token_id ? "target"
		: id === context.counterfactual_token_id ? "counterfactual" : "other";

	// These strings come from the immutable request saved with this result,
	// never the current form or decoded labels guessed from vocabulary IDs.
	function roleLabelsFor(meta, params) {
		const context = meta?.probability_context;
		const prompts = params?.[meta?.mode === "denoising" ? "cf_prompts" : "prompts"];
		if (!context || params?.method !== meta?.method || params?.mode !== meta?.mode
			|| prompts?.[context.prompt_index] !== context.reference_prompt) return {};
		return { model: params.model, target: params.targets?.[context.prompt_index],
			counterfactual: params.cf_targets?.[context.prompt_index] };
	}

	function visualRolesFor(tokens, context, labels = {}) {
		const matches = new Map(tokens.map((token) => {
			const role = roleFor(token.token_id, context);
			return [token.token_id, { role, match: role === "other" ? "none" : "exact" }];
		}));
		// Only known GPT-2 model names opt into the presentation-only fallback.
		if (!/^gpt2(?:-(?:small|medium|large|xl))?$/.test(labels.model || "")) return matches;
		const keyFor = (label) => {
			if (typeof label !== "string" || /[\u0000-\u001f\u007f]/.test(label)) return null;
			const key = label.replace(/^ +/, "");
			return key && !key.startsWith("<|") ? key : null;
		};
		const roles = [["target", context.target_token_id], ["counterfactual", context.counterfactual_token_id]]
			.map(([role, id]) => ({ role, id, label: labels[role], key: keyFor(labels[role]) }));
		for (const source of roles) {
			if (!Number.isInteger(source.id) || source.id < 0 || !source.key || tokens.some((token) => token.token_id === source.id)
				|| roles.some((other) => other !== source && other.key === source.key)) continue;
			const candidates = tokens.filter((token) => keyFor(token.token) === source.key);
			if (candidates.length !== 1 || candidates[0].token === source.label) continue;
			const candidate = candidates[0];
			// Never overwrite an exact match or another role's visual assignment.
			if (matches.get(candidate.token_id).role !== "other") continue;
			matches.set(candidate.token_id, { role: source.role, match: "leading-space", sourceId: source.id });
		}
		return matches;
	}
	const colorFor = (role) => ({ target: "var(--accent)", counterfactual: "var(--probability-cf)", other: "var(--ink-muted)" })[role];
	const visibleToken = (token) => (token.replace(/ /g, "␠").replace(/\t/g, "⇥")
		.replace(/\r/g, "␍").replace(/\n/g, "↵")
		.replace(/[\u0000-\u001f\u007f]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`)) || '""';

	function formatPercent(probability) {
		const percent = probability * 100;
		if (percent === 0) return "0%";
		if (percent < 0.001) return `${percent.toExponential(1)}%`;
		return `${Number(percent.toPrecision(percent < 1 ? 2 : 3))}%`;
	}
	function formatDelta(delta) {
		return `${delta < 0 ? "−" : "+"}${formatPercent(Math.abs(delta)).slice(0, -1)} pp`;
	}

	function readoutFor(path, meta = {}) {
		const unavailable = (message) => ({ available: false, message });
		const context = meta?.probability_context;
		if (!["tree", "path"].includes(meta?.method)) return unavailable("Path probabilities unavailable: this method does not retain isolated path interventions.");
		if (context?.available !== true) return unavailable(`Path probabilities unavailable: ${context?.reason || "not recorded in this result; rerun discovery."}`);
		const data = path?.probabilityComparison;
		if (data?.available !== true) return unavailable(`Path probabilities unavailable: ${data?.reason || "no final-output readout recorded for this path."}`);
		if (context.method !== meta.method || context.mode !== meta.mode || !context.id
			|| context.reference_label !== (meta.mode === "denoising" ? "Counterfactual" : "Clean")
			|| !["noising", "denoising"].includes(meta.mode)
			|| data.context_id !== context.id || data.path_id !== path.id
			|| data.convention !== "isolated_path_final_output_softmax"
			|| data.position !== context.position || !Number.isInteger(context.position) || context.position < 0
			|| context.prompt_index !== 0 || typeof context.reference_prompt !== "string"
			|| path.occurrenceIds?.length !== path.sequence?.length
			|| !same(data.node_ids, path.sequence) || !same(data.occurrence_ids, path.occurrenceIds)) {
			return unavailable("Path probabilities unavailable: mismatched or stale intervention data.");
		}
		const tokens = data.tokens;
		const validProbability = (p) => Number.isFinite(p) && p >= 0 && p <= 1;
		if (!Number.isInteger(context.top_k) || context.top_k < 1
			|| !Number.isInteger(data.vocab_size) || data.vocab_size < 1
			|| ["target_token_id", "counterfactual_token_id"].some((key) => context[key] !== null
				&& (!Number.isInteger(context[key]) || context[key] < 0 || context[key] >= data.vocab_size))
			|| !Array.isArray(tokens) || tokens.length !== Math.min(context.top_k, data.vocab_size)
			|| tokens.some((t) => !t || !Number.isInteger(t.token_id) || t.token_id < 0 || t.token_id >= data.vocab_size
				|| typeof t.token !== "string" || !validProbability(t.reference) || !validProbability(t.patched)
				|| !Number.isFinite(t.delta) || Math.abs(t.delta - (t.patched - t.reference)) > 1e-12)
			|| new Set(tokens.map((t) => t.token_id)).size !== tokens.length
			|| ["reference", "patched"].some((key) => tokens.reduce((sum, t) => sum + t[key], 0) > 1.000001)
			|| tokens.some((t, i) => i > 0 && (t.reference > tokens[i - 1].reference
				|| (t.reference === tokens[i - 1].reference && t.token_id < tokens[i - 1].token_id)))) {
			return unavailable("Path probabilities unavailable: invalid token probability pairs.");
		}
		return { available: true, context, tokens };
	}

	function scaleFor(paths, meta) {
		let maximum = 0;
		for (const path of paths) {
			const readout = readoutFor(path, meta);
			if (!readout.available) continue;
			for (const token of readout.tokens) maximum = Math.max(maximum, token.reference, token.patched);
		}
		return scaleForMaximum(maximum);
	}

	function scaleForMaximum(maximum) {
		const domain = Math.min(1, probabilityScale(Math.min(1, maximum > 0 ? maximum * 1.2 : 1)).maximum);
		return { maximum: domain, ticks: [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * domain) };
	}

	const valueFont = '400 10px system-ui, -apple-system, "Segoe UI", sans-serif';
	const textMetrics = (text) => ({ width: [...text].reduce((sum, c) => sum + (c === "%" ? 9 : c === "." ? 3 : 6), 2), ascent: 12, descent: 4 });

	function chartGeometry(width, tokens, scale = null, measure = textMetrics) {
		// Keep legible type on narrow layouts rather than shrinking an SVG viewBox.
		width = Math.max(220, width);
		scale ||= scaleForMaximum(Math.max(...tokens.flatMap((t) => [t.reference, t.patched])));
		// Ordinary percentage ticks need less gutter than scientific notation.
		const left = Math.max(40, ...scale.ticks.map((tick) => Math.ceil(measure(formatPercent(tick)).width * .9 + 9)));
		const plotHeight = 84;
		let right = 8;
		const domain = scale.maximum;
		let top = 24;
		let bars, barWidth;
		// Reserve only the edge space actually needed by centered end labels.
		// Do not clamp label x positions away from their corresponding bar centers.
		for (let pass = 0; pass < 8; pass++) {
			const groupWidth = (width - left - right) / tokens.length;
			barWidth = Math.min(22, groupWidth * 0.28);
			const gap = Math.max(3, Math.min(14, groupWidth / 2 - barWidth));
			bars = tokens.flatMap((token, i) => ["reference", "patched"].map((condition, j) => {
				const center = left + (i + 0.5) * groupWidth;
				const x = center + (j ? gap / 2 : -gap / 2 - barWidth);
				const height = token[condition] / domain * plotHeight;
				const text = formatPercent(token[condition]);
				return { token, condition, x, height, label: { ...measure(text), text, x: x + barWidth / 2 } };
			}));
			const overflow = Math.max(0, ...bars.map(({ label }) => label.x + label.width / 2 - (width - 2)));
			if (overflow < 0.01) break;
			right += overflow;
		}
		const obstacles = scale.ticks.map((tick) => ({ x: left - 7 - measure(formatPercent(tick)).width / 2,
			width: measure(formatPercent(tick)).width, y: top + plotHeight - tick / domain * plotHeight + 3, ascent: 10, descent: 3 }));
		// Place taller bars first, keeping values at the nearest clear position.
		// Measured glyph bounds avoid the excessive staggering from fixed advances.
		for (const bar of [...bars].sort((a, b) => b.height - a.height)) {
			const label = bar.label;
			label.y = top + plotHeight - bar.height - label.descent - 3;
			for (;;) {
				let y = label.y;
				for (const other of obstacles) {
					if (Math.abs(label.x - other.x) < (label.width + other.width) / 2 + 2
						&& label.y + label.descent > other.y - other.ascent - 2 + 1e-6
						&& label.y - label.ascent < other.y + other.descent + 2) y = Math.min(y, other.y - other.ascent - label.descent - 2);
				}
				for (const other of bars) {
					if (other !== bar && other.height > 0 && label.x + label.width / 2 > other.x - 1
						&& label.x - label.width / 2 < other.x + barWidth + 1
						&& label.y + label.descent > top + plotHeight - other.height - 2 + 1e-6
						&& label.y - label.ascent < top + plotHeight + 1) y = Math.min(y, top + plotHeight - other.height - label.descent - 2);
				}
				if (y === label.y) break;
				label.y = y;
			}
			obstacles.push(label);
		}
		const labels = bars.map((bar) => bar.label);
		const extraTop = Math.max(0, ...labels.map((l) => (l.x - l.width / 2 < 46 ? 17 : 2) - (l.y - l.ascent)));
		top += extraTop;
		for (const label of labels) label.y += extraTop;
		return { width, left, right, top, plotHeight, baseline: top + plotHeight, domain, bars, barWidth,
			ticks: scale.ticks };
	}

	function dispose(container) {
		observers.get(container)?.disconnect();
		observers.delete(container);
	}

	function appendHtml(parent, tag, className, text) {
		const node = parent.ownerDocument.createElement(tag);
		node.setAttribute("class", className);
		if (text !== undefined) node.textContent = text;
		parent.appendChild(node);
		return node;
	}

	function renderGuide(container, meta, paths = [], params = null) {
		container.replaceChildren();
		const readouts = paths.map((path) => readoutFor(path, meta)).filter((readout) => readout.available);
		container.hidden = !readouts.length;
		if (container.hidden) return;
		const context = readouts[0].context;
		const legend = appendHtml(container, "div", "path-probability-legend");
		const conditions = appendHtml(legend, "div", "path-probability-legend-group");
		conditions.setAttribute("role", "group");
		conditions.setAttribute("aria-label", "Probability conditions: fill styles");
		const roles = appendHtml(legend, "div", "path-probability-legend-group");
		roles.setAttribute("role", "group");
		roles.setAttribute("aria-label", "Token roles: bar colors");
		const labels = roleLabelsFor(meta, params);
		for (const [role, label, striped, key] of [
			["other", context.reference_label === "Clean" ? "Clean" : "Counterfactual ref.", false], ["other", "Patched", true],
			["target", "Target", false, "target_token_id"], ["counterfactual", "Counterfactual", false, "counterfactual_token_id"], ["other", "Other", false],
		]) {
			const id = key ? context[key] : null;
			const item = appendHtml(label === "Other" || key ? roles : conditions, "span", "path-probability-legend-item");
			const swatch = appendHtml(item, "i", `path-probability-swatch${striped ? " striped" : ""}`);
			swatch.style.color = colorFor(role);
			swatch.setAttribute("aria-hidden", "true");
			appendHtml(item, "span", "", label);
			if (key) {
				const source = labels[role];
				item.setAttribute("title", id === null ? `${label} token ID was not recorded for this result.`
					: `${label} configured token #${id}${typeof source === "string" ? `: ${JSON.stringify(source)}` : ""}. Exact IDs take precedence; unambiguous GPT-2 leading-space variants may share its visual color only.`);
			} else item.setAttribute("title", striped ? "Patched probability: diagonal hatch" : role === "other" && label !== "Other" ? `${label} probability: solid fill` : "Other selected tokens");
		}
	}

	function render(container, path, meta, { sharedGuide = false, params = null, scale = null } = {}) {
		dispose(container);
		container.replaceChildren();
		const doc = container.ownerDocument;
		const append = (tag, className, text, parent = container) => appendHtml(parent, tag, className, text);
		const readout = readoutFor(path, meta);
		if (!readout.available) {
			append("p", "path-probability-unavailable", readout.message);
			return;
		}
		const { tokens, context } = readout;
		if (!sharedGuide) renderGuide(append("div", "path-probability-guide"), meta, [path], params);
		const heading = append("div", "path-probability-heading");
		append("strong", "path-probability-title", "Final prediction probabilities", heading);
		const order = append("span", "path-probability-context", `Top ${tokens.length} by |Δp| · sorted by ${context.reference_label.toLowerCase()} P`, heading);
		order.setAttribute("title", `Result prompt ${context.prompt_index + 1}, prediction position ${context.position}. Reference prompt: ${JSON.stringify(context.reference_prompt)}`);
		const roles = visualRolesFor(tokens, context, roleLabelsFor(meta, params));
		const plot = append("div", "path-probability-plot");
		const tokenLabels = append("div", "path-probability-tokens");
		tokenLabels.style.gridTemplateColumns = `repeat(${tokens.length}, minmax(0, 1fr))`;
		const tokenNames = [];
		for (const token of tokens) {
			const cell = append("div", "path-probability-token", undefined, tokenLabels);
			cell.setAttribute("data-token-id", token.token_id);
			const identity = `Token #${token.token_id}: ${JSON.stringify(token.token)}`;
			const match = roles.get(token.token_id);
			const detail = match.match === "leading-space" ? `${identity}. ${match.role} visual-only leading-space match to configured ID #${match.sourceId}.` : identity;
			cell.setAttribute("data-role", match.role);
			cell.setAttribute("data-role-match", match.match);
			cell.setAttribute("title", detail);
			cell.setAttribute("aria-label", detail);
			tokenNames.push(append("span", "path-probability-token-text", visibleToken(token.token), cell));
			append("small", "path-probability-delta", `Δ ${formatDelta(token.delta)}`, cell);
		}
		const id = `path-probability-${++chartId}`;
		const canvas = doc.createElement("canvas").getContext?.("2d");
		if (canvas) canvas.font = valueFont;
		const measurements = new Map();
		const measure = (text) => {
			if (!measurements.has(text)) {
				const m = canvas?.measureText(text);
				measurements.set(text, m ? { width: Math.max(m.width, m.actualBoundingBoxLeft + m.actualBoundingBoxRight) + 2,
					ascent: Math.ceil(m.fontBoundingBoxAscent || m.actualBoundingBoxAscent || 12),
					descent: Math.ceil(m.fontBoundingBoxDescent || m.actualBoundingBoxDescent || 4) } : textMetrics(text));
			}
			return measurements.get(text);
		};
		let drawnWidth;
		const draw = () => {
			const width = Math.max(220, plot.clientWidth || container.clientWidth || 300);
			if (drawnWidth === width) return;
			drawnWidth = width;
			plot.replaceChildren();
			const g = chartGeometry(width, tokens, scale, measure);
			const svg = (tag, attrs, parent, text) => {
				const node = doc.createElementNS(SVGNS, tag);
				for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
				if (text !== undefined) node.textContent = text;
				parent.appendChild(node);
				return node;
			};
			const chart = svg("svg", { class: "path-probability-chart", width: g.width, height: g.baseline + 10,
				viewBox: `0 0 ${g.width} ${g.baseline + 10}`, role: "img",
				"aria-label": `${context.reference_label} and patched final token probabilities; shared percentage axis starting at zero`,
				"data-axis-min": 0, "data-axis-max": g.domain }, plot);
			const defs = svg("defs", {}, chart);
			for (const role of ["target", "counterfactual", "other"]) {
				const pattern = svg("pattern", { id: `${id}-${role}`, patternUnits: "userSpaceOnUse", width: 6, height: 6 }, defs);
				svg("rect", { width: 6, height: 6, fill: colorFor(role) }, pattern);
				svg("path", { d: "M-1 1L1 -1M0 6L6 0M5 7L7 5", stroke: "var(--path-card-background, var(--surface-1))", "stroke-width": 1.7 }, pattern);
			}
			svg("text", { class: "path-probability-axis-title", x: 0, y: 12 }, chart, "Prob. (%)");
			for (const tick of g.ticks) {
				const y = g.baseline - tick / g.domain * g.plotHeight;
				svg("line", { class: "path-probability-gridline", x1: g.left, x2: g.width - g.right, y1: y, y2: y }, chart);
				svg("text", { class: "path-probability-tick", x: g.left - 7, y: y + 3, "text-anchor": "end" }, chart, formatPercent(tick));
			}
			svg("line", { class: "path-probability-axis", x1: g.left, x2: g.left, y1: g.top, y2: g.baseline }, chart);
			svg("line", { class: "path-probability-axis", x1: g.left, x2: g.width - g.right, y1: g.baseline, y2: g.baseline }, chart);
			for (const bar of g.bars) {
				const match = roles.get(bar.token.token_id), role = match.role;
				const group = svg("g", { class: "path-probability-bar-group", "data-token-id": bar.token.token_id,
					"data-condition": bar.condition, "data-role": role, "data-role-match": match.match }, chart);
				svg("rect", { class: "path-probability-bar", x: bar.x, y: g.baseline - bar.height,
					width: g.barWidth, height: bar.height, "data-probability": bar.token[bar.condition],
					fill: bar.condition === "patched" ? `url(#${id}-${role})` : colorFor(role) }, group);
				// True zero remains zero height. No minimum-height distortion of small bars.
				svg("text", { class: "path-probability-value", x: bar.label.x, y: bar.label.y, "text-anchor": "middle" }, group, bar.label.text);
			}
			tokenLabels.style.marginLeft = `${g.left}px`;
			tokenLabels.style.marginRight = `${g.right}px`;
			// Keep ordinary names intact, bound unusual long labels to roughly two
			// lines, and retain the full token and exact ID in title/accessible text.
			const limit = Math.max(6, Math.floor(((g.width - g.left - g.right) / tokens.length - 4) / 6) * 2);
			for (const [i, name] of tokenNames.entries()) {
				const letters = Array.from(visibleToken(tokens[i].token));
				name.textContent = letters.length > limit ? letters.slice(0, limit - 1).join("") + "…" : letters.join("");
			}
		};
		draw();
		const Observer = doc.defaultView?.ResizeObserver;
		if (Observer) {
			const observer = new Observer(draw);
			observer.observe(plot);
			observers.set(container, observer);
		}
	}

	return { formatPercent, formatDelta, roleFor, roleLabelsFor, visualRolesFor, colorFor, visibleToken, readoutFor, scaleFor, chartGeometry, renderGuide, render, dispose };
})();

if (typeof module !== "undefined") module.exports = PathProbabilities;
