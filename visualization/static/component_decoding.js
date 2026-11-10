"use strict";

/* Component selection + token bars used to decode full components. */
const ComponentDecoding = (() => {
	const SVGNS = "http://www.w3.org/2000/svg";
	const observers = new WeakMap();
	const LABEL_CHAR_WIDTH = 6.7;
	const visibleToken = (token) => token.replace(/\r\n?/g, "\n").replace(/ /g, "␠")
		.replace(/\t/g, "⇥").replace(/\n/g, "↵");

	function formatProbability(probability) {
		return probability > 0 && probability < 0.0001 ? probability.toExponential(3) : probability.toFixed(4);
	}

	function probabilityScale(maximum, intervals = 4) {
		if (!Number.isFinite(maximum) || !(maximum > 0)) maximum = 1;
		const raw = maximum / Math.max(1, intervals);
		const power = 10 ** Math.max(-323, Math.floor(Math.log10(raw)));
		const step = [1, 2, 5, 10].find((factor) => factor * power >= raw) * power;
		const count = Math.ceil(maximum / step);
		const domain = count * step;
		const digits = Math.max(0, -Math.floor(Math.log10(step)));
		
		const ticks = Array.from({ length: count + 1 }, (_, i) => {
			const value = i * step;
			return { value, label: i === 0 ? "0" : step < 0.0001 ? value.toExponential(1) : value.toFixed(digits) };
		});
		return { maximum: domain, ticks };
	}

	function chartGeometry(width, tokens) {
		const left = Math.min(120, Math.max(52, width * 0.35));
		const scale = probabilityScale(Math.max(...tokens.map((item) => item.probability)));
		let plotWidth = Math.max(1, width - left - 8);
		
		for (let pass = 0; pass <= tokens.length; pass++) {
			let next = plotWidth;
			for (const item of tokens) {
				const fraction = item.probability / scale.maximum;
				const barWidth = fraction * plotWidth;
				const valueWidth = formatProbability(item.probability).length * LABEL_CHAR_WIDTH;
				if (fraction > 0 && barWidth < valueWidth + 10 && left + barWidth + 4 + valueWidth > width - 2) {
					next = Math.min(next, Math.max(1, (width - left - 6 - valueWidth) / fraction));
				}
			}
			if (next === plotWidth) break;
			plotWidth = next;
		}
		
		const ticks = scale.ticks;
		for (let stride = 1; stride < ticks.length; stride++) {
			const selected = ticks.filter((_, i) => i % stride === 0 || i === ticks.length - 1);
			let previousRight = -Infinity;
			const fits = selected.every((tick, i) => {
				const x = tick.value / scale.maximum * plotWidth;
				const labelWidth = tick.label.length * LABEL_CHAR_WIDTH;
				const start = i === 0 ? x : x - labelWidth * (i === selected.length - 1 ? 1 : 0.5);
				const clear = start >= previousRight + 2;
				previousRight = start + labelWidth;
				return clear;
			});
			if (fits || stride === ticks.length - 1) {
				scale.ticks = selected;
				break;
			}
		}
		return { left, plotWidth, scale, rowHeight: 18, top: 2, height: tokens.length * 18 + 4 };
	}

	// Reusable histogram: only token/probability pairs, no component or model metadata.
	function renderHistogram(plot, axis, tokens) {
		const width = plot.clientWidth || 300;
		const { left, plotWidth, scale, rowHeight, top, height } = chartGeometry(width, tokens);
		plot.replaceChildren();
		axis.replaceChildren();
		const svg = (tag, attrs, parent, text) => {
			const element = plot.ownerDocument.createElementNS(SVGNS, tag);
			for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
			if (text !== undefined) element.textContent = text;
			parent.appendChild(element);
			return element;
		};
		const bars = svg("svg", { class: "component-decoding-bars", width, height, viewBox: `0 0 ${width} ${height}`,
			role: "img", "aria-label": `Top ${tokens.length} decoded tokens, descending full-vocabulary probability` }, plot);
		axis.setAttribute("width", width);
		axis.setAttribute("viewBox", `0 0 ${width} 44`);
		
		for (const tick of scale.ticks) {
			const x = left + tick.value / scale.maximum * plotWidth;
			svg("line", { class: "component-decoding-gridline", x1: x, x2: x, y1: 0, y2: height }, bars);
			svg("line", { class: "component-decoding-axis-line", x1: x, x2: x, y1: 1, y2: 6 }, axis);
			svg("text", { class: "component-decoding-tick", x, y: 19,
				"text-anchor": tick.value === 0 ? "start" : tick === scale.ticks.at(-1) ? "end" : "middle" }, axis, tick.label);
		}
		svg("line", { class: "component-decoding-axis-line", x1: left, x2: left, y1: 0, y2: height }, bars);
		svg("line", { class: "component-decoding-axis-line", x1: left, x2: left + plotWidth, y1: 1, y2: 1 }, axis);
		svg("text", { class: "component-decoding-axis-title", x: left + plotWidth / 2, y: 38, "text-anchor": "middle" }, axis, "Probability");
		
		for (const [index, item] of tokens.entries()) {
			const cy = top + (index + 0.5) * rowHeight;
			const barWidth = item.probability / scale.maximum * plotWidth;
			const row = svg("g", { class: "component-decoding-row", "data-probability": item.probability }, bars);
			svg("title", {}, row, `${JSON.stringify(item.token)}: ${item.probability}`);
			const label = visibleToken(item.token) || '""';
			const token = svg("text", { class: "component-decoding-token", x: left - 8, y: cy + 3.5,
				"text-anchor": "end", "aria-label": label, "data-token": item.token }, row, label);
			
			const measure = (text) => {
				token.textContent = text;
				return token.getComputedTextLength?.() || Array.from(text).length * LABEL_CHAR_WIDTH;
			};
			if (measure(label) > left - 12) {
				const letters = Array.from(label);
				let low = 0, high = letters.length;
				while (low < high) {
					const mid = Math.ceil((low + high) / 2);
					if (measure(letters.slice(0, mid).join("") + "…") <= left - 12) low = mid;
					else high = mid - 1;
				}
				token.textContent = letters.slice(0, low).join("") + "…";
			}
			svg("rect", { class: "component-decoding-bar", x: left, y: cy - 7, width: barWidth, height: 14 }, row);
			const value = formatProbability(item.probability);
			const inside = barWidth >= value.length * LABEL_CHAR_WIDTH + 10;
			svg("text", { class: `component-decoding-probability${inside ? " inside" : ""}`,
				x: inside ? left + barWidth - 4 : left + barWidth + 4, y: cy + 3.5,
				"text-anchor": inside ? "end" : "start", "data-probability": item.probability }, row, value);
		}
	}

	function readoutFor(component, node, meta = {}) {
		const unavailable = (message) => ({ available: false, message });
		if (!["attn", "mlp"].includes(component?.kind)) {
			return unavailable("Decoding is currently available for individual attention heads and MLP outputs only.");
		}
		
		if (!node || (component.node_id || component.id) !== node.id
			|| ["kind", "layer", "position", ...(component.kind === "attn" ? ["head"] : [])]
				.some((key) => component[key] !== node[key])
			|| !Number.isInteger(component.layer) || component.layer < 0
			|| (component.kind === "attn" && (!Number.isInteger(component.head) || component.head < 0))) {
			return unavailable("Matching component metadata is unavailable in this result.");
		}
		const tokens = meta?.tokens;
		if (!Array.isArray(tokens) || !tokens.length
			|| (component.position !== null && (!Number.isInteger(component.position) || component.position < 0))) {
			return unavailable("A valid clean-prompt token position is required for decoding.");
		}
		
		const fallback = component.position === null;
		const position = fallback ? tokens.length - 1 : component.position;
		const lens = node.component_logit_lens;
		if (lens?.available !== true) return unavailable("No cached component decoding is available in this result.");
		if (lens.convention !== "isolated_component_softmax" || lens.position !== position
			|| typeof tokens[position] !== "string" || lens.position_token !== tokens[position]) {
			return unavailable("The cached decoding does not match this component's clean-prompt position.");
		}
		const top = lens.top_tokens;
		if (!Array.isArray(top) || !top.length || top.some((item) => !item || typeof item.token !== "string"
			|| !Number.isFinite(item.probability) || item.probability < 0 || item.probability > 1)
			|| top.reduce((sum, item) => sum + item.probability, 0) > 1.000001) {
			return unavailable("Valid token probabilities are unavailable for this component.");
		}
		return { available: true, position, positionToken: lens.position_token, fallback,
			topTokens: [...top].sort((a, b) => b.probability - a.probability).slice(0, 9) };
	}

	function render(container, { component = null, node = null, meta = {}, name } = {}) {
		observers.get(container)?.disconnect();
		observers.delete(container);
		container.replaceChildren();
		const append = (tag, className, text, parent = container) => {
			const element = container.ownerDocument.createElement(tag);
			element.setAttribute("class", className);
			if (text !== undefined) element.textContent = text;
			parent.appendChild(element);
			return element;
		};
		if (!component) {
			append("p", "component-decoding-empty", "Select a component in the path diagram to inspect its Top Tokens.");
			return;
		}
		append("p", "component-decoding-name", name(component));
		const readout = readoutFor(component, node, meta);
		if (!readout.available) {
			append("p", "component-decoding-empty", readout.message);
			return;
		}
		const coordinate = component.kind === "attn" ? "Query/output position" : "Token position";
		append("p", "component-decoding-position", `${coordinate}: ${readout.position} (${JSON.stringify(visibleToken(readout.positionToken))})`);
		if (readout.fallback) append("p", "component-decoding-note", "All-position component: using the last clean-prompt token (existing logit-lens fallback).");
		append("p", "component-decoding-scale", `Top ${readout.topTokens.length} tokens · full-vocabulary softmax`);
		const chart = append("div", "component-decoding-chart");
		const plot = append("div", "component-decoding-plot", undefined, chart);
		plot.setAttribute("tabindex", "0");
		plot.setAttribute("aria-label", "Decoded tokens, independently scrollable; probability axis stays visible");
		const axis = container.ownerDocument.createElementNS(SVGNS, "svg");
		axis.setAttribute("class", "component-decoding-axis");
		axis.setAttribute("height", "44");
		axis.setAttribute("role", "img");
		axis.setAttribute("aria-label", "Probability axis, starting at zero");
		chart.appendChild(axis);
		let drawnWidth;
		const draw = () => {
			const width = plot.clientWidth || 300;
			if (width === drawnWidth) return;
			drawnWidth = width;
			renderHistogram(plot, axis, readout.topTokens);
		};
		append("p", "component-decoding-disclaimer", "Exploratory component projection, not the full model's next-token probabilities or the selected path's causal contribution.");
		draw();
		const Observer = container.ownerDocument.defaultView?.ResizeObserver;
		if (Observer) {
			const observer = new Observer(draw);
			observer.observe(plot);
			observers.set(container, observer);
		}
	}

	return { readoutFor, formatProbability, probabilityScale, chartGeometry, renderHistogram, render };
})();

if (typeof module !== "undefined") module.exports = ComponentDecoding;
