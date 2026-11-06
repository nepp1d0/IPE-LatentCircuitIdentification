"use strict";

/* Structural adaptation of drawPathVisualization in
 * web_app/static/js/main.js (the original secondary IPE plot).*/
const PathDiagram = (() => {
	const SVGNS = "http://www.w3.org/2000/svg";
	function componentStyle(kind) {
		switch (kind) {
			case "attn": return { symbol: "circle", color: "#B0C4DE" };
			case "mlp": return { symbol: "square", color: "#90EE90" };
			case "embed": return { symbol: "square", color: "#FFDAB9" };
			case "final": return { symbol: "square", color: "#FFA07A" };
			default: return { symbol: "diamond", color: "#D3D3D3" };
		}
	}

	function occurrenceVariant(n, variant) {
		const saved = variant(n);
		const patch = typeof saved === "string" && (/^[QKV]+$/.test(saved) || saved === "none") ? saved : null;
		const flags = ["patch_query", "patch_key", "patch_value"];
		// An explicit original variant may fill absent flags, but must not
		// contradict any supplied boolean flag.
		return patch && flags.some((flag, i) => typeof n[flag] === "boolean" && n[flag] !== patch.includes("QKV"[i])) ? null : patch;
	}

	function metadataRows(n, { variant, meta = {}, original = true }) {
		// Grid nodes can supply a physical component name, never an occurrence's
		// intervention coordinates or patch flags.
		const source = original ? n : {};
		const positional = (value, single, all) => {
			if (value === null) return { label: all };
			if (!Number.isInteger(value) || value < 0) return { label: `${single}: Unavailable` };
			const token = Array.isArray(meta?.tokens) ? meta.tokens[value] : undefined;
			const visible = typeof token === "string" && token.length
				? token.replace(/\r\n?/g, "\n").replace(/ /g, "␠").replace(/\t/g, "⇥").replace(/\n/g, "↵") : null;
			return { label: `${single}: ${value}`, token: visible === null ? null : ` (${JSON.stringify(visible)})` };
		};
		const rows = [positional(source.position, n.kind === "attn" ? "Query/output position" : "Token position",
			n.kind === "attn" ? "Query/output positions: all eligible" : "Token positions: all")];
		if (n.kind !== "attn") return rows;
		const patch = original ? occurrenceVariant(source, variant) : null;
		const patchesKV = patch ? patch.includes("K") || patch.includes("V") : null;
		const patchLabel = patch === null ? "Patch: Unavailable" : `Patch ${patch}`;
		if (patchesKV === false && !Number.isInteger(source.kv_position)) {
			// Q-only (or no patched streams) needs no single K/V patch source.
			rows.push({ label: patchLabel });
		} else {
			// Custom Q-only occurrences can still restrict their output to a
			// particular read source. Preserve it without calling it a K/V patch.
			const sourceRow = positional(source.kv_position, patchesKV === false ? "Read source position" : "Source position",
				"Source positions: all eligible");
			rows.push({ ...sourceRow, label: `${patchLabel} · ${sourceRow.label}` });
		}
		return rows;
	}

	function wrapSegments(segments, maxWidth, measure) {
		const lines = [];
		let line = [];
		const text = () => line.map((s) => s.text).join("");
		const append = (segment) => {
			const last = line.at(-1);
			if (last && last.secondary === segment.secondary) last.text += segment.text;
			else line.push({ ...segment });
		};
		const flush = () => {
			while (line.length && !line.at(-1).text.trimEnd()) line.pop();
			if (line.length) {
				line.at(-1).text = line.at(-1).text.trimEnd();
				lines.push(line);
			}
			line = [];
		};
		for (const segment of segments) {
			for (const word of segment.text.match(/\s+|\S+/gu) || []) {
				if (!word.trim()) {
					if (line.length) append({ ...segment, text: word });
					continue;
				}
				if (measure(text() + word) > maxWidth) flush();
				let rest = Array.from(word);
				while (rest.length && measure(rest.join("")) > maxWidth) {
					let low = 1, high = rest.length;
					while (low < high) {
						const mid = Math.ceil((low + high) / 2);
						if (measure(rest.slice(0, mid).join("")) <= maxWidth) low = mid;
						else high = mid - 1;
					}
					append({ ...segment, text: rest.splice(0, low).join("") });
					flush();
				}
				if (rest.length) append({ ...segment, text: rest.join("") });
			}
		}
		flush();
		return lines;
	}

	function render(container, path, { name, variant, meta = {}, occurrences = [] }) {
		container.replaceChildren();
		if (!path) return;
		const document = container.ownerDocument;
		const svgEl = (tag, attrs, parent, text) => {
			const node = document.createElementNS(SVGNS, tag);
			for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
			if (text !== undefined) node.textContent = text;
			parent.appendChild(node);
			return node;
		};
		// Verify provenance by original object identity and its projected grid ID;
		// a legacy merged-grid fallback must not masquerade as an occurrence.
		const originals = new Set(occurrences);
		const components = path.components.map((n, i) => {
			const original = originals.has(n) && n.node_id === path.sequence[i];
			return { n, original, rows: metadataRows(n, { variant, meta, original }) };
		}).reverse();
		const hasTokens = components.some((c) => c.rows.some((r) => r.token));
		if (hasTokens) {
			const context = document.createElement("p");
			context.setAttribute("class", "path-token-context");
			context.textContent = "Token labels: first clean prompt. ␠ marks a space.";
			container.appendChild(context);
		}
		const svg = svgEl("svg", { viewBox: "0 0 268 64", width: "100%", height: 64,
			role: "img" }, container);
		const links = svgEl("g", {}, svg), centers = [], descriptions = [];
		let cy = 24, bottom = 64;
		components.forEach(({ n, rows }) => {
			centers.push(cy);
			const style = componentStyle(n.kind);
			const group = svgEl("g", { class: "path-component", "data-kind": n.kind,
				"data-node-id": n.node_id || n.id, "data-occurrence-id": n.id }, svg);
			const layer = n.layer !== null && n.layer !== undefined ? ` · Layer ${n.layer}` : "";
			const description = [name(n) + layer, ...rows.map((r) => r.label + (r.token || ""))].join("; ");
			descriptions.push(description);
			svgEl("title", {}, group, description);
			const attrs = { fill: style.color, stroke: "#555555", "stroke-width": 2 };
			if (style.symbol === "circle") svgEl("circle", { ...attrs, cx: 18, cy, r: 9 }, group);
			else if (style.symbol === "square") svgEl("rect", { ...attrs, x: 9, y: cy - 9, width: 18, height: 18 }, group);
			else svgEl("polygon", { ...attrs, points: `18,${cy - 10} 28,${cy} 18,${cy + 10} 8,${cy}` }, group);
			let y = cy - 7;
			for (const [index, row] of [{ label: name(n) + layer }, ...rows].entries()) {
				const text = svgEl("text", { class: index === 0 ? "path-component-name" : "path-component-meta" }, group);
				const segments = [{ text: row.label, secondary: false }, ...(row.token ? [{ text: row.token, secondary: true }] : [])];
				const measure = (value) => {
					text.textContent = value;
					return text.getComputedTextLength?.() || Array.from(value).length * (index === 0 ? 7 : 5.5);
				};
				const lines = wrapSegments(segments, 224, measure);
				text.replaceChildren();
				for (const line of lines) {
					const tspan = svgEl("tspan", { x: 36, y }, text);
					for (const segment of line) svgEl("tspan", segment.secondary ? { class: "path-component-token" } : {}, tspan, segment.text);
					y += index === 0 ? 16 : 14;
				}
			}
			bottom = y + 8;
			cy = Math.max(cy + 64, y + 18);
		});
		for (let i = 1; i < components.length; i++) {
			const target = components[i - 1];
			const attrs = { class: "path-component-link", x1: 18, x2: 18,
				y1: centers[i - 1], y2: centers[i], stroke: "#607D8B", "stroke-width": 2 };
			if (target.original && target.n.kind === "attn" && occurrenceVariant(target.n, variant)?.includes("Q")) attrs["stroke-dasharray"] = "3 3";
			svgEl("line", attrs, links);
		}
		const height = Math.max(64, components.length * 64, bottom);
		svg.setAttribute("viewBox", `0 0 268 ${height}`);
		svg.setAttribute("height", height);
		svg.setAttribute("aria-label", `Path ${path.ordinal} (${path.id}), FINAL to EMB${hasTokens ? " (token labels from the first clean prompt)" : ""}: ${descriptions.join(" → ")}`);
	}

	return { componentStyle, metadataRows, wrapSegments, render };
})();

if (typeof module !== "undefined") module.exports = PathDiagram;
