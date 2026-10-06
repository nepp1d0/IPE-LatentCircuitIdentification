"use strict";

/* IPE circuit discovery frontend.
 *
 * The main view is an SVG grid: token positions on the x axis, layers on the
 * y axis (FINAL at the top, per layer MLP above ATTN, EMB at the bottom).
 * The whole model is drawn as faint placeholder cells; discovered components
 * are rendered as head-level chips on top of it and the circuit's edges as
 * bezier links between chips (a horizontal jog = a K/V read at an earlier
 * position). Results stream in live over SSE while the search runs.
 */

const $ = (s) => document.querySelector(s);

const GUT = 64;      // left gutter for row labels
const COLW = 76;     // column width (one token position)
const ROWH = 32;     // row height (one layer band)
const HDR = 44;      // header height (position index + token)
const SVGNS = "http://www.w3.org/2000/svg";

const state = {
	config: null,
	meta: null,                 // {tokens, n_layers, n_heads, positional}
	previewVersion: 0,          // invalidates tokenize responses after edits/tab changes
	nodes: new Map(),           // id -> node dict
	edges: new Map(),           // "src->dst" -> {source, target, count, contribution}
	maxAbs: 0,
	threshold: 0,
	method: "tree",
	savedResults: { tree: null, path: null, acdc: null }, // latest successful run per algorithm
	drafts: { tree: null, path: null, acdc: null }, // unfinished form for a method with no saved results
	strategy: "threshold",
	mode: "noising",
	jobId: null,
	es: null,
	activeJob: null,              // request snapshot and stream metadata for the running job
	running: false,
	methodStatus: { tree: null, path: null, acdc: null },
	methodProgress: { tree: null, path: null, acdc: null },
	displayingLiveRun: false,
	result: null,
	view: { x: 16, y: 8, k: 1 },
	dirty: false,
	hover: null,
	admitQueue: [],          // incoming admit events, revealed one by one
	fresh: new Map(),        // node id -> reveal timestamp (drives the pulse)
};

/* ------------------------------------------------------------------ utils */

function el(tag, attrs = {}, parent = null) {
	const node = document.createElementNS(SVGNS, tag);
	for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
	if (parent) parent.appendChild(node);
	return node;
}

function lines(text) {
	return text.split("\n").map((s) => s.replace(/\r/, "")).filter((s) => s.length > 0);
}

function debounce(fn, ms) {
	let t = null;
	return (...args) => {
		clearTimeout(t);
		t = setTimeout(() => fn(...args), ms);
	};
}

function cssVar(name) {
	return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
	const h = hex.replace("#", "");
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function mix(a, b, t) {
	const A = hexToRgb(a), B = hexToRgb(b);
	const c = A.map((v, i) => Math.round(v + (B[i] - v) * t));
	return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/* Diverging color for a signed contribution: neutral midpoint -> blue for
 * positive, red for negative, saturating at the strongest |contribution|. */
function colorFor(c) {
	const live = state.activeJob?.liveState;
	const maxAbs = live && state.nodes === live.nodes ? live.maxAbs : state.maxAbs;
	if (c === null || c === undefined || maxAbs === 0) return null;
	const t = Math.min(1, Math.abs(c) / maxAbs);
	const pole = c >= 0 ? cssVar("--pos") : cssVar("--neg");
	return mix(cssVar("--mid"), pole, 0.3 + 0.7 * t);
}

function fmt(c) {
	return (c >= 0 ? "+" : "") + c.toFixed(4);
}

function nodeLabel(n) {
	if (n.kind === "final") return "FINAL";
	if (n.kind === "embed") return "EMB";
	if (n.kind === "mlp") return `MLP${n.layer}`;
	return `A${n.layer}H${n.head === null || n.head === undefined ? "*" : n.head}`;
}

/* ------------------------------------------------------------ graph state */

function clearDisplay(message = "Type a clean prompt to preview the model grid.") {
	state.nodes = new Map();
	state.edges = new Map();
	state.admitQueue = [];
	state.fresh = new Map();
	state.maxAbs = 0;
	state.meta = null;
	state.result = null;
	state.threshold = 0;
	state.hover = null;
	state.displayingLiveRun = false;
	$("#threshold-slider").value = 0;
	$("#threshold-val").textContent = "0";
	$("#summary").innerHTML = "";
	$("#result-box").hidden = true;
	$("#tooltip").hidden = true;
	$("#empty-state").textContent = message;
	scheduleRender();
}

function createLiveGraphState() {
	return {
		meta: null,
		nodes: new Map(),
		edges: new Map(),
		maxAbs: 0,
		threshold: 0,
		admitQueue: [],
		fresh: new Map(),
	};
}

/* Reveal admitted nodes one by one (draining faster if a burst piles up, e.g.
 * from the beam variants which admit a whole depth at once). */
function applyAdmit(ev, graphState = state) {
	mergeNode(ev.parent, graphState);
	mergeNode(ev.node, graphState);
	graphState.fresh.set(ev.node.id, performance.now());
	mergeEdge(ev.node.id, ev.parent.id, ev.contribution, ev.node.kind === "embed", graphState);
	if (ev.node.kind === "embed") markCompleteUp(ev.node.id, graphState);
}

function drainAdmitQueue(graphState, maxItems = null) {
	const n = maxItems === null
		? Math.max(1, Math.ceil(graphState.admitQueue.length / 25))
		: Math.min(maxItems, graphState.admitQueue.length);
	for (let i = 0; i < n && graphState.admitQueue.length; i++) {
		applyAdmit(graphState.admitQueue.shift(), graphState);
	}
}

setInterval(() => {
	const job = state.activeJob;
	const graphState = job?.liveState || state;
	if (!graphState.admitQueue.length) return;
	drainAdmitQueue(graphState);
	if (!job || isDisplayedJob(job)) scheduleRender();
}, 60);

function mergeNode(d, graphState = state) {
	const cur = graphState.nodes.get(d.id);
	if (!cur) {
		graphState.nodes.set(d.id, {
			...d,
			merged: d.merged || 1,
			complete: d.complete !== undefined ? d.complete : true,
			variants: d.variants ? [...d.variants] : (d.variant ? [d.variant] : []),
			kv_positions: d.kv_positions ? [...d.kv_positions]
				: (d.kv_position !== null && d.kv_position !== undefined ? [d.kv_position] : []),
		});
	} else {
		cur.merged += 1;
		if (d.contribution !== null && d.contribution !== undefined) {
			if (cur.contribution === null || cur.contribution === undefined
				|| Math.abs(d.contribution) > Math.abs(cur.contribution)) {
				cur.contribution = d.contribution;
			}
		}
		if (d.variant && !cur.variants.includes(d.variant)) cur.variants.push(d.variant);
		if (d.kv_position !== null && d.kv_position !== undefined
			&& !cur.kv_positions.includes(d.kv_position)) cur.kv_positions.push(d.kv_position);
	}
	const c = graphState.nodes.get(d.id).contribution;
	if (c !== null && c !== undefined) graphState.maxAbs = Math.max(graphState.maxAbs, Math.abs(c));
}

function mergeEdge(srcId, dstId, contribution, complete = false, graphState = state) {
	if (srcId === dstId) return;
	const key = `${srcId}->${dstId}`;
	const cur = graphState.edges.get(key);
	if (!cur) {
		graphState.edges.set(key, {
			source: srcId, target: dstId, count: 1,
			contribution: contribution ?? null, complete,
		});
	} else {
		cur.count += 1;
		cur.complete = cur.complete || complete;
		if (contribution !== null && contribution !== undefined) {
			if (cur.contribution === null || Math.abs(contribution) > Math.abs(cur.contribution)) {
				cur.contribution = contribution;
			}
		}
	}
}

/* When a branch reaches an embedding, everything upstream of it (toward FINAL)
 * is part of a complete branch: flip those edges to the coral accent. */
function markCompleteUp(startId, graphState = state) {
	const bySource = new Map();
	for (const e of graphState.edges.values()) {
		if (!bySource.has(e.source)) bySource.set(e.source, []);
		bySource.get(e.source).push(e);
	}
	const stack = [startId];
	const seen = new Set();
	while (stack.length) {
		const cur = stack.pop();
		if (seen.has(cur)) continue;
		seen.add(cur);
		for (const e of bySource.get(cur) || []) {
			e.complete = true;
			stack.push(e.target);
		}
	}
}

function loadGraph(graph, graphState = state, shouldRender = true) {
	graphState.nodes.clear();
	graphState.edges.clear();
	graphState.maxAbs = 0;
	for (const n of graph.nodes) {
		graphState.nodes.set(n.id, { ...n });
		if (n.contribution !== null && n.contribution !== undefined) {
			graphState.maxAbs = Math.max(graphState.maxAbs, Math.abs(n.contribution));
		}
	}
	for (const e of graph.edges) {
		graphState.edges.set(`${e.source}->${e.target}`, { ...e });
	}
	// Belt and braces for the final view: any branch that reaches an embedding
	// stays coral after the search, whatever the server-side flags say.
	for (const n of graphState.nodes.values()) {
		if (n.kind === "embed") markCompleteUp(n.id, graphState);
	}
	if (shouldRender) scheduleRender();
}

/* --------------------------------------------------------------- geometry */

function rowList() {
	const rows = [{ key: "final", label: "FINAL" }];
	for (let l = state.meta.n_layers - 1; l >= 0; l--) {
		rows.push({ key: `mlp${l}`, label: `L${l} mlp` });
		rows.push({ key: `attn${l}`, label: `L${l} attn` });
	}
	rows.push({ key: "embed", label: "EMB" });
	return rows;
}

function rowKeyFor(n) {
	if (n.kind === "final") return "final";
	if (n.kind === "embed") return "embed";
	return `${n.kind}${n.layer}`;
}

/* Two x-axis layouts. The IPE searches are positional, so the columns are token
 * positions. ACDC is position-agnostic but head-resolved, so laying its result
 * out by token would pile all 144 heads into one column: there the columns are
 * head indices instead, plus a trailing column for the head-less components
 * (MLP / EMB / FINAL). */
function byHead() {
	return state.meta.layout === "head";
}

function colList() {
	if (byHead()) {
		const cols = [];
		for (let h = 0; h < state.meta.n_heads; h++) cols.push({ p: h, label: "head" });
		cols.push({ p: null, label: "mlp / emb" });
		return cols;
	}
	if (!state.meta.positional) return [{ p: null, label: "any" }];
	const cols = state.meta.tokens.map((t, i) => ({ p: i, label: t }));
	for (const n of state.nodes.values()) {
		if (n.position === null || n.position === undefined) {
			cols.push({ p: null, label: "any" });
			break;
		}
	}
	return cols;
}

/* Which column a node sits in, under the active layout. */
function colKeyFor(n) {
	const v = byHead() ? n.head : n.position;
	return v === undefined ? null : v;
}

function nodeVisible(n) {
	if (n.kind === "final") return true;
	if (n.contribution === null || n.contribution === undefined) return true;
	return Math.abs(n.contribution) >= state.threshold;
}

/* ----------------------------------------------------------------- render */

function scheduleRender() {
	state.dirty = true;
}

setInterval(() => {
	if (state.dirty) {
		state.dirty = false;
		render();
	}
}, 120);

function render() {
	const svg = $("#grid");
	svg.innerHTML = "";
	if (!state.meta) {
		$("#empty-state").style.display = "flex";
		return;
	}
	$("#empty-state").style.display = "none";

	const vp = el("g", {
		id: "viewport",
		transform: `translate(${state.view.x},${state.view.y}) scale(${state.view.k})`,
	}, svg);

	const rows = rowList();
	const cols = colList();
	const rowIdx = new Map(rows.map((r, i) => [r.key, i]));
	const colIdx = new Map(cols.map((c, i) => [c.p, i]));
	const colX = (i) => GUT + i * COLW;
	const rowY = (i) => HDR + i * ROWH;
	const width = GUT + cols.length * COLW;

	/* header: position index + token, then a rule */
	const gh = el("g", {}, vp);
	cols.forEach((c, i) => {
		const cx = colX(i) + COLW / 2;
		el("text", { class: "hdr-pos", x: cx, y: 12 }, gh).textContent = c.p === null ? "*" : c.p;
		const tok = (c.label || "").trim() || "·";
		el("text", { class: "hdr-tok", x: cx, y: 28 }, gh)
			.textContent = tok.length > 9 ? tok.slice(0, 8) + "…" : tok;
	});
	el("line", { class: "hdr-rule", x1: GUT - 6, y1: HDR - 8, x2: width, y2: HDR - 8 }, gh);

	/* row labels + faint skeleton cells (the whole model as background) */
	const gc = el("g", {}, vp);
	rows.forEach((r, ri) => {
		el("text", { class: "row-label", x: GUT - 10, y: rowY(ri) + ROWH / 2 }, gc).textContent = r.label;
		cols.forEach((c, ci) => {
			el("rect", {
				class: "cell", x: colX(ci) + 2, y: rowY(ri) + 2,
				width: COLW - 6, height: ROWH - 6, rx: 4,
			}, gc);
		});
	});

	/* chip layout: group visible nodes per cell, spread them side by side */
	const visible = [...state.nodes.values()].filter(nodeVisible);
	const cells = new Map(); // "rowKey|colIdx" -> [node,...]
	for (const n of visible) {
		const ri = rowIdx.get(rowKeyFor(n));
		const ci = colIdx.get(colKeyFor(n));
		if (ri === undefined || ci === undefined) continue;
		const key = `${ri}|${ci}`;
		if (!cells.has(key)) cells.set(key, []);
		cells.get(key).push(n);
	}

	const anchors = new Map(); // id -> {cx, top, bottom}
	const gE = el("g", {}, vp); // edges under chips
	const gN = el("g", {}, vp);

	for (const [key, nodesInCell] of cells) {
		const [ri, ci] = key.split("|").map(Number);
		nodesInCell.sort((a, b) => (a.head ?? -1) - (b.head ?? -1));
		const cw = (COLW - 8) / nodesInCell.length;
		nodesInCell.forEach((n, i) => {
			const x = colX(ci) + 4 + i * cw;
			const y = rowY(ri) + 4;
			const h = ROWH - 10;
			anchors.set(n.id, { cx: x + cw / 2, top: y, bottom: y + h });
			const freshAt = state.fresh.get(n.id);
			const age = freshAt === undefined ? Infinity : performance.now() - freshAt;
			if (age > 600) state.fresh.delete(n.id);
			const g = el("g", {
				class: "chip" + (n.complete === false ? " incomplete" : "")
					+ (age <= 600 ? " pulse" : ""),
				"data-id": n.id,
			}, gN);
			const fill = colorFor(n.contribution);
			const rect = el("rect", {
				x, y, width: Math.max(cw - 2, 10), height: h, rx: 4,
				fill: fill || "var(--surface-1)",
				stroke: fill ? (n.contribution >= 0 ? "var(--pos)" : "var(--neg)") : "var(--baseline)",
			}, g);
			// Resume the pulse mid-animation after a re-render, not from the top.
			if (age <= 600) rect.style.animationDelay = `-${(age / 1000).toFixed(3)}s`;
			const label = nodeLabel(n);
			const t = el("text", { x: x + cw / 2, y: y + h / 2 }, g);
			t.textContent = (cw < 34 && label.length > 4) ? label.replace(/^A/, "").replace("MLP", "M") : label;
			g.addEventListener("mouseenter", (ev) => onHoverNode(n, ev));
			g.addEventListener("mousemove", moveTooltip);
			g.addEventListener("mouseleave", clearHover);
		});
	}

	/* edges: child (lower band) -> parent (upper band), bezier through the
	 * vertical midpoint; a horizontal offset reads as a K/V positional jog.
	 * Branches that reach an embedding are drawn in the coral accent, thicker;
	 * while the search runs every edge pumps coral dashes root -> leaves. */
	for (const e of state.edges.values()) {
		const s = anchors.get(e.source), t = anchors.get(e.target);
		if (!s || !t) continue;
		const midY = (s.top + t.bottom) / 2;
		const d = `M ${s.cx} ${s.top} C ${s.cx} ${midY}, ${t.cx} ${midY}, ${t.cx} ${t.bottom}`;
		const attrs = { class: "edge", d, "data-key": `${e.source}->${e.target}` };
		if (e.complete) {
			attrs.stroke = "var(--accent)";
			attrs["stroke-width"] = 2.2 + Math.min(2.4, Math.log2(e.count + 1));
			attrs.opacity = 0.85;
		} else if (state.displayingLiveRun) {
			attrs.stroke = "var(--accent)";
			attrs["stroke-width"] = 1.4;
			attrs.opacity = 0.45;
		} else {
			attrs.stroke = "var(--baseline)";
			attrs["stroke-width"] = 1.2;
			attrs["stroke-dasharray"] = "3 4";
			attrs.opacity = 0.55;
		}
		if (state.displayingLiveRun) attrs.class += " searching";
		const p = el("path", attrs, gE);
		if (state.displayingLiveRun) {
			// Keep the dash phase continuous across re-renders.
			p.style.animationDelay = `-${((performance.now() / 1000) % 0.9).toFixed(3)}s`;
		}
		p.addEventListener("mouseenter", (ev) => onHoverEdge(e, ev));
		p.addEventListener("mousemove", moveTooltip);
		p.addEventListener("mouseleave", clearHover);
	}

	applyHighlight();
}

/* -------------------------------------------------------------- highlight */

function relatedSet(id) {
	const up = new Map(), down = new Map();
	for (const e of state.edges.values()) {
		if (!up.has(e.source)) up.set(e.source, []);
		up.get(e.source).push(e);
		if (!down.has(e.target)) down.set(e.target, []);
		down.get(e.target).push(e);
	}
	const nodes = new Set([id]);
	const edges = new Set();
	const walk = (start, adj, next) => {
		const stack = [start];
		while (stack.length) {
			const cur = stack.pop();
			for (const e of adj.get(cur) || []) {
				edges.add(`${e.source}->${e.target}`);
				const n = next(e);
				if (!nodes.has(n)) { nodes.add(n); stack.push(n); }
			}
		}
	};
	walk(id, up, (e) => e.target);    // toward FINAL
	walk(id, down, (e) => e.source);  // toward the leaves
	return { nodes, edges };
}

function applyHighlight() {
	const svg = $("#grid");
	const chips = svg.querySelectorAll(".chip");
	const edges = svg.querySelectorAll(".edge");
	if (!state.hover) {
		chips.forEach((c) => c.classList.remove("dim"));
		edges.forEach((e) => { e.classList.remove("dim"); e.classList.remove("hot"); });
		return;
	}
	const rel = state.hover;
	chips.forEach((c) => c.classList.toggle("dim", !rel.nodes.has(c.dataset.id)));
	edges.forEach((e) => {
		const on = rel.edges.has(e.dataset.key);
		e.classList.toggle("dim", !on);
		e.classList.toggle("hot", on);
	});
}

function onHoverNode(n, ev) {
	state.hover = relatedSet(n.id);
	applyHighlight();
	const tokens = state.meta ? state.meta.tokens : [];
	const rowsOut = [`<div class="tt-title">${nodeLabel(n)}</div>`];
	if (n.position !== null && n.position !== undefined) {
		rowsOut.push(`position ${n.position} <span class="tt-dim">${JSON.stringify(tokens[n.position] ?? "")}</span>`);
	} else {
		rowsOut.push(`<span class="tt-dim">position-agnostic</span>`);
	}
	if (n.contribution !== null && n.contribution !== undefined) rowsOut.push(`contribution <b>${fmt(n.contribution)}</b>`);
	if (n.kind === "embed" && n.contribution === 0) {
		rowsOut.push(`<span class="tt-dim">zero isolated effect — the clean and counterfactual
			embeddings are identical at this position, so patching it does nothing</span>`);
	}
	if (n.variants && n.variants.length) rowsOut.push(`patched streams: ${n.variants.join(", ")}`);
	if (n.kv_positions && n.kv_positions.length) {
		rowsOut.push(`K/V read at: ${n.kv_positions.map((p) => `${p} ${JSON.stringify((tokens[p] ?? "").trim())}`).join(", ")}`);
	}
	if (n.merged > 1 && !byHead()) rowsOut.push(`<span class="tt-dim">${n.merged} tree nodes merged</span>`);
	if (n.complete === false) rowsOut.push(`<span class="tt-dim">on a pruned branch (never reached EMB)</span>`);
	showTooltip(rowsOut.join("<br>"), ev);
}

function onHoverEdge(e, ev) {
	state.hover = { nodes: new Set([e.source, e.target]), edges: new Set([`${e.source}->${e.target}`]) };
	applyHighlight();
	const s = state.nodes.get(e.source), t = state.nodes.get(e.target);
	const rows = [`<div class="tt-title">${s ? nodeLabel(s) : e.source} &rarr; ${t ? nodeLabel(t) : e.target}</div>`];
	if (e.contribution !== null && e.contribution !== undefined) {
		rows.push(byHead()
			? `strongest effect of cutting it: <b>${fmt(e.contribution)}</b>`
			: `strongest branch: <b>${fmt(e.contribution)}</b>`);
	}
	if (e.count > 1) {
		rows.push(`<span class="tt-dim">${e.count} ${byHead() ? "hook-level" : "tree"} edges merged</span>`);
	}
	rows.push(e.complete
		? `<span class="tt-dim">on a complete branch (reaches EMB)</span>`
		: `<span class="tt-dim">pruned branch</span>`);
	showTooltip(rows.join("<br>"), ev);
}

function clearHover() {
	state.hover = null;
	applyHighlight();
	$("#tooltip").hidden = true;
}

function showTooltip(html, ev) {
	const tt = $("#tooltip");
	tt.innerHTML = html;
	tt.hidden = false;
	moveTooltip(ev);
}

function moveTooltip(ev) {
	const tt = $("#tooltip");
	if (tt.hidden) return;
	const wrap = $("#canvas-wrap").getBoundingClientRect();
	let x = ev.clientX - wrap.left + 14;
	let y = ev.clientY - wrap.top + 14;
	if (x + tt.offsetWidth > wrap.width - 8) x = ev.clientX - wrap.left - tt.offsetWidth - 10;
	if (y + tt.offsetHeight > wrap.height - 8) y = ev.clientY - wrap.top - tt.offsetHeight - 10;
	tt.style.left = `${x}px`;
	tt.style.top = `${y}px`;
}

/* ---------------------------------------------------------------- pan/zoom */

function setupPanZoom() {
	const svg = $("#grid");
	let panning = null;
	svg.addEventListener("pointerdown", (e) => {
		if (e.target.closest(".chip") || e.target.closest(".edge")) return;
		panning = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y };
		svg.classList.add("panning");
		svg.setPointerCapture(e.pointerId);
	});
	svg.addEventListener("pointermove", (e) => {
		if (!panning) return;
		state.view.x = panning.vx + (e.clientX - panning.x);
		state.view.y = panning.vy + (e.clientY - panning.y);
		$("#viewport")?.setAttribute("transform",
			`translate(${state.view.x},${state.view.y}) scale(${state.view.k})`);
	});
	svg.addEventListener("pointerup", () => { panning = null; svg.classList.remove("panning"); });
	svg.addEventListener("wheel", (e) => {
		e.preventDefault();
		const rect = svg.getBoundingClientRect();
		const mx = e.clientX - rect.left, my = e.clientY - rect.top;
		const k0 = state.view.k;
		const k = Math.min(3, Math.max(0.25, k0 * (e.deltaY < 0 ? 1.12 : 0.89)));
		state.view.x = mx - ((mx - state.view.x) / k0) * k;
		state.view.y = my - ((my - state.view.y) / k0) * k;
		state.view.k = k;
		$("#viewport")?.setAttribute("transform",
			`translate(${state.view.x},${state.view.y}) scale(${state.view.k})`);
	}, { passive: false });
	$("#zoom-in").addEventListener("click", () => { state.view.k = Math.min(3, state.view.k * 1.2); scheduleRender(); });
	$("#zoom-out").addEventListener("click", () => { state.view.k = Math.max(0.25, state.view.k / 1.2); scheduleRender(); });
	$("#zoom-reset").addEventListener("click", () => { state.view = { x: 16, y: 8, k: 1 }; scheduleRender(); });
}

/* -------------------------------------------------------------------- API */

async function api(path, body) {
	const res = await fetch(path, {
		method: body === undefined ? "GET" : "POST",
		headers: { "Content-Type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
	});
	if (!res.ok) {
		let detail = res.statusText;
		try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
		throw new Error(detail);
	}
	return res.json();
}

async function loadConfig() {
	state.config = await api("/api/config");
	$("#device-chip").textContent = `device: ${state.config.device}`;
	const modelSel = $("#model");
	for (const m of state.config.models) {
		const o = document.createElement("option");
		o.value = o.textContent = m;
		modelSel.appendChild(o);
	}
	const metricSel = $("#metric");
	for (const m of state.config.metrics) {
		const o = document.createElement("option");
		o.value = o.textContent = m;
		metricSel.appendChild(o);
	}
	const acdcSel = $("#acdc-metric");
	for (const m of state.config.acdc_metrics || []) {
		const o = document.createElement("option");
		o.value = o.textContent = m;
		acdcSel.appendChild(o);
	}
	updateModeHint();
	applyMethod();
}

const tokenizePreview = debounce(async () => {
	if (state.result || state.displayingLiveRun
		|| (state.activeJob?.method === state.method && state.activeJob.displayActive)) return;
	const method = state.method;
	const prompt = lines($("#prompts").value)[0];
	if (!prompt) return;
	const model = $("#model").value;
	const positional = $("#positional").checked;
	const version = ++state.previewVersion;
	$("#empty-state").textContent = "Tokenizing (first call loads the model)…";
	$("#empty-state").style.display = "flex";
	try {
		const out = await api("/api/tokenize", { model, prompt });
		if (version !== state.previewVersion || state.result || state.displayingLiveRun
			|| state.method !== method || lines($("#prompts").value)[0] !== prompt
			|| $("#model").value !== model || $("#positional").checked !== positional
			|| (state.activeJob?.method === method && state.activeJob.displayActive)) return;
		state.meta = {
			tokens: out.tokens,
			n_layers: out.n_layers,
			n_heads: out.n_heads,
			positional,
			layout: method === "acdc" ? "head" : "position",
		};
		scheduleRender();
	} catch (e) {
		if (version !== state.previewVersion || state.method !== method
			|| state.result || state.displayingLiveRun
			|| lines($("#prompts").value)[0] !== prompt || $("#model").value !== model
			|| (state.activeJob?.method === method && state.activeJob.displayActive)) return;
		$("#empty-state").textContent = `Tokenize failed: ${e.message}`;
	}
}, 600);

/* ----------------------------------------------------------------- search */

function setRunning(on) {
	state.running = on;
	$("#run").disabled = on;
	$("#cancel").hidden = !on;
	refreshMethodStatus();
}

function setStatus(msg, method = state.method) {
	state.methodStatus[method] = msg;
	if (method === state.method) refreshMethodStatus();
}

function setProgress(frac, txt, method = state.method) {
	state.methodProgress[method] = { frac, text: txt };
	if (method === state.method) refreshMethodStatus();
}

function refreshMethodStatus() {
	const method = state.method;
	const status = state.methodStatus[method];
	const progress = state.methodProgress[method];
	const runningHere = state.running && state.activeJob?.method === method;
	$("#status").hidden = status === null && !runningHere;
	$("#status-msg").textContent = status || "";
	$("#progress-fill").style.width = `${Math.round((progress?.frac ?? 0) * 100)}%`;
	$("#progress-txt").textContent = progress?.text || "";
}

async function runSearch() {
	const body = {
		model: $("#model").value,
		prompts: lines($("#prompts").value),
		targets: lines($("#targets").value),
		cf_prompts: lines($("#cf-prompts").value),
		cf_targets: lines($("#cf-targets").value),
		method: state.method,
		strategy: state.strategy,
		mode: state.mode,
		min_contribution: parseFloat($("#min-contribution").value) || 0.05,
		max_width: parseInt($("#max-width").value, 10) || 20,
		metric: $("#metric").value,
		positional: $("#positional").checked,
		include_negative: $("#include-negative").checked,
		acdc_metric: $("#acdc-metric").value,
		acdc_threshold: parseFloat($("#acdc-threshold").value) || 0,
		acdc_abs_threshold: $("#acdc-abs-threshold").checked,
	};
	if (!body.prompts.length) return setStatus("Enter at least one clean prompt.");

	if (state.method === "acdc") {
		const needs = state.config ? state.config.acdc_metrics_needing_targets : [];
		const needsCf = state.config ? state.config.acdc_metrics_needing_cf_targets : [];
		if (needs.includes(body.acdc_metric) && body.targets.length !== body.prompts.length) {
			$("#status").hidden = false;
			return setStatus(`ACDC metric ${body.acdc_metric} needs one target per prompt.`);
		}
		if (needsCf.includes(body.acdc_metric) && body.cf_targets.length !== body.prompts.length) {
			$("#status").hidden = false;
			return setStatus(`ACDC metric ${body.acdc_metric} needs one counterfactual target per prompt.`);
		}
		return submit(body);
	}

	if (body.metric !== "kl_divergence" && body.targets.length !== body.prompts.length) {
		$("#status").hidden = false;
		return setStatus("Provide one target per prompt.");
	}
	if (body.mode === "denoising") {
		if (!body.cf_prompts.length) {
			$("#status").hidden = false;
			return setStatus("Denoising needs counterfactual prompts to patch clean activations back into.");
		}
		if (state.config && !state.config.denoising_metrics.includes(body.metric)) {
			$("#status").hidden = false;
			return setStatus(`Denoising is only supported for metrics: ${state.config.denoising_metrics.join(", ")}.`);
		}
	}

	return submit(body);
}

async function submit(body) {
	state.previewVersion += 1;
	const job = {
		method: body.method,
		params: JSON.parse(JSON.stringify(body)),
		meta: null,
		liveState: createLiveGraphState(),
		completed: false,
		displayActive: true,
	};
	state.activeJob = job;
	setRunning(true);
	setStatus("Submitting job…", job.method);
	setProgress(0, "", job.method);
	clearDisplay("Searching for a circuit…");
	job.displayActive = true;

	try {
		const out = await api("/api/search", body);
		if (state.activeJob !== job) return;
		state.jobId = out.job_id;
		openStream(out.job_id, job);
	} catch (e) {
		if (state.activeJob !== job) return;
		state.activeJob = null;
		state.jobId = null;
		setStatus(`Error: ${e.message}`, job.method);
		setRunning(false);
		if (state.method === job.method) restoreMethodResult(job.method);
	}
}

function isDisplayedJob(job) {
	return state.activeJob === job && job.displayActive && state.method === job.method;
}

function attachLiveGraph(job, restoreControls = false) {
	const live = job.liveState;
	drainAdmitQueue(live, live.admitQueue.length);
	state.meta = live.meta ? { ...live.meta } : null;
	state.nodes = live.nodes;
	state.edges = live.edges;
	state.maxAbs = live.maxAbs;
	state.threshold = live.threshold;
	state.admitQueue = live.admitQueue;
	state.fresh = live.fresh;
	state.result = null;
	state.hover = null;
	state.displayingLiveRun = Boolean(live.meta);
	$("#summary").innerHTML = "";
	$("#result-box").hidden = true;
	if (restoreControls) restoreParams(job.params, job.method);
	scheduleRender();
}

function openStream(jobId, job) {
	if (state.es) state.es.close();
	const es = new EventSource(`/api/search/${jobId}/events`);
	state.es = es;
	es.onmessage = (msg) => {
		const ev = JSON.parse(msg.data);
		switch (ev.event) {
			case "status":
				setStatus(ev.message || ev.status, job.method);
				break;
			case "meta":
				job.meta = {
					tokens: ev.tokens, n_layers: ev.n_layers,
					n_heads: ev.n_heads, positional: ev.positional,
					layout: ev.layout || (job.method === "acdc" ? "head" : "position"),
				};
				job.liveState.meta = { ...job.meta };
				setStatus("Searching…", job.method);
				if (isDisplayedJob(job)) {
					attachLiveGraph(job);
				}
				break;
			case "progress":
				// ACDC: one event per pruning step, occasionally carrying a
				// snapshot of the graph as it shrinks.
				setProgress(ev.frac ?? 0, ev.text || "", job.method);
				if (ev.message) setStatus(ev.message, job.method);
				if (ev.graph) loadGraph(ev.graph, job.liveState, isDisplayedJob(job));
				break;
			case "depth_start":
				setProgress(0, `depth ${ev.depth} · frontier ${ev.frontier_size}`, job.method);
				break;
			case "leaf_done":
				setProgress(ev.leaf / ev.n_leaves, `depth ${ev.depth} · leaf ${ev.leaf}/${ev.n_leaves}`, job.method);
				break;
			case "depth_end":
				setProgress(1, `depth ${ev.depth} done · admitted ${ev.admitted}`, job.method);
				break;
			case "admit":
				job.liveState.admitQueue.push(ev);
				break;
			case "path_complete":
				const graphState = job.liveState;
				for (const n of ev.path) {
					mergeNode(n, graphState);
					graphState.fresh.set(n.id, performance.now());
				}
				for (let i = 0; i + 1 < ev.path.length; i++) {
					mergeEdge(ev.path[i].id, ev.path[i + 1].id,
						ev.path[i].contribution ?? ev.contribution, true, graphState);
				}
				if (isDisplayedJob(job)) scheduleRender();
				break;
			case "result":
				job.completed = true;
				const method = ev.meta.method || job.method;
				const resultMeta = {
					...(job.meta || {}),
					...ev.meta,
					method,
					layout: ev.meta.layout || job.meta?.layout
						|| (method === "acdc" ? "head" : "position"),
				};
				if (Object.prototype.hasOwnProperty.call(state.savedResults, method)) {
					state.savedResults[method] = {
						graph: ev.graph,
						meta: resultMeta,
						params: job.params,
					};
					state.drafts[method] = null;
				}
				if (state.method === method) restoreSavedResult(method);
				setProgress(1, "", method);
				setStatus(`Done in ${ev.meta.runtime}s.`, method);
				break;
			case "end":
				es.close();
				if (!job.completed && state.method === job.method) restoreMethodResult(job.method);
				finishJob(job);
				if (ev.status !== "complete") setStatus(`Search ${ev.status}.`, job.method);
				break;
		}
	};
	es.onerror = () => {
		// The stream ends when the job does; report only if still running.
		if (state.running) setStatus("Stream interrupted — check the server log.", job.method);
		es.close();
		if (!job.completed && state.method === job.method) restoreMethodResult(job.method);
		finishJob(job);
	};
}

function finishJob(job) {
	if (state.activeJob !== job) return;
	state.activeJob = null;
	state.jobId = null;
	state.es = null;
	state.displayingLiveRun = false;
	setRunning(false);
	scheduleRender();
}

function restoreMethodResult(method) {
	const saved = state.savedResults[method];
	if (saved) restoreSavedResult(method);
	else clearDisplay("No completed circuit for this algorithm. Run discovery to create one.");
}

function restoreSavedResult(method) {
	const saved = state.savedResults[method];
	if (!saved) {
		clearDisplay("No completed circuit for this algorithm. Run discovery to create one.");
		return;
	}
	if (state.activeJob && state.activeJob.method === method) {
		state.activeJob.displayActive = false;
	}
	state.drafts[method] = null;
	state.method = method;
	clearDisplay();
	state.meta = { ...saved.meta };
	loadGraph(saved.graph);
	state.result = { graph: saved.graph, meta: saved.meta };
	restoreParams(saved.params, method);
	showSummary(saved.meta);
	scheduleRender();
}

function captureFormState() {
	return {
		model: $("#model").value,
		prompts: $("#prompts").value,
		targets: $("#targets").value,
		cf_prompts: $("#cf-prompts").value,
		cf_targets: $("#cf-targets").value,
		strategy: state.strategy,
		mode: state.mode,
		min_contribution: $("#min-contribution").value,
		max_width: $("#max-width").value,
		metric: $("#metric").value,
		positional: $("#positional").checked,
		include_negative: $("#include-negative").checked,
		acdc_metric: $("#acdc-metric").value,
		acdc_threshold: $("#acdc-threshold").value,
		acdc_abs_threshold: $("#acdc-abs-threshold").checked,
	};
}

function restoreDraft(draft, method) {
	applyFormState(draft, method);
}

function restoreParams(params, method) {
	if (!params) return;
	applyFormState({
		...params,
		prompts: (params.prompts || []).join("\n"),
		targets: (params.targets || []).join("\n"),
		cf_prompts: (params.cf_prompts || []).join("\n"),
		cf_targets: (params.cf_targets || []).join("\n"),
	}, method);
}

function resetFormToDefaults(method) {
	applyFormState({
		model: state.config?.models?.[0] ?? $("#model").options[0]?.value ?? "",
		prompts: "",
		targets: "",
		cf_prompts: "",
		cf_targets: "",
		strategy: "threshold",
		mode: "noising",
		min_contribution: "0.05",
		max_width: "20",
		metric: state.config?.metrics?.[0] ?? $("#metric").options[0]?.value ?? "",
		positional: true,
		include_negative: true,
		acdc_metric: state.config?.acdc_metrics?.[0] ?? $("#acdc-metric").options[0]?.value ?? "",
		acdc_threshold: "0.0575",
		acdc_abs_threshold: false,
	}, method);
}

function applyFormState(values, method) {
	$("#model").value = values.model || state.config?.models?.[0] || $("#model").options[0]?.value || "";
	$("#prompts").value = values.prompts ?? "";
	$("#targets").value = values.targets ?? "";
	$("#cf-prompts").value = values.cf_prompts ?? "";
	$("#cf-targets").value = values.cf_targets ?? "";
	$("#min-contribution").value = values.min_contribution ?? "0.05";
	$("#max-width").value = values.max_width ?? "20";
	$("#metric").value = values.metric || state.config?.metrics?.[0] || $("#metric").options[0]?.value || "";
	$("#positional").checked = values.positional ?? true;
	$("#include-negative").checked = values.include_negative ?? true;
	$("#acdc-metric").value = values.acdc_metric || state.config?.acdc_metrics?.[0]
		|| $("#acdc-metric").options[0]?.value || "";
	$("#acdc-threshold").value = values.acdc_threshold ?? "0.0575";
	$("#acdc-abs-threshold").checked = values.acdc_abs_threshold ?? false;
	state.method = method;
	state.strategy = values.strategy ?? "threshold";
	state.mode = values.mode ?? "noising";
	setSegValue("#method-seg", state.method);
	setSegValue("#strategy-seg", state.strategy);
	setSegValue("#mode-seg", state.mode);
	applyMethod();
	updateStrategyFields();
	updateModeHint();
}

function showSummary(meta) {
	const bits = [
		`<b>${meta.method}</b> search (${meta.strategy}), <b>${meta.mode}</b>, metric <b>${meta.metric}</b>`,
		`runtime <b>${meta.runtime}s</b>`,
		`${meta.n_nodes} grid nodes, ${meta.n_edges} edges`,
	];
	if (meta.joint_tree_contribution !== null && meta.joint_tree_contribution !== undefined) {
		bits.push(`joint tree contribution <b>${fmt(meta.joint_tree_contribution)}</b>`);
	}
	if (meta.acdc_edges !== undefined) {
		bits.push(`ACDC kept <b>${meta.acdc_edges}</b> of ${meta.acdc_initial_edges} hook-level edges`);
		bits.push(`final metric <b>${meta.final_metric.toFixed(4)}</b>`);
	}
	$("#summary").innerHTML = bits.join("<br>");
	const slider = $("#threshold-slider");
	slider.max = state.maxAbs.toFixed(3);
	slider.step = (state.maxAbs / 200 || 0.001).toFixed(5);
	slider.value = 0;
	$("#threshold-val").textContent = "0";
	state.threshold = 0;
	$("#result-box").hidden = false;
}

/* -------------------------------------------------------------- wire up UI */

function updateModeHint() {
	const el = $("#mode-hint");
	if (!el) return;
	if (state.mode === "denoising") {
		const metric = $("#metric").value;
		const ok = !state.config || state.config.denoising_metrics.includes(metric);
		el.textContent = "patch clean activations into the counterfactual run; "
			+ "measures restored performance"
			+ (ok ? "" : ` — needs metric: ${state.config.denoising_metrics.join(", ")}`);
		el.classList.toggle("warn", !ok);
	} else {
		el.textContent = "patch counterfactual activations into the clean run; measures degraded performance";
		el.classList.remove("warn");
	}
}

function setSegValue(id, value) {
	$(id).querySelectorAll("button").forEach((button) => {
		button.classList.toggle("on", button.dataset.value === value);
	});
}

function updateStrategyFields() {
	$("#min-contribution-field").hidden = state.strategy !== "threshold";
	$("#max-width-field").hidden = state.strategy !== "topk";
}

function selectMethod(method) {
	if (method === state.method) return;
	state.previewVersion += 1;
	const previousMethod = state.method;
	state.drafts[previousMethod] = state.savedResults[previousMethod]
		? null
		: captureFormState();
	const liveJob = state.activeJob && !state.activeJob.completed
		&& state.activeJob.method === method
		? state.activeJob
		: null;
	if (state.activeJob) state.activeJob.displayActive = false;
	state.method = method;
	if (liveJob) {
		clearDisplay("Search in progress…");
		liveJob.displayActive = true;
		attachLiveGraph(liveJob, true);
	} else if (state.savedResults[method]) {
		restoreSavedResult(method);
	} else {
		clearDisplay("No completed circuit for this algorithm. Run discovery to create one.");
		if (state.drafts[method]) {
			restoreDraft(state.drafts[method], method);
			if (lines($("#prompts").value)[0]) tokenizePreview();
		} else {
			resetFormToDefaults(method);
		}
	}
	refreshMethodStatus();
}

/* Show only the knobs the selected algorithm actually reads. ACDC ignores the
 * IPE strategy / patching direction / positional switches entirely: it has one
 * threshold, one metric and one (position-agnostic) pruning pass. */
function applyMethod() {
	const acdc = state.method === "acdc";
	document.querySelectorAll(".ipe-only").forEach((e) => { e.hidden = acdc; });
	document.querySelectorAll(".acdc-only").forEach((e) => { e.hidden = !acdc; });
	const hint = $("#method-hint");
	if (acdc) {
		hint.textContent = "greedy edge pruning over the whole computational graph "
			+ "(benchmark/Automatic-Circuit-Discovery) — position-agnostic, and slow: "
			+ "it evaluates one forward pass per candidate edge";
		$("#acdc-hint").textContent = $("#cf-prompts").value.trim()
			? "counterfactual ablation from the prompts above"
			: "no counterfactual prompts → zero ablation";
	} else {
		hint.textContent = state.method === "tree"
			? "grow one suffix-sharing tree of message paths back from the logits"
			: "score complete root→embedding paths independently";
	}
}

function setupSeg(id, onChange) {
	const seg = $(id);
	seg.querySelectorAll("button").forEach((b) => {
		b.addEventListener("click", () => {
			seg.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
			b.classList.add("on");
			onChange(b.dataset.value);
		});
	});
}

function init() {
	loadConfig().catch((e) => setStatus(`Config failed: ${e.message}`));
	setupPanZoom();

	setupSeg("#method-seg", selectMethod);
	setupSeg("#strategy-seg", (v) => {
		state.strategy = v;
		updateStrategyFields();
	});
	setupSeg("#mode-seg", (v) => { state.mode = v; updateModeHint(); });
	$("#metric").addEventListener("change", updateModeHint);
	updateModeHint();

	$("#prompts").addEventListener("input", tokenizePreview);
	$("#cf-prompts").addEventListener("input", applyMethod);
	$("#model").addEventListener("change", tokenizePreview);
	$("#positional").addEventListener("change", () => {
		if (!state.result && !state.displayingLiveRun && state.meta) {
			state.meta.positional = $("#positional").checked;
			scheduleRender();
		}
	});

	$("#run").addEventListener("click", runSearch);
	$("#cancel").addEventListener("click", async () => {
		if (state.jobId) {
			setStatus("Cancelling…", state.activeJob?.method || state.method);
			await api(`/api/search/${state.jobId}/cancel`, {});
		}
	});

	$("#threshold-slider").addEventListener("input", (e) => {
		state.threshold = parseFloat(e.target.value);
		$("#threshold-val").textContent = state.threshold.toFixed(3);
		scheduleRender();
	});

	$("#download").addEventListener("click", () => {
		if (!state.result) return;
		const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
		const a = document.createElement("a");
		a.href = URL.createObjectURL(blob);
		a.download = "ipe_circuit.json";
		a.click();
		URL.revokeObjectURL(a.href);
	});

	window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleRender);
}

init();
