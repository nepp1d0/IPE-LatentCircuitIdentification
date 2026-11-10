"use strict";

const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const P = require("../../visualization/static/path_exploration.js");
const PathDiagram = require("../../visualization/static/path_diagram.js");
const PathPresentation = require("../../visualization/static/path_presentation.js");
const ComponentDecoding = require("../../visualization/static/component_decoding.js");
const { fixture, occurrenceFixture } = require("./path_fixtures.cjs");

/* Minimal SVG/DOM harness: exercise the actual app's rendering/event handlers
 * without introducing jsdom, a browser dependency, or network/model requests. */
class Element {
	constructor(tag) {
		this.tag = tag;
		this.attrs = {};
		this.children = [];
		this.events = {};
		this.style = { setProperty(name, value) { this[name] = value; } };
		this.value = "";
		this.options = [];
		const classes = new Set();
		this.classList = {
			contains: (c) => classes.has(c),
			add: (...cs) => cs.forEach((c) => classes.add(c)),
			remove: (...cs) => cs.forEach((c) => classes.delete(c)),
			toggle: (c, on = !classes.has(c)) => on ? classes.add(c) : classes.delete(c),
			set: (value) => { classes.clear(); value.split(/\s+/).filter(Boolean).forEach((c) => classes.add(c)); },
		};
		this.dataset = {};
	}
	setAttribute(name, value) {
		this.attrs[name] = String(value);
		if (name === "class") this.classList.set(value);
		if (name.startsWith("data-")) {
			this.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
		}
	}
	getAttribute(name) { return this.attrs[name]; }
	set textContent(value) { this._text = String(value); this.children = []; }
	get textContent() { return (this._text || "") + this.children.map((child) => child.textContent).join(""); }
	appendChild(child) { this.children.push(child); child.parent = this; return child; }
	replaceChildren(...children) { this.children = []; children.forEach((child) => this.appendChild(child)); }
	set innerHTML(value) { this.html = value; this.children = []; this._text = ""; }
	get innerHTML() { return this.html || ""; }
	addEventListener(type, handler) { (this.events[type] ||= []).push(handler); }
	emit(type, event = {}) {
		for (const fn of this.events[type] || []) fn({ target: this, stopPropagation() {}, preventDefault() {}, ...event });
	}
	closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector); }
	matches(selector) { return selector.startsWith(".") ? this.classList.contains(selector.slice(1)) : this.tag === selector; }
	querySelectorAll(selector) {
		return this.children.flatMap((child) => [
			...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
		]);
	}
	getBoundingClientRect() { return this.bounds || { left: 0, top: 0, width: 1000, height: 1000 }; }
	contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
	setPointerCapture(id) { this.capturedPointer = id; }
	hasPointerCapture(id) { return this.capturedPointer === id; }
	releasePointerCapture() { this.capturedPointer = null; }
	scrollIntoView() { this.scrolledIntoView = true; }
	focus() { this.ownerDocument.activeElement = this; this.emit("focus"); }
}

function appHarness(inspectorWidth = "480px") {
	const elements = new Map();
	const get = (selector) => {
		if (selector === "#viewport") {
			return elements.get("#grid")?.children.find((child) => child.attrs.id === "viewport") || null;
		}
		if (!elements.has(selector)) elements.set(selector, make(selector === "#grid" ? "svg" : "div"));
		return elements.get(selector);
	};
	const document = Object.assign(new Element("document"), { querySelector: get, querySelectorAll: () => [],
		createElementNS: (_, tag) => make(tag), createElement: (tag) => make(tag) });
	const make = (tag) => { const node = new Element(tag); node.ownerDocument = document; return node; };
	const window = new Element("window"), mediaQueries = new Map();
	window.innerWidth = 1440;
	window.innerHeight = 1000;
	window.matchMedia = (query) => {
		if (!mediaQueries.has(query)) {
			const media = new Element("media");
			media.matches = query.includes("min-width") && window.innerWidth >= 1101;
			mediaQueries.set(query, media);
		}
		return mediaQueries.get(query);
	};
	get("#app").bounds = { left: 0, top: 0, width: 1440, height: 1000 };
	get("#sidebar").bounds = { left: 0, top: 52, width: 330, height: 948 };
	get("#path-explorer").getBoundingClientRect = () => ({ left: 960, top: 52,
		width: Number.parseFloat(get("#app").style["--path-inspector-width"] || "480"), height: 948 });
	get("#path-info-wrap").appendChild(get("#path-info"));
	get("#path-info-wrap").appendChild(get("#path-info-popover"));
	get("#component-decoding-info-wrap").appendChild(get("#component-decoding-info"));
	get("#component-decoding-info-wrap").appendChild(get("#component-decoding-info-popover"));
	get("#component-decoding-info-popover").hidden = true;
	get("#path-sort-wrap").appendChild(get("#path-sort-toggle"));
	get("#path-sort-wrap").appendChild(get("#path-sort-popover"));
	get("#path-sort-popover").appendChild(get("#path-sort"));
	get("#path-sort-popover").hidden = true;
	for (const value of ["discovery", "signed", "absolute"]) {
		const option = make("button");
		option.value = value;
		option.setAttribute("aria-checked", String(value === "discovery"));
		get("#path-sort").appendChild(option);
	}
	const resizeWindow = (width) => {
		window.innerWidth = get("#app").bounds.width = width;
		for (const [query, media] of mediaQueries) {
			media.matches = query.includes("min-width") && width >= 1101;
			media.emit("change");
		}
		window.emit("resize");
	};
	const context = vm.createContext({ PathExploration: P, PathDiagram, PathPresentation, ComponentDecoding, console,
		EventSource: class { close() {} },
		document,
		window,
		getComputedStyle: () => ({ getPropertyValue: (key) => ({ "--pos": "#5A92BF", "--neg": "#BF4D43", "--mid": "#E8E6DC",
			"--path-inspector-width": inspectorWidth })[key] }),
		setInterval() {}, setTimeout() {}, clearTimeout() {}, performance: { now: () => 1000 },
	});
	const source = fs.readFileSync(require.resolve("../../visualization/static/app.js"), "utf8").replace(/\ninit\(\);\s*$/, "");
	vm.runInContext(source + `\nglobalThis.app = { state, render, clearHover, clearDisplay, onClickNode,
		restoreSavedResult, selectMethod, setupPanZoom, openStream, attachLiveGraph,
		createLiveGraphState, init }; loadConfig = async () => {};`, context);
	context.app.init();
	return { ...context.app, get, document, resizeWindow };
}

function saved(graph, method = "path") {
	return { graph, meta: { method, positional: true, tokens: ["BOS", "word"], n_layers: 2, n_heads: 2 }, params: null };
}

function snapshotGraph(get) {
	const grid = get("#grid");
	return {
		nodes: grid.querySelectorAll(".chip").map((c) => [c.dataset.id, c.children[0].attrs]),
		edges: grid.querySelectorAll(".edge").map((e) => e.attrs),
	};
}

const pathButtons = (app) => app.get("#path-list").querySelectorAll(".path-select");

function laneSnapshot(grid) {
	return Object.fromEntries(grid.querySelectorAll(".path-stroke").map((p) => [`${p.dataset.pathId}|${p.dataset.key}`, p.attrs.d]));
}

describe("Inspector header and help interactions", () => {

test("compact inspector header shows dynamic component metadata and separates complete/incomplete counts", () => {
	const app = appHarness();
	const graph = fixture();
	Object.assign(graph.nodes.find((n) => n.id === "a"), { layer: 5, head: 9, position: 10 });
	graph.path_provenance.paths.push({ id: "pruned", complete: false, node_ids: ["spare", "a", "f"] });
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	assert.equal(app.get("#path-heading").textContent, "Paths through A5H9");
	assert.equal(app.get("#path-node").textContent, "Layer 5 · Head 9 · Token 10");
	assert.equal(app.get("#path-count").children[0].tag, "strong");
	assert.equal(app.get("#path-count").children[0].textContent, "7 complete paths");
	assert.equal(app.get("#path-count").querySelectorAll(".path-incomplete-count")[0].textContent, " · 1 incomplete branch excluded");
	assert.equal(app.get("#path-help").hidden, true);
	assert.equal(app.get("#path-key").hidden, false);
	app.onClickNode({ id: "b" });
	assert.equal(app.get("#path-heading").textContent, "Paths through MLP0");
	assert.equal(app.get("#path-node").textContent, "Layer 0 · Token 1");
	app.onClickNode({ id: "e0" });
	assert.equal(app.get("#path-heading").textContent, "Paths through EMB");
	assert.equal(app.get("#path-node").textContent, "Embedding · Token 0");
	app.onClickNode({ id: "f" });
	assert.equal(app.get("#path-node").textContent, "Final residual · Token 1");
	app.onClickNode({ id: "spare" });
	assert.match(app.get("#path-count").textContent, /^0 complete paths/);
	assert.equal(app.get("#path-help").hidden, false);
	assert.equal(app.get("#path-key").hidden, true);
});

test("header handles all-token/all-head metadata and singular complete path counts without changing selection", () => {
	const app = appHarness();
	const graph = fixture();
	Object.assign(graph.nodes.find((n) => n.id === "a"), { layer: 0, head: null, position: null });
	graph.path_provenance.paths = graph.path_provenance.paths.slice(0, 1);
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	assert.equal(app.get("#path-node").textContent, "Layer 0 · All heads · All tokens");
	assert.equal(app.get("#path-count").textContent, "1 complete path");
	assert.equal(app.state.pathExplorer.selectedId, "path-1");
});

test("header information supports hover, pinned click, keyboard and outside dismissal", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const wrap = app.get("#path-info-wrap"), info = app.get("#path-info"), popover = app.get("#path-info-popover");
	wrap.emit("pointerenter");
	assert.equal(popover.hidden, false);
	assert.equal(info.attrs["aria-expanded"], "true");
	wrap.emit("pointerleave");
	assert.equal(popover.hidden, true);
	info.emit("click");
	wrap.emit("pointerleave");
	assert.equal(popover.hidden, false);
	app.document.emit("pointerdown", { target: info });
	assert.equal(popover.hidden, false);
	app.document.emit("keydown", { key: "Escape" });
	assert.equal(popover.hidden, true);
	info.emit("focus");
	assert.equal(popover.hidden, false);
	info.emit("blur");
	assert.equal(popover.hidden, true);
	info.emit("click");
	app.document.emit("pointerdown", { target: app.get("#grid") });
	assert.equal(popover.hidden, true);
	info.emit("click");
	app.get("#path-clear").emit("click");
	assert.equal(popover.hidden, true);
	assert.equal(info.attrs["aria-expanded"], "false");
});

});

describe("Inspector resizing and responsive state", () => {

test("inspector resizing clamps width without touching graph geometry, pan/zoom or path selection", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	pathButtons(app)[2].emit("click");
	app.render();
	const handle = app.get("#path-resize-handle"), root = app.get("#app");
	const before = JSON.stringify(app.state.view), lanes = laneSnapshot(app.get("#grid"));
	assert.equal(root.style["--path-inspector-width"], "480px");
	handle.emit("pointerdown", { button: 0, pointerId: 1, clientX: 960 });
	handle.emit("pointermove", { pointerId: 1, clientX: 790 });
	assert.equal(root.style["--path-inspector-width"], "650px");
	assert.equal(app.get("#grid").classList.contains("panning"), false);
	handle.emit("pointermove", { pointerId: 1, clientX: -1000 });
	assert.equal(root.style["--path-inspector-width"], "720px");
	handle.emit("pointermove", { pointerId: 1, clientX: 2000 });
	assert.equal(root.style["--path-inspector-width"], "340px");
	handle.emit("pointerup", { pointerId: 1 });
	assert.equal(root.classList.contains("resizing-path-inspector"), false);
	assert.equal(handle.hasPointerCapture(1), false);
	assert.equal(app.state.pathExplorer.selectedId, "path-3");
	assert.equal(JSON.stringify(app.state.view), before);
	assert.deepEqual(laneSnapshot(app.get("#grid")), lanes);
	handle.emit("keydown", { key: "ArrowLeft" });
	assert.equal(root.style["--path-inspector-width"], "360px");
	handle.emit("keydown", { key: "End" });
	assert.equal(handle.attrs["aria-valuenow"], "720");
	handle.emit("dblclick");
	assert.equal(root.style["--path-inspector-width"], "480px");
	handle.emit("pointerdown", { button: 0, pointerId: 2, clientX: 990 });
	handle.emit("pointercancel", { pointerId: 2 });
	assert.equal(root.classList.contains("resizing-path-inspector"), false);
});

test("viewport resizing protects graph space, disables mobile dragging and restores desktop width preference", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const handle = app.get("#path-resize-handle"), root = app.get("#app");
	handle.emit("keydown", { key: "End" });
	app.resizeWindow(1200);
	assert.equal(root.style["--path-inspector-width"], "550px");
	assert.equal(handle.attrs["aria-valuemax"], "550");
	app.resizeWindow(390);
	assert.equal(handle.attrs["aria-disabled"], "true");
	assert.equal(handle.tabIndex, -1);
	handle.emit("pointerdown", { button: 0, pointerId: 1, clientX: 500 });
	assert.equal(root.classList.contains("resizing-path-inspector"), false);
	app.resizeWindow(1440);
	assert.equal(root.style["--path-inspector-width"], "720px");
	assert.equal(handle.tabIndex, 0);
});

test("inspector initialization and reset use the CSS width rather than a second hardcoded default", () => {
	const app = appHarness("510px");
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const handle = app.get("#path-resize-handle"), root = app.get("#app");
	assert.equal(root.style["--path-inspector-width"], "510px");
	handle.emit("keydown", { key: "End" });
	assert.equal(root.style["--path-inspector-width"], "720px");
	handle.emit("dblclick");
	assert.equal(root.style["--path-inspector-width"], "510px");
});

});

describe("Graph rendering and synchronized path selection", () => {

test("all dense shared-edge lanes and hit regions keep their geometry when an off-center path is selected", () => {
	const app = appHarness();
	const graph = fixture();
	graph.path_provenance.paths = Array.from({ length: 24 }, (_, i) => ({
		id: `path-${i + 1}`, complete: true, node_ids: ["e0", "a", "b", "f"],
	}));
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	app.render();
	const grid = app.get("#grid"), before = laneSnapshot(grid);
	const hits = () => Object.fromEntries(grid.querySelectorAll(".path-hit").map((p) => [
		`${p.dataset.pathId}|${p.dataset.key}`, [p.attrs.d, p.attrs["stroke-width"]],
	]));
	const hitBefore = hits();
	pathButtons(app)[23].emit("click");
	app.render();
	assert.deepEqual(laneSnapshot(grid), before);
	assert.deepEqual(hits(), hitBefore);
	for (const p of grid.querySelectorAll(".path-selected")) {
		assert.equal(p.dataset.pathId, "path-24");
		assert.equal(p.attrs.stroke, "var(--accent)");
		assert.equal(p.attrs.opacity, "0.85");
		assert.notEqual(p.attrs.d, grid.querySelectorAll(".edge").find((e) => e.dataset.key === p.dataset.key).attrs.d);
	}
});

test("default graph stays unchanged; active selection uses exactly the existing completed-edge style", () => {
	const app = appHarness();
	const graph = fixture();
	const cached = saved(graph);
	const original = JSON.stringify(cached);
	app.state.savedResults.path = cached;
	app.restoreSavedResult("path");
	app.render();
	const baseline = snapshotGraph(app.get);
	const grid = app.get("#grid");
	assert.equal(app.get("#path-explorer").hidden, true);
	assert.equal(app.get("#app").classList.contains("exploring-paths"), false);
	assert.equal(grid.querySelectorAll(".path-stroke").length, 0);
	grid.querySelectorAll(".chip").find((n) => n.dataset.id === "e0").emit("mouseenter", { clientX: 100, clientY: 100 });
	assert.equal(grid.querySelectorAll(".edge").find((e) => e.dataset.key === "e0->f").classList.contains("hot"), true);
	app.clearHover();
	assert.equal(grid.querySelectorAll(".edge").some((e) => e.classList.contains("hot")), false);
	grid.querySelectorAll(".chip").find((n) => n.dataset.id === "a").emit("click");
	app.render();
	assert.deepEqual(snapshotGraph(app.get), baseline);
	assert.match(app.get("#path-count").textContent, /7 complete paths/);
	assert.equal(app.state.pathExplorer.selectedId, "path-1");
	assert.equal(app.get("#path-explorer").hidden, false);
	assert.equal(app.get("#app").classList.contains("exploring-paths"), true);
	assert.equal(app.get("#path-list").children.length, 7);
	assert.equal(pathButtons(app)[0].attrs["aria-pressed"], "true");
	assert.match(pathButtons(app)[0].textContent, /Path 1.*Selected reference/);
	assert.equal(grid.querySelectorAll(".edge").every((e) => e.classList.contains("dim")), true);
	assert.equal(grid.querySelectorAll(".chip").find((n) => n.dataset.id === "spare").classList.contains("dim"), true);
	const shared = grid.querySelectorAll(".path-stroke").filter((e) => e.dataset.key === "a->b");
	assert.equal(shared.length, 4);
	assert.equal(new Set(shared.map((e) => e.attrs.d)).size, 4);
	assert.equal(new Set(shared.map((e) => e.attrs.stroke)).size, 2);
	assert.equal(shared.filter((e) => e.classList.contains("path-alternative"))
		.every((e) => e.attrs.stroke === "var(--path-alternative)"), true);
	for (const selected of grid.querySelectorAll(".path-selected")) {
		const originalEdge = grid.querySelectorAll(".edge").find((e) => e.dataset.key === selected.dataset.key);
		for (const attr of ["d", "stroke", "stroke-width", "opacity"]) {
			assert.equal(selected.attrs[attr], originalEdge.attrs[attr]);
		}
	}
	assert.equal(grid.querySelectorAll(".path-stroke").some((e) => e.dataset.key === "e0->f"), false);
	grid.querySelectorAll(".chip").find((n) => n.dataset.id === "a").emit("mouseenter", { clientX: 100, clientY: 100 });
	assert.match(app.get("#tooltip").innerHTML, /Component Logit Lens/);
	assert.match(app.get("#tooltip").innerHTML, /&lt;token&gt;/);
	app.clearHover();
	assert.equal(app.state.pathExplorer.selectedId, "path-1");
	assert.equal(app.get("#tooltip").hidden, true);
	app.get("#path-clear").emit("click");
	app.render();
	assert.deepEqual(snapshotGraph(app.get), baseline);
	assert.equal(grid.querySelectorAll(".path-stroke").length, 0);
	assert.equal(grid.querySelectorAll(".path-hit").length, 0);
	assert.equal(app.get("#path-explorer").hidden, true);
	assert.equal(app.get("#app").classList.contains("exploring-paths"), false);
	assert.equal(JSON.stringify(cached), original);
});

test("sidebar and graph path clicks select the same original identity without resetting list scroll", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	app.render();
	const grid = app.get("#grid");
	const list = app.get("#path-list");
	list.scrollTop = 100;
	const lanes = laneSnapshot(grid);
	pathButtons(app)[6].emit("click");
	app.render();
	assert.equal(app.state.pathExplorer.selectedId, "path-7");
	assert.equal(pathButtons(app)[6].attrs["aria-pressed"], "true");
	assert.equal(list.scrollTop, 100);
	assert.deepEqual(laneSnapshot(grid), lanes);
	assert.equal(grid.querySelectorAll(".path-selected").every((p) => p.dataset.pathId === "path-7"), true);
	const hit = grid.querySelectorAll(".path-hit").find((p) => p.dataset.pathId === "path-3");
	grid.emit("pointerdown", { target: hit, clientX: 10, clientY: 20, pointerId: 1 });
	assert.equal(grid.classList.contains("panning"), false);
	hit.emit("click");
	app.render();
	assert.equal(app.state.pathExplorer.selectedId, "path-3");
	assert.equal(pathButtons(app)[2].attrs["aria-pressed"], "true");
	assert.equal(pathButtons(app).filter((b) => b.attrs["aria-pressed"] === "true").length, 1);
	assert.equal(pathButtons(app)[2].scrolledIntoView, true);
	assert.equal(grid.querySelectorAll(".path-selected").every((p) => p.dataset.pathId === "path-3"), true);
	assert.deepEqual(laneSnapshot(grid), lanes);
});

});

describe("Sidebar comparisons, scores and sorting controls", () => {

test("sidebar summaries disambiguate Q/KV paths that have identical display geometry", () => {
	const app = appHarness();
	const graph = fixture();
	graph.path_provenance.occurrences = [
		{ id: "q", node_id: "a", kind: "attn", layer: 0, head: 0, position: 1, variant: "Q", kv_position: null },
		{ id: "kv", node_id: "a", kind: "attn", layer: 0, head: 0, position: 1, variant: "KV", kv_position: 0 },
	];
	graph.path_provenance.paths[0].occurrence_ids = ["e", "q", "b", "f"];
	graph.path_provenance.paths[2].occurrence_ids = ["e", "kv", "b", "f"];
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	assert.match(pathButtons(app)[2].textContent, /A0H0: Q → KV/);
	assert.match(pathButtons(app)[2].textContent, /Same route · different patching/);
	assert.match(pathButtons(app)[2].textContent, /source: all eligible → 0/);
	assert.equal(app.state.pathEntries.get("path-3").details.hidden, true, "two fully visible differences need no redundant expander");
	pathButtons(app)[2].emit("click");
	assert.match(pathButtons(app)[0].textContent, /A0H0: KV → Q/);
	assert.equal(app.get("#selected-path-meta").textContent, "Path 3");
});

test("only additional differences expand, and update against the new reference without losing expansion state", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	graph.path_provenance.paths[0].score = 0;
	graph.path_provenance.paths[2].score = -0.123456789;
	const kv = graph.path_provenance.occurrences.find((n) => n.id === "path-3-occ-1");
	Object.assign(kv, { variant: "KV", kv_position: 0, patch_type: "counterfactual" });
	const original = JSON.stringify(graph);
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const entry = app.state.pathEntries.get("path-3");
	assert.deepEqual(entry.summary.children.map((n) => n.textContent), ["A0H0: Q → KV", "A0H0 source: all eligible → 0", "A0H0 patch: zero → counterfactual"]);
	assert.equal(entry.details.hidden, true, "all three differences are directly visible");
	assert.equal(entry.changes.children.length, 0);
	pathButtons(app)[1].emit("click");
	assert.equal(entry.summary.children.length, 4);
	assert.equal(entry.details.hidden, false);
	assert.equal(entry.inspect.textContent, "Inspect remaining differences (1 more)");
	assert.equal(entry.changes.children.length, 1);
	const allLabels = P.comparePaths(app.state.pathExplorer.index.paths.get("path-2"), app.state.pathExplorer.index.paths.get("path-3"))
		.differences.map(PathPresentation.differenceLabel);
	assert.deepEqual([...entry.summary.children, ...entry.changes.children].map((n) => n.textContent), allLabels, "every difference keeps its original order");
	entry.details.open = true;
	pathButtons(app)[4].emit("click");
	assert.equal(entry.summary.children.length, 4);
	assert.equal(entry.details.hidden, true, "exactly four differences need no expander");
	assert.equal(entry.changes.children.length, 0);
	pathButtons(app)[1].emit("click");
	assert.equal(entry.details.open, true);
	assert.equal(entry.details.hidden, false);
	assert.equal(entry.changes.children.length, 1);
	pathButtons(app)[2].emit("click");
	assert.equal(entry.details.hidden, true);
	assert.match(pathButtons(app)[0].textContent, /\+0/);
	assert.match(pathButtons(app)[0].textContent, /Scoring mode unknown/);
	assert.equal(pathButtons(app)[2].querySelectorAll(".path-score")[0].textContent, "−0.123");
	assert.equal(app.get("#selected-path-meta").textContent, "Path 3");
	assert.equal(JSON.stringify(graph), original);
});

test("sidebar displays only the matching original score and uses completed-run metric metadata", () => {
	const app = appHarness();
	const graph = fixture("tree");
	const scores = [1.25, -2.5, 0, undefined, NaN, Infinity, -Infinity];
	graph.path_provenance.paths.forEach((path, i) => {
		path.score = scores[i];
		path.score_kind = "isolated_path_effect";
	});
	graph.path_provenance.paths.push({ id: "pruned", complete: false, node_ids: ["spare", "f"], score: 100 });
	const result = saved(graph, "tree");
	Object.assign(result.meta, { metric: "logit_difference", mode: "noising", joint_tree_contribution: 999 });
	app.state.savedResults.tree = result;
	app.restoreSavedResult("tree");
	app.get("#metric").value = "kl_divergence"; // editing the form must not relabel old scores
	app.onClickNode({ id: "a" });
	const buttons = pathButtons(app);
	assert.equal(buttons.length, 7);
	assert.equal(app.get("#path-score-context").textContent, "Effect on logit difference · Noising · logit units");
	for (const [i, expected] of ["+1.25", "−2.5", "+0", "Unavailable", "Unavailable", "Unavailable", "Unavailable"].entries()) {
		assert.equal(buttons[i].dataset.pathId, `tree-${i + 1}`);
		const score = buttons[i].querySelectorAll(".path-score")[0];
		assert.equal(score.textContent, expected);
		assert.ok(score.title);
		assert.equal(buttons[i].title, undefined, "no full-sequence tooltip may be inherited by the summary");
		assert.doesNotMatch(buttons[i].textContent, /tree-|Path effect|Logit Difference|999|0\.5|%/);
		assert.equal(buttons[i].querySelectorAll(".path-card-heading")[0].children[0].textContent, `Path ${i + 1}`);
	}
	assert.match(buttons[0].querySelectorAll(".path-score")[0].title, /Positive scores indicate that perturbing this path decreases/);
	assert.equal(app.state.pathExplorer.index.incompleteByNode.get("spare"), 1);
});

test("sidebar sorting leaves graph lanes, original ordinals and graph/sidebar selection synchronized", () => {
	const app = appHarness();
	const graph = fixture();
	[1, -3, 2, 0, null, null, 2].forEach((score, i) => { graph.path_provenance.paths[i].score = score; });
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	pathButtons(app)[3].emit("click");
	app.render();
	const grid = app.get("#grid"), lanes = laneSnapshot(grid);
	const geometry = snapshotGraph(app.get);
	const sort = app.get("#path-sort");
	const option = (value) => sort.querySelectorAll("button").find((n) => n.value === value);
	assert.equal(option("discovery").attrs["aria-checked"], "true");
	assert.equal(option("signed").disabled, false);
	option("signed").emit("click");
	assert.deepEqual(pathButtons(app).map((b) => b.dataset.pathId), ["path-3", "path-7", "path-1", "path-4", "path-2", "path-5", "path-6"]);
	assert.equal(app.state.pathExplorer.selectedId, "path-4");
	assert.match(pathButtons(app)[3].textContent, /Path 4.*Selected reference/);
	assert.equal(option("signed").attrs["aria-checked"], "true");
	assert.equal(option("discovery").attrs["aria-checked"], "false");
	assert.equal(app.get("#path-sort-current").textContent, "Signed");
	option("absolute").emit("click");
	assert.deepEqual(pathButtons(app).map((b) => b.dataset.pathId), ["path-2", "path-3", "path-7", "path-1", "path-4", "path-5", "path-6"]);
	pathButtons(app)[0].emit("click");
	app.render();
	assert.equal(app.state.pathExplorer.selectedId, "path-2");
	assert.deepEqual(laneSnapshot(grid), lanes);
	assert.deepEqual(snapshotGraph(app.get), geometry);
	grid.querySelectorAll(".path-hit").find((p) => p.dataset.pathId === "path-7").emit("click");
	assert.equal(app.state.pathExplorer.selectedId, "path-7");
	assert.equal(pathButtons(app)[2].attrs["aria-pressed"], "true");
	assert.equal(pathButtons(app)[2].scrolledIntoView, true);
	option("discovery").emit("click");
	assert.deepEqual(pathButtons(app).map((b) => b.dataset.pathId), graph.path_provenance.paths.map((p) => p.id));
	assert.equal(app.state.pathExplorer.selectedId, "path-7");
	app.get("#path-clear").emit("click");
	assert.ok(sort.querySelectorAll("button").every((n) => n.disabled));
});

test("Sort menu supports focus navigation, commits full-row options, closes and returns focus", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const toggle = app.get("#path-sort-toggle"), popover = app.get("#path-sort-popover");
	const options = app.get("#path-sort").querySelectorAll("button");
	toggle.emit("click");
	assert.equal(popover.hidden, false);
	assert.equal(toggle.attrs["aria-expanded"], "true");
	assert.equal(app.document.activeElement, options[0]);
	options[0].emit("keydown", { key: "ArrowUp" });
	assert.equal(app.document.activeElement, options[2], "ArrowUp wraps to the last option");
	options[2].emit("keydown", { key: "ArrowDown" });
	assert.equal(app.document.activeElement, options[0], "ArrowDown wraps to the first option");
	options[0].emit("keydown", { key: "End" });
	assert.equal(app.document.activeElement, options[2]);
	options[2].emit("keydown", { key: "Home" });
	assert.equal(app.document.activeElement, options[0]);
	assert.equal(options.filter((n) => n.tabIndex === 0).length, 1);
	assert.equal(app.state.pathExplorer.sortOrder, "discovery", "focus navigation must not commit a sort");
	options[2].emit("click");
	assert.equal(options[2].attrs["aria-checked"], "true");
	assert.equal(app.get("#path-sort-current").textContent, "Absolute");
	assert.match(toggle.attrs["aria-label"], /Current order: Absolute/);
	assert.equal(popover.hidden, true, "choosing an option closes the menu");
	assert.equal(app.document.activeElement, toggle);
	toggle.emit("click");
	assert.equal(app.document.activeElement, options[2], "reopening focuses the active option");
	app.document.emit("pointerdown", { target: options[2] });
	assert.equal(popover.hidden, false);
	app.document.emit("keydown", { key: "Escape" });
	assert.equal(popover.hidden, true);
	assert.equal(app.document.activeElement, toggle);
	toggle.emit("keydown", { key: "ArrowUp" });
	assert.equal(app.document.activeElement, options[2]);
	app.get("#path-sort-wrap").emit("focusout", { relatedTarget: app.get("#path-clear") });
	assert.equal(popover.hidden, true);
	toggle.emit("keydown", { key: "ArrowDown" });
	assert.equal(app.document.activeElement, options[0]);
	app.document.emit("pointerdown", { target: app.get("#canvas-wrap") });
	assert.equal(popover.hidden, true);
	toggle.emit("click");
	app.get("#path-clear").emit("click");
	assert.equal(toggle.disabled, true);
	assert.equal(popover.hidden, true);
	assert.equal(app.get("#path-score-context").hidden, true);
});

test("compact cards show tiny signed scores and concise differences without summary hover tooltips", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	graph.path_provenance.paths.forEach((p) => { p.score_kind = "isolated_path_effect"; });
	graph.path_provenance.paths[0].score = 1.43051e-6;
	graph.path_provenance.paths[1].score = -1.43051e-6;
	const original = JSON.stringify(graph);
	const result = saved(graph);
	Object.assign(result.meta, { metric: "logit_difference", mode: "denoising" });
	app.state.savedResults.path = result;
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const buttons = pathButtons(app);
	assert.equal(buttons[0].querySelectorAll(".path-score")[0].textContent, "+1.43 × 10⁻⁶");
	assert.equal(buttons[1].querySelectorAll(".path-score")[0].textContent, "−1.43 × 10⁻⁶");
	assert.match(buttons[0].querySelectorAll(".path-score")[0].title, /Original value: 0.00000143051/);
	for (const entry of app.state.pathEntries.values()) {
		assert.equal(entry.button.title, undefined);
		assert.equal(entry.summary.title, undefined);
		assert.ok(entry.summary.children.every((n) => n.title === undefined));
		assert.ok(entry.summary.children.length <= 4);
		assert.doesNotMatch(entry.button.textContent, /path-\d|Path effect|Logit Difference|token position|\[Q/);
	}
	assert.equal(app.get("#path-score-context").textContent, "Effect on logit difference · Denoising · logit units");
	assert.equal(app.get("#selected-path-meta").textContent, "Path 1");
	buttons[1].emit("click");
	assert.equal(app.get("#selected-path-meta").textContent, "Path 2");
	assert.equal(JSON.stringify(graph), original);
});

});

describe("Selected path diagram rendering", () => {

test("selected diagram adapts the original marker styles and query links, preserving the full occurrence order", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	Object.assign(graph.path_provenance.occurrences.find((n) => n.id === "path-3-occ-1"), { variant: "KV", kv_position: 0 });
	const original = JSON.stringify(graph);
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	assert.equal(app.get("#selected-path-details").hidden, true);
	app.onClickNode({ id: "a" });
	const diagram = app.get("#selected-path-diagram");
	const groups = () => diagram.querySelectorAll(".path-component");
	assert.equal(app.get("#selected-path-details").hidden, false);
	assert.deepEqual(groups().map((g) => g.dataset.kind), ["final", "mlp", "attn", "embed"]);
	assert.deepEqual(groups().map((g) => g.dataset.occurrenceId), ["path-1-occ-3", "path-1-occ-2", "path-1-occ-1", "path-1-occ-0"]);
	assert.equal(groups()[0].querySelectorAll("rect")[1].attrs.fill, "#FFA07A");
	assert.equal(groups()[1].querySelectorAll("rect")[1].attrs.fill, "#90EE90");
	assert.equal(groups()[2].querySelectorAll("circle")[0].attrs.fill, "#B0C4DE");
	assert.equal(groups()[3].querySelectorAll("rect")[1].attrs.fill, "#FFDAB9");
	assert.match(groups()[2].textContent, /A0H0 · Layer 0.*Query\/output position: 1.*Patch Q/);
	assert.doesNotMatch(groups()[2].textContent, /unspecified|Source position/);
	assert.ok(groups()[2].querySelectorAll(".path-component-token").some((n) => n.textContent.includes('"word"')));
	assert.match(diagram.querySelectorAll(".path-token-context")[0].textContent, /first clean prompt/);
	assert.equal(diagram.querySelectorAll(".path-component-link").length, 3);
	assert.equal(diagram.querySelectorAll(".path-component-link").filter((n) => n.attrs["stroke-dasharray"]).length, 1);
	assert.ok(groups().every((g) => g.attrs.role === "button" && g.attrs.tabindex === "0"));
	app.get("#path-list").scrollTop = 500;
	diagram.scrollTop = 120;
	pathButtons(app)[2].emit("click");
	assert.equal(app.get("#path-list").scrollTop, 500);
	assert.equal(diagram.scrollTop, 0);
	assert.match(groups()[2].textContent, /Patch KV · Source position: 0/);
	assert.ok(groups()[2].querySelectorAll(".path-component-token").some((n) => n.textContent.includes('"BOS"')));
	assert.equal(diagram.querySelectorAll(".path-component-link").some((n) => n.attrs["stroke-dasharray"]), false);
	app.get("#path-clear").emit("click");
	assert.equal(app.get("#selected-path-details").hidden, true);
	assert.equal(diagram.children.length, 0);
	assert.equal(JSON.stringify(graph), original, "diagram rendering must not modify scores, provenance or graph data");
});

test("diagram does not collapse original occurrences or guess missing patching metadata", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	const path = graph.path_provenance.paths[0];
	path.node_ids.splice(2, 0, "a");
	path.occurrence_ids.splice(2, 0, "repeated-a");
	graph.path_provenance.occurrences.push({ id: "repeated-a", node_id: "a", kind: "attn", layer: 0, head: 0, position: 1 });
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const nodes = app.get("#selected-path-diagram").querySelectorAll(".path-component");
	assert.equal(nodes.length, 5);
	assert.equal(nodes.filter((n) => n.dataset.nodeId === "a").length, 2);
	assert.match(nodes.find((n) => n.dataset.occurrenceId === "repeated-a").textContent, /Patch: Unavailable/);
	assert.match(pathButtons(app)[0].textContent, /Some intervention metadata unavailable/);
});

test("diagram never uses merged intervention metadata when the original occurrence is absent", () => {
	const app = appHarness();
	const graph = fixture();
	Object.assign(graph.nodes.find((n) => n.id === "a"), { variant: "KV", kv_position: 0,
		patch_query: false, patch_key: true, patch_value: true, kv_positions: [0, 1] });
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const diagram = app.get("#selected-path-diagram");
	const attention = diagram.querySelectorAll(".path-component").find((n) => n.dataset.kind === "attn");
	assert.match(attention.textContent, /Query\/output position: Unavailable.*Patch: Unavailable.*Source position: Unavailable/);
	assert.doesNotMatch(attention.textContent, /Patch KV|Source position: 0/);
	assert.equal(attention.querySelectorAll(".path-component-token").length, 0);
});

test("diagram rejects an occurrence whose grid identity does not match the path step", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	graph.path_provenance.occurrences.find((n) => n.id === "path-1-occ-1").node_id = "second-head";
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const diagram = app.get("#selected-path-diagram");
	const attention = diagram.querySelectorAll(".path-component").find((n) => n.dataset.kind === "attn");
	assert.match(attention.textContent, /Query\/output position: Unavailable.*Patch: Unavailable/);
	assert.equal(attention.querySelectorAll(".path-component-token").length, 0);
	assert.equal(diagram.querySelectorAll(".path-component-link").some((n) => n.attrs["stroke-dasharray"]), false);
});

test("contradictory original flags cannot create a Q label or dotted query connection", () => {
	const app = appHarness();
	const graph = occurrenceFixture();
	Object.assign(graph.path_provenance.occurrences.find((n) => n.id === "path-1-occ-1"), { patch_query: false });
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	const diagram = app.get("#selected-path-diagram");
	const attention = diagram.querySelectorAll(".path-component").find((n) => n.dataset.kind === "attn");
	assert.match(attention.textContent, /Patch: Unavailable/);
	assert.doesNotMatch(attention.textContent, /Patch Q/);
	assert.equal(diagram.querySelectorAll(".path-component-link").some((n) => n.attrs["stroke-dasharray"]), false);
});

});

describe("Component decoding within the selected path", () => {

	function decodedFixture(method = "path", positional = true) {
		const graph = occurrenceFixture();
		graph.path_provenance.source = method;
		for (const n of [...graph.nodes, ...graph.path_provenance.occurrences]) {
			if (!positional) n.position = null;
		}
		for (const n of graph.nodes.filter((n) => ["attn", "mlp"].includes(n.kind))) {
			n.component_logit_lens = { available: true, convention: "isolated_component_softmax",
				position: n.position ?? 1, position_token: n.position === 0 ? "BOS" : "word",
				top_tokens: [0.2, 0.1, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 1e-10, 0].map((probability, i) => ({ rank: i + 1, token: ` ${n.id}<${i}>`, probability })) };
		}
		return graph;
	}
	const components = (app) => app.get("#selected-path-diagram").querySelectorAll(".path-component");
	const chart = (app) => app.get("#component-decoding-content");
	const selectComponent = (app, kind) => components(app).find((g) => g.dataset.kind === kind).emit("click");

	for (const method of ["tree", "path"]) test(`${method}: clicking/keyboard-decoding leaves path, graph and cached scores unchanged`, () => {
		const app = appHarness(), graph = decodedFixture(method);
		const cached = saved(graph, method), original = JSON.stringify(cached);
		app.state.savedResults[method] = cached;
		app.restoreSavedResult(method);
		app.onClickNode({ id: "a" });
		app.render();
		const geometry = snapshotGraph(app.get), lanes = laneSnapshot(app.get("#grid"));
		const view = JSON.stringify(app.state.view), reference = app.state.pathExplorer.selectedId;
		assert.match(chart(app).textContent, /Select a component/);
		app.get("#selected-path-diagram").scrollTop = 80;
		app.get("#path-list").scrollTop = 100;
		const attention = components(app).find((g) => g.dataset.kind === "attn");
		attention.focus();
		attention.emit("click");
		assert.equal(attention.attrs["aria-pressed"], "true");
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 9);
		assert.match(chart(app).textContent, /A0H0Query\/output position: 1 \("word"\)/);
		assert.match(chart(app).textContent, /0\.2000/);
		const bars = chart(app).querySelectorAll(".component-decoding-bar");
		assert.equal(Number(bars[1].attrs.width) / Number(bars[0].attrs.width), 0.5);
		assert.equal(chart(app).querySelectorAll(".component-decoding-probability")[0].dataset.probability, "0.2");
		assert.equal(chart(app).querySelectorAll(".component-decoding-axis-title")[0].textContent, "Probability");
		assert.equal(chart(app).querySelectorAll(".component-decoding-gridline").length,
			chart(app).querySelectorAll(".component-decoding-tick").length);
		assert.equal(chart(app).querySelectorAll(".component-decoding-token")[0].textContent, "␠a<0>");
		assert.equal(chart(app).querySelectorAll("script").length, 0);
		assert.equal(app.document.activeElement, attention, "component selection must not replace focused diagram elements");
		const mlp = components(app).find((g) => g.dataset.kind === "mlp");
		mlp.emit("keydown", { key: "ArrowDown" });
		assert.equal(attention.attrs["aria-pressed"], "true");
		mlp.emit("keydown", { key: "Enter" });
		assert.equal(attention.attrs["aria-pressed"], "false");
		assert.equal(mlp.attrs["aria-pressed"], "true");
		assert.match(chart(app).textContent, /MLP0Token position: 1/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-token")[0].textContent, "␠b<0>");
		attention.emit("keydown", { key: " " });
		assert.equal(attention.attrs["aria-pressed"], "true");
		assert.equal(app.get("#selected-path-diagram").scrollTop, 80);
		assert.equal(app.get("#path-list").scrollTop, 100);
		assert.equal(app.state.pathExplorer.selectedId, reference);
		assert.equal(app.state.pathExplorer.nodeId, "a");
		assert.equal(JSON.stringify(app.state.view), view);
		assert.deepEqual(snapshotGraph(app.get), geometry);
		assert.deepEqual(laneSnapshot(app.get("#grid")), lanes);
		assert.equal(JSON.stringify(cached), original);
	});

	test("two heads in the same summary select their own cached payload without changing the path", () => {
		const app = appHarness(), graph = decodedFixture();
		const second = graph.nodes.find((n) => n.id === "second-head");
		const path = graph.path_provenance.paths[0];
		path.node_ids[2] = second.id;
		Object.assign(graph.path_provenance.occurrences.find((n) => n.id === path.occurrence_ids[2]),
			{ ...second, id: path.occurrence_ids[2], node_id: second.id });
		graph.edges.push({ source: "a", target: second.id, count: 1 }, { source: second.id, target: "f", count: 1 });
		app.state.savedResults.path = saved(graph);
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		const heads = components(app).filter((g) => g.dataset.kind === "attn");
		heads[0].emit("click");
		assert.match(chart(app).textContent, /A0H1/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-token")[0].attrs["aria-label"], "␠second-head<0>");
		heads[1].emit("click");
		assert.match(chart(app).textContent, /A0H0/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-token")[0].textContent, "␠a<0>");
		assert.equal(app.state.pathExplorer.selectedId, "path-1");
	});

	test("position zero is decoded as BOS and contradictory occurrence coordinates never reuse a grid readout", () => {
		const app = appHarness(), graph = decodedFixture();
		const mlp = graph.nodes.find((n) => n.id === "b");
		mlp.position = mlp.component_logit_lens.position = 0;
		mlp.component_logit_lens.position_token = "BOS";
		for (const n of graph.path_provenance.occurrences.filter((n) => n.node_id === "b")) n.position = 0;
		graph.path_provenance.occurrences.find((n) => n.id === "path-1-occ-1").position = 0;
		app.state.savedResults.path = saved(graph);
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "mlp");
		assert.match(chart(app).textContent, /Token position: 0 \("BOS"\)/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 9);
		selectComponent(app, "attn");
		assert.match(chart(app).textContent, /Matching component metadata is unavailable/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 0);
	});

	test("decoding information toggles and dismisses without changing component or path selection", () => {
		const app = appHarness();
		app.state.savedResults.path = saved(decodedFixture());
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		const info = app.get("#component-decoding-info"), popover = app.get("#component-decoding-info-popover");
		const reference = app.state.pathExplorer.selectedId, selection = app.state.componentSelection;
		const before = chart(app).textContent;
		assert.equal(popover.hidden, true);
		info.emit("click");
		assert.equal(info.attrs["aria-expanded"], "true");
		assert.equal(popover.hidden, false);
		app.document.emit("pointerdown", { target: popover });
		assert.equal(popover.hidden, false);
		app.document.emit("keydown", { key: "Escape" });
		assert.equal(popover.hidden, true);
		assert.equal(info.attrs["aria-expanded"], "false");
		assert.equal(app.document.activeElement, info);
		info.emit("click");
		info.emit("click");
		assert.equal(popover.hidden, true);
		info.emit("click");
		app.document.emit("pointerdown", { target: app.get("#grid") });
		assert.equal(popover.hidden, true);
		info.emit("click");
		app.get("#component-decoding-info-wrap").emit("focusout", { relatedTarget: app.get("#path-clear") });
		assert.equal(popover.hidden, true);
		assert.equal(app.state.pathExplorer.selectedId, reference);
		assert.equal(app.state.componentSelection, selection);
		assert.equal(chart(app).textContent, before);
		info.emit("click");
		pathButtons(app)[1].emit("click");
		assert.equal(popover.hidden, true, "switching paths clears the popover");
		info.emit("click");
		app.get("#path-clear").emit("click");
		assert.equal(popover.hidden, true, "clearing the Inspector clears the popover");
	});

	test("decoding information fits the scrolling Inspector and repositions while open", () => {
		const app = appHarness(), info = app.get("#component-decoding-info");
		const panel = app.get("#path-explorer"), popover = app.get("#component-decoding-info-popover");
		panel.getBoundingClientRect = () => ({ left: 280, top: 400, width: 620, height: 250 });
		info.bounds = { left: 850, top: 450, width: 22, height: 22 };
		info.emit("click");
		assert.equal(popover.dataset.side, "below");
		assert.equal(popover.style.maxHeight, "170px");
		info.bounds.top = 600;
		panel.emit("scroll");
		assert.equal(popover.dataset.side, "above");
		assert.equal(popover.style.maxHeight, "192px");
		info.bounds.top = 450;
		app.resizeWindow(900);
		assert.equal(popover.dataset.side, "below");
		assert.equal(popover.style.maxHeight, "170px");
		assert.equal(popover.hidden, false);
	});

	test("histogram renderer accepts token/probability pairs without component-specific metadata", () => {
		const app = appHarness();
		const plot = app.document.createElement("div"), axis = app.document.createElementNS("http://www.w3.org/2000/svg", "svg");
		const tokens = [{ token: " generic", probability: 0.189 }, { token: "other", probability: 1e-12 }];
		ComponentDecoding.renderHistogram(plot, axis, tokens);
		assert.equal(plot.querySelectorAll(".component-decoding-row").length, 2);
		assert.equal(plot.querySelectorAll(".component-decoding-probability")[0].textContent, "0.1890");
		assert.equal(plot.querySelectorAll(".component-decoding-probability")[1].textContent, "1.000e-12");
		assert.equal(axis.querySelectorAll(".component-decoding-axis-title")[0].textContent, "Probability");
		assert.doesNotMatch(plot.querySelectorAll("svg")[0].attrs["aria-label"], /component|path/i);
		assert.equal(app.state.componentSelection, null);
	});

	test("sorting preserves component selection; path/method/result changes and live discovery clear stale data", () => {
		const app = appHarness(), graph = decodedFixture();
		app.state.savedResults.path = saved(graph);
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		const selection = app.state.componentSelection;
		app.get("#path-sort").querySelectorAll("button")[1].emit("click");
		assert.equal(app.state.componentSelection, selection);
		assert.match(chart(app).textContent, /A0H0/);
		const oldButton = components(app).find((g) => g.dataset.kind === "attn");
		pathButtons(app).find((b) => b.dataset.pathId === "path-2").emit("click");
		assert.equal(app.state.componentSelection, null);
		assert.match(chart(app).textContent, /Select a component/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 0);
		oldButton.emit("click"); // A delayed event from the previous diagram must be ignored.
		assert.equal(app.state.componentSelection, null);
		selectComponent(app, "mlp");
		assert.match(chart(app).textContent, /MLP1/);
		app.restoreSavedResult("path");
		assert.equal(app.state.componentSelection, null);
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		const live = { method: "path", params: {}, meta: null, liveState: app.createLiveGraphState() };
		app.attachLiveGraph(live);
		assert.equal(app.state.componentSelection, null);
		assert.equal(app.get("#selected-path-details").hidden, true);
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		app.state.savedResults.acdc = saved(fixture("acdc"), "acdc");
		app.selectMethod("acdc");
		app.onClickNode({ id: "a" });
		assert.equal(app.state.componentSelection, null);
		assert.equal(app.get("#path-explorer").hidden, true);
	});

	test("non-positional fallback is explicit and different K/V occurrences reuse the full-head readout", () => {
		const app = appHarness(), graph = decodedFixture("path", false);
		Object.assign(graph.path_provenance.occurrences.find((n) => n.id === "path-3-occ-1"), { variant: "KV", kv_position: null });
		const result = saved(graph);
		result.meta.positional = false;
		app.state.savedResults.path = result;
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		const decoding = chart(app).textContent;
		assert.match(decoding, /Query\/output position: 1/);
		assert.match(decoding, /last clean-prompt token/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-disclaimer")[0].textContent,
			"Exploratory component projection, not the full model's next-token probabilities or the selected path's causal contribution.");
		assert.equal(chart(app).querySelectorAll(".component-decoding-context").length, 0);
		assert.doesNotMatch(decoding, /Zero-based positions|⇥ tab|↵ newline/);
		assert.doesNotMatch(decoding, /not restricted to the path's K\/V source/);
		pathButtons(app)[2].emit("click");
		selectComponent(app, "attn");
		assert.equal(chart(app).textContent, decoding);
		assert.match(components(app).find((g) => g.dataset.kind === "attn").textContent, /Source positions: all eligible/);
	});

	test("unsupported/missing/invalid readouts and empty circuits replace, rather than retain, the previous chart", () => {
		const app = appHarness(), graph = decodedFixture();
		graph.nodes.find((n) => n.id === "b").component_logit_lens = { available: false };
		app.state.savedResults.path = saved(graph);
		app.restoreSavedResult("path");
		app.onClickNode({ id: "a" });
		selectComponent(app, "attn");
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 9);
		selectComponent(app, "mlp");
		assert.match(chart(app).textContent, /No cached component decoding/);
		assert.equal(chart(app).querySelectorAll(".component-decoding-row").length, 0);
		for (const kind of ["embed", "final"]) {
			selectComponent(app, kind);
			assert.match(chart(app).textContent, /individual attention heads and MLP outputs only/);
		}
		graph.nodes.find((n) => n.id === "a").component_logit_lens.top_tokens[0].probability = NaN;
		selectComponent(app, "attn");
		assert.match(chart(app).textContent, /Valid token probabilities are unavailable/);
		app.onClickNode({ id: "spare" });
		assert.equal(app.state.componentSelection, null);
		assert.equal(app.get("#selected-path-details").hidden, true);
		app.state.savedResults.path = saved({ nodes: [], edges: [], path_provenance: { version: 1, source: "path", paths: [] } });
		app.restoreSavedResult("path");
		assert.equal(app.get("#selected-path-details").hidden, true);
		assert.equal(app.get("#selected-path-diagram").children.length, 0);
	});
});

describe("Application lifecycle, filtering and navigation state", () => {

test("a new node replaces exploration and nodes without original paths do not invent graph routes", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.render();
	const baseline = snapshotGraph(app.get);
	app.onClickNode({ id: "a" });
	pathButtons(app)[6].emit("click");
	app.onClickNode({ id: "e1" });
	assert.equal(app.state.pathExplorer.selectedId, "path-2");
	assert.equal(app.get("#path-list").children.length, 3);
	app.onClickNode({ id: "spare" }); // merged graph has spare -> FINAL, provenance does not
	app.render();
	assert.equal(app.state.pathExplorer.selectedId, null);
	assert.equal(app.get("#path-list").children.length, 0);
	assert.match(app.get("#path-help").textContent, /No complete discovered paths/);
	assert.equal(app.get("#threshold-slider").disabled, false);
	assert.equal(app.get("#grid").querySelectorAll(".path-stroke").length, 0);
	assert.deepEqual(snapshotGraph(app.get), baseline);
});

test("completed-only state, method/result resets and independent per-method caches", () => {
	const app = appHarness();
	const graph = fixture();
	app.state.savedResults.path = saved(graph);
	app.state.savedResults.tree = saved(fixture("tree"), "tree");
	app.state.savedResults.acdc = saved(fixture("acdc"), "acdc");
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	app.state.pathExplorer.select("path-1");
	app.selectMethod("tree");
	assert.equal(app.state.pathExplorer.active, false);
	app.onClickNode({ id: "a" });
	assert.equal(app.state.pathExplorer.eligible[0], "tree-1");
	app.selectMethod("path");
	assert.equal(app.state.pathExplorer.active, false);
	assert.equal(app.state.result.graph, graph);
	app.onClickNode({ id: "a" });
	app.restoreSavedResult("path");
	assert.equal(app.state.pathExplorer.active, false);
	app.selectMethod("acdc");
	app.onClickNode({ id: "a" });
	assert.equal(app.state.pathExplorer.active, false);
	assert.equal(app.get("#path-explorer").hidden, true);
	app.clearDisplay();
	app.onClickNode({ id: "a" });
	assert.equal(app.state.pathExplorer, null);
	assert.equal(app.get("#path-explorer").hidden, true);
});

test("completed Tree flags never propagate through unrelated merged branches", () => {
	const app = appHarness();
	const graph = fixture("tree");
	graph.edges.find((e) => e.source === "a" && e.target === "c").complete = false;
	app.state.savedResults.tree = saved(graph, "tree");
	app.restoreSavedResult("tree");
	assert.equal(app.state.edges.get("a->c").complete, false);
	app.render();
	assert.equal(app.get("#grid").querySelectorAll(".edge")
		.find((e) => e.dataset.key === "a->c").attrs["stroke-dasharray"], "3 4");
});

test("background method completion leaves exploration intact; attaching a live run resets it", () => {
	const app = appHarness();
	app.state.savedResults.path = saved(fixture());
	app.restoreSavedResult("path");
	app.onClickNode({ id: "a" });
	app.state.pathExplorer.select("path-1");
	const job = { method: "tree", params: {}, meta: null, liveState: app.createLiveGraphState() };
	app.state.activeJob = job;
	app.state.running = true;
	app.openStream("background-tree", job);
	const result = saved(fixture("tree"), "tree");
	app.state.es.onmessage({ data: JSON.stringify({ event: "result", graph: result.graph, meta: result.meta }) });
	assert.equal(app.state.pathExplorer.selectedId, "path-1");
	assert.equal(app.state.savedResults.tree.graph.path_provenance.source, "tree");
	assert.equal(app.state.savedResults.path.graph.path_provenance.source, "path");
	app.attachLiveGraph(job);
	assert.equal(app.state.pathExplorer, null);
	assert.equal(app.state.result, null);
	assert.equal(app.get("#path-explorer").hidden, true);
});

test("filter is paused and restored, chips never reflow, zoom/pan retain the SVG viewport", () => {
	const app = appHarness();
	const graph = fixture();
	graph.nodes.find((n) => n.id === "second-head").contribution = 0.01;
	app.state.savedResults.path = saved(graph);
	app.restoreSavedResult("path");
	app.state.threshold = 0.2;
	app.render();
	const grid = app.get("#grid");
	assert.equal(grid.querySelectorAll(".chip").some((n) => n.dataset.id === "second-head"), false);
	const aBefore = grid.querySelectorAll(".chip").find((n) => n.dataset.id === "a").children[0].attrs;
	app.onClickNode({ id: "a" });
	app.render();
	assert.equal(app.get("#threshold-slider").disabled, true);
	assert.deepEqual(grid.querySelectorAll(".chip").find((n) => n.dataset.id === "a").children[0].attrs, aBefore);
	assert.equal(grid.querySelectorAll(".chip").some((n) => n.dataset.id === "second-head"), true);
	const before = { ...app.state.view };
	grid.emit("wheel", { deltaY: -1, clientX: 150, clientY: 200, preventDefault() {} });
	assert.ok(app.state.view.k > before.k);
	assert.match(app.get("#viewport").attrs.transform, /scale\(1\.12\)/);
	grid.emit("pointerdown", { clientX: 10, clientY: 20, pointerId: 1 });
	const x = app.state.view.x;
	grid.emit("pointermove", { clientX: 40, clientY: 20 });
	assert.equal(app.state.view.x, x + 30);
	assert.equal(app.get("#viewport").attrs.transform,
		`translate(${app.state.view.x},${app.state.view.y}) scale(${app.state.view.k})`);
	grid.emit("pointerup");
	const view = { ...app.state.view };
	app.get("#path-clear").emit("click");
	app.render();
	assert.equal(app.state.threshold, 0.2);
	assert.equal(app.get("#threshold-slider").disabled, false);
	assert.equal(grid.querySelectorAll(".chip").some((n) => n.dataset.id === "second-head"), false);
	assert.deepEqual({ ...app.state.view }, view);
});

});
