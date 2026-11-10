"use strict";

// Optional real-CSS regression test. Uses an installed Chromium browser's CDP,
// not a new frontend dependency. Set IPE_TEST_BROWSER to its executable path.
const { describe, test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const http = require("node:http");
const { spawn } = require("node:child_process");

const browserPath = process.env.IPE_TEST_BROWSER || (process.platform === "win32"
	? ["C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
		"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"].find((file) => fs.existsSync(file)) : null);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function graphFixture() {
	const node = (id, kind, layer, position, head) => ({ id, kind, layer, position, head,
		contribution: 0.5, complete: true, merged: 1,
		component_logit_lens: ["attn", "mlp"].includes(kind) ? {
			available: true, convention: "isolated_component_softmax", position,
			position_token: position === 10 ? " to" : "<BOS>",
			top_tokens: [0.25, 0.125, 0.03125, 0.02, 0.01, 0.005, 0.002, 0.001, 1e-10, 0].map((probability, i) => ({ rank: i + 1,
				token: i === 0 ? ' <script>window.badToken=true</script>' + "🧠LongToken".repeat(30) : ` ${id}-${i}`,
				probability })),
		} : { available: false } });
	const paths = Array.from({ length: 9 }, (_, i) => ({ id: `path-${i + 1}`, complete: true,
		node_ids: i === 8 ? ["e0", "b", "a", "m", "f"] : [i % 2 ? "e1" : "e0", "a", "m", "f"],
		score: (i + 1) / 10, score_kind: "isolated_path_effect" }));
	paths[8].score = 1.43051e-6;
	for (let i = 0; i < 5; i++) paths.push({ id: `pruned-${i}`, complete: false, node_ids: ["pruned", "a", "m", "f"] });
	const edges = new Map();
	for (const p of paths) for (let i = 1; i < p.node_ids.length; i++) {
		const source = p.node_ids[i - 1], target = p.node_ids[i];
		edges.set(`${source}->${target}`, { source, target, count: 1, contribution: 0.5, complete: source !== "pruned" });
	}
	const nodes = [node("e0", "embed", null, 0), node("e1", "embed", null, 1),
		node("b", "mlp", 0, 0), node("a", "attn", 5, 10, 9), node("m", "mlp", 6, 10),
		node("f", "final", 6, 10), node("pruned", "mlp", 1, 2)];
	const occurrences = [], byId = new Map(nodes.map((n) => [n.id, n]));
	const variants = [
		{ patch_query: true, patch_key: false, patch_value: false, kv_position: null },
		{ patch_query: false, patch_key: true, patch_value: true, kv_position: 0 },
		{ patch_query: true, patch_key: true, patch_value: false, kv_position: 3 },
		{ patch_query: false, patch_key: false, patch_value: true, kv_position: 7 },
		{ patch_query: true, patch_key: true, patch_value: true, kv_position: null },
		{ patch_query: false, patch_key: true, patch_value: false },
		{}, // Original occurrence with genuinely unavailable patching metadata.
	];
	paths.forEach((p, index) => {
		p.occurrence_ids = p.node_ids.map((id, i) => {
			const occurrence = { ...byId.get(id), id: `${p.id}-occ-${i}`, node_id: id, patch_type: "zero",
				...(id === "a" ? variants[index] || variants[0] : {}) };
			if (index === 2 && ["a", "m", "f"].includes(id)) occurrence.patch_type = "counterfactual";
			if (index !== 7 || id !== "a") occurrences.push(occurrence);
			return occurrence.id;
		});
	});
	// A missing occurrence must not recover these deliberately misleading flags.
	Object.assign(byId.get("a"), { variant: "KV", patch_query: false, patch_key: true, patch_value: true, kv_position: 4 });
	return { nodes, edges: [...edges.values()],
		path_provenance: { version: 1, source: "path", order: "discovery", paths, occurrences } };
}

describe("Path Inspector browser layout and navigation", () => {

test("Inspector container width and Clear/header stability preserve manual circuit navigation", {
	skip: !browserPath || typeof WebSocket === "undefined" ? "Requires an installed Chromium browser and Node's WebSocket API" : false,
	timeout: 60000,
}, async () => {
	const root = path.resolve(__dirname, "../../visualization");
	const server = http.createServer((req, res) => {
		res.setHeader("Cache-Control", "no-store");
		if (req.url === "/api/config") {
			res.setHeader("Content-Type", "application/json");
			return res.end(JSON.stringify({ models: ["gpt2-small"], metrics: ["logit_difference"],
				denoising_metrics: ["logit_difference"], acdc_metrics: ["kl_div"], device: "test" }));
		}
		const name = req.url === "/" ? "static/index.html" : req.url.split("?")[0];
		const file = path.join(root, name);
		if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
		res.setHeader("Content-Type", { ".html": "text/html", ".css": "text/css", ".js": "text/javascript" }[path.extname(file)] || "text/plain");
		res.end(fs.readFileSync(file));
	});
	await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
	const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ipe-inspector-layout-"));
	const browser = spawn(browserPath, ["--headless=new", "--no-first-run", "--no-default-browser-check",
		"--disable-gpu", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
	let socket, closeBrowser;
	try {
		const portFile = path.join(profile, "DevToolsActivePort");
		for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await delay(100);
		const port = fs.readFileSync(portFile, "utf8").split("\n")[0];
		const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
		socket = new WebSocket(target.webSocketDebuggerUrl);
		await new Promise((resolve, reject) => { socket.addEventListener("open", resolve); socket.addEventListener("error", reject); });
		let nextId = 0;
		const pending = new Map(), errors = [];
		socket.addEventListener("message", (event) => {
			const message = JSON.parse(event.data);
			if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails);
			if (!message.id) return;
			const { resolve, reject } = pending.get(message.id);
			pending.delete(message.id);
			message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
		});
		const send = (method, params = {}) => new Promise((resolve, reject) => {
			const id = ++nextId;
			pending.set(id, { resolve, reject });
			socket.send(JSON.stringify({ id, method, params }));
		});
		closeBrowser = () => send("Browser.close");
		const evaluate = async (expression) => {
			const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
			assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
			return result.result.value;
		};
		const viewport = async (width, height = 800) => {
			await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
			await delay(80);
		};
		const measure = () => evaluate(`(() => {
			const rect = (s) => { const r=document.querySelector(s).getBoundingClientRect(); return [r.x,r.y,r.width,r.height]; };
			return { panel:rect('#path-explorer'), sidebar:rect('#sidebar'), graph:rect('#canvas-wrap'),
				actions:['#path-clear','#path-info'].map(rect), heading:rect('#path-heading'), sort:rect('#path-sort-toggle'),
				cardWidth:document.querySelector('.path-select')?.getBoundingClientRect().width,
				bodyWidth:document.body.getBoundingClientRect().width, scrollWidth:document.documentElement.scrollWidth,
				view:JSON.stringify(state.view), selection:state.pathExplorer.selectedId,
				geometry:JSON.stringify([...document.querySelectorAll('#grid .chip')].map(n=>[n.dataset.id,n.children[0].outerHTML])),
				lanes:JSON.stringify([...document.querySelectorAll('.path-stroke')].map(n=>[n.dataset.pathId,n.dataset.key,n.getAttribute('d')])
					.sort((a,b)=>(a[0]+a[1]).localeCompare(b[0]+b[1]))) };
		})()`);
		await send("Runtime.enable");
		await viewport(1440);
		await send("Page.navigate", { url: `http://127.0.0.1:${server.address().port}/` });
		await delay(350);
		const tokens = Array.from({ length: 11 }, (_, i) => String(i));
		tokens[0] = "<BOS>";
		tokens[3] = " long" + "🧠WideUnbrokenToken".repeat(25);
		tokens[7] = ' <script>window.badToken=true</script><img src=x onerror="window.badToken=true">&"';
		tokens[10] = " to";
		await evaluate(`state.savedResults.path={graph:${JSON.stringify(graphFixture())},meta:{method:'path',positional:true,
			tokens:${JSON.stringify(tokens)},cf_tokens:Array(11).fill('WRONG_COUNTERFACTUAL_TOKEN'),
			n_layers:7,n_heads:12,metric:'logit_difference',mode:'denoising'},params:null};
			restoreSavedResult('path');state.view={x:87,y:-46,k:1.37};onClickNode({id:'a'});render();`);
		const initial = await measure();
		assert.equal(initial.panel[2], 480, "the actual panel container, not only its contents, must be 480px");
		const [x, y, w, h] = initial.actions[0];
		await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: x + w / 2, y: y + h / 2 });
		assert.deepEqual((await measure()).actions, initial.actions, "Clear hover must not move header actions");
		await evaluate(`document.querySelector('#path-clear').focus({preventScroll:true})`);
		assert.deepEqual((await measure()).actions, initial.actions, "Clear focus must not move header actions");
		await evaluate(`document.querySelector('#path-info').click()`);
		assert.deepEqual((await measure()).sort, initial.sort, "the information popover must not reflow the header");
		const help = await evaluate(`(() => {
			const info=document.querySelector('#path-info'), popover=document.querySelector('#path-info-popover');
			return { title:popover.querySelector('strong').textContent, items:[...popover.querySelectorAll('li')].map(n=>n.textContent),
				note:popover.querySelector('.path-info-note').textContent, describedBy:info.getAttribute('aria-describedby'),
				noteSize:parseFloat(getComputedStyle(popover.querySelector('.path-info-note')).fontSize),
				bodySize:parseFloat(getComputedStyle(popover).fontSize) };
		})()`);
		assert.equal(help.title, "Exploring paths");
		assert.deepEqual(help.items, [
			"Click a circuit component to explore the complete discovered paths passing through it.",
			"Select a path from the graph or the list to highlight it and inspect its components.",
			"Compare paths by their structural and Q/KV patching differences.",
			"Path contribution scores measure the effect of a path intervention on the selected output metric, not probabilities or importance percentages.",
		]);
		assert.equal(help.note, "Only complete discovered paths are selectable. The contribution filter is temporarily paused during exploration.");
		assert.equal(help.describedBy, "path-info-popover");
		assert.ok(help.noteSize < help.bodySize);
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape" });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true);
		await evaluate(`document.querySelector('#path-clear').focus();document.querySelector('#path-info').focus()`);
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), false, "keyboard focus exposes the help");
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r" });
		await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
		assert.equal(await evaluate(`document.querySelector('#path-info').dataset.pinned`), "true", "keyboard activation pins the help");
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape" });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true);
		const navigator = () => evaluate(`(() => {
			const box=n=>{const r=n.getBoundingClientRect();return [r.x,r.y,r.width,r.height];};
			const cards=[...document.querySelectorAll('.path-select')];
			return { context:document.querySelector('#path-score-context').textContent,
				legend:box(document.querySelector('#path-key')),sort:box(document.querySelector('#path-sort-toggle')),
				cards:cards.map(n=>({ id:n.dataset.pathId,text:n.textContent,box:box(n),number:box(n.querySelector('.path-card-heading strong')),
					score:box(n.querySelector('.path-score')),scoreText:n.querySelector('.path-score').textContent,
					scoreWeight:getComputedStyle(n.querySelector('.path-score')).fontWeight, title:n.getAttribute('title'),
					summaryTitles:[...n.querySelectorAll('.path-card-summary [title]')].length,
					summaryLines:n.querySelectorAll('.path-card-summary > span').length,
					diffCount:PathExploration.comparePaths(state.pathExplorer.index.paths.get(state.pathExplorer.selectedId),state.pathExplorer.index.paths.get(n.dataset.pathId)).differences.length,
					remaining:n.parentElement.querySelectorAll('details li').length, detailsHidden:n.parentElement.querySelector('details').hidden,
					tooltips:[...n.querySelectorAll('[title]')].every(n=>n.classList.contains('path-score')),
					overflow:n.scrollWidth>n.clientWidth })),
				subtitle:document.querySelector('#selected-path-meta').textContent,
				activeOrder:document.querySelector('#path-sort [aria-checked="true"]').value };
		})()`);
		let nav = await navigator();
		assert.equal(nav.context, "Effect on logit difference · Denoising · logit units");
		assert.equal(nav.subtitle, "Path 1");
		assert.equal(nav.activeOrder, "discovery");
		assert.ok(Math.abs(nav.legend[1]+nav.legend[3]/2-nav.sort[1]-nav.sort[3]/2)<1, "Sort must align with the legend");
		assert.ok(nav.sort[0]>nav.legend[0]+nav.legend[2], "Sort belongs to the right of the legend");
		assert.equal(nav.cards[8].scoreText, "+1.43 × 10⁻⁶");
		for (const card of nav.cards) {
			assert.equal(card.title, null);
			assert.equal(card.summaryTitles, 0);
			assert.equal(card.tooltips, true, "only the score retains a tooltip, not the summary or its ancestors");
			assert.equal(card.overflow, false);
			if (card.diffCount) assert.equal(card.summaryLines, Math.min(4, card.diffCount));
			assert.equal(card.remaining, Math.max(0, card.diffCount-4));
			assert.equal(card.detailsHidden, card.diffCount<=4);
			assert.doesNotMatch(card.text, /path-\d|Path effect|Logit Difference|logit units|token position/);
			assert.ok(Number(card.scoreWeight)>=600);
			assert.ok(card.score[0]>card.number[0]+card.number[2]);
			assert.ok(card.box[0]+card.box[2]-card.score[0]-card.score[2]<12, "score is aligned right");
		}
		const pressKey = async (key, virtual, text) => {
			const code = key === " " ? "Space" : key;
			await send("Input.dispatchKeyEvent", { type: "keyDown", key, code, windowsVirtualKeyCode: virtual, text });
			await send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode: virtual });
		};
		const clickAt = async (x, y) => {
			await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
			await send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1 });
			await send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
		};
		const decoding = () => evaluate(`(() => {
			const content=document.querySelector('#component-decoding-content');
			const box=n=>{const r=n.getBoundingClientRect();return [r.x,r.y,r.width,r.height];};
			const bounds=n=>{const r=n.getBBox();return [r.x,r.y,r.width,r.height];};
			const axis=content.querySelector('.component-decoding-axis'),svg=content.querySelector('.component-decoding-bars');
			const plot=content.querySelector('.component-decoding-plot'),note=content.querySelector('.component-decoding-disclaimer');
			return { text:content.textContent, values:[...content.querySelectorAll('.component-decoding-probability')].map(n=>n.textContent),
				bars:[...content.querySelectorAll('.component-decoding-bar')].map(n=>Number(n.getAttribute('width'))),
				probabilities:[...content.querySelectorAll('.component-decoding-row')].map(n=>Number(n.dataset.probability)),
				axis:axis&&box(axis), content:box(content),svgWidth:svg&&Number(svg.getAttribute('width')),
				svg:svg&&box(svg),plot:plot&&box(plot),lastBar:svg&&box(svg.querySelector('.component-decoding-row:last-child rect')),
				note:note&&box(note),noteOverflow:note&&note.scrollHeight>note.clientHeight,
				barBoxes:svg&&[...svg.querySelectorAll('text')].map(bounds),
				labels:svg&&[...svg.querySelectorAll('.component-decoding-probability')].map(n=>({
					inside:n.classList.contains('inside'),fill:getComputedStyle(n).fill,box:bounds(n),
					bar:bounds(n.parentElement.querySelector('rect')),token:bounds(n.parentElement.querySelector('.component-decoding-token'))})),
				outsideFill:svg&&getComputedStyle(svg.querySelector('.component-decoding-token')).fill,
				axisBoxes:axis&&[...axis.querySelectorAll('text')].map(bounds),
				tickBoxes:axis&&[...axis.querySelectorAll('.component-decoding-tick')].map(bounds),
				ticks:axis&&[...axis.querySelectorAll('.component-decoding-tick')].map(n=>Number(n.textContent)),
				grid:svg&&[...svg.querySelectorAll('.component-decoding-gridline')].map(n=>Number(n.getAttribute('x1'))),
				tickX:axis&&[...axis.querySelectorAll('.component-decoding-tick')].map(n=>Number(n.getAttribute('x'))),
				left:box(document.querySelector('.path-component-section')),right:box(document.querySelector('#component-decoding')),
				heights:[document.querySelector('#selected-path-diagram').clientHeight,content.clientHeight],
				overflow:content.scrollWidth>content.clientWidth,unsafe:content.querySelectorAll('script,img').length,
				pressed:[...document.querySelectorAll('.path-component[aria-pressed="true"]')].map(n=>n.dataset.kind),
				name:content.querySelector('.component-decoding-name')?.textContent };
		})()`);
		const assertHistogram = (d, context, count = 9) => {
			assert.equal(d.values.length, count, context);
			assert.equal(d.ticks[0], 0);
			assert.ok(d.ticks.at(-1) >= d.probabilities[0], "axis never truncates the largest probability");
			assert.deepEqual(d.grid, d.tickX, "vertical gridlines align with probability ticks");
			assert.equal(d.overflow, false, context);
			for (const [x, y, w, h] of d.barBoxes) {
				assert.ok(x>=0 && x+w<=d.svgWidth+0.5, `token/value clipping: ${context} (${x}, ${w}, ${d.svgWidth})`);
				assert.ok(y>=0 && h<=20 && y+h<=d.svg[3]+0.5, `row text fits SVG height: ${context}`);
			}
			for (const label of d.labels) {
				assert.equal(label.fill, label.inside ? "rgb(255, 255, 255)" : d.outsideFill, `light inside / dark outside: ${context}`);
				const [x, y, w, h] = label.box, [bx, by, bw, bh] = label.bar;
				if (label.inside) assert.ok(x>=bx && x+w<=bx+bw && y>=by && y+h<=by+bh, `inside value fits its bar: ${context}`);
				else assert.ok(x>=bx+bw+2, `short-bar value stays outside the bar: ${context}`);
				assert.ok(x>label.token[0]+label.token[2], `value never overlaps its token label: ${context}`);
			}
			for (const [x, y, w, h] of d.axisBoxes) {
				assert.ok(x>=0 && x+w<=d.svgWidth+0.5 && y>=0 && y+h<=44, `axis clipping: ${context}`);
			}
			for (let i=1;i<d.tickBoxes.length;i++) assert.ok(d.tickBoxes[i-1][0]+d.tickBoxes[i-1][2]<d.tickBoxes[i][0], `tick overlap: ${context}`);
			if (d.probabilities[0] === 0) assert.ok(d.bars.every((w)=>w===0), "zero probabilities produce zero-length bars");
			else for (let i=1;i<d.bars.length;i++) assert.ok(Math.abs(d.bars[i]/d.bars[0]-d.probabilities[i]/d.probabilities[0])<1e-9, "bars share the unmodified probability scale");
			assert.ok(d.axis[1]>=d.content[1] && d.axis[1]+d.axis[3]<=d.content[1]+d.content[3]+0.5, `axis stays within panel: ${context}`);
			assert.equal(d.svg[3], count*18+4, `plot height follows token count: ${context}`);
			assert.ok(d.plot[3]<=d.svg[3]+0.5, `plot never stretches beyond its rows: ${context}`);
			assert.ok(Math.abs(d.axis[1]-d.plot[1]-d.plot[3])<0.5, `axis directly below plot viewport: ${context}`);
			if (d.plot[3]>=d.svg[3]) assert.ok(Math.abs(d.axis[1]-d.lastBar[1]-d.lastBar[3]-4)<0.5, `small consistent margin after last bar: ${context}`);
			assert.ok(Math.abs(d.note[1]-d.axis[1]-d.axis[3]-5)<0.5, `note directly below histogram: ${context}`);
			assert.ok(d.note[1]+d.note[3]<=d.content[1]+d.content[3]+0.5, `methodological note stays visible: ${context}`);
			assert.equal(d.noteOverflow, false, `methodological note is not clipped: ${context}`);
		};
		assert.match((await decoding()).text, /Select a component/);
		// Real pointer activation on the summary label, then keyboard activation.
		const componentPoint = await evaluate(`(() => {
			const n=document.querySelector('.path-component[data-kind="attn"]');
			n.scrollIntoView({block:'nearest'});const r=n.querySelector('.path-component-name').getBoundingClientRect();
			return [r.x+r.width/2,r.y+r.height/2];
		})()`);
		await clickAt(...componentPoint);
		let decoded = await decoding();
		assert.equal(decoded.name, "A5H9");
		assert.deepEqual(decoded.values, ["0.2500", "0.1250", "0.0313", "0.0200", "0.0100", "0.0050", "0.0020", "0.0010", "1.000e-10"]);
		assertHistogram(decoded, "default panel");
		assert.ok(decoded.labels.some(label=>label.inside) && decoded.labels.some(label=>!label.inside), "default panel covers both value placements");
		assert.equal(await evaluate(`document.querySelector('.component-decoding-disclaimer').textContent`),
			"Exploratory component projection, not the full model's next-token probabilities or the selected path's causal contribution.");
		assert.doesNotMatch(decoded.text, /Zero-based positions|⇥ tab|↵ newline/);
		assert.match(decoded.text, /Top 9 tokens · full-vocabulary softmax/);
		assert.deepEqual(await evaluate(`['.component-decoding-token','.component-decoding-probability','.component-decoding-tick']
			.map(s=>getComputedStyle(document.querySelector(s)).fontSize)`), ["11px", "11px", "11px"]);
		const expectedExplanation = [
			"The chart shows the top tokens obtained by projecting the selected component's output into the vocabulary space using a logit lens.",
			"For attention heads, the full output at the query position is decoded, rather than the isolated message of the selected path.",
			"Token positions are zero-based and include BOS. Token labels refer to the first clean prompt: ␠ indicates a space, ⇥ a tab, and ↵ a newline.",
		];
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info-popover strong').textContent`), "About Component Decoding");
		assert.deepEqual(await evaluate(`[...document.querySelectorAll('#component-decoding-info-popover p')].map(n=>n.textContent)`), expectedExplanation);
		await evaluate(`document.querySelector('#component-decoding-info').focus({preventScroll:true})`);
		await pressKey("Enter", 13, "\r");
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info-popover').hidden`), false);
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info').getAttribute('aria-expanded')`), "true");
		assert.deepEqual((await decoding()).axis, decoded.axis, "information popover must not resize the chart");
		await pressKey("Escape", 27);
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info-popover').hidden`), true);
		assert.equal(await evaluate(`document.activeElement.id`), "component-decoding-info");
		await pressKey(" ", 32, " ");
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info-popover').hidden`), false);
		await pressKey("Tab", 9);
		assert.equal(await evaluate(`document.querySelector('#component-decoding-info-popover').hidden`), true);
		await evaluate(`document.querySelector('.component-decoding-plot').scrollTop=10000`);
		const scrolled = await decoding();
		assert.deepEqual(scrolled.axis, decoded.axis, "scrolling tokens never hides or moves the axis");
		assert.ok(Math.abs(scrolled.axis[1]-scrolled.lastBar[1]-scrolled.lastBar[3]-4)<1, "last bar sits directly above the axis after scrolling to the bottom");
		await evaluate(`document.querySelector('.component-decoding-plot').scrollTop=0;
			document.querySelector('#path-resize-handle').dispatchEvent(new KeyboardEvent('keydown',{key:'End',bubbles:true}))`);
		await delay(80);
		const wideHistogram = await decoding();
		assertHistogram(wideHistogram, "resized without changing component");
		assert.ok(wideHistogram.svgWidth>decoded.svgWidth, "ResizeObserver redraws the shared axis and bars");
		await evaluate(`document.querySelector('#path-resize-handle').dispatchEvent(new MouseEvent('dblclick'))`);
		await delay(80);
		const originalTokens = await evaluate(`state.result.graph.nodes.find(n=>n.id==='a').component_logit_lens.top_tokens`);
		await viewport(1440, 1200);
		for (const count of [2, 7, 9]) {
			await evaluate(`state.result.graph.nodes.find(n=>n.id==='a').component_logit_lens.top_tokens=${JSON.stringify(originalTokens)}.slice(0,${count});updateComponentDecoding()`);
			const natural = await decoding();
			assertHistogram(natural, `${count} rows with ample height`, count);
			assert.equal(natural.plot[3], natural.svg[3], "short and full readouts use their natural height when space permits");
		}
		await viewport(1440);
		for (const maximum of [0, 1e-10, 0.00011, 0.189, 0.5, 1]) {
			await evaluate(`(() => {
				const lens=state.result.graph.nodes.find(n=>n.id==='a').component_logit_lens;
				const factors=${JSON.stringify(maximum === 1 ? [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
					: maximum === 0.5 ? [1, 0.98, 0.01, 0, 0, 0, 0, 0, 0, 0] : [1, 0.7, 0.55, 0.3, 0.1, 0.05, 0.01, 0.001, 1e-10, 0])};
				lens.top_tokens=lens.top_tokens.map((item,i)=>({...item,probability:${maximum}*factors[i]}));
				updateComponentDecoding();
			})()`);
			const distribution = await decoding();
			assertHistogram(distribution, `maximum ${maximum}`);
			if (maximum === 0.189) assert.equal(distribution.ticks.at(-1), 0.2);
		}
		await evaluate(`state.result.graph.nodes.find(n=>n.id==='a').component_logit_lens.top_tokens=${JSON.stringify(originalTokens)};updateComponentDecoding()`);
		assert.deepEqual(decoded.pressed, ["attn"]);
		assert.match(decoded.text, /Query\/output position: 10 \("␠to"\)/);
		assert.equal(decoded.overflow, false);
		assert.equal(decoded.unsafe, 0);
		await evaluate(`document.querySelector('.path-component[data-kind="mlp"]').focus()`);
		await pressKey("Enter", 13, "\r");
		assert.equal((await decoding()).name, "MLP6");
		assert.equal(await evaluate(`document.activeElement.dataset.kind`), "mlp");
		await evaluate(`document.querySelector('.path-component[data-kind="attn"]').focus()`);
		await pressKey(" ", 32, " ");
		assert.equal((await decoding()).name, "A5H9");
		for (const key of ["view", "selection", "geometry", "lanes"]) assert.equal((await measure())[key], initial[key], key);
		const menuHidden = () => evaluate(`document.querySelector('#path-sort-popover').hidden`);
		// Keyboard focus moves without sorting until Enter/Space activates a row.
		await evaluate(`document.querySelector('#path-sort-toggle').focus()`);
		await pressKey("Enter", 13, "\r");
		assert.equal(await menuHidden(), false);
		assert.equal(await evaluate(`document.activeElement.value`), "discovery");
		for (const [key, virtual, expected] of [["End", 35, "absolute"], ["Home", 36, "discovery"],
			["ArrowUp", 38, "absolute"], ["ArrowDown", 40, "discovery"], ["ArrowDown", 40, "signed"]]) {
			await pressKey(key, virtual);
			assert.equal(await evaluate(`document.activeElement.value`), expected);
			assert.equal((await navigator()).activeOrder, "discovery", "focus navigation must not sort");
		}
		assert.equal(await evaluate(`getComputedStyle(document.activeElement).outlineStyle`), "solid", "keyboard focus is visible");
		await pressKey("Enter", 13, "\r");
		assert.equal(await menuHidden(), true);
		assert.equal(await evaluate(`document.activeElement.id`), "path-sort-toggle");
		nav = await navigator();
		assert.equal(nav.activeOrder, "signed");
		assert.equal(nav.cards[0].id, "path-8");
		assert.equal(nav.subtitle, "Path 1");
		assert.equal(await evaluate(`document.querySelector('#path-sort-current').textContent`), "Signed");
		await pressKey(" ", 32, " ");
		assert.equal(await menuHidden(), false);
		assert.equal(await evaluate(`document.activeElement.value`), "signed");
		await pressKey(" ", 32, " ");
		assert.equal(await menuHidden(), true, "selecting the current option also closes the menu");
		await pressKey("Enter", 13, "\r");
		await pressKey("Escape", 27);
		assert.equal(await menuHidden(), true);
		assert.equal(await evaluate(`document.activeElement.id`), "path-sort-toggle");
		for (const key of ["view", "selection", "geometry", "lanes"]) assert.equal((await measure())[key], initial[key], key);
		// Every part of the trigger is clickable, including icon, text and arrow.
		for (const selector of ['#path-sort-toggle > span:first-child', '#path-sort-current', '#path-sort-toggle > span:last-child']) {
			const point = await evaluate(`(() => {const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()`);
			await clickAt(...point);
			assert.equal(await menuHidden(), false);
			await clickAt(...point);
			assert.equal(await menuHidden(), true);
		}
		await evaluate(`document.querySelector('#path-sort-toggle').click()`);
		const options = await evaluate(`[...document.querySelectorAll('#path-sort button')].map(n=>({role:n.getAttribute('role'),
			label:n.querySelector('strong').textContent,description:n.querySelector('small').textContent}))`);
		assert.deepEqual(options.map(n=>n.role), Array(3).fill("menuitemradio"));
		assert.deepEqual(options.map(n=>[n.label,n.description]), [
			["Discovery order", "Original discovery order."], ["Signed contribution", "Highest signed path contribution first."],
			["Absolute contribution", "Largest absolute path contribution first."],
		]);
		await pressKey("Escape", 27);
		for (const value of ["absolute", "signed", "discovery"]) {
			await evaluate(`document.querySelector('#path-sort-toggle').click()`);
			const point = await evaluate(`(() => {const r=document.querySelector('#path-sort button[value="${value}"]').getBoundingClientRect();return [r.right-4,r.y+r.height/2];})()`);
			await clickAt(...point); // Right-side row padding, outside the indicator or label.
			assert.equal(await menuHidden(), true);
			assert.equal((await navigator()).activeOrder, value);
			assert.deepEqual(await evaluate(`[...document.querySelectorAll('#path-sort button')].map(n=>getComputedStyle(n.querySelector('.path-sort-check')).visibility)`),
				["discovery", "signed", "absolute"].map(n=>n===value?"visible":"hidden"));
			for (const key of ["view", "selection", "geometry", "lanes"]) assert.equal((await measure())[key], initial[key], key);
		}
		assert.equal((await navigator()).activeOrder, "discovery");
		await evaluate(`document.querySelector('#path-sort-toggle').click()`);
		await pressKey("End", 35);
		await pressKey("Tab", 9);
		assert.equal(await menuHidden(), true, "Tab can leave the menu");
		await evaluate(`document.querySelector('#path-sort-toggle').click()`);
		const outside = await evaluate(`(() => {const r=document.querySelector('#path-count').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()`);
		await clickAt(...outside);
		assert.equal(await menuHidden(), true, "outside clicks dismiss the menu");
		await evaluate(`document.querySelector('.path-select[data-path-id="path-3"]').parentElement.querySelector('summary').click()`);
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-3"]').parentElement.querySelector('details').open`), true);
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-3"]').parentElement.querySelectorAll('details li').length`), 1);
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-3"] .path-card-summary').children.length`), 4);
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-3"]').parentElement.querySelector('summary').textContent`), "Inspect remaining differences (1 more)");
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-6"]').parentElement.querySelector('details').hidden`), true, "no expander for fully visible differences");
		await evaluate(`document.querySelector('#path-info').click()`);
		await evaluate(`document.querySelector('#path-info').click();onClickNode({id:'b'});render()`);
		const shortList = await measure();
		assert.deepEqual(shortList.actions, initial.actions, "changing component names must not move actions");
		assert.deepEqual(shortList.sort, initial.sort, "excluded-branch text must not change the count row height");
		assert.equal(shortList.cardWidth, initial.cardWidth, "appearing/disappearing list scrollbars must not change card width");
		await evaluate(`onClickNode({id:'a'});render()`);
		await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: initial.panel[0], y: 240 });
		await send("Input.dispatchMouseEvent", { type: "mousePressed", x: initial.panel[0], y: 240, button: "left", buttons: 1, clickCount: 1 });
		assert.equal(await evaluate(`document.querySelector('#path-resize-handle').hasPointerCapture(1)`), true);
		await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: initial.panel[0] - 100, y: 240, button: "left", buttons: 1 });
		await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: initial.panel[0] - 100, y: 240, button: "left", clickCount: 1 });
		const resized = await measure();
		assert.equal(resized.panel[2], 580);
		decoded = await decoding();
		assert.ok(decoded.right[0] > decoded.left[0] + decoded.left[2], "wide Inspector places decoding to the right");
		assert.equal(decoded.right[1], decoded.left[1]);
		assert.equal(resized.graph[2], initial.graph[2] - 100);
		for (const key of ["view", "selection", "geometry", "lanes"]) assert.equal(resized[key], initial[key], key);
		await evaluate(`document.querySelector('#path-resize-handle').dispatchEvent(new MouseEvent('dblclick'));`);
		assert.equal((await measure()).panel[2], 480, "reset must use the CSS default");
		const diagram = () => evaluate(`(() => {
			const container=document.querySelector('#selected-path-diagram'), svg=container.querySelector('svg');
			const groups=[...svg.querySelectorAll('.path-component')];
			const attention=groups.find(n=>n.dataset.kind==='attn');
			const visible=n=>[...n.querySelectorAll('text')].map(n=>[...n.children].map(n=>n.textContent).join(' ')).join(' ');
			const bounds=n=>{const b=n.getBBox();return [b.x,b.y,b.width,b.height];};
			return { attention:visible(attention), tokenLabels:[...attention.querySelectorAll('.path-component-token')].map(n=>n.textContent).join(''),
				context:container.querySelector('.path-token-context')?.textContent,
				height:svg.viewBox.baseVal.height, boxes:[...svg.querySelectorAll('text')].map(bounds),
				groups:groups.map(bounds), kinds:groups.map(n=>n.dataset.kind),
				links:svg.querySelectorAll('.path-component-link').length,
				dotted:svg.querySelectorAll('.path-component-link[stroke-dasharray]').length,
				tokenFill:getComputedStyle(attention.querySelector('.path-component-token') || attention).fill,
				metaFill:getComputedStyle(attention.querySelector('.path-component-meta')).fill,
				clientWidth:container.clientWidth, scrollWidth:container.scrollWidth,
				unsafe:container.querySelectorAll('script,img').length, badToken:!!window.badToken,
				aria:svg.getAttribute('aria-label') };
		})()`);
		const assertDiagramBounds = (d, context) => {
			assert.ok(d.scrollWidth <= d.clientWidth, `diagram horizontal overflow: ${context}`);
			for (const [x, y, w, h] of d.boxes) {
				// Glyph ink can extend slightly beyond its measured text advance.
				assert.ok(x >= 34 && x + w <= 262, `text exceeds diagram width: ${context} (${x}, ${w})`);
				assert.ok(y >= 0 && y + h <= d.height, `text exceeds diagram height: ${context}`);
			}
			for (let i = 1; i < d.groups.length; i++) {
				assert.ok(d.groups[i - 1][1] + d.groups[i - 1][3] < d.groups[i][1], `overlapping components: ${context}`);
			}
			assert.equal(d.links, d.kinds.length - 1);
			assert.equal(d.unsafe, 0);
			assert.equal(d.badToken, false);
			assert.doesNotMatch(d.aria, /WRONG_COUNTERFACTUAL_TOKEN|unspecified/);
		};
		const expected = [
			/Patch Q$/,
			/Patch KV · Source position: 0/,
			/Patch QK · Source position: 3/,
			/Patch V · Source position: 7/,
			/Patch QKV · Source positions: all eligible/,
			/Patch K · Source position: Unavailable/,
			/Patch: Unavailable · Source position: Unavailable/,
			/Query\/output position: Unavailable.*Patch: Unavailable · Source position: Unavailable/,
		];
		let shortHeight;
		for (let i = 0; i < expected.length; i++) {
			// Activate real sidebar buttons to verify the original occurrences reach the renderer.
			await evaluate(`document.querySelector('.path-select[data-path-id="path-${i + 1}"]').click();render()`);
			const d = await diagram();
			assert.match(d.attention, expected[i]);
			assertDiagramBounds(d, `path-${i + 1}`);
			assert.deepEqual(d.kinds, ["final", "mlp", "attn", "embed"]);
			assert.equal(d.dotted, [0, 2, 4].includes(i) ? 1 : 0);
			if (i < 7) {
				assert.match(d.attention, /Query\/output position: 10/);
				assert.ok(d.tokenLabels.includes('"␠to"'));
				assert.notEqual(d.tokenFill, d.metaFill, "tokens must be visually secondary");
			} else assert.equal(d.tokenLabels, "", "merged metadata must not supply token coordinates");
			assert.match(d.context, /first clean prompt/);
			if (i === 0) {
				shortHeight = d.height;
				assert.doesNotMatch(d.attention, /Source position/);
			}
			if (i === 1) assert.ok(d.tokenLabels.includes('"<BOS>"'), "source position 0 is the clean BOS token");
			if (i === 2) {
				assert.ok(d.height > shortHeight, "long tokens expand vertical spacing instead of overflowing");
				assert.ok(d.tokenLabels.includes(tokens[3].replaceAll(" ", "␠")), "wrapping preserves the complete token");
			}
			if (i === 3) assert.ok(d.tokenLabels.includes("<script>"), "token markup is visible as text, not HTML");
		}
		for (const key of ["view", "geometry", "lanes"]) assert.equal((await measure())[key], initial[key], key);
		for (const width of [1440, 1280, 1101, 1100, 900, 681, 680, 390, 320]) {
			await viewport(width);
			await evaluate(`onClickNode({id:'a'});selectPath('path-3');render()`);
			const open = await measure();
			assert.ok(open.scrollWidth <= width, `horizontal overflow at ${width}px`);
			nav = await navigator();
			assert.ok(nav.cards.every((c)=>!c.overflow), `card overflow at ${width}px`);
			await evaluate(`document.querySelector('#path-sort-toggle').click()`);
			const popup = await evaluate(`(() => {const r=document.querySelector('#path-sort-popover').getBoundingClientRect();return [r.x,r.right];})()`);
			assert.ok(popup[0]>=0 && popup[1]<=width, `Sort popover outside viewport at ${width}px`);
			assert.deepEqual((await measure()).sort, open.sort, "Sort popup must not reflow the legend");
			await evaluate(`document.querySelector('#path-sort-toggle').click()`);
			assertDiagramBounds(await diagram(), `${width}px`);
			await evaluate(`document.querySelector('.path-component[data-kind="attn"]').dispatchEvent(new MouseEvent('click',{bubbles:true}))`);
			decoded = await decoding();
			assertHistogram(decoded, `${width}px`);
			await evaluate(`document.querySelector('#component-decoding-info').scrollIntoView({block:'center'});
				document.querySelector('#component-decoding-info').click()`);
			const infoBounds = await evaluate(`(() => {
				const n=document.querySelector('#component-decoding-info-popover'),r=n.getBoundingClientRect();
				return {hidden:n.hidden,left:r.left,right:r.right,top:r.top,bottom:r.bottom,
					overflow:n.scrollHeight>n.clientHeight,
					visible:n.contains(document.elementFromPoint(r.x+r.width/2,r.y+2)) && n.contains(document.elementFromPoint(r.x+r.width/2,r.bottom-2))};
			})()`);
			assert.equal(infoBounds.hidden, false);
			assert.ok(infoBounds.left>=0 && infoBounds.right<=width && infoBounds.top>=0 && infoBounds.bottom<=800,
				`decoding information outside viewport at ${width}px: ${JSON.stringify(infoBounds)}`);
			assert.equal(infoBounds.visible, true, `decoding information clipped at ${width}px`);
			if (infoBounds.overflow) {
				await evaluate(`document.querySelector('#component-decoding-info-popover').scrollTop=0`);
				await send("Input.dispatchMouseEvent", { type: "mouseWheel", x: (infoBounds.left+infoBounds.right)/2,
					y: (infoBounds.top+infoBounds.bottom)/2, deltaX: 0, deltaY: 500 });
				await delay(80);
				const scrollState = await evaluate(`(() => {const n=document.querySelector('#component-decoding-info-popover');return {hidden:n.hidden,scrollTop:n.scrollTop,height:n.clientHeight,scrollHeight:n.scrollHeight};})()`);
				assert.ok(!scrollState.hidden && scrollState.scrollTop>0,
					`long explanation remains open and scrollable at ${width}px: ${JSON.stringify({infoBounds,scrollState})}`);
			}
			await pressKey("Escape", 27);
			assert.equal(decoded.overflow, false, `decoding horizontal overflow at ${width}px`);
			assert.ok(decoded.heights.every((h)=>h>0), `both plots have usable scroll areas at ${width}px`);
			if (decoded.right[0] === decoded.left[0]) assert.ok(decoded.right[1] >= decoded.left[1] + decoded.left[3], "narrow sections stack");
			assert.equal(open.view, initial.view, "opening the Inspector must not reset manual pan/zoom");
			await evaluate(`document.querySelector('#path-clear').click();render()`);
			const closed = await measure();
			assert.equal(await evaluate(`document.querySelector('#path-explorer').hidden`), true);
			assert.equal(closed.sidebar[2], open.sidebar[2], `Clear changed parameter column width at ${width}px`);
			assert.equal(closed.graph[0], open.graph[0], `Clear moved the graph origin at ${width}px`);
			assert.equal(closed.bodyWidth, open.bodyWidth, `Clear toggled page scrollbar space at ${width}px`);
			assert.equal(closed.view, initial.view);
			assert.equal(closed.geometry, open.geometry);
		}
		const nonpositional = graphFixture();
		for (const n of [...nonpositional.nodes, ...nonpositional.path_provenance.occurrences]) {
			n.position = null;
			if (n.kind === "attn") n.kv_position = null;
		}
		await evaluate(`state.savedResults.path={graph:${JSON.stringify(nonpositional)},
			meta:{...state.savedResults.path.meta,positional:false},params:null};restoreSavedResult('path');onClickNode({id:'a'});render()`);
		let d = await diagram();
		assert.match(d.attention, /Query\/output positions: all eligible.*Patch Q/);
		assert.equal(d.context, undefined);
		assert.equal(d.tokenLabels, "");
		assertDiagramBounds(d, "non-positional Q");
		await evaluate(`selectPath('path-2');render()`);
		d = await diagram();
		assert.match(d.attention, /Query\/output positions: all eligible.*Patch KV · Source positions: all eligible/);
		assert.equal(d.context, undefined);
		assert.equal(d.tokenLabels, "");
		assertDiagramBounds(d, "non-positional KV");
		assert.deepEqual(errors, []);
	} finally {
		await closeBrowser?.().catch(() => {});
		socket?.close();
		browser.kill();
		server.close();
		await delay(500);
		fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
	}
});

});
