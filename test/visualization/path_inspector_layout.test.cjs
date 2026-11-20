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
		const assertInfoAppearance = async () => {
			await evaluate(`document.querySelector('#path-sort-toggle').focus({preventScroll:true})`);
			await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: 0, y: 0 });
			await send("DOM.enable"); await send("CSS.enable");
			const { root } = await send("DOM.getDocument");
			const nodes = await Promise.all(["#path-info", "#component-decoding-info"].map(async (selector) =>
				(await send("DOM.querySelector", { nodeId: root.nodeId, selector })).nodeId));
			try {
				for (const state of [[], ["hover"], ["focus-visible"]]) {
					await Promise.all(nodes.map((nodeId) => send("CSS.forcePseudoState", { nodeId, forcedPseudoClasses: state })));
					const appearance = await evaluate(`(() => {
						const style=(n,keys)=>Object.fromEntries(keys.map(key=>[key,getComputedStyle(n)[key]]));
						return [...document.querySelectorAll('.path-info-button')].map(n=>({text:n.textContent,
							button:style(n,['width','height','display','placeItems','padding','border','borderRadius','backgroundColor','color',
								'fontFamily','fontSize','fontStyle','fontWeight','lineHeight','outlineStyle','outlineWidth','outlineColor','outlineOffset']),
							gap:getComputedStyle(n.parentElement.parentElement).gap,
							popover:style(document.getElementById(n.getAttribute('aria-controls')),
								['width','maxWidth','padding','border','borderRadius','backgroundColor','color','fontSize','lineHeight','boxShadow'])}));
					})()`);
					assert.equal(appearance.length, 2);
					assert.deepEqual(appearance[0], appearance[1], `information controls share all visual styling: ${state}`);
					assert.equal(appearance[0].text, 'i');
					assert.equal(appearance[0].button.width,'24px'); assert.equal(appearance[0].button.height,'24px');
					assert.equal(appearance[0].button.borderRadius,'50%'); assert.equal(appearance[0].button.padding,'0px');
					assert.match(appearance[0].button.border,/^1px solid /);
					assert.equal(appearance[0].button.fontStyle,'italic'); assert.equal(appearance[0].button.fontSize,'13px');
					assert.equal(appearance[0].gap,'8px');
					if(state.includes('focus-visible')) {
						assert.equal(appearance[0].button.outlineStyle,'solid'); assert.equal(appearance[0].button.outlineWidth,'2px');
					}
				}
			} finally {
				await Promise.all(nodes.map((nodeId) => send("CSS.forcePseudoState", { nodeId, forcedPseudoClasses: [] })));
			}
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
		const [infoX,infoY,infoW,infoH]=initial.actions[1];
		await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: infoX+infoW/2, y: infoY+infoH/2 });
		await delay(60);
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true, "real pointer hover does not open information");
		assert.equal(await evaluate(`document.querySelector('#path-info').getAttribute('aria-expanded')`), "false");
		for(const type of ['mousePressed','mouseReleased']) await send("Input.dispatchMouseEvent",
			{type,x:infoX+infoW/2,y:infoY+infoH/2,button:'left',buttons:type==='mousePressed'?1:0,clickCount:1});
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), false, "real pointer click opens information");
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
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true, "keyboard focus alone does not expose information");
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r" });
		await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), false, "Enter produces the same click activation");
		assert.equal(await evaluate(`document.querySelector('#path-info').getAttribute('aria-expanded')`), "true");
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape" });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true);
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: " ", code: "Space", windowsVirtualKeyCode: 32, text: " " });
		await send("Input.dispatchKeyEvent", { type: "keyUp", key: " ", code: "Space", windowsVirtualKeyCode: 32 });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), false, "Space also activates the native button");
		await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
		await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
		assert.equal(await evaluate(`document.querySelector('#path-info-popover').hidden`), true, "Tab dismisses information when focus leaves its wrapper");
		const navigator = () => evaluate(`(() => {
			const box=n=>{const r=n.getBoundingClientRect();return [r.x,r.y,r.width,r.height];};
			const cards=[...document.querySelectorAll('.path-select')];
			return { context:document.querySelector('#path-score-context').textContent,
				count:box(document.querySelector('#path-count')),legend:box(document.querySelector('#path-key')),sort:box(document.querySelector('#path-sort-toggle')),
				cards:cards.map(n=>({ id:n.dataset.pathId,text:n.textContent,box:box(n),number:box(n.querySelector('.path-card-heading strong')),
					score:box(n.querySelector('.path-score')),scoreText:n.querySelector('.path-score').textContent,
					scoreWeight:getComputedStyle(n.querySelector('.path-score')).fontWeight, title:n.getAttribute('title'),
					summaryTitles:[...n.querySelectorAll('.path-card-summary [title]')].length,
					summaryLines:n.querySelectorAll('.path-card-summary > span').length,
					diffCount:PathExploration.comparePaths(state.pathExplorer.index.paths.get(state.pathExplorer.selectedId),state.pathExplorer.index.paths.get(n.dataset.pathId)).differences.length,
					expanders:n.parentElement.querySelectorAll('details,summary').length,
					tooltips:[...n.querySelectorAll('[title]')].every(n=>n.classList.contains('path-score')),
					overflow:n.scrollWidth>n.clientWidth })),
				subtitle:document.querySelector('#selected-path-meta').textContent,
				activeOrder:document.querySelector('#path-sort [aria-checked="true"]').value };
		})()`);
		let nav = await navigator();
		assert.equal(nav.context, "Effect on logit difference · Denoising · logit units");
		assert.equal(nav.subtitle, "Path 1");
		assert.equal(nav.activeOrder, "discovery");
		assert.ok(Math.abs(nav.count[1]+nav.count[3]/2-nav.sort[1]-nav.sort[3]/2)<1, "Sort aligns with the path count");
		assert.ok(nav.sort[0]>nav.count[0]+nav.count[2], "Sort belongs to the right of the count");
		assert.equal(nav.cards[8].scoreText, "+1.43 × 10⁻⁶");
		for (const card of nav.cards) {
			assert.equal(card.title, null);
			assert.equal(card.summaryTitles, 0);
			assert.equal(card.tooltips, true, "only the score retains a tooltip, not the summary or its ancestors");
			assert.equal(card.overflow, false);
			if (card.diffCount) assert.equal(card.summaryLines, card.diffCount);
			assert.equal(card.expanders, 0);
			assert.doesNotMatch(card.text, /Inspect remaining|\(\d+ more\)/);
			assert.doesNotMatch(card.text, /path-\d|Path effect|Logit Difference|logit units|token position/);
			assert.equal(Number(card.scoreWeight), card.id === "path-1" ? 650 : 500);
			if (Math.abs(card.score[1] - card.number[1]) < 3) {
				assert.ok(card.score[0] - card.number[0] - card.number[2] <= 9, "score sits immediately next to the identifier");
			} else {
				assert.ok(card.score[1] >= card.number[1] + card.number[3], "long scores wrap below the identifier rather than clipping");
				assert.ok(Math.abs(card.score[0] - card.number[0]) < 1);
			}
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
		assert.equal(await evaluate(`document.querySelector('.path-select[data-path-id="path-3"] .path-card-summary').children.length`), 5);
		assert.equal(await evaluate(`document.querySelectorAll('#path-list details,#path-list summary').length`), 0, "all differences render without expansion controls");
		assert.doesNotMatch(await evaluate(`document.querySelector('#path-list').textContent`), /Inspect remaining|\(\d+ more\)/);
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
		// Real SVG/CSS checks for always-visible path probabilities, including
		// many cards, tiny/zero bars, long token labels, resizing and both themes.
		await evaluate(`(() => {
			const context={available:true,id:'probability-layout',method:'path',mode:'noising',top_k:5,
				prompt_index:0,position:10,reference_prompt:'test prompt',reference_label:'Clean',
				target_token_id:5335,counterfactual_token_id:1757};
			Object.assign(state.savedResults.path.meta,{mode:'noising',probability_context:context});
			state.savedResults.path.params={model:'gpt2-small',method:'path',mode:'noising',prompts:['test prompt'],targets:['Mary'],cf_targets:['John']};
			for(const p of state.savedResults.path.graph.path_provenance.paths){
				const tokens=[.3,.29,.1,1e-10,0].map((reference,i)=>({token_id:[5335,1757,50256,197,1][i],
					token:[' Mary',' John','<|endoftext|>','\\t\\n',' <script>window.badToken=true</script>'+ '🧠Long'.repeat(20)][i],
					reference,patched:[.29,.3,.09,1e-9,0][i]}));
				tokens.forEach(t=>t.delta=t.patched-t.reference);
				p.probability_comparison={available:true,context_id:context.id,path_id:p.id,position:10,
					convention:'isolated_path_final_output_softmax',vocab_size:50257,
					node_ids:[...p.node_ids],occurrence_ids:[...p.occurrence_ids],tokens};
			}
			restoreSavedResult('path');onClickNode({id:'a'});
		})()`);
		for (const [roleMatch, theme] of [["exact", "light"], ["exact", "dark"], ["leading-space", "light"], ["leading-space", "dark"]]) {
			await evaluate(`Object.assign(state.savedResults.path.meta.probability_context,
				{target_token_id:${roleMatch === "exact" ? 5335 : 24119},counterfactual_token_id:${roleMatch === "exact" ? 1757 : 7554}});
				restoreSavedResult('path');onClickNode({id:'a'});`);
			await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: theme }] });
			if(roleMatch==='exact') await assertInfoAppearance();
			for (const width of [1440, 1280, 1101, 1100, 900, 681, 680, 390, 320]) {
				await viewport(width);
				await delay(80);
				const plots = await evaluate(`(() => {
					const boxes=n=>{const b=n.getBBox();return [b.x,b.y,b.width,b.height]};
					const rect=n=>{const b=n.getBoundingClientRect();return [b.x,b.y,b.width,b.height]};
					return [...document.querySelectorAll('.path-probabilities')].map(container=>{
						const svg=container.querySelector('svg'),texts=[...svg.querySelectorAll('.path-probability-value')];
						const card=container.closest('.path-entry'),info=card.querySelector('.path-card-info');
						const panel=container.closest('#path-explorer'),panelStyle=getComputedStyle(panel);
						return {hidden:container.hidden,width:svg.width.baseVal.value,height:svg.height.baseVal.value,
							heading:rect(container.querySelector('.path-probability-heading')),
							headingTitle:rect(container.querySelector('.path-probability-title')),
							headingOrder:rect(container.querySelector('.path-probability-context')),
							headingText:container.querySelector('.path-probability-heading').textContent,plot:rect(svg),
							wide:panel.clientWidth-parseFloat(panelStyle.paddingLeft)-parseFloat(panelStyle.paddingRight)>=430,
							card:rect(card),info:rect(info),chart:rect(container),selected:card.dataset.selected,
							scoreColor:getComputedStyle(card.querySelector('.path-score')).color,
							border:[getComputedStyle(card).borderTopColor,getComputedStyle(card).borderRightColor,getComputedStyle(card).borderBottomColor,getComputedStyle(card).borderLeftColor],
							chartInside:container.parentElement===card && card.contains(info),
							scrolling:[container,info,card].some(n=>['auto','scroll'].includes(getComputedStyle(n).overflowY)),
							availableWidth:container.querySelector('.path-probability-plot').clientWidth,
							values:texts.map(boxes),allText:[...svg.querySelectorAll('text')].map(boxes),
							valueCenters:texts.map(n=>+n.getAttribute('x')),
							valueStyles:texts.map(n=>({size:getComputedStyle(n).fontSize,weight:getComputedStyle(n).fontWeight,fill:getComputedStyle(n).fill})),
							barBounds:[...svg.querySelectorAll('.path-probability-bar')].map(boxes),
							deltas:container.querySelectorAll('.path-probability-delta').length,
							deltaStyles:[...container.querySelectorAll('.path-probability-delta')].map(n=>({text:n.textContent,color:getComputedStyle(n).color,size:parseFloat(getComputedStyle(n).fontSize),box:rect(n)})),
							tokens:container.querySelectorAll('.path-probability-token').length,
							labels:[...container.querySelectorAll('.path-probability-token-text')].map(n=>({text:n.textContent,title:n.parentElement.title,overflow:n.scrollWidth>n.clientWidth,
								color:getComputedStyle(n).color,size:parseFloat(getComputedStyle(n).fontSize),box:rect(n)})),
							grid:[...svg.querySelectorAll('.path-probability-gridline')].map(n=>({x1:+n.getAttribute('x1'),x2:+n.getAttribute('x2'),y:+n.getAttribute('y1'),
								stroke:getComputedStyle(n).stroke,width:getComputedStyle(n).strokeWidth,opacity:getComputedStyle(n).opacity})),
							ticks:[...svg.querySelectorAll('.path-probability-tick')].map(n=>({text:n.textContent,y:+n.getAttribute('y')})),
							bars:[...svg.querySelectorAll('.path-probability-bar')].map(n=>n.getAttribute('fill')),
							colors:[...svg.querySelectorAll('.path-probability-bar')].map(n=>{
								const fill=n.getAttribute('fill');return getComputedStyle(fill.startsWith('url(#')?svg.querySelector(fill.slice(4,-1)+' rect'):n).fill;
							}),
							patterns:[...svg.querySelectorAll('pattern')].map(n=>n.id),
							roles:[...svg.querySelectorAll('.path-probability-bar-group')].map(n=>n.dataset.role),
							matches:[...svg.querySelectorAll('.path-probability-bar-group')].map(n=>n.dataset.roleMatch),
							axisMin:svg.dataset.axisMin,axisMax:svg.dataset.axisMax,
							unsafe:container.querySelectorAll('script,img').length,badToken:!!window.badToken};
					});
				})()`);
				assert.equal(plots.length, 9);
				assert.equal(await evaluate(`document.querySelectorAll('#path-probability-guide .path-probability-legend').length`), 1);
				assert.equal(await evaluate(`document.querySelectorAll('#path-list .path-probability-legend').length`), 0);
				const guide = await evaluate(`(() => {
					const guide=document.querySelector('#path-probability-guide');
					const rect=n=>{const r=n.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom}};
					const header=document.querySelector('.path-inspector-header'),rows=[...header.children];
					const groups=[document.querySelector('#path-key'),...guide.querySelectorAll('.path-probability-legend-group')];
					const metric=document.querySelector('#path-score-context').getBoundingClientRect();
					return {text:guide.textContent,parent:guide.parentElement.id,header:rect(header),
						rowClasses:rows.map(n=>n.id||n.className),rows:rows.map(rect),
						title:rect(document.querySelector('#path-heading')),context:rect(document.querySelector('#path-node')),
						headingText:rect(document.querySelector('.path-heading-text')),
						headingAlign:getComputedStyle(document.querySelector('.path-heading-text')).alignItems,
						titleSize:getComputedStyle(document.querySelector('#path-heading')).fontSize,
						titleWeight:getComputedStyle(document.querySelector('#path-heading')).fontWeight,
						contextSize:getComputedStyle(document.querySelector('#path-node')).fontSize,
						contextColor:getComputedStyle(document.querySelector('#path-node')).color,
						contextAlign:getComputedStyle(document.querySelector('#path-node')).textAlign,
						actions:rect(document.querySelector('.path-header-actions')),
						count:rect(document.querySelector('#path-count')),sort:rect(document.querySelector('#path-sort-toggle')),
						groups:groups.map(n=>({text:n.textContent.replace(/\\s+/g,''),box:rect(n),label:n.getAttribute('aria-label'),
							divider:getComputedStyle(n).borderLeftWidth,dividerColor:getComputedStyle(n).borderLeftColor})),
						alternative:getComputedStyle(document.querySelector('.path-line-sample.alternative')).borderTopColor,
						other:getComputedStyle(guide.querySelector('.path-probability-legend-group:last-child .path-probability-legend-item:last-child i')).color,
						metricWidth:metric.width,metricHeight:metric.height};
				})()`);
				assert.equal(guide.text, 'CleanPatchedTargetCounterfactualOther');
				assert.equal(guide.parent, 'path-legend-row');
				assert.deepEqual(guide.rowClasses,['path-heading-row','path-count-row','path-legend-row']);
				assert.deepEqual(guide.groups.map(n=>n.text),['SelectedAlternatives','CleanPatched','TargetCounterfactualOther']);
				assert.deepEqual(guide.groups.map(n=>n.label),['Path highlighting','Probability conditions: fill styles','Token roles: bar colors']);
				assert.deepEqual(guide.groups.map(n=>n.divider),['0px','1px','1px']);
				assert.equal(guide.groups[1].dividerColor,guide.groups[2].dividerColor);
				assert.notEqual(guide.alternative,guide.other,'path alternatives and other-token bars keep their distinct grays');
				assert.equal(guide.contextAlign,'left'); assert.equal(guide.headingAlign,'baseline');
				assert.equal(guide.titleSize,'17px'); assert.equal(guide.titleWeight,'600'); assert.equal(guide.contextSize,'10px');
				assert.equal(guide.contextColor,theme==='light'?'rgb(94, 93, 89)':'rgb(194, 192, 182)');
				assert.equal(guide.headingText.x,guide.rows[0].x);
				assert.ok(guide.actions.x>=guide.headingText.right,'controls stay beyond the left-aligned title/context group');
				assert.ok(Math.abs(guide.actions.right-guide.rows[0].right)<.5,'Clear and info remain at the far right');
				if(guide.headingText.width>=guide.title.width+guide.context.width+10-.5) {
					assert.ok(Math.abs(guide.context.x-guide.title.right-10)<.5,'subtitle immediately follows the title with a 10px gap');
					assert.ok(guide.context.y>=guide.title.y && guide.context.bottom<=guide.title.bottom+2,'title and subtitle align on the same row');
				} else {
					assert.ok(guide.context.y>=guide.title.bottom,'subtitle wraps below the title when needed');
					assert.equal(guide.context.x,guide.headingText.x);
				}
				assert.ok(guide.count.right<=guide.sort.x && Math.abs(guide.sort.right-guide.rows[1].right)<.5);
				for(let i=1;i<3;i++) assert.ok(guide.rows[i].y>=guide.rows[i-1].bottom && guide.rows[i].y-guide.rows[i-1].bottom<=6,'three tightly spaced header rows');
				if(guide.rows[2].width>=440) {
					assert.equal(new Set(guide.groups.map(n=>n.box.y)).size,1,'combined legend fits on one row when width permits');
					assert.ok(guide.header.height<100,'compact header is shorter than the previous stacked title/context/legends');
				}
				for(const group of guide.groups) assert.ok(group.box.x>=guide.rows[2].x && group.box.right<=guide.rows[2].right+.5,'legend groups wrap without overflow');
				assert.equal(guide.metricWidth, 1); assert.equal(guide.metricHeight, 1);
				const patternIds = plots.flatMap((p) => p.patterns);
				assert.equal(new Set(patternIds).size, patternIds.length);
				for (const p of plots) {
					const context = `${theme} ${width}px`;
					assert.equal(p.hidden, false);
					assert.equal(p.tokens, 5);
					assert.equal(p.values.length, 10);
					assert.equal(p.bars.length, 10);
					assert.equal(p.axisMin, "0");
					assert.equal(p.axisMax, "0.4");
					assert.deepEqual(p.ticks.map(n=>n.text), ['0%','10%','20%','30%','40%']);
					assert.equal(p.grid.length, 5);
					for (const [i,line] of p.grid.entries()) {
						assert.equal(line.x1,p.grid[0].x1); assert.equal(line.x2,p.grid[0].x2);
						assert.equal(line.stroke,p.grid[0].stroke); assert.equal(line.width,'1px'); assert.equal(line.opacity,p.grid[0].opacity);
						assert.ok(Math.abs(p.ticks[i].y-line.y-3)<1e-6);
						if(i) assert.ok(Math.abs(p.grid[i-1].y-line.y-21)<1e-6, 'five evenly spaced gridlines over the unchanged 84px plot');
					}
					assert.equal(p.deltas, 5);
					assert.match(p.headingText, /Final prediction probabilitiesTop 5 by \|Δp\| · sorted by clean P/);
					assert.ok(p.heading[1]+p.heading[3] <= p.plot[1], "heading sits directly above its chart inside the card");
					if (p.heading[2] >= 350) assert.ok(Math.abs(p.headingTitle[1]-p.headingOrder[1])<3, "heading explanation shares the title row when it fits");
					for (const style of p.valueStyles) {
						assert.equal(style.size, '10px'); assert.equal(style.weight, '400');
						assert.equal(style.fill, theme === 'light' ? 'rgb(94, 93, 89)' : 'rgb(194, 192, 182)');
					}
					assert.equal(p.chartInside, true);
					assert.equal(p.scrolling, false, "no additional nested scrolling in cards");
					if (p.wide) {
						assert.ok(Math.abs(p.info[1] - p.chart[1]) < 1, `wide cards must use two columns: ${context}`);
						assert.ok(p.chart[0] >= p.info[0] + p.info[2], `side-by-side sections: ${context}`);
						const total=p.info[2]+p.chart[2];
						assert.ok(p.info[2]>=108 && p.info[2]<=Math.max(120,total*.3)-8, 'chart gains width while protecting metadata');
						if(total>=450) assert.ok(Math.abs(p.info[2]/total-.24)<.005, 'roomy cards use a 24%/76% split');
						if (p.selected === 'true') assert.ok(p.card[3] < 220, `two-column card remains compact with wrapped heading/token text: ${context}`);
					} else assert.ok(p.chart[1] >= p.info[1] + p.info[3], `narrow cards stack without overlap: ${context}`);
					assert.ok(p.chart[0] >= p.card[0] && p.chart[0]+p.chart[2] <= p.card[0]+p.card[2], "selection border encloses chart");
					if (p.selected === "true") {
						assert.ok(p.border.every(c=>c==='rgb(217, 119, 87)'));
						assert.equal(p.scoreColor, 'rgb(217, 119, 87)');
					} else assert.notEqual(p.scoreColor, 'rgb(217, 119, 87)');
					assert.equal(p.colors[0], 'rgb(217, 119, 87)');
					assert.equal(p.colors[1], p.colors[0]);
					assert.equal(p.colors[2], theme === 'light' ? 'rgb(40, 125, 136)' : 'rgb(102, 183, 190)');
					assert.equal(p.colors[3], p.colors[2]);
					assert.equal(p.colors[4], theme === 'light' ? 'rgb(138, 136, 127)' : 'rgb(143, 141, 133)');
					assert.ok(p.height <= 132, "compact plot does not reserve excess top margin");
					assert.ok(p.width <= p.availableWidth, `chart overflow: ${context}`);
					assert.ok(p.labels.every((n) => !n.overflow), `token overflow: ${context}`);
					for (const [i,label] of p.labels.entries()) {
						assert.equal(label.color, theme==='light'?'rgb(25, 25, 25)':'rgb(245, 244, 238)', 'all token names use normal neutral ink');
						const delta=p.deltaStyles[i];
						assert.equal(delta.color,theme==='light'?'rgb(94, 93, 89)':'rgb(194, 192, 182)');
						assert.ok(delta.size<label.size); assert.match(delta.text,/^Δ [−+].* pp$/);
						assert.ok(delta.box[1]>=label.box[1]+label.box[3], 'delta stays directly below its token');
					}
					assert.ok(p.labels[4].title.includes("🧠Long".repeat(20)), "full long token and exact ID remain accessible");
					assert.ok(p.labels[4].text.endsWith('…'), "long labels are bounded rather than making arbitrarily tall cards");
					assert.equal(p.unsafe, 0); assert.equal(p.badToken, false);
					assert.deepEqual(p.roles, ["target", "target", "counterfactual", "counterfactual", ...Array(6).fill("other")]);
					assert.ok(p.matches.slice(0, 4).every(match=>match===roleMatch));
					for (const [i, fill] of p.bars.entries()) assert.equal(fill.startsWith("url(#"), Boolean(i % 2));
					for (const [x,y,w,h] of p.allText) assert.ok(x >= 0 && x+w <= p.width+.5 && y >= 0 && y+h <= p.height+.5,
						`probability text clipping: ${context} ${JSON.stringify([x,y,w,h])}`);
					for (let i=0;i<p.values.length;i++) for (let j=0;j<i;j++) {
						const a=p.values[i],b=p.values[j];
						assert.ok(a[0]+a[2] <= b[0] || b[0]+b[2] <= a[0] || a[1]+a[3] <= b[1] || b[1]+b[3] <= a[1],
							`probability label overlap: ${context}`);
					}
					for (const [i,bar] of p.barBounds.entries()) {
						assert.ok(Math.abs(p.valueCenters[i]-bar[0]-bar[2]/2)<.001, 'value anchor centered on its own bar');
						assert.ok(p.values[i][1]+p.values[i][3]<=bar[1], 'value stays above its own bar');
					}
					assert.ok(p.values.filter((value,i)=>p.barBounds[i][1]-value[1]-value[3]<5).length>=4, 'most labels stay close unless collision clearance is necessary');
					for (const a of p.values) for (const b of p.barBounds.filter(b=>b[3]>0)) assert.ok(
						a[0]+a[2] <= b[0] || b[0]+b[2] <= a[0] || a[1]+a[3] <= b[1] || b[1]+b[3] <= a[1],
						`value overlaps a bar: ${context} value=${JSON.stringify(a)} bar=${JSON.stringify(b)}`);
				}
			}
		}
		// Multiple selected labels with the same leading-space-normalized name
		// must not silently receive the target role, in either theme.
		for (const theme of ['light','dark']) {
			await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: theme }] });
			await evaluate(`state.savedResults.path.graph.path_provenance.paths.forEach(p=>p.probability_comparison.tokens[2].token='  Mary');
				restoreSavedResult('path');onClickNode({id:'a'});`);
			const ambiguous = await evaluate(`(() => {
				const svg=document.querySelector('.path-probability-chart');
				return {roles:[...svg.querySelectorAll('.path-probability-bar-group')].map(n=>n.dataset.role),
					color:getComputedStyle(svg.querySelector('.path-probability-bar')).fill};
			})()`);
			assert.deepEqual(ambiguous.roles, ['other','other','counterfactual','counterfactual',...Array(6).fill('other')]);
			assert.equal(ambiguous.color, theme === 'light' ? 'rgb(138, 136, 127)' : 'rgb(143, 141, 133)');
		}
		await evaluate(`state.savedResults.path.graph.path_provenance.paths.forEach(p=>p.probability_comparison.tokens[2].token='<|endoftext|>');
			restoreSavedResult('path');onClickNode({id:'a'});`);
		const chartPoint = await evaluate(`(() => {
			const bar=document.querySelector('.path-entry[data-path-id="path-2"] .path-probability-bar');
			bar.scrollIntoView({block:'center'});const r=bar.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];
		})()`);
		await clickAt(...chartPoint);
		assert.equal(await evaluate(`state.pathExplorer.selectedId`), 'path-2', "real chart clicks select the owning path");
		assert.equal(await evaluate(`document.querySelector('.path-entry[data-path-id="path-2"]').dataset.selected`), 'true');
		assert.equal(await evaluate(`document.querySelector('.path-entry[data-path-id="path-1"]').dataset.selected`), 'false');
		const sharedScale = await evaluate(`(() => {
			const high=state.savedResults.path.graph.path_provenance.paths[0].probability_comparison.tokens[0];
			high.patched=.61;high.delta=high.patched-high.reference;
			restoreSavedResult('path');onClickNode({id:'e1'});
			const scale=state.result.probabilityScale;
			selectPath('path-4');document.querySelector('#path-sort button[value="absolute"]').click();
			document.querySelector('#path-list').scrollTop=200;
			document.querySelector('#path-clear').click();
			const filter=document.querySelector('#threshold-slider');filter.value='.2';filter.dispatchEvent(new Event('input'));
			onClickNode({id:'e1'});
			return {maximum:scale.maximum,same:scale===state.result.probabilityScale,
				highHidden:!state.pathEntries.has('path-1'),
				axes:[...document.querySelectorAll('.path-probability-chart')].map(n=>n.dataset.axisMax),
				ticks:[...document.querySelector('.path-probability-chart').querySelectorAll('.path-probability-tick')].map(n=>n.textContent)};
		})()`);
		assert.equal(sharedScale.maximum,.8); assert.equal(sharedScale.same,true); assert.equal(sharedScale.highHidden,true);
		assert.ok(sharedScale.axes.length>0 && sharedScale.axes.every(max=>max==='0.8'));
		assert.deepEqual(sharedScale.ticks,['0%','20%','40%','60%','80%']);
		for (const maximum of [0, 1e-40, 1e-10, .00011, .189, .5, 1]) {
			await evaluate(`(() => {
				const factors=${JSON.stringify(maximum === 1 ? [1,0,0,0,0] : [1,.5,.3,.1,0])};
				for(const p of state.savedResults.path.graph.path_provenance.paths) {
					p.probability_comparison.tokens.forEach((t,i)=>{
						t.reference=${maximum}*factors[i];t.patched=t.reference*.99;t.delta=t.patched-t.reference;
					});
					p.probability_comparison.tokens.sort((a,b)=>b.reference-a.reference || a.token_id-b.token_id);
				}
				restoreSavedResult('path');onClickNode({id:'a'});
			})()`);
			for (const theme of ['light','dark']) for (const width of [1440,680,390,320]) {
				await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: theme }] });
				await viewport(width);
				const smallScale = await evaluate(`(() => {
					const svg=document.querySelector('.path-probability-chart');
					const box=n=>{const b=n.getBBox();return [b.x,b.y,b.width,b.height]};
					return {width:svg.width.baseVal.value,height:svg.height.baseVal.value,
						values:svg.querySelectorAll('.path-probability-value').length,
						valueBounds:[...svg.querySelectorAll('.path-probability-value')].map(box),
						axisBounds:[...svg.querySelectorAll('.path-probability-tick,.path-probability-axis-title')].map(box),
						bars:[...svg.querySelectorAll('.path-probability-bar')].map(box),
						text:[...svg.querySelectorAll('text')].map(box)};
				})()`);
				assert.equal(smallScale.values, 10);
				for (const [x,y,w,h] of smallScale.text) assert.ok(x>=0 && x+w<=smallScale.width+.5 && y>=0 && y+h<=smallScale.height+.5,
					`axis/value clipping at maximum ${maximum}, ${theme} ${width}px: ${JSON.stringify([x,y,w,h])}`);
				for (const [i,a] of smallScale.valueBounds.entries()) {
					for (const b of [...smallScale.valueBounds.slice(0,i),...smallScale.axisBounds,...smallScale.bars.filter(b=>b[3]>0)]) assert.ok(
						a[0]+a[2]<=b[0] || b[0]+b[2]<=a[0] || a[1]+a[3]<=b[1] || b[1]+b[3]<=a[1],
						`overlap at maximum ${maximum}, ${theme} ${width}px: ${JSON.stringify({a,b})}`);
				}
			}
		}
		if (process.env.IPE_TEST_SCREENSHOT) {
			await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
			await viewport(1440, 1000);
			await evaluate(`(() => {
				for(const p of state.savedResults.path.graph.path_provenance.paths) {
					for(const t of p.probability_comparison.tokens) {
						t.reference=({5335:.3,1757:.29,50256:.1,197:1e-10,1:0})[t.token_id];
						t.patched=({5335:.29,1757:.3,50256:.09,197:1e-9,1:0})[t.token_id];t.delta=t.patched-t.reference;
					}
					p.probability_comparison.tokens.sort((a,b)=>b.reference-a.reference || a.token_id-b.token_id);
				}
				restoreSavedResult('path');onClickNode({id:'a'});
				document.querySelector('#path-list').scrollTop=0;document.querySelector('#path-explorer').scrollIntoView();
			})()`);
			const screenshot = await send("Page.captureScreenshot", { format: "png" });
			fs.writeFileSync(process.env.IPE_TEST_SCREENSHOT, Buffer.from(screenshot.data, "base64"));
		}
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
