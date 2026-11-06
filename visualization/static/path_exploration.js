"use strict";

const PathExploration = (() => {
	const LANE_GAP = 3.5;
	const MAX_OFFSET = 24;
	const edgeKey = (source, target) => `${source}->${target}`;

	function buildIndex(graph, method) {
		const provenance = graph?.path_provenance;
		const available = ["path", "tree"].includes(method)
			&& provenance?.version === 1 && provenance.source === method;
		const index = { available, paths: new Map(), byNode: new Map(), byEdge: new Map(),
			incompleteByNode: new Map(), order: provenance?.order };
		if (!available) return index;
		const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
		const edges = new Set(graph.edges.map((e) => edgeKey(e.source, e.target)));
		const occurrences = new Map((provenance.occurrences || []).map((n) => [n.id, n]));
		for (const original of provenance.paths || []) {
			const ids = original.node_ids;
			if (!Array.isArray(ids) || !ids.length || ids.some((id) => !nodes.has(id))) continue;
			const nodeIds = new Set(ids);
			if (original.complete !== true) {
				for (const id of nodeIds) {
					index.incompleteByNode.set(id, (index.incompleteByNode.get(id) || 0) + 1);
				}
				continue;
			}
			// Only consecutive original pairs are eligible, never all edges induced
			// by these display nodes. Same-grid occurrences have no display edge.
			const edgeKeys = new Set();
			for (let i = 0; i + 1 < ids.length; i++) {
				if (ids[i] !== ids[i + 1]) edgeKeys.add(edgeKey(ids[i], ids[i + 1]));
			}
			if ([...edgeKeys].some((key) => !edges.has(key)) || index.paths.has(original.id)) continue;
			const path = { id: original.id, ordinal: index.paths.size + 1, nodeIds, edgeKeys,
				sequence: [...ids], score: Number.isFinite(original.score) ? original.score : null,
				scoreKind: original.score_kind || "unknown",
				// Keep the original ordered sequence (including occurrence variants)
				// for summaries; membership sets are only for graph highlighting.
				components: ids.map((id, i) => occurrences.get(original.occurrence_ids?.[i]) || nodes.get(id)) };
			index.paths.set(path.id, path);
			for (const [keys, membership] of [[nodeIds, index.byNode], [edgeKeys, index.byEdge]]) {
				for (const key of keys) {
					if (!membership.has(key)) membership.set(key, []);
					membership.get(key).push(path.id);
				}
			}
		}
		return index;
	}

	class Explorer {
		constructor(index) {
			this.index = index;
			this.clear();
		}
		clear() {
			this.nodeId = null;
			this.selectedId = null;
			this.eligible = [];
			this.sortOrder = "discovery";
			this._view = null;
		}
		open(nodeId) {
			if (!this.index.available) return false;
			this.nodeId = nodeId;
			this.eligible = this.index.byNode.get(nodeId) || [];
			this.selectedId = this.eligible[0] || null;
			this._view = null;
			return true;
		}
		get active() { return this.nodeId !== null; }
		select(id) {
			if (!this.eligible.includes(id)) return false;
			this.selectedId = id;
			this._view = null;
			return true;
		}
		setSortOrder(order) {
			if (!["discovery", "signed", "absolute"].includes(order)) return false;
			this.sortOrder = order;
			return true;
		}
		listedPaths() {
			// Sort only sidebar rows. The view, lane slots, ordinals and selection
			// continue to use original discovery order and original identities.
			const paths = [...this.view().paths];
			if (this.sortOrder === "discovery") return paths;
			return paths.sort((a, b) => {
				const validA = Number.isFinite(a.score), validB = Number.isFinite(b.score);
				if (validA !== validB) return validA ? -1 : 1;
				const rank = (path) => this.sortOrder === "absolute" ? Math.abs(path.score) : path.score;
				return (validA ? rank(b) - rank(a) : 0) || a.ordinal - b.ordinal;
			});
		}
		view() {
			if (this._view) return this._view;
			// Reserve lanes by discovery order, never by selection. The first
			// path owns the center lane even when another path becomes orange.
			const laneGap = Math.min(LANE_GAP, (MAX_OFFSET - 5) / Math.max(1, Math.ceil((this.eligible.length - 1) / 2) - 1));
			const paths = this.eligible.map((id, slot) => ({ ...this.index.paths.get(id),
				selected: id === this.selectedId,
				offset: slot === 0 ? 0 : (slot % 2 ? 1 : -1) * (5 + Math.floor((slot - 1) / 2) * laneGap) }));
			const nodes = new Map(), edges = new Map(), focusNodes = new Set();
			for (const path of paths) {
				for (const [keys, membership] of [[path.nodeIds, nodes], [path.edgeKeys, edges]]) {
					for (const key of keys) {
						if (!membership.has(key)) membership.set(key, []);
						membership.get(key).push(path);
					}
				}
				if (this.selectedId === path.id) {
					for (const id of path.nodeIds) focusNodes.add(id);
				}
			}
			this._view = { paths, nodes, edges, focusNodes, laneGap };
			return this._view;
		}
	}

	function scorePresentation(path, meta = {}) {
		const metrics = {
			logit_difference: ["Logit Difference", "logit units"],
			target_logit_percentage: ["Target Logit Percentage", "relative %"],
			target_probability_percentage: ["Target Probability Percentage", "relative %"],
			indirect_effect: ["Indirect Effect", "relative effect"],
			kl_divergence: ["KL Divergence", "nats"],
		};
		const [metric, unit] = metrics[meta.metric] || [meta.metric || "Metric unavailable", ""];
		const isolated = path.scoreKind === "isolated_path_effect";
		const joint = path.scoreKind === "joint_candidate_effect";
		const label = isolated ? "Path effect" : joint ? "In-context admission score" : "Original score (mode unknown)";
		if (!Number.isFinite(path.score)) {
			return { text: `${label} · ${metric} · unavailable`,
				tooltip: "No finite original complete-path score is available. Merged node/edge contributions and the joint tree contribution are not substituted." };
		}
		const magnitude = Math.abs(path.score);
		const value = magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e6)
			? magnitude.toExponential(5) : Number(magnitude.toPrecision(6)).toString();
		const signed = `${path.score < 0 || Object.is(path.score, -0) ? "−" : "+"}${value}`;
		let explanation;
		if (joint) {
			explanation = "Original candidate admission score in the frozen tree context at discovery; not an isolated path effect or the final joint tree contribution.";
		} else if (!isolated) {
			explanation = "The original score's scoring mode is unknown. Do not interpret it as an isolated path effect.";
		} else {
			const noising = meta.mode === "noising", denoising = meta.mode === "denoising";
			if (meta.metric === "logit_difference" && noising) {
				explanation = "Positive scores indicate that perturbing this path decreases the clean target's logit advantage over the counterfactual target. Negative scores indicate the opposite.";
			} else if (meta.metric === "logit_difference" && denoising) {
				explanation = "Positive scores indicate that patching clean activations along this path into the counterfactual run increases the clean target's logit advantage over the counterfactual target. Negative scores indicate the opposite.";
			} else if (meta.metric === "target_logit_percentage" && noising) {
				explanation = "Positive scores indicate a decrease in the target logit relative to the absolute clean target logit; negative scores indicate an increase. Values are relative changes × 100, not probabilities.";
			} else if (meta.metric === "target_probability_percentage" && noising) {
				explanation = "Positive scores indicate a relative decrease in the configured target probabilities; negative scores indicate an increase. Values are relative changes × 100, not probabilities.";
			} else if (meta.metric === "indirect_effect" && (noising || denoising)) {
				explanation = denoising
					? "Baseline-subtracted Indirect Effect for patching clean activations into the counterfactual run. Positive scores increase the direction-adjusted relative-probability metric from its counterfactual baseline; negative scores indicate the opposite. This is not a probability or percentage."
					: "Direction-adjusted Indirect Effect computed from relative probability changes for both configured targets under path perturbation. Positive scores indicate an effect in the noising direction; negative scores indicate the opposite. This is not a probability or percentage.";
			} else if (meta.metric === "kl_divergence" && noising) {
				explanation = "KL divergence of the patched output distribution from the clean output distribution (KL(patched || clean)), in nats. Larger values mean more distributional change, not necessarily worse target performance.";
			} else {
				explanation = "Original isolated complete-path score. Directional interpretation is unavailable for this metric/patching combination.";
			}
			explanation += " This is a batch-averaged isolated complete-path intervention, not independent component importance, an aggregated node/edge score, or the joint tree contribution.";
		}
		return { text: `${label} · ${metric} · ${signed}${unit ? ` ${unit}` : ""}`,
			tooltip: `${explanation} Original value: ${path.score}.` };
	}

	function componentName(n) {
		if (n.kind === "attn") return `A${n.layer}H${n.head ?? "*"}`;
		if (n.kind === "mlp") return `MLP${n.layer}`;
		return { final: "FINAL", embed: "EMB" }[n.kind] || n.kind || "Unknown";
	}

	function patchVariant(n) {
		const flags = ["patch_query", "patch_key", "patch_value"];
		if (flags.every((flag) => typeof n[flag] === "boolean")) {
			return flags.map((flag, i) => n[flag] ? "QKV"[i] : "").join("") || "none";
		}
		// Never infer an occurrence's intervention from merged `variants`.
		return typeof n.variant === "string" && /^[QKV]+$/.test(n.variant)
			? [..."QKV"].filter((flag) => n.variant.includes(flag)).join("") : null;
	}

	function componentLabel(n) {
		let label = componentName(n);
		if (n.position !== null && n.position !== undefined) label += `@${n.position}`;
		if (n.kind === "attn") {
			const variant = patchVariant(n);
			if (variant !== null) {
				label += `[${variant}${n.kv_position === null || n.kv_position === undefined ? "" : `:${n.kv_position}`}]`;
			}
		}
		return label;
	}

	function physicalKey(n) {
		return JSON.stringify([n.kind, n.kind === "embed" ? null : n.layer ?? null,
			n.kind === "attn" ? n.head ?? null : null,
			n.kind === "unknown" ? n.node_id || n.id : null]);
	}

	function interventionKey(n) {
		return JSON.stringify([n.position, patchVariant(n), n.kv_position, n.patch_type]);
	}

	/* Align shared physical components with a weighted longest common
	 * subsequence. Exact metadata breaks ties between repeated components.
	 * Single unmatched pairs are replacements; larger unmatched runs are
	 * explicit removals/additions rather than invented index correspondences.
	 * Neither occurrence object IDs nor contribution values influence alignment. */
	function comparePaths(reference, alternative) {
		const a = [...reference.components].reverse(), b = [...alternative.components].reverse();
		const weight = Math.max(a.length, b.length) + 1;
		const keysA = a.map(physicalKey), keysB = b.map(physicalKey);
		const specsA = a.map(interventionKey), specsB = b.map(interventionKey);
		const dp = Array.from({ length: a.length + 1 }, () => new Uint32Array(b.length + 1));
		for (let i = a.length - 1; i >= 0; i--) {
			for (let j = b.length - 1; j >= 0; j--) {
				const match = keysA[i] === keysB[j]
					? weight + Number(specsA[i] === specsB[j]) + dp[i + 1][j + 1] : 0;
				dp[i][j] = Math.max(match, dp[i + 1][j], dp[i][j + 1]);
			}
		}
		const differences = [], removed = [], added = [];
		const flush = () => {
			if (removed.length === 1 && added.length === 1) {
				differences.push({ type: "component", label: `${componentLabel(removed[0])} → ${componentLabel(added[0])}` });
			} else {
				for (const n of removed) differences.push({ type: "removed", label: `Removed ${componentLabel(n)}` });
				for (const n of added) differences.push({ type: "added", label: `Added ${componentLabel(n)}` });
			}
			removed.length = added.length = 0;
		};
		const value = (v) => v === null ? "unspecified" : String(v);
		const compareFields = (left, right) => {
			const field = (type, name, from, to) => {
				if (from !== undefined && to !== undefined && from !== to) {
					differences.push({ type, label: `${componentName(left)}${name}: ${value(from)} → ${value(to)}` });
				}
			};
			field("position", " token position", left.position, right.position);
			if (left.kind === "attn") {
				field("patching", "", patchVariant(left) ?? undefined, patchVariant(right) ?? undefined);
				field("kv_position", " K/V position", left.kv_position, right.kv_position);
			}
			field("patch_type", " patch type", left.patch_type ?? undefined, right.patch_type ?? undefined);
		};
		let i = 0, j = 0;
		while (i < a.length && j < b.length) {
			const match = keysA[i] === keysB[j]
				&& dp[i][j] === weight + Number(specsA[i] === specsB[j]) + dp[i + 1][j + 1];
			if (match) {
				flush();
				compareFields(a[i++], b[j++]);
			} else if (dp[i + 1][j] >= dp[i][j + 1]) removed.push(a[i++]);
			else added.push(b[j++]);
		}
		removed.push(...a.slice(i));
		added.push(...b.slice(j));
		flush();
		const metadataIncomplete = [...a, ...b].some((n) => n.position === undefined || n.patch_type == null
			|| (n.kind === "attn" && (patchVariant(n) === null || n.kv_position === undefined)));
		// Consecutive occurrences may collapse onto one display node. Ignore
		// those zero-length links only for projected-geometry equality, never
		// for the original sequence or its intervention comparison.
		const displayRoute = (path) => path.sequence.filter((id, k) => k === 0 || id !== path.sequence[k - 1]);
		const routeA = displayRoute(reference), routeB = displayRoute(alternative);
		const sameGeometry = routeA.length === routeB.length && routeA.every((id, k) => id === routeB[k]);
		return { differences, sameGeometry, metadataIncomplete,
			patchingOnly: differences.length > 0 && differences.every((d) => ["patching", "kv_position", "patch_type"].includes(d.type)) };
	}

	/* Sample the existing cubic and offset along its local unit normal. A plain
	 * x-translation collapses lanes on horizontal portions of curved K/V reads.
	 * Slot offsets are fixed across the whole path, not recomputed per edge:
	 * lanes never swap sides when another path joins or branches away. */
	function parallelPoints(source, target, offset) {
		const midY = (source.top + target.bottom) / 2;
		const x0 = source.cx, y0 = source.top, x1 = target.cx, y1 = target.bottom;
		const steps = Math.max(24, Math.min(160, Math.ceil((Math.abs(x1 - x0) + Math.abs(y1 - y0)) / 4)));
		const points = [];
		for (let i = 0; i <= steps; i++) {
			const t = i / steps, u = 1 - t;
			const x = u * u * u * x0 + 3 * u * u * t * x0 + 3 * u * t * t * x1 + t * t * t * x1;
			const y = u * u * u * y0 + 3 * u * u * t * midY + 3 * u * t * t * midY + t * t * t * y1;
			const dx = 6 * u * t * (x1 - x0);
			const dy = 3 * u * u * (midY - y0) + 3 * t * t * (y1 - midY);
			const length = Math.hypot(dx, dy) || 1;
			points.push({ x: x - dy / length * offset, y: y + dx / length * offset });
		}
		return points;
	}

	function parallelPath(source, target, offset) {
		return parallelPoints(source, target, offset)
			.map((p, i) => `${i ? "L" : "M"} ${p.x.toFixed(3)} ${p.y.toFixed(3)}`).join(" ");
	}

	return { buildIndex, Explorer, scorePresentation, comparePaths, componentName, componentLabel, patchVariant, parallelPoints, parallelPath };
})();

if (typeof module !== "undefined") module.exports = PathExploration;
