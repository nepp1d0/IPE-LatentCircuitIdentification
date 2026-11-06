"use strict";

const PathPresentation = (() => {
	function formatScore(score) {
		if (!Number.isFinite(score)) return "Unavailable";
		const sign = score < 0 || Object.is(score, -0) ? "−" : "+";
		const magnitude = Math.abs(score);
		if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
			const [coefficient, exponent] = magnitude.toExponential(2).split("e");
			const superscript = String(Number(exponent)).replace(/[-\d]/g, (c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]);
			return `${sign}${Number(coefficient)} × 10${superscript}`;
		}
		return sign + Number(magnitude.toPrecision(3)).toString();
	}

	function scoreContext(paths, meta = {}) {
		const metrics = {
			logit_difference: ["logit difference", "logit units"],
			target_logit_percentage: ["relative target-logit change", "relative change × 100"],
			target_probability_percentage: ["relative target-probability change", "relative change × 100"],
			indirect_effect: ["indirect effect", "relative effect"],
			kl_divergence: ["KL divergence", "nats"],
		};
		const [metric, unit] = metrics[meta.metric] || [meta.metric || "metric unavailable", ""];
		const isolated = paths.length > 0 && paths.every((p) => p.scoreKind === "isolated_path_effect");
		const joint = paths.length > 0 && paths.every((p) => p.scoreKind === "joint_candidate_effect");
		const label = isolated ? `Effect on ${metric}` : `${joint ? "In-context scores" : "Original scores"} · ${metric}`;
		const direction = { noising: "Noising", denoising: "Denoising" }[meta.mode] || "Direction unavailable";
		return [label, direction, unit].filter(Boolean).join(" · ");
	}

	function scoreNote(path) {
		return path.scoreKind === "isolated_path_effect" ? ""
			: path.scoreKind === "joint_candidate_effect" ? "In-context admission score" : "Scoring mode unknown";
	}

	function differenceLabel(diff) {
		let label = diff.label;
		if (diff.type === "position") label = label.replace(" token position:", ":").replace(/\bunspecified\b/g, "all");
		if (diff.type === "kv_position") label = label.replace(" K/V position:", " source:").replace(/\bunspecified\b/g, "all eligible");
		if (diff.type === "patch_type") label = label.replace(" patch type:", " patch:");
		return label.replace(/\[([QKV]+|none)(:[^\]]+)?\]/g, (_, patch, source = "") => ` (${patch}${source})`);
	}

	return { formatScore, scoreContext, scoreNote, differenceLabel };
})();

if (typeof module !== "undefined") module.exports = PathPresentation;
