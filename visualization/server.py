"""Local web UI for IPE circuit discovery.

Serves a single-page frontend (static/) and a small JSON/SSE API around the
path- and tree-based searches:

	POST /api/tokenize            tokenize a prompt (loads the model on first use)
	POST /api/search              start a discovery job -> {job_id}
	GET  /api/search/{id}/events  Server-Sent Events stream of search progress
	GET  /api/search/{id}         job status / final result (for reconnects)
	POST /api/search/{id}/cancel  request cancellation of a running job
	GET  /api/config              available models, metrics, methods, device

Run from the repo root:

	PYTHONPATH=src uvicorn visualization.server:app --port 8321
	# or simply
	python visualization/server.py

One search runs at a time (a single model forward pass already saturates the
device); extra jobs queue on a lock. The loaded HookedTransformer is cached per
model name and reused across jobs.
"""

import gc
import json
import math
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

# Allow `python visualization/server.py` without an installed ipe package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
	sys.path.insert(0, str(_SRC))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformer_lens import HookedTransformer

from ipe.experiment import ExperimentManager
from ipe.paths import evaluate_tree
from ipe.graph_search import (
	PathMessagePatching,
	PathMessagePatching_LimitedLevelWidth,
	TreeMessagePatching,
	TreeMessagePatching_LimitedLevelWidth,
)
from ipe.webutils.serialization import graph_from_paths, graph_from_tree, node_to_dict
from ipe.webutils.component_logit_lens import LiveComponentLogitLens
from ipe.webutils import acdc_bridge

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Llama/Gemma entries are gated on HuggingFace: accept the license there and
# `huggingface-cli login` (or set HF_TOKEN) before selecting them.
MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large", "pythia-160m", "pythia-410m",
		  "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-2-2b-it"]
METRICS = ["logit_difference", "indirect_effect", "target_logit_percentage",
		   "target_probability_percentage", "kl_divergence"]
MODES = ["noising", "denoising"]
# The vendored ACDC baseline (benchmark/Automatic-Circuit-Discovery) brings its
# own metrics and its own (position-agnostic) notion of a circuit.
ACDC_METRICS = acdc_bridge.ACDC_METRICS
# ExperimentManager silently forces patch_clean_into_cf=False (i.e. noising) for
# any other metric (experiment.py:120), so denoising is only truthful for these.
DENOISING_METRICS = ["logit_difference", "indirect_effect"]

app = FastAPI(title="IPE circuit discovery")


# --------------------------------------------------------------------- models

_model_cache: dict[str, HookedTransformer] = {}
_model_lock = threading.Lock()


def get_model(name: str) -> HookedTransformer:
	"""Load (once) and cache a HookedTransformer, configured as in the
	experiments (fp16 on GPU, centered unembedding)."""
	if name not in MODELS:
		raise HTTPException(400, f"Unknown model '{name}'. Available: {MODELS}")
	with _model_lock:
		if name not in _model_cache:
			model = HookedTransformer.from_pretrained(
				name,
				device=DEVICE,
				torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
				center_unembed=True,
			)
			model.eval()
			_model_cache[name] = model
		return _model_cache[name]


# ----------------------------------------------------------------------- jobs

class SearchCancelled(BaseException):
	"""Raised inside the search's on_event callback to abort a job. Derives from
	BaseException on purpose: the search swallows Exception from observers."""


def _scrub(x):
	"""Replace non-finite floats with None. json.dumps emits bare NaN/Infinity,
	which is invalid JSON for the browser's JSON.parse — one NaN metric (easy
	with fp16) would otherwise silently kill the event that carries it."""
	if isinstance(x, float):
		return x if math.isfinite(x) else None
	if isinstance(x, dict):
		clean = {k: _scrub(v) for k, v in x.items()}
		lens = clean.get("component_logit_lens")
		if isinstance(lens, dict) and any(
			item.get("probability") is None for item in lens.get("top_tokens", [])
		):
			clean["component_logit_lens"] = {"available": False}
		return clean
	if isinstance(x, (list, tuple)):
		return [_scrub(v) for v in x]
	return x


class Job:
	def __init__(self, job_id: str, params: dict):
		self.id = job_id
		self.params = params
		self.status = "queued"  # queued | running | complete | error | cancelled
		self.error: str | None = None
		self.result: dict | None = None
		self.events: list[dict] = []
		self.cond = threading.Condition()
		self.cancel_requested = False
		self.created = time.time()

	def push(self, event: dict) -> None:
		event = _scrub(event)
		with self.cond:
			self.events.append(event)
			self.cond.notify_all()


_jobs: OrderedDict[str, Job] = OrderedDict()
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # one search at a time
MAX_JOBS = 20


def _register_job(job: Job) -> None:
	with _jobs_lock:
		_jobs[job.id] = job
		while len(_jobs) > MAX_JOBS:
			old_id, old = next(iter(_jobs.items()))
			if old.status in ("complete", "error", "cancelled"):
				_jobs.pop(old_id)
			else:
				break


def _get_job(job_id: str) -> Job:
	job = _jobs.get(job_id)
	if job is None:
		raise HTTPException(404, "Job not found")
	return job


# ------------------------------------------------------------------- requests

class TokenizeRequest(BaseModel):
	model: str
	prompt: str


class SearchRequest(BaseModel):
	model: str = "gpt2-small"
	prompts: list[str] = Field(min_length=1)
	# ACDC's kl_div needs no target at all; the per-method _validate below still
	# enforces one target per prompt for every case that does need them.
	targets: list[str] = []
	cf_prompts: list[str] = []
	cf_targets: list[str] = []
	method: str = "tree"           # tree | path | acdc
	strategy: str = "threshold"    # threshold | topk
	mode: str = "noising"          # noising | denoising
	min_contribution: float = 0.05
	max_width: int = 20000
	metric: str = "logit_difference"
	positional: bool = True
	include_negative: bool = True
	# ACDC only (method == "acdc"); the knobs above do not apply to it.
	acdc_metric: str = "kl_div"
	acdc_threshold: float = 0.0575
	acdc_abs_threshold: bool = False


def _validate_acdc(req: SearchRequest, model: HookedTransformer) -> None:
	if req.acdc_metric not in ACDC_METRICS:
		raise HTTPException(400, f"Unknown ACDC metric '{req.acdc_metric}'. Available: {ACDC_METRICS}")
	if req.cf_prompts and len(req.cf_prompts) != len(req.prompts):
		raise HTTPException(400, "cf_prompts must match prompts in number")

	# ACDC always runs the prompts as one batch, so they have to tokenize to a
	# common length whatever the (IPE-only) positional switch says.
	ref_len = len(model.to_str_tokens(req.prompts[0], prepend_bos=True))
	for p in req.prompts + req.cf_prompts:
		n = len(model.to_str_tokens(p, prepend_bos=True))
		if n != ref_len:
			raise HTTPException(400, f"Prompt {p!r} has {n} tokens, expected {ref_len} "
									 "(ACDC batches all prompts, so they must tokenize to "
									 "the same length)")

	if req.acdc_metric in acdc_bridge.ACDC_METRICS_NEEDING_TARGETS:
		if len(req.targets) != len(req.prompts):
			raise HTTPException(400, f"ACDC metric '{req.acdc_metric}' needs one target per prompt")
	if req.acdc_metric in acdc_bridge.ACDC_METRICS_NEEDING_CF_TARGETS:
		if len(req.cf_targets) != len(req.prompts):
			raise HTTPException(400, f"ACDC metric '{req.acdc_metric}' needs one counterfactual "
									 "target per prompt (it is the 'wrong' answer of the logit difference)")
	for t in req.targets + req.cf_targets:
		if t and len(model.to_str_tokens(t, prepend_bos=False)) != 1:
			raise HTTPException(400, f"Target {t!r} is not a single token "
									 "(tip: most tokens need a leading space)")


def _validate(req: SearchRequest, model: HookedTransformer) -> None:
	if req.method == "acdc":
		return _validate_acdc(req, model)
	if req.method not in ("tree", "path"):
		raise HTTPException(400, "method must be 'tree', 'path' or 'acdc'")
	if req.strategy not in ("threshold", "topk"):
		raise HTTPException(400, "strategy must be 'threshold' or 'topk'")
	if req.mode not in MODES:
		raise HTTPException(400, f"mode must be one of {MODES}")
	if req.metric not in METRICS:
		raise HTTPException(400, f"Unknown metric '{req.metric}'. Available: {METRICS}")
	if req.mode == "denoising":
		if not req.cf_prompts:
			raise HTTPException(400, "denoising needs counterfactual prompts to source "
									 "clean activations from (zero patching only noises)")
		if req.metric not in DENOISING_METRICS:
			raise HTTPException(400, f"denoising is only supported for metrics {DENOISING_METRICS}; "
									 f"'{req.metric}' would silently fall back to noising")
	if req.cf_prompts and len(req.cf_prompts) != len(req.prompts):
		raise HTTPException(400, "cf_prompts must match prompts in number")
	if len(req.targets) != len(req.prompts):
		raise HTTPException(400, "targets must match prompts in number")
	if req.cf_prompts and len(req.cf_targets) != len(req.cf_prompts):
		raise HTTPException(400, "cf_targets must match cf_prompts in number")

	ref_len = len(model.to_str_tokens(req.prompts[0], prepend_bos=True))
	for p in req.prompts + req.cf_prompts:
		n = len(model.to_str_tokens(p, prepend_bos=True))
		if req.positional and n != ref_len:
			raise HTTPException(400, f"Prompt {p!r} has {n} tokens, expected {ref_len} "
									 "(all prompts and counterfactuals must tokenize to the "
									 "same length for a positional search)")
	if req.metric != "kl_divergence":
		for t in req.targets + req.cf_targets:
			if len(model.to_str_tokens(t, prepend_bos=False)) != 1:
				raise HTTPException(400, f"Target {t!r} is not a single token "
										 "(tip: most tokens need a leading space)")


# -------------------------------------------------------------------- worker

def _run_acdc(job: Job, req: SearchRequest) -> None:
	"""Body of an ACDC job. Runs under the same lock / SSE plumbing as the IPE
	searches; only the events differ (ACDC prunes a full graph instead of
	growing one, so there is nothing to 'admit' node by node)."""
	last_emit = 0.0

	def on_event(event: dict) -> None:
		nonlocal last_emit
		if job.cancel_requested:
			raise SearchCancelled()
		if event.get("event") == "progress":
			# ACDC steps are cheap to report but there are ~2k of them; throttle
			# unless the event carries a graph snapshot.
			now = time.monotonic()
			if "graph" not in event and now - last_emit < 0.2:
				return
			last_emit = now
		if "graph" in event:
			lens.enrich(event["graph"])
		job.push(event)

	with _run_lock:
		job.status = "running"
		job.push({"event": "status", "status": "running", "message": f"Loading {req.model}..."})
		model = get_model(req.model)
		job.push({"event": "status", "status": "running", "message": "Preparing live component readouts..."})
		lens = LiveComponentLogitLens(model, req.prompts[0])
		if job.cancel_requested:
			raise SearchCancelled()

		tokens = model.to_str_tokens(req.prompts[0], prepend_bos=True)
		job.push({"event": "meta", "tokens": tokens,
				  "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads,
				  "positional": False, "layout": "head"})

		graph, info = acdc_bridge.run_acdc(
			model,
			req.prompts, req.targets, req.cf_prompts, req.cf_targets,
			metric_name=req.acdc_metric,
			threshold=req.acdc_threshold,
			abs_value_threshold=req.acdc_abs_threshold,
			on_event=on_event,
		)
		job.push({"event": "status", "status": "running",
				  "message": "Decoding discovered components from the clean prompt..."})
		lens.enrich(graph, final=True)
		del lens

		job.result = _scrub({
			"graph": graph,
			"meta": {
				"tokens": tokens,
				"n_layers": model.cfg.n_layers,
				"n_heads": model.cfg.n_heads,
				"positional": False,
				"layout": "head",
				"method": "acdc",
				"strategy": f"threshold {info['threshold']}",
				"mode": f"{info['ablation']} ablation",
				"metric": info["metric"],
				"runtime": info["runtime"],
				"joint_tree_contribution": None,
				"n_nodes": len(graph["nodes"]),
				"n_edges": len(graph["edges"]),
				"acdc_edges": info["acdc_edges"],
				"acdc_initial_edges": info["initial_acdc_edges"],
				"final_metric": info["final_metric"],
			},
		})
		job.status = "complete"
		job.push({"event": "result", **job.result})

		gc.collect()
		if DEVICE.type == "cuda":
			torch.cuda.empty_cache()


def _run_search(job: Job, req: SearchRequest) -> None:
	last_leaf_emit = 0.0

	def on_event(event: dict) -> None:
		nonlocal last_leaf_emit
		if job.cancel_requested:
			raise SearchCancelled()
		kind = event.get("event")
		if kind == "leaf_done":
			# Progress ticks can be very frequent; throttle to ~10/s.
			now = time.monotonic()
			if now - last_leaf_emit < 0.1 and event["leaf"] != event["n_leaves"]:
				return
			last_leaf_emit = now
			job.push(event)
		elif kind == "admit":
			node = node_to_dict(event["node"])
			parent = node_to_dict(event["parent"])
			lens.enrich({"nodes": [node, parent]})
			job.push({
				"event": "admit",
				"depth": event["depth"],
				"node": node,
				"parent": parent,
				"contribution": event["contribution"],
			})
		elif kind == "path_complete":
			path = [node_to_dict(n) for n in event["path"]]
			lens.enrich({"nodes": path})
			job.push({
				"event": "path_complete",
				"depth": event["depth"],
				"path": path,
				"contribution": event["contribution"],
			})
		else:  # depth_start / depth_end
			job.push(event)

	try:
		if req.method == "acdc":
			return _run_acdc(job, req)
		with _run_lock:
			job.status = "running"
			job.push({"event": "status", "status": "running", "message": f"Loading {req.model}..."})
			model = get_model(req.model)
			job.push({"event": "status", "status": "running", "message": "Running clean/counterfactual forward passes..."})

			has_cf = bool(req.cf_prompts)
			# Prompts go in un-inverted: the user's clean run is the library's
			# clean run. patch_clean_into_cf is then the knob, with no confusing
			# double negation on top of a prompt swap:
			#   denoising -> True  -> base run is the counterfactual (corrupted)
			#                         run, clean activations are patched into it;
			#                         we measure how much performance is restored.
			#   noising   -> False -> base run is the clean run, counterfactual
			#                         activations are patched into it; we measure
			#                         how much performance is degraded.
			# (ExperimentManager honours True only for the DENOISING_METRICS; other
			# metrics are rejected up front in _validate so the knob never lies.)
			denoising = req.mode == "denoising"
			experiment = ExperimentManager(
				model=model,
				prompts=req.prompts,
				targets=req.targets,
				cf_prompts=req.cf_prompts if has_cf else None,
				cf_targets=req.cf_targets if has_cf else None,
				algorithm="PathMessagePatching",
				search_strategy="Threshold",
				algorithm_params={"min_contribution": req.min_contribution,
								  "include_negative": req.include_negative},
				metric=req.metric,
				positional_search=req.positional,
				patch_type="counterfactual" if has_cf else "zero",
				patch_clean_into_cf=denoising,
			)
			metric = experiment.metric
			root = experiment.root
			job.push({"event": "status", "status": "running", "message": "Preparing live component readouts..."})
			lens = LiveComponentLogitLens(model, req.prompts[0], experiment.cache)
			if job.cancel_requested:
				raise SearchCancelled()

			tokens = model.to_str_tokens(req.prompts[0], prepend_bos=True)
			job.push({"event": "meta", "tokens": tokens,
					  "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads,
					  "positional": req.positional})

			t0 = time.time()
			if req.method == "tree":
				search = (TreeMessagePatching if req.strategy == "threshold"
						  else TreeMessagePatching_LimitedLevelWidth)
				kwargs = ({"min_contribution": req.min_contribution} if req.strategy == "threshold"
						  else {"max_width": req.max_width})
				out_root = search(model, metric, root, include_negative=req.include_negative,
								  joint_scoring=False, on_event=on_event, **kwargs)
				graph = graph_from_tree(out_root, joint_scoring=False)
				joint = float(evaluate_tree(out_root, metric))
			else:
				search = (PathMessagePatching if req.strategy == "threshold"
						  else PathMessagePatching_LimitedLevelWidth)
				kwargs = ({"min_contribution": req.min_contribution} if req.strategy == "threshold"
						  else {"max_width": req.max_width})
				paths = search(model, metric, root, include_negative=req.include_negative,
							   on_event=on_event, **kwargs)
				graph = graph_from_paths(paths)
				joint = None
			runtime = time.time() - t0
			job.push({"event": "status", "status": "running",
					  "message": "Decoding discovered components from the clean prompt..."})
			lens.enrich(graph, final=True)
			del lens

			job.result = _scrub({
				"graph": graph,
				"meta": {
					"tokens": tokens,
					"n_layers": model.cfg.n_layers,
					"n_heads": model.cfg.n_heads,
					"positional": req.positional,
					"method": req.method,
					"strategy": req.strategy,
					"mode": req.mode,
					"metric": req.metric,
					"runtime": round(runtime, 2),
					"joint_tree_contribution": joint,
					"n_nodes": len(graph["nodes"]),
					"n_edges": len(graph["edges"]),
				},
			})
			job.status = "complete"
			job.push({"event": "result", **job.result})

			del experiment
			gc.collect()
			if DEVICE.type == "cuda":
				torch.cuda.empty_cache()
	except SearchCancelled:
		job.status = "cancelled"
		job.push({"event": "status", "status": "cancelled", "message": "Search cancelled."})
	except Exception as e:
		job.status = "error"
		job.error = f"{type(e).__name__}: {e}"
		job.push({"event": "status", "status": "error", "message": job.error})
	finally:
		with job.cond:
			job.cond.notify_all()


# ------------------------------------------------------------------ endpoints

@app.get("/api/config")
def config():
	return {
		"models": MODELS,
		"loaded_models": list(_model_cache),
		"metrics": METRICS,
		"methods": ["tree", "path", "acdc"],
		"strategies": ["threshold", "topk"],
		"modes": MODES,
		"denoising_metrics": DENOISING_METRICS,
		"acdc_metrics": ACDC_METRICS,
		"acdc_metrics_needing_targets": acdc_bridge.ACDC_METRICS_NEEDING_TARGETS,
		"acdc_metrics_needing_cf_targets": acdc_bridge.ACDC_METRICS_NEEDING_CF_TARGETS,
		"acdc_available": acdc_bridge.ACDC_ROOT.is_dir(),
		"device": str(DEVICE),
	}


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
	model = get_model(req.model)
	tokens = model.to_str_tokens(req.prompt, prepend_bos=True)
	return {"tokens": tokens, "length": len(tokens),
			"n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads}


@app.post("/api/search")
def start_search(req: SearchRequest):
	model = get_model(req.model)
	_validate(req, model)
	job = Job(str(uuid.uuid4()), req.model_dump())
	_register_job(job)
	threading.Thread(target=_run_search, args=(job, req), daemon=True).start()
	return {"job_id": job.id, "status": job.status}


@app.get("/api/search/{job_id}")
def job_status(job_id: str):
	job = _get_job(job_id)
	out = {"job_id": job.id, "status": job.status, "n_events": len(job.events),
		   "params": job.params, "error": job.error}
	if job.result is not None:
		out["result"] = job.result
	return out


@app.post("/api/search/{job_id}/cancel")
def cancel_job(job_id: str):
	job = _get_job(job_id)
	job.cancel_requested = True
	return {"job_id": job.id, "status": job.status, "cancel_requested": True}


@app.get("/api/search/{job_id}/events")
def job_events(job_id: str):
	job = _get_job(job_id)

	def stream():
		# Never yield while holding job.cond: the generator suspends at yield,
		# and the worker needs the lock to push events.
		idx = 0
		while True:
			with job.cond:
				if idx >= len(job.events) and job.status in ("queued", "running"):
					job.cond.wait(timeout=15.0)
				batch = job.events[idx:]
				idx += len(batch)
				done = job.status not in ("queued", "running")
			for event in batch:
				yield f"data: {json.dumps(event)}\n\n"
			if done and not batch:
				yield f"data: {json.dumps({'event': 'end', 'status': job.status})}\n\n"
				return
			if not batch:
				yield ": keep-alive\n\n"

	return StreamingResponse(stream(), media_type="text/event-stream",
							 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
	return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8321)
