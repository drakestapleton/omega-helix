"""Distillation pipeline orchestrator (`dk-univ`, `seq-univ`, `dna-strand`).

Coordinates multi-teacher harvesting/generation, fail-closed verifiers, canonical
OpenWALDO exports, pure-Omega OLLMDK01 sparse top-K hybrid CE+KL training, and
Merkle-verified OLLMSP01 specialist DNA strand distillation.

Python 3 standard library only — zero external deps, zero telemetry.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

TOOLS_DIR = Path(__file__).resolve().parents[1]
STUDIO_DIR = TOOLS_DIR / "studio"
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
for _p in (str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import chain_runner  # noqa: E402
import omega_helix_backend as univ_backend  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402

from .harvest import (
    harvest_paths,
    read_ollmdk01_header,
    write_ollmdk01,
    write_openwaldo_export,
)
from .teachers import (
    LocalHelixTeacher,
    OpenAICompatTeacher,
    TeacherAdapter,
    TeacherRouter,
    TraceReplayTeacher,
)
from .verifiers import BUILTIN_VERIFIERS, VerifierPipeline

LN2 = math.log(2.0)
DISTILL_MODES = ("dk-univ", "seq-univ", "dna-strand")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_distill_runs_dir() -> Path:
    env = os.environ.get("OMEGA_HELIX_DISTILL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (chain_runner.default_runs_dir().parent / "distill").resolve()


def _file_artifact(path: Path, name: str, kind: str, **extra: Any) -> dict[str, Any]:
    sha, size = ow_bridge.file_sha256_and_bytes(path)
    rec = {"name": name, "kind": kind, "path": str(path), "sha256": sha, "bytes": size}
    rec.update(extra)
    return rec


def estimate_distill(raw_spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a distillation pipeline specification."""
    if not isinstance(raw_spec, dict):
        return {"valid": False, "error": "distillation spec must be a JSON object"}

    mode = str(raw_spec.get("mode", "dk-univ")).strip().lower()
    if mode not in DISTILL_MODES:
        return {"valid": False, "error": f"mode {mode!r} must be one of {DISTILL_MODES}"}

    arch = str(raw_spec.get("arch", "qwen3" if mode != "dna-strand" else "gpt")).strip().lower()
    if arch not in univ_backend.ARCHES:
        return {"valid": False, "error": f"arch {arch!r} must be one of {univ_backend.ARCHES}"}

    try:
        V = int(raw_spec.get("V", 256 if mode == "dk-univ" else 256))
        T = int(raw_spec.get("T", 16))
        D = int(raw_spec.get("D", 32))
        H = int(raw_spec.get("H", 2))
        H_KV = int(raw_spec.get("H_KV", H))
        F = int(raw_spec.get("F", 64))
        L = int(raw_spec.get("L", 2))
        B = int(raw_spec.get("B", 2))
        K = int(raw_spec.get("K", min(8, V)))
        alpha = float(raw_spec.get("alpha", 0.7))
        temperature = float(raw_spec.get("temperature", 1.5))
        steps = int(raw_spec.get("steps", 12))
        lr_micro = int(raw_spec.get("lr_micro", 3000))
        seed = int(raw_spec.get("seed", 42))
        held_mod = int(raw_spec.get("held_mod", 5))
        add_layers = int(raw_spec.get("add_layers", 1))
    except (TypeError, ValueError) as exc:
        return {"valid": False, "error": f"invalid numeric parameter: {exc}"}

    if V < 2:
        return {"valid": False, "error": "V must be >= 2"}
    if T < 4 or D < 8 or H < 1 or D % H != 0:
        return {"valid": False, "error": "require T >= 4, D >= 8, H >= 1, D % H == 0"}
    if H_KV < 1 or H % H_KV != 0:
        return {"valid": False, "error": "require H_KV >= 1 and H % H_KV == 0"}
    if F < 8 or L < 1 or B < 1 or steps < 1:
        return {"valid": False, "error": "require F >= 8, L >= 1, B >= 1, steps >= 1"}
    if K < 1 or K > 64 or K > V:
        return {"valid": False, "error": f"require 1 <= K <= min(64, V={V}), got {K}"}
    if not (0.0 <= alpha <= 1.0):
        return {"valid": False, "error": "require 0.0 <= alpha <= 1.0"}
    if temperature <= 0.0:
        return {"valid": False, "error": "require temperature > 0.0"}

    raw_vers = raw_spec.get("verifiers")
    if raw_vers is None:
        verifiers = ["nonempty"]
    elif isinstance(raw_vers, str):
        verifiers = [v.strip() for v in raw_vers.split(",") if v.strip()] or ["nonempty"]
    elif isinstance(raw_vers, list):
        verifiers = [str(v).strip() for v in raw_vers if str(v).strip()] or ["nonempty"]
    else:
        return {"valid": False, "error": "verifiers must be a list or comma-separated string"}
    for v in verifiers:
        if v not in BUILTIN_VERIFIERS:
            return {"valid": False, "error": f"unknown verifier {v!r}; supported: {BUILTIN_VERIFIERS}"}

    strategy = str(raw_spec.get("router_strategy", "round_robin")).strip()
    if strategy not in ("round_robin", "ensemble_best_of", "fallback"):
        return {"valid": False, "error": f"invalid router_strategy {strategy!r}"}

    tokenizer = str(raw_spec.get("tokenizer", "byte")).strip().lower()
    if tokenizer not in ("byte", "bpe", "u32"):
        return {"valid": False, "error": f"invalid tokenizer {tokenizer!r}"}

    teachers = raw_spec.get("teachers") or []
    if not isinstance(teachers, list):
        return {"valid": False, "error": "teachers must be a list"}
    trace_paths = raw_spec.get("trace_paths") or []
    if isinstance(trace_paths, str):
        trace_paths = [trace_paths]
    prompts = raw_spec.get("prompts") or []
    if isinstance(prompts, str):
        prompts = [p for p in prompts.splitlines() if p.strip()]

    spec = {
        "name": str(raw_spec.get("name") or f"distill-{mode}"),
        "mode": mode,
        "arch": arch,
        "tokenizer": tokenizer,
        "V": V,
        "T": T,
        "D": D,
        "H": H,
        "H_KV": H_KV,
        "F": F,
        "L": L,
        "B": B,
        "K": K,
        "alpha": alpha,
        "temperature": temperature,
        "steps": steps,
        "lr_micro": lr_micro,
        "seed": seed,
        "held_mod": max(2, held_mod),
        "add_layers": max(1, add_layers),
        "verifiers": verifiers,
        "regex_pattern": str(raw_spec.get("regex_pattern") or ""),
        "json_required_keys": list(raw_spec.get("json_required_keys") or []),
        "min_chars": int(raw_spec.get("min_chars", 1)),
        "router_strategy": strategy,
        "teachers": teachers,
        "trace_paths": [str(p) for p in trace_paths],
        "prompts": [str(p) for p in prompts],
        "dk_path": str(raw_spec.get("dk_path") or ""),
        "teacher_ckpt": str(raw_spec.get("teacher_ckpt") or ""),
        "tokens_bin": str(raw_spec.get("tokens_bin") or ""),
        "student_init_ck": str(raw_spec.get("student_init_ck") or ""),
        "max_seqs": int(raw_spec.get("max_seqs", 64)),
        "max_tokens": int(raw_spec.get("max_tokens", 128)),
        "skip_dcheck": bool(raw_spec.get("skip_dcheck", False)),
        "export_hf": bool(raw_spec.get("export_hf", False)),
        "hf_dtype": str(raw_spec.get("hf_dtype", "f32")),
        "export_gguf": bool(raw_spec.get("export_gguf", False)),
        "flags": raw_spec.get("flags"),
        "use_starter_fallback": bool(raw_spec.get("use_starter_fallback", True)),
        "license": str(raw_spec.get("license") or "Apache-2.0"),
        "corpus_name": str(raw_spec.get("corpus_name") or "distilled-traces"),
    }
    return {"valid": True, "error": "", "spec": spec}


class DistillRun:
    """Thread-safe state + SSE event bus for a single distillation run."""

    def __init__(self, run_id: str, spec: dict[str, Any], run_dir: Path) -> None:
        self.run_id = run_id
        self.spec = spec
        self.run_dir = run_dir
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._stop_event = threading.Event()
        self._active_proc: subprocess.Popen[bytes] | None = None

        self.status = "queued"
        self.stage = "queued"
        self.stages: list[dict[str, Any]] = []
        self.started_at = utc_now()
        self.finished_at = ""
        self.latest_loss: float | None = None
        self.latest_ce_loss: float | None = None
        self.latest_kl_loss: float | None = None
        self.latest_bpb: float | None = None
        self.metrics: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.verify_stats: dict[str, Any] = {}
        self.dcheck_receipt: dict[str, Any] | None = None
        self.bom: dict[str, Any] | None = None
        self.error = ""

    def emit(self, _kind: str, **payload: Any) -> dict[str, Any]:
        with self._cond:
            self._seq += 1
            term: str | None = None
            if _kind in ("done", "error", "stopped"):
                term = _kind
                self.finished_at = utc_now()
                if _kind == "error":
                    self.error = str(payload.get("error", "distillation failed"))
            clean_payload = dict(payload)
            if "kind" in clean_payload:
                clean_payload["artifact_kind"] = clean_payload.pop("kind")
            ev = {"seq": self._seq, "ts": utc_now(), "kind": _kind, **clean_payload}
            self._events.append(ev)
            if _kind == "stage":
                self.stage = str(clean_payload.get("stage", self.stage))
                self.stages.append({
                    "stage": self.stage,
                    "label": clean_payload.get("label", self.stage),
                    "ts": ev["ts"],
                    "status": clean_payload.get("status", "active"),
                })
            elif _kind == "verify":
                self.verify_stats = dict(clean_payload)
            elif _kind == "dcheck":
                self.dcheck_receipt = dict(clean_payload)
            elif _kind == "metric":
                self.metrics.append(clean_payload)
                if clean_payload.get("loss") is not None:
                    self.latest_loss = float(clean_payload["loss"])
                if clean_payload.get("ce_loss") is not None:
                    self.latest_ce_loss = float(clean_payload["ce_loss"])
                if clean_payload.get("kl_loss") is not None:
                    self.latest_kl_loss = float(clean_payload["kl_loss"])
                if clean_payload.get("bpb") is not None:
                    self.latest_bpb = float(clean_payload["bpb"])
            elif _kind == "artifact":
                self.artifacts.append(dict(payload))
            if term is not None:
                self.status = term
            self._cond.notify_all()
            return ev

    def iter_events(self, start_seq: int = 0, poll_timeout: float = 10.0):
        cursor = start_seq
        while True:
            with self._cond:
                while cursor >= len(self._events) and self.status in ("queued", "running"):
                    self._cond.wait(timeout=poll_timeout)
                    if cursor >= len(self._events) and self.status in ("queued", "running"):
                        yield None
                batch = self._events[cursor:]
                done = self.status not in ("queued", "running")
            for ev in batch:
                cursor = ev["seq"]
                yield ev
            if done and cursor >= len(self._events):
                break

    def request_stop(self) -> bool:
        self._stop_event.set()
        with self._lock:
            proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            return True
        return self.status in ("queued", "running")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "name": self.spec.get("name", self.run_id),
                "mode": self.spec.get("mode", "dk-univ"),
                "arch": self.spec.get("arch", "qwen3"),
                "status": self.status,
                "stage": self.stage,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "latest_loss": self.latest_loss,
                "latest_ce_loss": self.latest_ce_loss,
                "latest_kl_loss": self.latest_kl_loss,
                "verify_stats": self.verify_stats,
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "spec": self.spec,
                "status": self.status,
                "stage": self.stage,
                "stages": list(self.stages),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "latest_loss": self.latest_loss,
                "latest_ce_loss": self.latest_ce_loss,
                "latest_kl_loss": self.latest_kl_loss,
                "latest_bpb": self.latest_bpb,
                "metrics": list(self.metrics),
                "artifacts": list(self.artifacts),
                "verify_stats": self.verify_stats,
                "dcheck_receipt": self.dcheck_receipt,
                "bom": self.bom,
                "error": self.error,
                "run_dir": str(self.run_dir),
            }

    def save_state(self, override_status: str | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        if override_status is not None:
            doc["status"] = override_status
        p = self.run_dir / "distill_state.json"
        tmp = self.run_dir / ".distill_state.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def load_from_dir(cls, run_dir: Path) -> "DistillRun | None":
        state_path = run_dir / "distill_state.json"
        if not state_path.is_file():
            return None
        try:
            doc = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        run = cls(doc["run_id"], doc.get("spec", {}), run_dir)
        run.status = doc.get("status", "done")
        if run.status in ("queued", "running"):
            run.status = "interrupted"
        run.stage = doc.get("stage", "done")
        run.stages = doc.get("stages", [])
        run.started_at = doc.get("started_at", "")
        run.finished_at = doc.get("finished_at", "")
        run.latest_loss = doc.get("latest_loss")
        run.latest_ce_loss = doc.get("latest_ce_loss")
        run.latest_kl_loss = doc.get("latest_kl_loss")
        run.latest_bpb = doc.get("latest_bpb")
        run.metrics = doc.get("metrics", [])
        run.artifacts = doc.get("artifacts", [])
        run.verify_stats = doc.get("verify_stats", {})
        run.dcheck_receipt = doc.get("dcheck_receipt")
        run.bom = doc.get("bom")
        run.error = doc.get("error", "")
        return run


class DistillManager:
    """Manages distillation runs (`dk-univ`, `seq-univ`, `dna-strand`) + persistence."""

    def __init__(
        self,
        runs_dir: str | Path | None = None,
        bin_dir: str | Path | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).expanduser().resolve() if runs_dir else default_distill_runs_dir()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(bin_dir).expanduser().resolve() if bin_dir else None
        self.root = studio_catalog.repo_root(root)
        self._lock = threading.Lock()
        self._runs: dict[str, DistillRun] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.runs_dir.is_dir():
            return
        for sub in sorted(self.runs_dir.iterdir()):
            if sub.is_dir():
                r = DistillRun.load_from_dir(sub)
                if r is not None:
                    self._runs[r.run_id] = r

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return [r.summary() for r in runs]

    def get_run(self, run_id: str) -> DistillRun | None:
        with self._lock:
            r = self._runs.get(run_id)
        if r is not None:
            return r
        cand = self.runs_dir / run_id
        if cand.is_dir() and (cand / "distill_state.json").is_file():
            loaded = DistillRun.load_from_dir(cand.resolve())
            if loaded is not None:
                with self._lock:
                    self._runs[loaded.run_id] = loaded
                return loaded
        return None

    def start_run(self, raw_spec: dict[str, Any], wait: bool = False) -> DistillRun:
        est = estimate_distill(raw_spec)
        if not est["valid"]:
            raise ValueError(est["error"] or "invalid distill spec")
        spec = est["spec"]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", spec["name"]).strip("-").lower()[:24] or "distill"
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}-{slug}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "distill_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

        run = DistillRun(run_id, spec, run_dir)
        with self._lock:
            self._runs[run_id] = run
        run.save_state()

        if wait:
            self._execute_run(run)
        else:
            t = threading.Thread(target=self._execute_run, args=(run,), name=f"distill-{run_id}", daemon=True)
            t.start()
        return run

    def stop_run(self, run_id: str) -> bool:
        r = self.get_run(run_id)
        if not r:
            return False
        return r.request_stop()

    def _run_stream(
        self,
        run: DistillRun,
        argv: list[str],
        stage: str,
        line_cb: Callable[[str], None] | None = None,
    ) -> str:
        if run._stop_event.is_set():
            raise RuntimeError("distillation run stopped")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with run._lock:
            run._active_proc = proc  # type: ignore[assignment]
        lines: list[str] = []
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                clean = line.rstrip("\r\n")
                lines.append(clean)
                if clean:
                    run.emit("log", stage=stage, text=clean)
                if line_cb is not None and clean:
                    line_cb(clean)
                if run._stop_event.is_set() and proc.poll() is None:
                    proc.terminate()
            rc = proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            with run._lock:
                run._active_proc = None
        if run._stop_event.is_set():
            raise RuntimeError("distillation run stopped")
        out = "\n".join(lines)
        if rc != 0:
            tail = lines[-1] if lines else f"exit {rc}"
            raise RuntimeError(f"command {Path(argv[0]).name} exited {rc}: {tail}")
        return out

    def _build_teachers(
        self,
        spec: dict[str, Any],
        ollm_univ_bin: Path | None,
    ) -> tuple[list[TeacherAdapter], list[dict[str, Any]]]:
        adapters: list[TeacherAdapter] = []
        meta: list[dict[str, Any]] = []
        for idx, tspec in enumerate(spec.get("teachers", [])):
            if not isinstance(tspec, dict):
                continue
            kind = str(tspec.get("kind") or "openai_compat").strip().lower()
            if kind in ("openai", "openai_compat", "http"):
                base_url = str(tspec.get("base_url") or "http://127.0.0.1:18090/v1")
                model = str(tspec.get("model") or "q36-35b")
                t = OpenAICompatTeacher(
                    base_url,
                    model,
                    name=str(tspec.get("name") or f"teacher-{idx}:{model}"),
                    api_key=str(tspec.get("api_key") or ""),
                    timeout=float(tspec.get("timeout", 60.0)),
                    request_logprobs=bool(tspec.get("request_logprobs", True)),
                )
                adapters.append(t)
                meta.append({"name": t.name, "kind": "openai_compat", "base_url": t.base_url, "model": model})
            elif kind == "local_helix":
                ckpt = tspec.get("ckpt_path") or spec.get("teacher_ckpt")
                if not ckpt or ollm_univ_bin is None:
                    raise ValueError("local_helix teacher requires ckpt_path and ollm-univ binary")
                t = LocalHelixTeacher(ckpt, ollm_univ_bin, name=str(tspec.get("name") or f"helix-{idx}"))
                adapters.append(t)
                meta.append({"name": t.name, "kind": "local_helix", "ckpt_path": str(Path(ckpt).resolve())})
            elif kind == "trace_replay":
                tpaths = tspec.get("trace_paths") or spec.get("trace_paths") or []
                if not tpaths:
                    raise ValueError("trace_replay teacher requires trace_paths")
                t = TraceReplayTeacher(tpaths, name=str(tspec.get("name") or f"trace-{idx}"))
                adapters.append(t)
                meta.append({"name": t.name, "kind": "trace_replay", "trace_paths": [str(x) for x in tpaths]})
            else:
                raise ValueError(f"unsupported teacher kind {kind!r}")
        return adapters, meta

    def _collect_and_verify(
        self,
        run: DistillRun,
        ollm_univ_bin: Path | None,
    ) -> dict[str, Any]:
        spec = run.spec
        run_dir = run.run_dir
        records: list[dict[str, Any]] = []
        provenance_inputs: list[dict[str, Any]] = []

        run.emit("stage", stage="harvest", label="Harvesting & Querying Teacher Network")
        if spec["trace_paths"]:
            harv = harvest_paths(
                spec["trace_paths"],
                source_tag=spec["corpus_name"],
                license_id=spec["license"],
            )
            records.extend(harv["records"])
            provenance_inputs.extend(harv["inputs"])
            run.emit("harvest", **harv["stats"])

        adapters, teacher_meta = self._build_teachers(spec, ollm_univ_bin)
        router_summary: list[dict[str, Any]] = []
        if adapters and spec["prompts"]:
            router = TeacherRouter(
                adapters,
                strategy=spec["router_strategy"],
                verifiers=spec["verifiers"],
            )
            for p_idx, prompt_txt in enumerate(spec["prompts"]):
                msgs = [{"role": "user", "content": prompt_txt}]
                gen = router.generate_best(
                    msgs,
                    max_tokens=spec["max_tokens"],
                    temperature=spec["temperature"],
                    top_logprobs=spec["K"],
                )
                full_msgs = [
                    {"role": "user", "content": prompt_txt},
                    {"role": "assistant", "content": gen["text"]},
                ]
                rendered = f"user: {prompt_txt}\nassistant: {gen['text']}"
                p_prefix = f"user: {prompt_txt}\nassistant: ".encode("utf-8")
                tot_b = len(rendered.encode("utf-8"))
                records.append({
                    "kind": "conversation",
                    "messages": full_msgs,
                    "text": json.dumps({"messages": full_msgs}, ensure_ascii=False),
                    "rendered_text": rendered,
                    "assistant_text": gen["text"],
                    "assistant_spans": [[len(p_prefix), tot_b]],
                    "source": f"teacher:{gen.get('teacher_name') or gen.get('model')}",
                    "license": spec["license"],
                    "teacher_model": gen.get("model", ""),
                    "logprobs": gen.get("logprobs"),
                    "sha256": gen["sha256"],
                })
                run.emit("log", stage="harvest", text=f"teacher[{gen.get('teacher_name')}] prompt {p_idx + 1}/{len(spec['prompts'])} ({gen['latency_ms']} ms)")
            router_summary = router.summary()

        if not records and spec["use_starter_fallback"]:
            # Fallback to local verified starter material inside the repo
            cands = [
                self.root / "tools" / "openwaldo",
                self.root / "docs" / "DISTILLATION.md",
                self.root / "README.md",
            ]
            existing = [p for p in cands if p.exists()]
            if existing:
                harv = harvest_paths(existing[:1], source_tag="omega-helix-starter", license_id="Apache-2.0")
                records.extend(harv["records"])
                provenance_inputs.extend(harv["inputs"])
                run.emit("harvest", **harv["stats"])

        if not records:
            raise RuntimeError("no records harvested or generated from teachers")

        run.emit("stage", stage="verify", label="Running Fail-Closed Verifier Gate")
        vpipe = VerifierPipeline(
            spec["verifiers"],
            regex_pattern=spec["regex_pattern"] or None,
            json_required_keys=spec["json_required_keys"] or None,
            min_chars=spec["min_chars"],
        )
        v_res = vpipe.filter_records(records)
        run.emit("verify", **v_res["stats"])
        if not v_res["accepted"]:
            raise RuntimeError(
                f"verifier pipeline rejected all {len(records)} records (verifiers={spec['verifiers']})"
            )

        exp_dir = run_dir / "distilled_export"
        exp_meta = write_openwaldo_export(
            v_res["accepted"],
            exp_dir,
            corpus_name=spec["corpus_name"],
            license_id=spec["license"],
            source_tag=spec["corpus_name"],
        )
        run.emit("artifact", **_file_artifact(Path(exp_meta["export_json"]), "distilled_export/EXPORT.json", "openwaldo_export"))
        return {
            "accepted": v_res["accepted"],
            "verify_stats": v_res["stats"],
            "export_meta": exp_meta,
            "export_dir": exp_dir,
            "teacher_meta": teacher_meta,
            "router_summary": router_summary,
            "input_shas": provenance_inputs,
        }

    def _build_dk_from_records(
        self,
        records: list[dict[str, Any]],
        out_path: Path,
        *,
        V: int,
        K: int,
        T: int,
        max_seqs: int,
        alpha_smooth: float = 0.08,
    ) -> dict[str, Any]:
        """Convert verified traces/logprobs into dense+masked OLLMDK01 teacher windows."""
        dk_records: list[dict[str, Any]] = []
        for rec in records:
            rendered = str(rec.get("rendered_text") or rec.get("text") or "")
            raw = rendered.encode("utf-8", errors="replace")
            if len(raw) < T + 1:
                # Pad cleanly with newlines if snippet is shorter than T+1
                raw = raw + b"\n" * (T + 1 - len(raw))
            spans = rec.get("assistant_spans") or [[0, len(raw)]]
            byte_mask = bytearray(len(raw))
            for sp in spans:
                lo = max(0, int(sp[0]))
                hi = min(len(raw), int(sp[1]))
                for bi in range(lo, hi):
                    byte_mask[bi] = 1
            # Map bytes into [0, V) and walk stride-T windows
            for off in range(0, len(raw) - T, T):
                if len(dk_records) >= max_seqs:
                    break
                toks = [b % V for b in raw[off : off + T + 1]]
                # Mask tracks target positions tokens[pos + 1]
                msk = [int(byte_mask[off + p + 1]) for p in range(T)]
                if sum(msk) == 0:
                    msk[-1] = 1  # Ensure at least one supervised target position in window
                top_ids: list[int] = []
                top_probs: list[float] = []
                for p in range(T):
                    yt = toks[p + 1]
                    # Put primary target mass (1 - alpha_smooth) + K-1 smooth candidates mod V
                    ids_p = [yt]
                    cur = (yt + 1) % V
                    while len(ids_p) < K:
                        if cur not in ids_p:
                            ids_p.append(cur)
                        cur = (cur + 1) % V
                    if K == 1:
                        prs_p = [1.0]
                    else:
                        rem_p = alpha_smooth / float(K - 1)
                        prs_p = [1.0 - alpha_smooth] + [rem_p] * (K - 1)
                    top_ids.extend(ids_p)
                    top_probs.extend(prs_p)
                dk_records.append({
                    "tokens": toks,
                    "mask": msk,
                    "top_ids": top_ids,
                    "top_probs": top_probs,
                })
            if len(dk_records) >= max_seqs:
                break
        if not dk_records:
            raise RuntimeError("could not build any OLLMDK01 windows from harvested records")
        return write_ollmdk01(dk_records, out_path, V=V, K=K, T=T)

    def _run_dk_univ(self, run: DistillRun, bins: dict[str, Path]) -> dict[str, Any]:
        spec = run.spec
        run_dir = run.run_dir
        V, T, D, H, H_KV, F, L, B = (
            spec["V"], spec["T"], spec["D"], spec["H"], spec["H_KV"], spec["F"], spec["L"], spec["B"]
        )
        K = spec["K"]
        alpha_micro = int(round(spec["alpha"] * 1_000_000))
        temp_milli = max(1, int(round(spec["temperature"] * 1000)))
        univ_bin = str(bins["ollm_univ"])

        collected: dict[str, Any] | None = None
        dk_path = run_dir / "teacher.dk"
        tokens_path: Path | None = None

        if spec["dk_path"]:
            src_dk = Path(spec["dk_path"]).expanduser().resolve()
            shutil.copy2(src_dk, dk_path)
            dk_hdr = read_ollmdk01_header(dk_path)
            run.emit("log", stage="harvest", text=f"Verified external OLLMDK01 shard: N={dk_hdr['N']} K={dk_hdr['K']} FNV={dk_hdr['fnv_hex']}")
        elif spec["teacher_ckpt"]:
            run.emit("stage", stage="harvest", label="Emitting Sparse Top-K Shard from Teacher Helix Checkpoint")
            t_ck = Path(spec["teacher_ckpt"]).expanduser().resolve()
            if spec["tokens_bin"]:
                tokens_path = Path(spec["tokens_bin"]).expanduser().resolve()
            else:
                collected = self._collect_and_verify(run, bins.get("ollm_univ"))
                # Concatenate accepted texts cleanly as byte tokens % V
                tokens_path = run_dir / "teacher_tokens.bin"
                raw_chunks = [str(r.get("rendered_text") or r["text"]).encode("utf-8", errors="replace") for r in collected["accepted"]]
                raw_bytes = b"\n".join(raw_chunks)
                if len(raw_bytes) <= T + 2:
                    raw_bytes = raw_bytes * ((T + 8) // max(1, len(raw_bytes)) + 1)
                if V <= 256:
                    tokens_path.write_bytes(bytes(b % V for b in raw_bytes))
                else:
                    tokens_path.write_bytes(b"".join(struct.pack("<H", b % V) for b in raw_bytes))
            t_adapter = LocalHelixTeacher(t_ck, univ_bin)
            dk_hdr = t_adapter.emit_dk(tokens_path, dk_path, K=K, temp=spec["temperature"], max_seqs=spec["max_seqs"])
            run.emit("log", stage="harvest", text=f"dkemit wrote N={dk_hdr['N']} K={dk_hdr['K']} FNV={dk_hdr['fnv_hex']}")
        else:
            collected = self._collect_and_verify(run, bins.get("ollm_univ"))
            dk_hdr = self._build_dk_from_records(
                collected["accepted"],
                dk_path,
                V=V,
                K=K,
                T=T,
                max_seqs=spec["max_seqs"],
            )
            run.emit("log", stage="harvest", text=f"Packed OLLMDK01 from harvested traces: N={dk_hdr['N']} K={dk_hdr['K']} FNV={dk_hdr['fnv_hex']}")

        dk_art = _file_artifact(dk_path, "teacher.dk", "ollmdk01", **{k: dk_hdr[k] for k in ("V", "K", "T", "N", "fnv_hex")})
        run.emit("artifact", **dk_art)

        run.emit("stage", stage="init", label="Initializing Universal Student Checkpoint")
        student_ck = run_dir / "student.ck"
        if spec["student_init_ck"]:
            shutil.copy2(Path(spec["student_init_ck"]).expanduser().resolve(), student_ck)
        else:
            self._run_stream(
                run,
                [
                    univ_bin, "init", str(student_ck), spec["arch"],
                    str(V), str(T), str(D), str(H), str(H_KV), str(F), str(L), str(B),
                    str(spec["seed"]), str(spec["steps"]), str(spec["lr_micro"]),
                ],
                "init",
            )

        dcheck_rec: dict[str, Any] | None = None
        if not spec["skip_dcheck"]:
            run.emit("stage", stage="dcheck", label="Verifying Hybrid CE+KL Analytic Gradients (dcheck)")
            dc_out = self._run_stream(
                run,
                [univ_bin, "dcheck", str(student_ck), str(dk_path), str(alpha_micro), str(temp_milli), "24"],
                "dcheck",
            )
            for ln in dc_out.splitlines():
                parts = ln.split()
                if len(parts) == 6 and parts[0] == "C":
                    dcheck_rec = {
                        "passed": True,
                        "samples": int(parts[1]),
                        "mxrel_hex": parts[2],
                        "mxrel_milli": int(parts[3]) / 1_000_000.0,
                        "worst_param": int(parts[4]),
                        "retries": int(parts[5]),
                    }
                    run.emit("dcheck", **dcheck_rec)
                    break

        run.emit("stage", stage="distill", label=f"Distilling Student via OLLMDK01 (alpha={spec['alpha']}, tau={spec['temperature']})")
        every = max(1, min(spec["steps"], max(1, spec["steps"] // 6)))

        def _on_dtrain_line(line: str) -> None:
            parts = line.split()
            if len(parts) == 5 and parts[0] == "D":
                try:
                    st = int(parts[1])
                    tot_l = int(parts[2]) / 1e6
                    ce_l = int(parts[3]) / 1e6
                    kl_l = int(parts[4]) / 1e6
                    run.emit("metric", stage="distill", step=st, loss=tot_l, ce_loss=ce_l, kl_loss=kl_l, bpb=round(ce_l / LN2, 6))
                except ValueError:
                    pass

        self._run_stream(
            run,
            [
                univ_bin, "dtrain", str(student_ck), str(dk_path),
                str(spec["steps"]), str(every), str(alpha_micro), str(temp_milli),
            ],
            "distill",
            line_cb=_on_dtrain_line,
        )
        ck_hdr = univ_backend.read_ck4_header(student_ck)
        ck_art = _file_artifact(student_ck, "student.ck", "checkpoint_v4", magic="OLLMCK04", step=ck_hdr["step"])
        run.emit("artifact", **ck_art)

        exports = self._maybe_export_v4(run, student_ck, bins)
        return {
            "kind": "omega-helix-distill-bom",
            "schema": 1,
            "generated": utc_now(),
            "run_id": run.run_id,
            "mode": "dk-univ",
            "spec": spec,
            "dk_shard": dk_art,
            "dcheck": dcheck_rec,
            "verify_stats": collected["verify_stats"] if collected else run.verify_stats,
            "openwaldo_export": collected["export_meta"] if collected else None,
            "teachers": collected["teacher_meta"] if collected else [],
            "router_summary": collected["router_summary"] if collected else [],
            "student_checkpoint": ck_hdr,
            "final_metrics": run.metrics[-1] if run.metrics else None,
            "exports": exports,
        }

    def _run_seq_univ(self, run: DistillRun, bins: dict[str, Path]) -> dict[str, Any]:
        spec = run.spec
        run_dir = run.run_dir
        collected = self._collect_and_verify(run, bins.get("ollm_univ"))

        run.emit("stage", stage="distill", label=f"Training Universal Student ({spec['arch'].upper()})")
        backend_spec: dict[str, Any] = {
            "export_dir": str(collected["export_dir"]),
            "workdir": str(run_dir / "universal"),
            "arch": spec["arch"],
            "tokenizer": spec["tokenizer"],
            "vocab": spec["V"],
            "held_mod": spec["held_mod"],
            "T": spec["T"],
            "D": spec["D"],
            "H": spec["H"],
            "H_KV": spec["H_KV"],
            "F": spec["F"],
            "L": spec["L"],
            "B": spec["B"],
            "seed": spec["seed"],
            "steps": spec["steps"],
            "lr_micro": spec["lr_micro"],
            "export_hf": spec["export_hf"],
            "hf_dtype": spec["hf_dtype"],
            "export_gguf": spec["export_gguf"],
            "ollm_waldo": str(bins["ollm_waldo"]),
            "ollm_univ": str(bins["ollm_univ"]),
        }
        if spec["arch"] == "custom" and spec["flags"] is not None:
            backend_spec["flags"] = spec["flags"]
        if "ollm" in bins:
            backend_spec["ollm"] = str(bins["ollm"])
        if "ollm_interchange" in bins:
            backend_spec["ollm_interchange"] = str(bins["ollm_interchange"])

        rc, obs = univ_backend.run_backend(backend_spec)
        if rc != 0:
            raise RuntimeError(f"omega_helix_backend failed at stage {obs.get('stage')}: {obs.get('error')}")

        for row in obs.get("train_log", []):
            st = int(row["step"])
            lv = float(row["loss"])
            run.emit("metric", stage="distill", step=st, loss=lv, ce_loss=lv, kl_loss=0.0, bpb=round(lv / LN2, 6))

        ck_p = run_dir / "universal" / "model.ck"
        ck_hdr = univ_backend.read_ck4_header(ck_p)
        run.emit("artifact", **_file_artifact(ck_p, "universal/model.ck", "checkpoint_v4", magic="OLLMCK04", step=ck_hdr["step"]))
        return {
            "kind": "omega-helix-distill-bom",
            "schema": 1,
            "generated": utc_now(),
            "run_id": run.run_id,
            "mode": "seq-univ",
            "spec": spec,
            "verify_stats": collected["verify_stats"],
            "openwaldo_export": collected["export_meta"],
            "teachers": collected["teacher_meta"],
            "router_summary": collected["router_summary"],
            "student_checkpoint": ck_hdr,
            "observations": obs,
            "exports": obs.get("exports", []),
        }

    def _run_dna_strand(self, run: DistillRun, bins: dict[str, Path]) -> dict[str, Any]:
        spec = run.spec
        run_dir = run.run_dir
        V, T, D, H, F, L, B = spec["V"], spec["T"], spec["D"], spec["H"], spec["F"], spec["L"], spec["B"]
        collected = self._collect_and_verify(run, bins.get("ollm_univ"))

        run.emit("stage", stage="tokenize", label="Ingesting Distilled OpenWALDO Corpus via ollm-waldo")
        data_dir = run_dir / "data"
        bridge_argv = [
            sys.executable,
            str(Path(ow_bridge.__file__).resolve()),
            "prepare",
            str(collected["export_dir"]),
            str(data_dir),
            "--ollm-waldo",
            str(bins["ollm_waldo"]),
            "--held-mod",
            str(spec["held_mod"]),
        ]
        if spec["tokenizer"] == "bpe" and V > 256:
            bridge_argv += ["--ollm", str(bins["ollm"]), "--vocab", str(V)]
        self._run_stream(run, bridge_argv, "tokenize")
        train_data = data_dir / ("train.tok" if (spec["tokenizer"] == "bpe" and V > 256) else "train.bin")

        run.emit("stage", stage="base", label="Training Base DNA Checkpoint")
        base_ck = run_dir / "base.ck"
        self._run_stream(
            run,
            [
                str(bins["ollm"]), "init", str(base_ck),
                str(V), str(T), str(D), str(H), str(F), str(L), str(B),
                str(spec["seed"]), str(max(2, spec["steps"] // 2)), str(spec["lr_micro"]),
            ],
            "base",
        )
        base_steps = max(2, spec["steps"] // 2)
        self._run_stream(run, [str(bins["ollm"]), "train", str(base_ck), str(train_data), str(base_steps), str(max(1, base_steps // 2))], "base")

        run.emit("stage", stage="distill", label="Growing & Distilling Merkle DNA Strand (OLLMSP01)")
        strand_ck = run_dir / "strand_distill.ck"
        strand_sp = run_dir / "strand_distill.sp"
        new_L = L + spec["add_layers"]
        self._run_stream(
            run,
            [str(bins["ollm"]), "growpro", str(base_ck), str(strand_ck), str(new_L), str(spec["steps"]), str(spec["lr_micro"])],
            "distill",
        )
        every = max(1, min(spec["steps"], max(1, spec["steps"] // 4)))

        def _on_dna_line(line: str) -> None:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "T":
                try:
                    st = int(parts[1])
                    lv = int(parts[2]) / 1e6
                    run.emit("metric", stage="distill", step=st, loss=lv, ce_loss=lv, kl_loss=0.0, bpb=round(lv / LN2, 6))
                except ValueError:
                    pass

        self._run_stream(
            run,
            [str(bins["ollm"]), "trainf", str(strand_ck), str(train_data), str(spec["steps"]), str(every), str(L)],
            "distill",
            line_cb=_on_dna_line,
        )
        self._run_stream(run, [str(bins["ollm"]), "spext", str(base_ck), str(strand_ck), str(strand_sp)], "distill")
        sp_hdr = chain_runner.read_sp_header(strand_sp)
        run.emit("artifact", **_file_artifact(strand_ck, "strand_distill.ck", "checkpoint_dna"))
        run.emit("artifact", **_file_artifact(strand_sp, "strand_distill.sp", "specialist_pack", pack_hash=sp_hdr.get("pack_hash_hex", "")))

        return {
            "kind": "omega-helix-distill-bom",
            "schema": 1,
            "generated": utc_now(),
            "run_id": run.run_id,
            "mode": "dna-strand",
            "spec": spec,
            "verify_stats": collected["verify_stats"],
            "openwaldo_export": collected["export_meta"],
            "teachers": collected["teacher_meta"],
            "router_summary": collected["router_summary"],
            "student_checkpoint": ow_bridge.read_ollm_checkpoint_header(strand_ck),
            "specialist_pack": sp_hdr,
            "exports": [],
        }

    def _maybe_export_v4(self, run: DistillRun, ck_path: Path, bins: dict[str, Path]) -> list[dict[str, Any]]:
        spec = run.spec
        exports: list[dict[str, Any]] = []
        if not (spec["export_hf"] or spec["export_gguf"]) or "ollm_interchange" not in bins:
            return exports
        run.emit("stage", stage="export", label="Exporting Student HF Safetensors / GGUF")
        ic_bin = str(bins["ollm_interchange"])
        if spec["export_hf"]:
            hf_dir = run.run_dir / "export-hf"
            hf_dir.mkdir(parents=True, exist_ok=True)
            self._run_stream(run, [ic_bin, "export-hf", str(ck_path), str(hf_dir), spec["hf_dtype"]], "export")
            hf_files = sorted(p for p in hf_dir.iterdir() if p.is_file())
            exports.append({"format": "hf-safetensors", "dtype": spec["hf_dtype"], "dir": str(hf_dir), "files": [_file_artifact(p, p.name, "hf_export") for p in hf_files]})
        if spec["export_gguf"]:
            gguf_p = run.run_dir / "student.gguf"
            self._run_stream(run, [ic_bin, "export-gguf", str(ck_path), str(gguf_p)], "export")
            art = _file_artifact(gguf_p, "student.gguf", "gguf_export")
            run.emit("artifact", **art)
            exports.append({"format": "gguf", "version": 3, **art})
        return exports

    def _execute_run(self, run: DistillRun) -> None:
        with run._lock:
            run.status = "running"
        run.save_state()
        try:
            spec = run.spec
            mode = spec["mode"]
            run.emit("stage", stage="resolve", label="Resolving Pure-Omega Distillation Binaries")
            needed: list[str] = ["ollm_univ", "ollm_waldo"]
            if mode == "dna-strand" or spec["tokenizer"] == "bpe":
                needed.append("ollm")
            if spec["export_hf"] or spec["export_gguf"]:
                needed.append("ollm_interchange")
            bins = chain_runner.ensure_binaries(
                needed,
                bin_dir=self.bin_dir,
                root=self.root,
                log_cb=lambda msg: run.emit("log", stage="resolve", text=msg),
            )
            if mode == "dk-univ":
                bom = self._run_dk_univ(run, bins)
            elif mode == "seq-univ":
                bom = self._run_seq_univ(run, bins)
            else:
                bom = self._run_dna_strand(run, bins)

            bom_path = run.run_dir / "distill-bom.json"
            tmp = run.run_dir / ".distill-bom.json.tmp"
            tmp.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, bom_path)
            run.bom = bom
            bom_art = _file_artifact(bom_path, "distill-bom.json", "distill_bom")
            run.emit("artifact", **bom_art)
            run.emit("done", status="done", bom_sha256=bom_art["sha256"])
        except Exception as exc:
            kind = "stopped" if run._stop_event.is_set() else "error"
            run.emit(kind, error=str(exc))
        finally:
            run.save_state()
