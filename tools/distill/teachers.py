"""Multi-teacher discovery, OpenAI-compatible HTTP client, local Helix & trace teachers.

Supports localhost serving stacks (omega-router :18090, q36-serve :18091, ollm CPU :18098,
Ollama :11434, vLLM :8000, llama.cpp :8080, LM Studio :1234), local pure-Omega Helix
checkpoints (`ollm-univ sample` / `dkemit`), and offline JSONL trace replay.

Python 3 standard library only — zero third-party deps, zero telemetry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .harvest import harvest_paths, read_ollmdk01_header
from .verifiers import verify_record

DEFAULT_LOCAL_ENDPOINTS: list[dict[str, str]] = [
    {"name": "omega-router", "base_url": "http://127.0.0.1:18090/v1"},
    {"name": "q36-serve", "base_url": "http://127.0.0.1:18091/v1"},
    {"name": "ollm-cpu", "base_url": "http://127.0.0.1:18098/v1"},
    {"name": "ollama", "base_url": "http://127.0.0.1:11434/v1"},
    {"name": "vllm", "base_url": "http://127.0.0.1:8000/v1"},
    {"name": "llama-cpp", "base_url": "http://127.0.0.1:8080/v1"},
    {"name": "lm-studio", "base_url": "http://127.0.0.1:1234/v1"},
]


def _extract_choice_text(choice: dict[str, Any]) -> str:
    msg = choice.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _extract_logprobs(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None
    content = lp.get("content")
    if not isinstance(content, list) or not content:
        return None
    out: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        top = item.get("top_logprobs")
        tops: list[dict[str, Any]] = []
        if isinstance(top, list):
            for t in top:
                if isinstance(t, dict) and "token" in t and "logprob" in t:
                    tops.append({"token": str(t["token"]), "logprob": float(t["logprob"])})
        out.append({
            "token": str(item.get("token", "")),
            "logprob": float(item.get("logprob", 0.0)),
            "top_logprobs": tops,
        })
    return out or None


def probe_local_teachers(
    endpoints: list[dict[str, str]] | None = None,
    timeout: float = 0.6,
) -> list[dict[str, Any]]:
    """Fast non-blocking probe of localhost OpenAI-compatible open-source model servers."""
    targets = endpoints or DEFAULT_LOCAL_ENDPOINTS
    results: list[dict[str, Any]] = []
    for ep in targets:
        name = str(ep.get("name") or ep.get("base_url") or "local")
        base_url = str(ep["base_url"]).rstrip("/")
        url = f"{base_url}/models"
        t0 = time.monotonic()
        reachable = False
        models: list[str] = []
        err_msg = ""
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                reachable = 200 <= resp.status < 300
                doc = json.loads(raw) if raw else {}
                data = doc.get("data") if isinstance(doc, dict) else None
                if isinstance(data, list):
                    models = [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
        except urllib.error.HTTPError as exc:
            err_msg = f"HTTP {exc.code}"
        except Exception as exc:
            err_msg = str(exc)
        results.append({
            "name": name,
            "base_url": base_url,
            "reachable": reachable,
            "models": models,
            "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
            "error": err_msg if not reachable else "",
        })
    return results


class TeacherAdapter:
    name: str = "teacher"
    provider: str = "base"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatTeacher(TeacherAdapter):
    """Pure stdlib client for any OpenAI-compatible teacher (/v1/chat/completions)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        name: str = "",
        api_key: str = "",
        timeout: float = 60.0,
        request_logprobs: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model = model
        self.name = name or f"openai:{model}"
        self.provider = "openai_compat"
        self.api_key = api_key
        self.timeout = float(timeout)
        self.request_logprobs = bool(request_logprobs)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if self.request_logprobs and top_logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = min(20, int(top_logprobs))

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
        doc = json.loads(raw)
        choices = doc.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"{self.base_url}: response missing choices")
        c0 = choices[0]
        return {
            "text": _extract_choice_text(c0),
            "model": str(doc.get("model") or self.model),
            "provider": self.provider,
            "teacher_name": self.name,
            "logprobs": _extract_logprobs(c0),
            "latency_ms": latency_ms,
        }


class LocalHelixTeacher(TeacherAdapter):
    """Wraps a local pure-Omega `ollm-univ` checkpoint for text sampling and `dkemit`."""

    def __init__(
        self,
        ckpt_path: str | Path,
        ollm_univ_bin: str | Path,
        *,
        name: str = "",
        seed: int = 42,
    ) -> None:
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.ollm_univ_bin = Path(ollm_univ_bin).expanduser().resolve()
        self.name = name or f"helix:{self.ckpt_path.name}"
        self.provider = "local_helix"
        self.seed = int(seed)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.8,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        prompt_text = "\n".join(m.get("content", "") for m in messages if m.get("content")).strip()
        raw_prompt = prompt_text.encode("utf-8")[:64] or b"hello"
        # Seeded u8 sample using hex/token IDs accepted by ollm-univ sample
        tok_ids = [str(b) for b in raw_prompt[:8]]
        temp_milli = max(1, int(round(temperature * 1000)))
        top_k = max(0, int(top_logprobs))
        t0 = time.monotonic()
        res = subprocess.run(
            [
                str(self.ollm_univ_bin),
                "sample",
                str(self.ckpt_path),
                str(max(1, int(max_tokens))),
                str(temp_milli),
                str(top_k),
                str(self.seed),
                *tok_ids,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
        if res.returncode != 0:
            raise RuntimeError(f"ollm-univ sample failed (rc={res.returncode}): {res.stderr.strip()}")
        gen_toks: list[int] = []
        for ln in res.stdout.splitlines():
            parts = ln.split()
            if len(parts) >= 2 and parts[0] == "S":
                gen_toks = [int(x) for x in parts[1:]]
                break
        decoded = bytes(t & 0xFF for t in gen_toks).decode("utf-8", errors="replace")
        return {
            "text": decoded,
            "tokens": gen_toks,
            "model": self.ckpt_path.name,
            "provider": self.provider,
            "teacher_name": self.name,
            "logprobs": None,
            "latency_ms": latency_ms,
        }

    def emit_dk(
        self,
        tokens_bin: str | Path,
        out_dk: str | Path,
        *,
        K: int = 8,
        temp: float = 1.5,
        max_seqs: int | None = None,
    ) -> dict[str, Any]:
        temp_milli = max(1, int(round(float(temp) * 1000)))
        argv = [
            str(self.ollm_univ_bin),
            "dkemit",
            str(self.ckpt_path),
            str(Path(tokens_bin).expanduser().resolve()),
            str(Path(out_dk).expanduser().resolve()),
            str(int(K)),
            str(temp_milli),
        ]
        if max_seqs is not None and int(max_seqs) > 0:
            argv.append(str(int(max_seqs)))
        res = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ollm-univ dkemit failed (rc={res.returncode}): {res.stderr.strip()}")
        return read_ollmdk01_header(out_dk)


class TraceReplayTeacher(TeacherAdapter):
    """Offline zero-network teacher replaying recorded JSONL/OpenWALDO assistant traces."""

    def __init__(self, trace_paths: list[str | Path], *, name: str = "trace-replay") -> None:
        self.trace_paths = [Path(p).expanduser().resolve() for p in trace_paths]
        self.name = name
        self.provider = "trace_replay"
        harvested = harvest_paths(self.trace_paths)
        self.records = harvested["records"]
        self._cursor = 0
        self._by_prompt: dict[str, dict[str, Any]] = {}
        for rec in self.records:
            msgs = rec.get("messages") or []
            user_txt = "\n".join(m["content"] for m in msgs if m["role"] == "user").strip()
            if user_txt:
                self._by_prompt.setdefault(user_txt, rec)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        if not self.records:
            raise RuntimeError("TraceReplayTeacher has zero records")
        t0 = time.monotonic()
        user_txt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()
        rec = self._by_prompt.get(user_txt)
        if rec is None:
            rec = self.records[self._cursor % len(self.records)]
            self._cursor += 1
        return {
            "text": str(rec.get("assistant_text") or rec["text"]),
            "model": str(rec.get("teacher_model") or self.name),
            "provider": self.provider,
            "teacher_name": self.name,
            "logprobs": rec.get("logprobs"),
            "latency_ms": round((time.monotonic() - t0) * 1000.0, 2),
        }


class TeacherRouter:
    """Multi-teacher routing (`round_robin`, `ensemble_best_of`, `fallback`) + telemetry."""

    def __init__(
        self,
        teachers: list[TeacherAdapter],
        *,
        strategy: str = "round_robin",
        verifiers: list[str] | tuple[str, ...] = ("nonempty",),
    ) -> None:
        if not teachers:
            raise ValueError("TeacherRouter requires at least one teacher")
        if strategy not in ("round_robin", "ensemble_best_of", "fallback"):
            raise ValueError(f"unsupported router strategy {strategy!r}")
        self.teachers = list(teachers)
        self.strategy = strategy
        self.verifiers = list(verifiers)
        self._idx = 0
        self.stats: dict[str, dict[str, Any]] = {
            t.name: {"calls": 0, "passes": 0, "errors": 0, "total_latency_ms": 0.0}
            for t in self.teachers
        }

    def generate_best(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        if self.strategy == "round_robin":
            teacher = self.teachers[self._idx % len(self.teachers)]
            self._idx += 1
            return self._call_and_score(teacher, messages, max_tokens=max_tokens, temperature=temperature, top_logprobs=top_logprobs)

        if self.strategy == "fallback":
            last_err: Exception | None = None
            for teacher in self.teachers:
                try:
                    out = self._call_and_score(teacher, messages, max_tokens=max_tokens, temperature=temperature, top_logprobs=top_logprobs)
                    if out["verification"]["passed"]:
                        return out
                    last_out = out
                except Exception as exc:
                    last_err = exc
                    continue
            if "last_out" in locals():
                return last_out
            raise RuntimeError(f"all fallback teachers failed: {last_err}")

        # ensemble_best_of: query every teacher and select highest verification score (then latency)
        candidates: list[dict[str, Any]] = []
        last_err = None
        for teacher in self.teachers:
            try:
                candidates.append(
                    self._call_and_score(teacher, messages, max_tokens=max_tokens, temperature=temperature, top_logprobs=top_logprobs)
                )
            except Exception as exc:
                last_err = exc
        if not candidates:
            raise RuntimeError(f"all ensemble teachers failed: {last_err}")
        candidates.sort(key=lambda c: (-float(c["verification"]["score"]), float(c.get("latency_ms", 0.0))))
        best = dict(candidates[0])
        best["ensemble_candidates"] = len(candidates)
        return best

    def _call_and_score(
        self,
        teacher: TeacherAdapter,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
    ) -> dict[str, Any]:
        st = self.stats[teacher.name]
        st["calls"] += 1
        try:
            res = teacher.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=top_logprobs,
            )
        except Exception:
            st["errors"] += 1
            raise
        st["total_latency_ms"] += float(res.get("latency_ms", 0.0))
        ver = verify_record(res.get("text", ""), self.verifiers)
        if ver["passed"]:
            st["passes"] += 1
        out = dict(res)
        out["verification"] = ver
        out["sha256"] = hashlib.sha256(str(res.get("text", "")).encode("utf-8")).hexdigest()
        return out

    def summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in self.teachers:
            st = self.stats[t.name]
            calls = st["calls"]
            out.append({
                "name": t.name,
                "provider": t.provider,
                "calls": calls,
                "passes": st["passes"],
                "errors": st["errors"],
                "avg_latency_ms": round(st["total_latency_ms"] / calls, 2) if calls else 0.0,
            })
        return out
