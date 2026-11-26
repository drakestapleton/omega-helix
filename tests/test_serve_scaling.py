#!/usr/bin/env python3
"""
Workstream 3 regression suite for native/serve.om:
- Continuous-batching multi-slot concurrency (zero spurious 503s, true saturation 503)
- Shared-trunk KV prefix caching across base, single strands, and spclip S=2 strands
  with bit-exact cold/offline parity, TTFT speedup, collision rejection, and LRU eviction
- 8K-32K long-context serving, generous default output limits (>= 4096), clean 400 refusal
- OpenAI feature preservation & disconnect cleanup
- Live GB10 GPU gserve concurrency & 8K verification (when OLLM_GB10_LIVE=1)
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
RESERVED_PORTS = {3080, 18090, 18091, 18097, 18098}


def _pick_safe_port() -> int:
    env_port = os.environ.get("OLLM_SERVE_PORT", "").strip()
    if env_port:
        p = int(env_port)
        if p in RESERVED_PORTS:
            raise RuntimeError(f"Refusing reserved Spark port {p}")
        return p
    for _ in range(64):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        if port not in RESERVED_PORTS:
            return port
    raise RuntimeError("Could not allocate safe ephemeral port")


def _wait_port_free(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        finally:
            s.close()
        time.sleep(0.05)


def _run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _find_omc() -> str:
    if os.environ.get("OMC"):
        return os.environ["OMC"]
    local = Path.home() / ".local" / "bin" / "omc"
    if local.exists():
        return str(local)
    spark_canon = Path.home() / "canon" / "omega-mv" / "seed" / "out" / "omc"
    if spark_canon.exists():
        return str(spark_canon)
    found = shutil.which("omc")
    if found:
        return found
    raise RuntimeError("Could not locate omc compiler")


class ManagedServer:
    def __init__(self, args: list[str], port: int) -> None:
        self.args = args
        self.port = port
        self.proc: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._t_out: threading.Thread | None = None
        self._t_err: threading.Thread | None = None
        self._ready = threading.Event()
        self._failed = False

    def __enter__(self) -> "ManagedServer":
        _wait_port_free(self.port)
        self.proc = subprocess.Popen(
            self.args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def read_out() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            try:
                for line in self.proc.stdout:
                    s = line.rstrip("\r\n")
                    self.lines.append(s)
                    if s.startswith("PORT "):
                        self._ready.set()
            finally:
                self._ready.set()

        def read_err() -> None:
            assert self.proc is not None and self.proc.stderr is not None
            for line in self.proc.stderr:
                self.stderr_lines.append(line.rstrip("\r\n"))

        self._t_out = threading.Thread(target=read_out, daemon=True)
        self._t_err = threading.Thread(target=read_err, daemon=True)
        self._t_out.start()
        self._t_err.start()

        if not self._ready.wait(timeout=45.0):
            self.close()
            raise RuntimeError(f"Server startup timeout: out={self.lines} err={self.stderr_lines}")
        if self.proc.poll() is not None and not any(l.startswith("PORT ") for l in self.lines):
            rc = self.proc.returncode
            self.close()
            raise RuntimeError(f"Server exited early rc={rc}: out={self.lines} err={self.stderr_lines}")
        return self

    def assert_alive(self) -> None:
        assert self.proc is not None
        rc = self.proc.poll()
        if rc is not None:
            raise RuntimeError(f"Server crashed unexpectedly with rc={rc}: out={self.lines[-10:]} err={self.stderr_lines[-10:]}")

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5.0)
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc = None
        if self._t_out is not None:
            self._t_out.join(timeout=2.0)
        if self._t_err is not None:
            self._t_err.join(timeout=2.0)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _http_json(
    port: int,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, str], dict, float]:
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers: dict[str, str] = {}
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        dt_ms = (time.monotonic() - t0) * 1000.0
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        data = json.loads(raw) if raw.strip() else {}
        return resp.status, hdrs, data, dt_ms
    finally:
        conn.close()


def _http_sse(
    port: int,
    path: str,
    payload: dict,
    timeout: float = 120.0,
) -> tuple[int, float, list[tuple[float, dict]], dict | None]:
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        chunks: list[tuple[float, dict]] = []
        usage: dict | None = None
        ttft_ms = 0.0
        if status != 200:
            raw = resp.read().decode("utf-8", errors="replace")
            return status, (time.monotonic() - t0) * 1000.0, [(0.0, {"error_body": raw})], None
        while True:
            line_b = resp.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:].strip()
            if item == "[DONE]":
                break
            obj = json.loads(item)
            now = time.monotonic()
            if ttft_ms == 0.0:
                ch = obj.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    ttft_ms = (now - t0) * 1000.0
            if "usage" in obj and obj["usage"] is not None:
                usage = obj["usage"]
            chunks.append((now, obj))
        if ttft_ms == 0.0:
            ttft_ms = (time.monotonic() - t0) * 1000.0
        return status, ttft_ms, chunks, usage
    finally:
        conn.close()


def _sse_text(chunks: list[tuple[float, dict]]) -> str:
    parts: list[str] = []
    for _, obj in chunks:
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            c = d.get("content")
            if isinstance(c, str):
                parts.append(c)
    return "".join(parts)


def _check_static_invariants() -> None:
    serve_path = ROOT / "native" / "serve.om"
    text = serve_path.read_text(encoding="utf-8")
    nlines = len(text.splitlines())
    if nlines >= 8192:
        raise AssertionError(f"native/serve.om has {nlines} lines (must be < 8192)")
    for mut in ("OMUTMETH", "OMUTPATH", "OMUTBIG", "OMUTJSON", "OMUTLOOP", "OMUTSEED", "OMUTMID"):
        if f"const {mut} ulong 1" not in text:
            raise AssertionError(f"Missing mutant guard {mut} in native/serve.om")


def _build_target(target: str, work: Path, omc: str) -> Path:
    build_out = work / f"bin_{target}"
    build_out.mkdir(parents=True, exist_ok=True)
    res = _run_cmd(
        ["sh", str(ROOT / "scripts" / "build.sh")],
        env={"TARGET": target, "OMC": omc, "OUT": str(build_out)},
        timeout=180.0,
    )
    if res.returncode != 0:
        raise RuntimeError(f"build.sh TARGET={target} failed rc={res.returncode}:\n{res.stdout}\n{res.stderr}")
    name = "ollm-gb10" if target == "gb10" else ("ollm-macos" if target == "macos" else f"ollm-{target}" if target != "aarch64" else "ollm")
    bin_path = build_out / name
    if not bin_path.exists():
        raise RuntimeError(f"Missing expected binary {bin_path}")
    return bin_path


def _bpe_token_count(ollm_bin: Path, merges: Path, v: int, text: str, scratch: Path) -> int:
    txt = scratch / "tok_check.txt"
    toks = scratch / "tok_check.toks"
    txt.write_text(text, encoding="utf-8")
    r = _run_cmd([str(ollm_bin), "bpeenc", str(merges), str(v), str(txt), str(toks)])
    if r.returncode != 0:
        raise RuntimeError(f"bpeenc failed rc={r.returncode}: {r.stdout} {r.stderr}")
    return toks.stat().st_size // 2


def _make_prompt_with_tokens(ollm_bin: Path, merges: Path, v: int, unit: str, target_toks: int, scratch: Path) -> tuple[str, int]:
    unit_cnt = _bpe_token_count(ollm_bin, merges, v, unit * 8, scratch)
    toks_per_rep = max(0.1, unit_cnt / 8.0)
    reps = max(1, int(target_toks / toks_per_rep))
    s = unit * reps
    c = _bpe_token_count(ollm_bin, merges, v, s, scratch)
    for _ in range(12):
        if c < target_toks - 10:
            need = max(1, int((target_toks - 4 - c) / toks_per_rep))
            s += unit * need
            c = _bpe_token_count(ollm_bin, merges, v, s, scratch)
        elif c > target_toks:
            over = max(1, int((c - (target_toks - 4)) / toks_per_rep))
            cur_reps = len(s) // len(unit)
            if cur_reps <= over:
                break
            s = unit * (cur_reps - over)
            c = _bpe_token_count(ollm_bin, merges, v, s, scratch)
        else:
            break
    return s, c


def _setup_models(ollm_bin: Path, work: Path) -> dict[str, Path]:
    corpus0 = work / "corpus0.txt"
    corpus1 = work / "corpus1.txt"
    base_txt = (
        "sovereign verified deterministic compute builds clean reproducible kernels across every single token step.\n"
        "quickly jumping wizards vex bold nymphs fighting over extra lazy brown dogs and quiet quartz stones.\n"
        * 40
    )
    s0_txt = "alpha specialist strand focuses sharply on symbolic proof verification and theorem bounds.\n" * 40
    s1_txt = "beta specialist strand routes hardware telemetry and memory bounds without allocation drift.\n" * 40
    corpus0.write_text(base_txt + s0_txt, encoding="utf-8")
    corpus1.write_text(s1_txt + base_txt, encoding="utf-8")

    merges = work / "merges.bin"
    toks0 = work / "toks0.bin"
    toks1 = work / "toks1.bin"
    v_fit = 320
    v = 300
    for cmd in (
        [str(ollm_bin), "bpefit", str(corpus0), str(merges), str(v_fit)],
        [str(ollm_bin), "bpeenc", str(merges), str(v), str(corpus0), str(toks0)],
        [str(ollm_bin), "bpeenc", str(merges), str(v), str(corpus1), str(toks1)],
    ):
        r = _run_cmd(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd} failed rc={r.returncode}: {r.stdout} {r.stderr}")

    base_ck = work / "base.ck"
    spec0_ck = work / "spec0.ck"
    spec1_ck = work / "spec1.ck"
    s0_sp = work / "s0.sp"
    s1_sp = work / "s1.sp"
    long_ck = work / "long.ck"
    for cmd in (
        [str(ollm_bin), "init", str(base_ck), str(v), "64", "16", "2", "32", "4", "1", "11", "4", "10000"],
        [str(ollm_bin), "train", str(base_ck), str(toks0), "2", "2"],
        [str(ollm_bin), "grow", str(base_ck), str(spec0_ck), "5", "6", "10000"],
        [str(ollm_bin), "trainf", str(spec0_ck), str(toks0), "4", "2", "4"],
        [str(ollm_bin), "spext", str(base_ck), str(spec0_ck), str(s0_sp)],
        [str(ollm_bin), "grow", str(base_ck), str(spec1_ck), "5", "6", "10000"],
        [str(ollm_bin), "trainf", str(spec1_ck), str(toks1), "4", "2", "4"],
        [str(ollm_bin), "spext", str(base_ck), str(spec1_ck), str(s1_sp)],
        [str(ollm_bin), "init", str(long_ck), str(v), "32", "8", "1", "16", "1", "1", "7", "2", "10000"],
    ):
        r = _run_cmd(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd} failed rc={r.returncode}: {r.stdout} {r.stderr}")

    return {
        "merges": merges,
        "base": base_ck,
        "s0": s0_sp,
        "s1": s1_sp,
        "long": long_ck,
    }


def _suite_a_concurrency(ollm_bin: Path, models: dict[str, Path], port: int) -> dict[str, object]:
    metrics: dict[str, object] = {}
    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(models["base"]), str(models["merges"]), "slots=8", "ctx=8192", "test=1"],
        port,
    ) as srv:
        results: list[tuple[int, int, str, float, list[tuple[float, dict]], dict | None]] = [None] * 6  # type: ignore

        def worker(idx: int) -> None:
            if idx < 2:
                payload = {
                    "messages": [{"role": "user", "content": f"stream request {idx}"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "ollm_test_output": "alpha beta gamma delta epsilon ",
                    "ollm_test_delay_ms": 40,
                    "temperature": 0,
                }
                st, _, ch, us = _http_sse(port, "/v1/chat/completions", payload)
                results[idx] = (idx, st, _sse_text(ch), 0.0, ch, us)
            elif idx < 4:
                payload = {
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": f"turn1 {idx}"},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": f"turn2 {idx}"},
                    ],
                    "ollm_test_output": f"reply-{idx}",
                    "ollm_test_delay_ms": 45,
                    "temperature": 0,
                }
                st, _, data, dt = _http_json(port, "POST", "/v1/chat/completions", payload)
                txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                results[idx] = (idx, st, txt, dt, [], data.get("usage"))
            else:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"part1-{idx} "},
                                {"type": "text", "text": "part2"},
                            ],
                        }
                    ],
                    "ollm_test_output": f"parts-{idx}",
                    "ollm_test_delay_ms": 45,
                    "temperature": 0,
                }
                st, _, data, dt = _http_json(port, "POST", "/v1/chat/completions", payload)
                txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                results[idx] = (idx, st, txt, dt, [], data.get("usage"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()

        max_inflight = 0
        seen_max_slots = 0
        for _ in range(30):
            try:
                st_h, _, hdata, _ = _http_json(port, "GET", "/health", timeout=5.0)
                if st_h == 200:
                    max_inflight = max(max_inflight, int(hdata.get("in_flight", 0)))
                    seen_max_slots = int(hdata.get("max_slots", 0))
            except Exception:
                pass
            if all(not t.is_alive() for t in threads):
                break
            time.sleep(0.02)

        for t in threads:
            t.join(timeout=30.0)
            if t.is_alive():
                raise AssertionError("Concurrent worker thread timed out")
        srv.assert_alive()

        statuses = [r[1] for r in results]
        if any(s != 200 for s in statuses):
            raise AssertionError(f"Expected 6 HTTP 200 responses, got {statuses}")
        if seen_max_slots != 8 or max_inflight < 4:
            raise AssertionError(f"Expected max_slots=8 & in_flight>=4 mid-flight, got max_slots={seen_max_slots} max_inflight={max_inflight}")
        c0_times = [ts for ts, obj in results[0][4] if ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")]
        c1_times = [ts for ts, obj in results[1][4] if ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")]
        if not c0_times or not c1_times or max(c0_times[0], c1_times[0]) >= min(c0_times[-1], c1_times[-1]):
            raise AssertionError(f"SSE streams were not interleaved: c0={c0_times} c1={c1_times}")
        if not results[0][5] or not results[1][5]:
            raise AssertionError("Missing stream_options.include_usage payload")
        metrics["concurrent_6_503s"] = 0
        metrics["max_in_flight_seen"] = max_inflight

    # Saturation 503 check with slots=2
    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(models["base"]), str(models["merges"]), "slots=2", "ctx=8192", "test=1"],
        port,
    ) as srv:
        res_sat: list[tuple[int, dict[str, str], dict]] = []

        def hold(i: int) -> None:
            st, _, data, _ = _http_json(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": f"hold-{i}"}],
                    "ollm_test_output": "abcdef",
                    "ollm_test_delay_ms": 120,
                    "temperature": 0,
                },
            )
            res_sat.append((st, {}, data))

        t1 = threading.Thread(target=hold, args=(1,))
        t2 = threading.Thread(target=hold, args=(2,))
        t1.start()
        t2.start()

        busy_seen = False
        for _ in range(30):
            _, _, hdata, _ = _http_json(port, "GET", "/health")
            if int(hdata.get("in_flight", 0)) == 2 and hdata.get("busy") is True:
                busy_seen = True
                break
            time.sleep(0.015)
        if not busy_seen:
            raise AssertionError("Did not observe in_flight=2 busy=true on slots=2")

        st3, hdr3, data3, _ = _http_json(
            port,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "overflow"}], "temperature": 0},
        )
        if st3 != 503 or hdr3.get("retry-after") != "1" or (data3.get("error") or {}).get("code") != "server_busy":
            raise AssertionError(f"Expected 503 server_busy with Retry-After:1, got st={st3} hdr={hdr3} data={data3}")

        t1.join(timeout=20.0)
        t2.join(timeout=20.0)
        srv.assert_alive()
        _, _, h_after, _ = _http_json(port, "GET", "/health")
        if int(h_after.get("in_flight", -1)) != 0 or h_after.get("busy") is not False:
            raise AssertionError(f"Slots leaked after completion: {h_after}")
        metrics["saturation_503_ok"] = True
    return metrics


def _suite_b_prefix_cache_and_strands(ollm_bin: Path, models: dict[str, Path], work: Path, port: int) -> dict[str, object]:
    merges = models["merges"]
    shared_prefix, prefix_toks = _make_prompt_with_tokens(
        ollm_bin, merges, 300, "sovereign deterministic compute ", 528, work
    )
    if prefix_toks < 512:
        raise AssertionError(f"Prefix too short: {prefix_toks}")

    metrics: dict[str, object] = {"shared_prefix_tokens": prefix_toks}
    with ManagedServer(
        [
            str(ollm_bin),
            "serve",
            str(port),
            str(models["base"]),
            str(merges),
            "slots=8",
            "ctx=8192",
            f"strand={models['s0']}",
            f"strand={models['s1']}",
        ],
        port,
    ) as srv:
        st_m, _, mdata, _ = _http_json(port, "GET", "/v1/models")
        if st_m != 200:
            raise AssertionError(f"/v1/models failed: {st_m}")
        model_ids = [d.get("id") for d in mdata.get("data") or []]
        for req_id in ("strand:0", "strand:1", "spclip"):
            if req_id not in model_ids:
                raise AssertionError(f"Missing {req_id} in /v1/models: {model_ids}")

        # Cold uncached baselines (cache_prefix: false) for base, strand:0, strand:1, spclip
        cold_outputs: dict[str, str] = {}
        cold_times: dict[str, float] = {}
        configs = [
            ("base", {"strand": "base"}),
            ("strand0", {"strand": 0}),
            ("strand1", {"strand": 1}),
            ("spclip", {"strand": "spclip", "spclip_mode": 1, "spclip_weights": [1000, 1000]}),
        ]
        for name, extra in configs:
            req = {
                "prompt": shared_prefix + f" query-{name}:",
                "max_tokens": 12,
                "temperature": 0,
                "seed": 42,
                "cache_prefix": False,
            }
            req.update(extra)
            st, _, data, dt = _http_json(port, "POST", "/v1/completions", req)
            if st != 200:
                raise AssertionError(f"Cold {name} failed st={st}: {data}")
            ptd = (data.get("usage") or {}).get("prompt_tokens_details") or {}
            if int(ptd.get("cached_tokens", -1)) != 0:
                raise AssertionError(f"cache_prefix=false still reported cached_tokens={ptd}")
            cold_outputs[name] = ((data.get("choices") or [{}])[0]).get("text", "")
            cold_times[name] = dt

        # Warm cached sequence: first base populates trunk KV for shared_prefix; subsequent base, strand:0, strand:1, spclip hit!
        warm_outputs: dict[str, str] = {}
        warm_times: dict[str, float] = {}
        # Populate with exact shared prefix + base suffix
        req_pop = {"prompt": shared_prefix + " seed:", "max_tokens": 8, "temperature": 0, "seed": 42, "strand": "base"}
        st_p, ttft_cold_sse, _, us_p = _http_sse(
            port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": shared_prefix}],
                "max_tokens": 8,
                "temperature": 0,
                "seed": 42,
                "stream": True,
                "stream_options": {"include_usage": True},
                "strand": "base",
            },
        )
        if st_p != 200 or not us_p:
            raise AssertionError(f"SSE populate failed: {st_p} {us_p}")
        st_pw, ttft_warm_sse, _, us_pw = _http_sse(
            port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": shared_prefix}],
                "max_tokens": 8,
                "temperature": 0,
                "seed": 42,
                "stream": True,
                "stream_options": {"include_usage": True},
                "strand": 0,
            },
        )
        if st_pw != 200 or not us_pw:
            raise AssertionError(f"SSE warm hit failed: {st_pw} {us_pw}")
        sse_cached = int((us_pw.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        if sse_cached < 512:
            raise AssertionError(f"SSE cross-strand warm hit cached_tokens={sse_cached} < 512")

        # Also test non-streaming /v1/completions cross-strand prefix cache populate + hits + bit-exact parity
        st_b0, _, db0, dt_b0 = _http_json(port, "POST", "/v1/completions", {"prompt": shared_prefix + " query-base:", "max_tokens": 12, "temperature": 0, "seed": 42, "strand": "base"})
        if st_b0 != 200:
            raise AssertionError(f"base warm/pop failed: {st_b0}")
        for name, extra in configs:
            req = {
                "prompt": shared_prefix + f" query-{name}:",
                "max_tokens": 12,
                "temperature": 0,
                "seed": 42,
            }
            req.update(extra)
            st, _, data, dt = _http_json(port, "POST", "/v1/completions", req)
            if st != 200:
                raise AssertionError(f"Warm {name} failed: {st} {data}")
            cached_toks = int(((data.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            if cached_toks < 512:
                raise AssertionError(f"Expected cached_tokens >= 512 for {name}, got {cached_toks}")
            txt = ((data.get("choices") or [{}])[0]).get("text", "")
            if txt != cold_outputs[name]:
                raise AssertionError(f"Parity mismatch on {name}: cached={txt!r} != cold={cold_outputs[name]!r}")
            warm_outputs[name] = txt
            warm_times[name] = dt

        if not (ttft_warm_sse < ttft_cold_sse):
            raise AssertionError(f"Expected warm SSE TTFT < cold SSE TTFT: warm={ttft_warm_sse:.2f}ms cold={ttft_cold_sse:.2f}ms")

        _, _, hdata, _ = _http_json(port, "GET", "/health")
        kvc = hdata.get("kv_cache") or {}
        if int(kvc.get("hits", 0)) < 3 or int(kvc.get("tokens_saved", 0)) < 1536:
            raise AssertionError(f"Unexpected /health kv_cache stats: {kvc}")

        # Bit-exactness vs offline CLI spsamp & spclip on <= T_ck prompt (single chunk <= 64 tokens)
        short_padded = "sovereign deterministic compute builds clean "
        _, _, ds0_a, _ = _http_json(port, "POST", "/v1/completions", {"prompt": short_padded, "max_tokens": 8, "temperature": 0, "seed": 42, "strand": 0})
        _, _, ds0_b, _ = _http_json(port, "POST", "/v1/completions", {"prompt": short_padded, "max_tokens": 8, "temperature": 0, "seed": 42, "strand": 0})
        _, _, dsc_b, _ = _http_json(port, "POST", "/v1/completions", {"prompt": short_padded, "max_tokens": 8, "temperature": 0, "seed": 42, "strand": "spclip", "spclip_mode": 1, "spclip_weights": [1000, 1000]})
        if ds0_a["choices"][0]["text"] != ds0_b["choices"][0]["text"]:
            raise AssertionError("Short prompt cold vs cached parity failed for strand 0")
        r_s0_pad = _run_cmd([str(ollm_bin), "spsamp", str(models["base"]), str(models["s0"]), "8", "0", short_padded, "42", str(merges)])
        r_clip_pad = _run_cmd([
            str(ollm_bin), "spclip", str(models["base"]), "1", short_padded, "8", "0", "42", str(merges),
            "2", "1000", "1", str(models["s0"]), "1000", "1", str(models["s1"]),
        ])
        if r_s0_pad.returncode != 0 or r_clip_pad.returncode != 0:
            raise RuntimeError(f"Offline parity CLI failed: spsamp={r_s0_pad.stderr} spclip={r_clip_pad.stderr}")
        if r_s0_pad.stdout != ds0_b["choices"][0]["text"]:
            raise AssertionError(f"Serve strand:0 != CLI spsamp: cli={r_s0_pad.stdout!r} srv={ds0_b['choices'][0]['text']!r}")
        if r_clip_pad.stdout != dsc_b["choices"][0]["text"]:
            raise AssertionError(f"Serve spclip != CLI spclip: cli={r_clip_pad.stdout!r} srv={dsc_b['choices'][0]['text']!r}")

        srv.assert_alive()
        metrics["cold_ttft_ms"] = round(ttft_cold_sse, 2)
        metrics["warm_ttft_ms"] = round(ttft_warm_sse, 2)
        metrics["ttft_speedup"] = round(ttft_cold_sse / max(0.01, ttft_warm_sse), 2)
        metrics["kv_hits"] = int(kvc["hits"])
        metrics["tokens_saved"] = int(kvc["tokens_saved"])
    return metrics


def _suite_c_collision_and_lru(ollm_bin: Path, models: dict[str, Path], port: int) -> dict[str, object]:
    merges = models["merges"]
    metrics: dict[str, object] = {}
    prompt_a = "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha "
    prompt_b = "omega omega omega omega omega omega omega omega omega omega omega omega "
    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(models["base"]), str(merges), "slots=4", "ctx=8192", "kvblks=64", "test=1"],
        port,
    ) as srv:
        st_b_cold, _, db_cold, _ = _http_json(
            port, "POST", "/v1/completions",
            {"prompt": prompt_b, "max_tokens": 8, "temperature": 0, "seed": 7, "cache_prefix": False},
        )
        st_a, _, _, _ = _http_json(
            port, "POST", "/v1/completions",
            {"prompt": prompt_a, "max_tokens": 8, "temperature": 0, "seed": 7, "ollm_test_collide": True},
        )
        st_b_col, _, db_col, _ = _http_json(
            port, "POST", "/v1/completions",
            {"prompt": prompt_b, "max_tokens": 8, "temperature": 0, "seed": 7, "ollm_test_collide": True},
        )
        if st_b_cold != 200 or st_a != 200 or st_b_col != 200:
            raise AssertionError(f"Collision test bad status: {st_b_cold} {st_a} {st_b_col}")
        cached_col = int(((db_col.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", -1))
        if cached_col != 0:
            raise AssertionError(f"Expected cached_tokens=0 on forced collision, got {cached_col}")
        if db_col["choices"][0]["text"] != db_cold["choices"][0]["text"]:
            raise AssertionError("Hash collision corrupted Prompt B output!")
        _, _, hdata, _ = _http_json(port, "GET", "/health")
        rej = int((hdata.get("kv_cache") or {}).get("collisions_rejected", 0))
        if rej < 1:
            raise AssertionError(f"Expected collisions_rejected >= 1, got {hdata}")
        srv.assert_alive()
        metrics["collisions_rejected"] = rej

    # LRU bounded eviction check with kvblks=8
    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(models["base"]), str(merges), "slots=4", "ctx=8192", "kvblks=8"],
        port,
    ) as srv:
        for i in range(10):
            p = f"lru distinct item number {i} " * 4
            st, _, _, _ = _http_json(port, "POST", "/v1/completions", {"prompt": p, "max_tokens": 2, "temperature": 0})
            if st != 200:
                raise AssertionError(f"LRU req {i} failed: {st}")
        _, _, hdata, _ = _http_json(port, "GET", "/health")
        kvc = hdata.get("kv_cache") or {}
        if int(kvc.get("capacity", 0)) != 8 or int(kvc.get("entries", 99)) > 8 or int(kvc.get("evictions", 0)) <= 0:
            raise AssertionError(f"LRU bounded eviction failed: {kvc}")
        srv.assert_alive()
        metrics["lru_evictions"] = int(kvc["evictions"])
    return metrics


def _suite_d_long_context(ollm_bin: Path, models: dict[str, Path], work: Path, port: int) -> dict[str, object]:
    merges = models["merges"]
    long_ck = models["long"]
    metrics: dict[str, object] = {}

    # CLI > 32768 must exit 2
    r_bad = _run_cmd([str(ollm_bin), "serve", str(port), str(long_ck), str(merges), "ctx=32769"])
    if r_bad.returncode != 2:
        raise AssertionError(f"Expected exit 2 for ctx=32769, got rc={r_bad.returncode}")

    p8k_ok, n8k_ok = _make_prompt_with_tokens(ollm_bin, merges, 300, "x ", 8180, work)
    p8k_over, n8k_over = _make_prompt_with_tokens(ollm_bin, merges, 300, "x ", 8200, work)
    if not (8160 <= n8k_ok + 1 < 8192) or (n8k_over + 1 < 8192):
        raise AssertionError(f"Unexpected 8K token counts: ok={n8k_ok} over={n8k_over}")

    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(long_ck), str(merges), "slots=2", "ctx=8192", "test=1"],
        port,
    ) as srv:
        st_ok, _, d_ok, _ = _http_json(port, "POST", "/v1/completions", {"prompt": p8k_ok, "max_tokens": 2, "temperature": 0})
        if st_ok != 200:
            raise AssertionError(f"8K boundary request failed st={st_ok}: {d_ok}")
        st_ov, _, d_ov, _ = _http_json(port, "POST", "/v1/completions", {"prompt": p8k_over, "max_tokens": 2, "temperature": 0})
        if st_ov != 400 or (d_ov.get("error") or {}).get("code") != "context_length_exceeded":
            raise AssertionError(f"Expected 400 context_length_exceeded over 8K, got st={st_ov} data={d_ov}")

        # Verify generous default output limit (>= 4096) when max_tokens is omitted
        big_out = "a" * 4200
        st_gen, _, d_gen, _ = _http_json(
            port,
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "ollm_test_output": big_out,
                "temperature": 0,
            },
        )
        if st_gen != 200:
            raise AssertionError(f"Default output >=4096 check failed st={st_gen}: {d_gen}")
        comp_toks = int((d_gen.get("usage") or {}).get("completion_tokens", 0))
        if comp_toks < 4096:
            raise AssertionError(f"Default max_tokens truncated at {comp_toks} < 4096")
        srv.assert_alive()
        metrics["ctx_8k_ok_tokens"] = n8k_ok + 1
        metrics["default_out_tokens"] = comp_toks

    p32k_ok, n32k_ok = _make_prompt_with_tokens(ollm_bin, merges, 300, "z ", 32750, work)
    p32k_over, n32k_over = _make_prompt_with_tokens(ollm_bin, merges, 300, "z ", 32785, work)
    if not (32700 <= n32k_ok + 1 < 32768) or (n32k_over + 1 < 32768):
        raise AssertionError(f"Unexpected 32K token counts: ok={n32k_ok} over={n32k_over}")

    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(long_ck), str(merges), "slots=2", "ctx=32768", "kvcache=16", "kvblks=8192"],
        port,
    ) as srv:
        st_32_cold, _, d_32_cold, dt_32_cold = _http_json(
            port, "POST", "/v1/completions", {"prompt": p32k_ok, "max_tokens": 2, "temperature": 0, "seed": 9}, timeout=180.0
        )
        if st_32_cold != 200:
            raise AssertionError(f"32K cold failed st={st_32_cold}: {d_32_cold}")
        st_32_warm, _, d_32_warm, dt_32_warm = _http_json(
            port, "POST", "/v1/completions", {"prompt": p32k_ok, "max_tokens": 2, "temperature": 0, "seed": 9}, timeout=180.0
        )
        if st_32_warm != 200:
            raise AssertionError(f"32K warm failed st={st_32_warm}: {d_32_warm}")
        c32 = int(((d_32_warm.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        if c32 < 32700 or dt_32_warm >= dt_32_cold:
            raise AssertionError(f"32K prefix cache check failed: cached={c32} cold={dt_32_cold:.1f}ms warm={dt_32_warm:.1f}ms")
        st_32_ov, _, d_32_ov, _ = _http_json(
            port, "POST", "/v1/completions", {"prompt": p32k_over, "max_tokens": 2, "temperature": 0}, timeout=60.0
        )
        if st_32_ov != 400 or (d_32_ov.get("error") or {}).get("code") != "context_length_exceeded":
            raise AssertionError(f"Expected 400 context_length_exceeded over 32K, got st={st_32_ov} data={d_32_ov}")
        srv.assert_alive()
        metrics["ctx_32k_ok_tokens"] = n32k_ok + 1
        metrics["ctx_32k_cold_ms"] = round(dt_32_cold, 1)
        metrics["ctx_32k_warm_ms"] = round(dt_32_warm, 1)
    return metrics


def _suite_e_openai_and_disconnect(ollm_bin: Path, models: dict[str, Path], port: int) -> None:
    with ManagedServer(
        [str(ollm_bin), "serve", str(port), str(models["base"]), str(models["merges"]), "slots=4", "ctx=8192", "test=1"],
        port,
    ) as srv:
        # Stop sequence & top_k / top_p
        st_s, _, d_s, _ = _http_json(
            port,
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "stop test"}],
                "ollm_test_output": "hello STOP world",
                "stop": ["STOP"],
                "top_k": 5,
                "top_p": 0.9,
                "temperature": 0.5,
            },
        )
        if st_s != 200 or d_s["choices"][0]["message"]["content"] != "hello " or d_s["choices"][0]["finish_reason"] != "stop":
            raise AssertionError(f"Stop/top_k check failed: {st_s} {d_s}")

        # Tool call emission
        st_t, _, d_t, _ = _http_json(
            port,
            "POST",
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "call a tool"}],
                "tools": [{"type": "function", "function": {"name": "ping"}}],
                "tool_choice": "auto",
                "ollm_test_output": '<tool_call>{"name":"ping","arguments":{"x":1}}</tool_call>',
                "temperature": 0,
            },
        )
        tcalls = d_t["choices"][0]["message"].get("tool_calls") or []
        if st_t != 200 or d_t["choices"][0]["finish_reason"] != "tool_calls" or len(tcalls) != 1:
            raise AssertionError(f"Tool call check failed: {st_t} {d_t}")

        # 404 & 405
        st_404, _, _, _ = _http_json(port, "GET", "/v1/doesnotexist")
        st_405, _, _, _ = _http_json(port, "GET", "/v1/chat/completions")
        if st_404 != 404 or st_405 != 405:
            raise AssertionError(f"Expected 404/405, got {st_404}/{st_405}")

        # Mid-stream client disconnect cleanup -> in_flight returns to 0
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(10.0)
            s.connect(("127.0.0.1", port))
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "drop mid stream"}],
                "stream": True,
                "ollm_test_output": "token0 token1 token2 token3 token4 ",
                "ollm_test_delay_ms": 60,
                "temperature": 0,
            }).encode("utf-8")
            raw_req = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(req_body)}\r\n\r\n"
            ).encode("ascii") + req_body
            s.sendall(raw_req)
            _ = s.recv(512)
        finally:
            s.close()

        cleaned = False
        for _ in range(50):
            _, _, hdata, _ = _http_json(port, "GET", "/health")
            if int(hdata.get("in_flight", -1)) == 0:
                cleaned = True
                break
            time.sleep(0.05)
        if not cleaned:
            raise AssertionError("Slot did not release after client disconnect")
        srv.assert_alive()


def _suite_f_gb10(omc: str, merges: Path, work: Path, port: int) -> dict[str, object]:
    gb10_bin = _build_target("gb10", work, omc)
    gpu_ck = work / "gpu.ck"
    gpu_corpus = work / "gpu_corpus.txt"
    gpu_merges = work / "gpu_merges.bin"
    gpu_txt = (
        (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
        + "\n"
        + (ROOT / "scripts" / "verify-om.sh").read_text(encoding="utf-8")
    )
    gpu_corpus.write_text(gpu_txt, encoding="utf-8")
    rm = _run_cmd([str(gb10_bin), "bpefit", str(gpu_corpus), str(gpu_merges), "512"])
    if rm.returncode != 0:
        raise RuntimeError(f"gpu bpefit failed rc={rm.returncode}: {rm.stdout} {rm.stderr}")
    r = _run_cmd([str(gb10_bin), "init", str(gpu_ck), "512", "64", "64", "2", "128", "2", "1", "13", "2", "10000"])
    if r.returncode != 0:
        raise RuntimeError(f"gpu init failed rc={r.returncode}: {r.stdout} {r.stderr}")
    p8k, _ = _make_prompt_with_tokens(gb10_bin, gpu_merges, 512, "g ", 8180, work)
    p8k_over, _ = _make_prompt_with_tokens(gb10_bin, gpu_merges, 512, "g ", 8200, work)

    with ManagedServer(
        [str(gb10_bin), "gserve", str(port), str(gpu_ck), str(gpu_merges), "slots=4", "ctx=8192"],
        port,
    ) as srv:
        sts: list[int] = [0] * 4

        def gworker(i: int) -> None:
            st, _, _, _ = _http_sse(
                port,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": f"gpu stream {i}"}],
                    "stream": True,
                    "max_tokens": 6,
                    "temperature": 0,
                    "seed": 100 + i,
                },
            )
            sts[i] = st

        ts = [threading.Thread(target=gworker, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60.0)
        if sts != [200, 200, 200, 200]:
            raise AssertionError(f"GB10 gserve 4 concurrent streams failed: {sts}")

        st_8k, _, _, dt_8k = _http_json(
            port, "POST", "/v1/completions", {"prompt": p8k, "max_tokens": 2, "temperature": 0}, timeout=120.0
        )
        if st_8k != 200:
            raise AssertionError(f"GB10 gserve 8K long-context failed: {st_8k}")
        st_ov, _, d_ov, _ = _http_json(
            port, "POST", "/v1/completions", {"prompt": p8k_over, "max_tokens": 2, "temperature": 0}
        )
        if st_ov != 400 or (d_ov.get("error") or {}).get("code") != "context_length_exceeded":
            raise AssertionError(f"GB10 gserve >8K refusal failed: {st_ov} {d_ov}")
        srv.assert_alive()
        return {"gb10_concurrent_4_200": True, "gb10_8k_ms": round(dt_8k, 1)}


def main() -> int:
    _check_static_invariants()
    omc = _find_omc()
    with tempfile.TemporaryDirectory(prefix="helix-serve-scale-") as tmpdir:
        work = Path(tmpdir)
        ollm_env = os.environ.get("OLLM_BIN", "").strip()
        ollm_bin = Path(ollm_env) if ollm_env else _build_target("aarch64", work, omc)
        magic = ollm_bin.read_bytes()[:4]
        if magic != b"\x7fELF":
            raise AssertionError(f"Expected ELF binary for TARGET=aarch64, got {magic!r}")

        if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
            print("SERVE_SCALING_BUILD_PASS (live HTTP execution requires Linux aarch64)")
            return 0

        port = _pick_safe_port()
        models = _setup_models(ollm_bin, work)
        m_a = _suite_a_concurrency(ollm_bin, models, port)
        m_b = _suite_b_prefix_cache_and_strands(ollm_bin, models, work, port)
        m_c = _suite_c_collision_and_lru(ollm_bin, models, port)
        m_d = _suite_d_long_context(ollm_bin, models, work, port)
        _suite_e_openai_and_disconnect(ollm_bin, models, port)
        m_f: dict[str, object] = {}
        if os.environ.get("OLLM_GB10_LIVE", "") == "1":
            m_f = _suite_f_gb10(omc, models["merges"], work, port)

        summary = {**m_a, **m_b, **m_c, **m_d, **m_f}
        print("SERVE_SCALING_METRICS " + json.dumps(summary, sort_keys=True))
        print("SERVE_SCALING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
