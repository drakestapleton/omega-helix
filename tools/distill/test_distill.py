#!/usr/bin/env python3
"""Comprehensive verification suite for Omega Helix Sovereign Distillation Pipeline Network."""

from __future__ import annotations

import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
STUDIO_DIR = TOOLS_DIR / "studio"
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
for _p in (str(TOOLS_DIR), str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chain_runner  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402
import server as studio_server  # noqa: E402
from distill.harvest import (  # noqa: E402
    UMAGIC_DK1,
    fnv1a64,
    harvest_paths,
    read_ollmdk01_header,
    write_ollmdk01,
    write_openwaldo_export,
)
from distill.pipeline import DistillManager, estimate_distill  # noqa: E402
from distill.teachers import (  # noqa: E402
    OpenAICompatTeacher,
    TeacherRouter,
    TraceReplayTeacher,
    probe_local_teachers,
)
from distill.verifiers import VerifierPipeline, find_omc, verify_record  # noqa: E402


def _ensure_bins() -> dict[str, Path]:
    needed = ["ollm_univ", "ollm_waldo", "ollm"]
    env_dir = os.environ.get("OMEGA_HELIX_BIN_DIR")
    return chain_runner.ensure_binaries(needed, bin_dir=env_dir, root=ROOT)


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._send(HTTPStatus.OK, {"object": "list", "data": [{"id": "mock-teacher-1"}]})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        if not self.path.endswith("/chat/completions"):
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        model = body.get("model", "mock-teacher-1")
        prompt = ""
        for m in body.get("messages") or []:
            if m.get("role") == "user":
                prompt = m.get("content", "")
        if "bad" in model:
            text = "not json at all {{{"
        else:
            text = f"The sovereign Omega Helix response for {prompt.strip() or 'test'} uses clean verified tokens.\n" * 3
        self._send(
            HTTPStatus.OK,
            {
                "id": "cmpl-mock",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "The",
                                    "logprob": -0.2,
                                    "top_logprobs": [
                                        {"token": "The", "logprob": -0.2},
                                        {"token": "A", "logprob": -2.1},
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )


def _start_mock_openai() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    host, port = srv.server_address[:2]
    return srv, t, f"http://{host}:{port}/v1"


class TestDistillationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bins = _ensure_bins()
        cls.univ_bin = str(cls.bins["ollm_univ"])
        cls.bin_dir = str(Path(cls.univ_bin).parent)

    def _run_univ(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.univ_bin, *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def _init_small_qwen3(self, ck: Path, seed: int = 42, steps: int = 6) -> None:
        r = self._run_univ(
            "init", str(ck), "qwen3",
            "16", "8", "16", "2", "2", "32", "2", "2",
            str(seed), str(steps), "3000",
        )
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_ollmdk01_emit_dcheck_dtrain_and_partial_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            tok = tdp / "tok.bin"
            tok.write_bytes(bytes([i % 16 for i in range(256)]))
            t_ck = tdp / "teacher.ck"
            s_ck = tdp / "student.ck"
            dk = tdp / "teacher.dk"
            self._init_small_qwen3(t_ck, seed=11, steps=4)
            self._init_small_qwen3(s_ck, seed=29, steps=6)

            r_emit = self._run_univ("dkemit", str(t_ck), str(tok), str(dk), "4", "1500", "8")
            self.assertEqual(r_emit.returncode, 0, r_emit.stderr)
            self.assertTrue(r_emit.stdout.startswith("K 8 16 4 8 "), r_emit.stdout)

            hdr = read_ollmdk01_header(dk)
            self.assertEqual(hdr["V"], 16)
            self.assertEqual(hdr["K"], 4)
            self.assertEqual(hdr["T"], 8)
            self.assertEqual(hdr["N"], 8)

            r_chk = self._run_univ("dcheck", str(s_ck), str(dk), "700000", "1500", "24")
            self.assertEqual(r_chk.returncode, 0, r_chk.stdout + r_chk.stderr)
            self.assertTrue(r_chk.stdout.startswith("C 24 "), r_chk.stdout)

            r_tr = self._run_univ("dtrain", str(s_ck), str(dk), "4", "2", "700000", "1500")
            self.assertEqual(r_tr.returncode, 0, r_tr.stderr)
            dlines = [ln for ln in r_tr.stdout.splitlines() if ln.startswith("D ")]
            self.assertEqual(len(dlines), 2, r_tr.stdout)
            for ln in dlines:
                _, st, tot_m, ce_m, kl_m = ln.split()
                expect_tot = round(0.3 * int(ce_m) + 0.7 * int(kl_m))
                self.assertLessEqual(abs(int(tot_m) - expect_tot), 2, ln)

            # Partial mask + unnormalized sparse probs (sum=0.65) across alpha=0, 0.5, 1.0
            dk_partial = tdp / "partial.dk"
            recs = []
            for row_i in range(4):
                toks = [(row_i + p) % 16 for p in range(9)]
                msk = [0, 0, 0, 0, 1, 1, 1, 1]
                tids: list[int] = []
                tprs: list[float] = []
                for p in range(8):
                    y = toks[p + 1]
                    tids.extend([y, (y + 1) % 16, (y + 2) % 16, (y + 3) % 16])
                    tprs.extend([0.35, 0.15, 0.10, 0.05])
                recs.append({"tokens": toks, "mask": msk, "top_ids": tids, "top_probs": tprs})
            write_ollmdk01(recs, dk_partial, V=16, K=4, T=8)

            s_ck2 = tdp / "student2.ck"
            self._init_small_qwen3(s_ck2, seed=71, steps=4)
            for alpha_m in ("0", "500000", "1000000"):
                rc = self._run_univ("dcheck", str(s_ck2), str(dk_partial), alpha_m, "1400", "16")
                self.assertEqual(rc.returncode, 0, f"alpha={alpha_m} failed: {rc.stdout} {rc.stderr}")

    def test_ollmdk01_negative_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ck = tdp / "s.ck"
            dk = tdp / "valid.dk"
            self._init_small_qwen3(ck, seed=5, steps=2)
            rec = {
                "tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "mask": [1] * 8,
                "top_ids": [0, 1, 2, 3] * 8,
                "top_probs": [0.4, 0.3, 0.2, 0.1] * 8,
            }
            write_ollmdk01([rec], dk, V=16, K=4, T=8)
            valid_raw = dk.read_bytes()

            def _check_mutated(name: str, data: bytes, expected_rc: int) -> None:
                p = tdp / f"{name}.dk"
                p.write_bytes(data)
                r = self._run_univ("dcheck", str(ck), str(p), "500000", "1000", "4")
                self.assertEqual(r.returncode, expected_rc, f"{name}: got {r.returncode}, expected {expected_rc}")

            # 10: truncated
            _check_mutated("trunc", valid_raw[:32], 10)
            # 11: bad magic, bad K=0
            _check_mutated("bad_magic", b"BADMAGIC" + valid_raw[8:], 11)
            bad_k = bytearray(valid_raw)
            struct.pack_into("<Q", bad_k, 24, 0)
            _check_mutated("bad_k0", bytes(bad_k), 11)
            # 12: V mismatch, mask > 1
            bad_v = bytearray(valid_raw)
            struct.pack_into("<Q", bad_v, 16, 32)
            _check_mutated("bad_v", bytes(bad_v), 12)
            bad_m = bytearray(valid_raw)
            msk_off = 64 + 4 * 9
            bad_m[msk_off] = 2
            struct.pack_into("<Q", bad_m, 56, fnv1a64(bytes(bad_m[64:])))
            _check_mutated("bad_mask", bytes(bad_m), 12)
            # 14: FNV-1a corruption
            bad_fnv = bytearray(valid_raw)
            bad_fnv[-1] ^= 0x01
            _check_mutated("bad_fnv", bytes(bad_fnv), 14)
            # 15: trailing extra byte
            _check_mutated("trailing", valid_raw + b"\x00", 15)
            # 25: out-of-range token and prob sum > 1.0001
            bad_tok = bytearray(valid_raw)
            struct.pack_into("<I", bad_tok, 64, 99)
            struct.pack_into("<Q", bad_tok, 56, fnv1a64(bytes(bad_tok[64:])))
            _check_mutated("bad_tok", bytes(bad_tok), 25)
            bad_sum = bytearray(valid_raw)
            pr_off = 64 + 4 * 9 + 8 + 4 * 32
            struct.pack_into("<f", bad_sum, pr_off, 0.99)
            struct.pack_into("<Q", bad_sum, 56, fnv1a64(bytes(bad_sum[64:])))
            _check_mutated("bad_sum", bytes(bad_sum), 25)

    def test_verifiers_and_real_omc_compile(self) -> None:
        self.assertTrue(verify_record('{"ok": true, "k": 1}', ["json_schema"], json_required_keys=["ok", "k"])["passed"])
        self.assertFalse(verify_record('{"ok": true}', ["json_schema"], json_required_keys=["ok", "k"])["passed"])
        self.assertTrue(verify_record("```python\ndef add(x, y):\n    return x + y\n```", ["python_ast"])["passed"])
        self.assertFalse(verify_record("def broken(:\n", ["python_ast"])["passed"])
        self.assertTrue(verify_record("ANSWER: 42", ["regex"], regex_pattern=r"ANSWER:\s+\d+")["passed"])

        omc = find_omc()
        if omc is not None:
            valid_om = "```om\nom2\nfn main to int\nret 0\nend\n```"
            broken_om = "```om\nom2\nfn main to int\nthis_is_not_valid\nend\n```"
            self.assertTrue(verify_record(valid_om, ["omc_compile"], omc_path=omc)["passed"])
            self.assertFalse(verify_record(broken_om, ["omc_compile"], omc_path=omc)["passed"])

        pipe = VerifierPipeline(["nonempty", "json_schema"], json_required_keys=["id"])
        res = pipe.filter_records([
            {"text": '{"id": 1}'},
            {"text": '{"missing": 2}'},
            {"text": ""},
        ])
        self.assertEqual(res["stats"]["passed"], 1)
        self.assertEqual(res["stats"]["rejected"], 2)

    def test_harvest_formats_dedup_and_openwaldo_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            t1 = tdp / "traces.jsonl"
            t1.write_text(
                "\n".join([
                    json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}, {"role": "assistant", "content": "Hello from trace one!"}]}),
                    json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello from trace one!"}]}),  # exact duplicate
                    json.dumps({"conversations": [{"from": "human", "value": "Explain KL"}, {"from": "gpt", "value": "Forward KL times tau squared."}]}),
                    json.dumps({"instruction": "Compute double", "input": "x = 3", "output": "6"}),
                ]) + "\n",
                encoding="utf-8",
            )
            harv = harvest_paths([t1])
            self.assertEqual(harv["stats"]["raw_records"], 4)
            self.assertEqual(harv["stats"]["duplicates_skipped"], 1)
            self.assertEqual(harv["stats"]["unique_records"], 3)

            exp_dir = tdp / "export"
            meta = write_openwaldo_export(harv["records"], exp_dir, corpus_name="test-distill")
            rep = ow_bridge.verify_export(exp_dir, check_records=True)
            self.assertEqual(rep["total_docs"], 3)
            self.assertEqual(meta["export_json_sha256"], rep["export_json_sha256"])
            self.assertTrue((exp_dir / "MASKS.jsonl").is_file())

    def test_teachers_mock_openai_router_and_probe(self) -> None:
        srv, _, base_url = _start_mock_openai()
        try:
            probed = probe_local_teachers([{"name": "mock", "base_url": base_url}], timeout=2.0)
            self.assertTrue(probed[0]["reachable"])
            self.assertIn("mock-teacher-1", probed[0]["models"])

            t_good = OpenAICompatTeacher(base_url, "mock-teacher-1", name="good")
            t_bad = OpenAICompatTeacher(base_url, "bad-teacher", name="bad")
            out = t_good.generate([{"role": "user", "content": "helix"}], top_logprobs=4)
            self.assertIn("sovereign Omega Helix", out["text"])
            self.assertIsNotNone(out["logprobs"])

            with tempfile.TemporaryDirectory() as td:
                tr = Path(td) / "replay.jsonl"
                tr.write_text(json.dumps({"prompt": "helix", "response": '{"ok": true}'}) + "\n", encoding="utf-8")
                t_rep = TraceReplayTeacher([tr])
                router = TeacherRouter([t_bad, t_rep], strategy="ensemble_best_of", verifiers=["json_schema"])
                best = router.generate_best([{"role": "user", "content": "helix"}])
                self.assertTrue(best["verification"]["passed"])
                self.assertEqual(best["teacher_name"], "trace-replay")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_pipeline_all_three_modes(self) -> None:
        srv, _, base_url = _start_mock_openai()
        try:
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                traces = tdp / "corpus.jsonl"
                lines = [
                    json.dumps({"prompt": f"task {i}", "response": ("Sovereign verified Omega Helix sequence text block.\n" * 6)})
                    for i in range(6)
                ]
                traces.write_text("\n".join(lines) + "\n", encoding="utf-8")
                t_ck = tdp / "t.ck"
                self._init_small_qwen3(t_ck, seed=3, steps=4)

                mgr = DistillManager(runs_dir=tdp / "distill_runs", bin_dir=self.bin_dir, root=ROOT)

                # 1a. dk-univ from local teacher checkpoint
                r1 = mgr.start_run({
                    "name": "dk-from-ckpt",
                    "mode": "dk-univ",
                    "arch": "qwen3",
                    "V": 16, "T": 8, "D": 16, "H": 2, "H_KV": 2, "F": 32, "L": 2, "B": 2,
                    "K": 4, "alpha": 0.7, "temperature": 1.4, "steps": 4,
                    "teacher_ckpt": str(t_ck),
                    "trace_paths": [str(traces)],
                }, wait=True)
                self.assertEqual(r1.status, "done", r1.error)
                self.assertTrue(r1.dcheck_receipt and r1.dcheck_receipt["passed"])
                self.assertTrue((r1.run_dir / "distill-bom.json").is_file())

                # 1b. dk-univ packed from traces
                r1b = mgr.start_run({
                    "name": "dk-from-traces",
                    "mode": "dk-univ",
                    "arch": "llama",
                    "V": 256, "T": 16, "D": 16, "H": 2, "H_KV": 2, "F": 32, "L": 2, "B": 2,
                    "K": 6, "alpha": 0.5, "temperature": 1.2, "steps": 4,
                    "trace_paths": [str(traces)],
                }, wait=True)
                self.assertEqual(r1b.status, "done", r1b.error)

                # 2. seq-univ via mock OpenAI teacher
                r2 = mgr.start_run({
                    "name": "seq-teacher",
                    "mode": "seq-univ",
                    "arch": "qwen3",
                    "V": 256, "T": 16, "D": 16, "H": 2, "H_KV": 2, "F": 32, "L": 2, "B": 2,
                    "steps": 4,
                    "teachers": [{"kind": "openai_compat", "base_url": base_url, "model": "mock-teacher-1"}],
                    "prompts": ["Distill this prompt A", "Distill this prompt B", "Distill this prompt C"],
                    "trace_paths": [str(traces)],
                }, wait=True)
                self.assertEqual(r2.status, "done", r2.error)
                self.assertTrue((r2.run_dir / "universal" / "model.ck").is_file())

                # 3. dna-strand -> OLLMSP01
                r3 = mgr.start_run({
                    "name": "dna-distill",
                    "mode": "dna-strand",
                    "arch": "gpt",
                    "V": 256, "T": 16, "D": 32, "H": 2, "H_KV": 2, "F": 64, "L": 2, "B": 2,
                    "steps": 4, "add_layers": 1,
                    "trace_paths": [str(traces)],
                }, wait=True)
                self.assertEqual(r3.status, "done", r3.error)
                self.assertTrue((r3.run_dir / "strand_distill.sp").is_file())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_studio_distill_http_sse_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            traces = tdp / "cli_traces.jsonl"
            traces.write_text(
                "\n".join([
                    json.dumps({"prompt": f"p{i}", "response": '{"status": "ok", "code": 1}\n' * 4})
                    for i in range(4)
                ]) + "\n",
                encoding="utf-8",
            )
            srv = studio_server.create_server(
                host="127.0.0.1",
                port=0,
                workspace_dir=tdp / "ws",
                bin_dir=self.bin_dir,
            )
            t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            t.start()
            host, port = srv.server_address[:2]
            base = f"http://{host}:{port}"
            try:
                with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
                    html = resp.read().decode("utf-8")
                    self.assertIn("tab-distill", html)
                    self.assertIn("panel-distill", html)

                with urllib.request.urlopen(f"{base}/api/distill/teachers/probe", timeout=5) as resp:
                    pr = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(pr["ok"])

                harv_req = urllib.request.Request(
                    f"{base}/api/distill/harvest",
                    data=json.dumps({
                        "inputs": [str(traces)],
                        "out_dir": str(tdp / "api_export"),
                        "verifiers": ["nonempty"],
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(harv_req, timeout=5) as resp:
                    hr = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(hr["ok"])
                    self.assertEqual(hr["verify_stats"]["passed"], 4)

                start_req = urllib.request.Request(
                    f"{base}/api/distill/runs",
                    data=json.dumps({
                        "name": "http-dk",
                        "mode": "dk-univ",
                        "arch": "qwen3",
                        "V": 256, "T": 16, "D": 16, "H": 2, "H_KV": 2, "F": 32, "L": 2, "B": 2,
                        "K": 4, "alpha": 0.7, "temperature": 1.5, "steps": 3,
                        "trace_paths": [str(traces)],
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(start_req, timeout=5) as resp:
                    sr = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(sr["ok"])
                    run_id = sr["run_id"]

                events: list[dict] = []
                with urllib.request.urlopen(f"{base}/api/distill/runs/{run_id}/events", timeout=30) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if line.startswith("data: "):
                            ev = json.loads(line[6:])
                            events.append(ev)
                            if ev.get("kind") in ("done", "error"):
                                break
                self.assertEqual(events[-1]["kind"], "done", events)
            finally:
                srv.shutdown()
                srv.server_close()

            # CLI integration
            env = dict(os.environ, OMEGA_HELIX_BIN_DIR=self.bin_dir, OMEGA_HELIX_DISTILL_DIR=str(tdp / "cli_dist"))
            cli_probe = subprocess.run([str(ROOT / "bin" / "omega-helix"), "distill", "probe", "--json"], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(cli_probe.returncode, 0, cli_probe.stderr)
            cli_harv = subprocess.run(
                [str(ROOT / "bin" / "omega-helix"), "distill", "harvest", "-i", str(traces), "-o", str(tdp / "cli_exp"), "--json"],
                env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(cli_harv.returncode, 0, cli_harv.stderr)
            cli_run = subprocess.run(
                [
                    str(ROOT / "bin" / "omega-helix"), "distill", "run",
                    "--mode", "dk-univ", "--arch", "qwen3",
                    "--V", "256", "--T", "16", "--D", "16", "--H", "2", "--H-KV", "2", "--F", "32", "--L", "2", "--B", "2",
                    "--K", "4", "--steps", "3", "--trace", str(traces), "--json",
                ],
                env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(cli_run.returncode, 0, cli_run.stderr)
            doc = json.loads(cli_run.stdout)
            self.assertEqual(doc["status"], "done")
            cli_stat = subprocess.run(
                [str(ROOT / "bin" / "omega-helix"), "distill", "status", doc["run_id"]],
                env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(cli_stat.returncode, 0, cli_stat.stderr)


if __name__ == "__main__":
    unittest.main()
