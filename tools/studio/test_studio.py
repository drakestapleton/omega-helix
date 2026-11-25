#!/usr/bin/env python3
"""Comprehensive stdlib unittest suite for Omega Helix Studio backend, server, and CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

STUDIO_DIR = Path(__file__).resolve().parent
TOOLS_DIR = STUDIO_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"

for _p in (str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import chain_runner as runner  # noqa: E402
import cli as studio_cli  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402
import server as studio_server  # noqa: E402


class StudioTestCase(unittest.TestCase):
    bin_dir: Path
    tmp_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="omega-studio-test-")).resolve()
        # Resolve or build binaries once for the entire test suite
        resolved = runner.ensure_binaries(["ollm", "ollm_univ", "ollm_interchange", "ollm_waldo"])
        cls.bin_dir = resolved["ollm"].parent
        cls.bins = resolved

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------ 1. Catalog

    def test_catalog_load_search_and_spdx_normalization(self) -> None:
        res = studio_catalog.search_catalog()
        self.assertTrue(res["ok"])
        self.assertEqual(res["total"], 73)
        self.assertEqual(res["count"], 73)
        self.assertGreaterEqual(len(res["facets"]["categories"]), 5)
        self.assertGreaterEqual(len(res["facets"]["licenses"]), 5)
        self.assertGreaterEqual(len(res["facets"]["prog_languages"]), 2)
        self.assertIn("local_presets", res)

        # SPDX normalization on stack-v2-edu (huge composite license list)
        s_res = studio_catalog.search_catalog(q="stack-v2-edu")
        self.assertGreaterEqual(s_res["count"], 1)
        edu = s_res["corpora"][0]
        self.assertEqual(edu["name"], "stack-v2-edu")
        self.assertGreater(edu["license_count"], 4)
        self.assertIn("more (", edu["license_summary"])
        self.assertIn("Apache-2.0", edu["license_tags"])
        self.assertIn("MIT", edu["license_tags"])

        # Category & license filtering
        code_res = studio_catalog.search_catalog(category="code")
        self.assertGreater(code_res["count"], 0)
        self.assertTrue(all(c["category"] == "code" for c in code_res["corpora"]))

        mit_res = studio_catalog.search_catalog(license="MIT")
        self.assertGreater(mit_res["count"], 0)
        self.assertTrue(all("MIT" in c["license_tags"] for c in mit_res["corpora"]))

        go_res = studio_catalog.search_catalog(prog="Go")
        self.assertGreater(go_res["count"], 0)

    def test_materialize_waldo_without_cli_returns_guidance(self) -> None:
        if shutil.which("waldo"):
            self.skipTest("waldo CLI is installed on this host")
        out = self.tmp_root / "waldo_mat"
        res = studio_catalog.materialize_waldo("code/permissive", out)
        self.assertFalse(res["ok"])
        self.assertFalse(res["installed"])
        self.assertTrue(res["guidance"])

    # ------------------------------------------------------------------ 2. Local Dataset Prep

    def test_prepare_local_dataset_preset_and_folder_verified_by_ollm_waldo(self) -> None:
        # Starter preset -> EXPORT.json -> openwaldo_bridge + ollm-waldo
        out_preset = self.tmp_root / "ds_preset"
        res = studio_catalog.prepare_local_dataset(
            "omega-starter",
            out_preset,
            name="starter",
            ollm_waldo=self.bins["ollm_waldo"],
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["kind"], "preset")
        self.assertGreater(res["total_docs"], 10)
        self.assertTrue(res["waldo_summary"].startswith("W "))
        self.assertTrue((out_preset / "EXPORT.json").is_file())

        # Inspect generated EXPORT.json dir
        insp = studio_catalog.inspect_dataset(path=str(out_preset))
        self.assertTrue(insp["ok"])
        self.assertEqual(insp["kind"], "export_dir")
        self.assertTrue(insp["export_verified"])

        # Re-ingest existing EXPORT.json dir
        res2 = studio_catalog.prepare_local_dataset(
            out_preset,
            self.tmp_root / "ds_reexport",
            ollm_waldo=self.bins["ollm_waldo"],
        )
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["kind"], "export_dir")

        # Raw local folder with .md + .om + .py files
        raw_dir = self.tmp_root / "raw_corpus"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            (raw_dir / f"doc_{i:02d}.md").write_text(
                f"# Omega Helix Doc {i}\nSovereign intelligence runs cleanly on owned hardware.\n" * 6,
                encoding="utf-8",
            )
        out_local = self.tmp_root / "ds_local"
        res3 = studio_catalog.prepare_local_dataset(
            raw_dir,
            out_local,
            name="raw",
            ollm_waldo=self.bins["ollm_waldo"],
        )
        self.assertTrue(res3["ok"])
        self.assertEqual(res3["total_docs"], 12)
        self.assertTrue(res3["waldo_summary"].startswith("W 12 "))

    # ------------------------------------------------------------------ 3. Estimation & Validation

    def test_estimate_chain_exact_params_and_validation(self) -> None:
        # Legacy DNA exact formula: V*D + T*D + L*(4*D^2 + 2*D*F + 2*D) + D
        # For V=256, T=16, D=16, H=2, F=32, L=2 -> 256*16 + 16*16 + 2*(1024+1024+32) + 16 = 4096+256+4160+16 = 8528
        est_dna = runner.estimate_chain({
            "name": "est-dna",
            "mode": "hybrid",
            "arch": "gpt",
            "tokenizer": "byte",
            "V": 256, "T": 16, "D": 16, "H": 2, "F": 32, "L": 2, "B": 2,
            "base_steps": 10, "lr_micro": 3000,
            "strands": [
                {"id": "a", "name": "Code", "add_layers": 1, "steps": 6},
                {"id": "b", "name": "Docs", "add_layers": 1, "steps": 6},
            ],
            "export_hf": True, "export_gguf": True,
        })
        self.assertTrue(est_dna["valid"], est_dna["error"])
        self.assertEqual(est_dna["base_params"], 8528)
        self.assertEqual(est_dna["params"], 8528 + 2 * (4 * 256 + 2 * 16 * 32 + 32))
        self.assertEqual(est_dna["tok_bytes"], 1)
        node_ids = {n["id"] for n in est_dna["dag"]["nodes"]}
        self.assertTrue({"base-data", "base-model", "merkle-lock", "strand-a", "strand-b", "spclip", "ck04-upgrade", "export-hf", "export-gguf", "bom"}.issubset(node_ids))

        # Universal Qwen3 u32 vocab
        est_u = runner.estimate_chain({
            "name": "est-qwen3",
            "mode": "universal",
            "arch": "qwen3",
            "tokenizer": "u32",
            "V": 70000, "T": 32, "D": 32, "H": 4, "H_KV": 2, "F": 64, "L": 2, "B": 2,
        })
        self.assertTrue(est_u["valid"], est_u["error"])
        self.assertEqual(est_u["tok_bytes"], 4)
        self.assertGreater(est_u["params"], 4_000_000)

        # Invalid: looped core + DNA strands refused before launch
        bad = runner.estimate_chain({
            "mode": "dna", "V": 256, "T": 16, "D": 16, "H": 2, "F": 32, "L": 4, "B": 2,
            "LP": 1, "LC": 2, "NP": 2,
            "strands": [{"id": "s1", "name": "S1", "add_layers": 1, "steps": 5}],
        })
        self.assertFalse(bad["valid"])
        self.assertIn("Looped core", bad["error"])

    # ------------------------------------------------------------------ 4. Full DNA Merkle + Hybrid Forge

    def test_full_dna_merkle_chain_forge_and_spclip_sample(self) -> None:
        runs_dir = self.tmp_root / "runs_dna"
        mgr = runner.ChainManager(runs_dir=runs_dir, bin_dir=self.bin_dir)

        spec = {
            "name": "test-dna-hybrid",
            "mode": "hybrid",
            "arch": "gpt",
            "tokenizer": "byte",
            "V": 256, "T": 16, "D": 16, "H": 2, "H_KV": 2, "F": 32, "L": 2, "B": 2,
            "base_steps": 10, "lr_micro": 4000,
            "base_dataset": {"kind": "preset", "value": "omega-docs"},
            "strands": [
                {
                    "id": "strand-code",
                    "name": "Code",
                    "color": "#22d3ee",
                    "add_layers": 1,
                    "growth": "growpro",
                    "use_fcachem": True,
                    "steps": 6,
                    "lr_micro": 2000,
                    "weight": 600,
                    "dataset": {"kind": "preset", "value": "omega-native"},
                },
                {
                    "id": "strand-docs",
                    "name": "Docs",
                    "color": "#a78bfa",
                    "add_layers": 1,
                    "growth": "growpro",
                    "use_fcachem": False,
                    "steps": 6,
                    "lr_micro": 2000,
                    "weight": 400,
                    "dataset": {"kind": "preset", "value": "omega-docs"},
                },
            ],
            "spclip_mode": 1,
            "export_hf": True,
            "hf_dtype": "bf16",
            "export_gguf": True,
        }

        run = mgr.start_chain(spec, wait=True)
        self.assertEqual(run.status, "done", f"DNA chain failed: {run.error}")
        self.assertEqual(len(run.strands), 2)
        for s in run.strands:
            self.assertTrue(Path(s["pack_path"]).is_file())
            self.assertEqual(len(s["pack_sha256"]), 64)
            self.assertEqual(len(s["base_hash"]), 16)
            self.assertEqual(len(s["pack_hash"]), 16)

        # Verify unified BOM + exports exist
        bom_path = run.run_dir / "openwaldo-helix-bom.json"
        self.assertTrue(bom_path.is_file())
        bom = json.loads(bom_path.read_text(encoding="utf-8"))
        self.assertEqual(len(bom["specialist_packs"]), 2)
        self.assertTrue((run.run_dir / "model_v4.ck").is_file())
        self.assertTrue((run.run_dir / "export-hf" / "model.safetensors").is_file())
        self.assertTrue((run.run_dir / "export-hf" / "config.json").is_file())
        gguf = run.run_dir / "model.gguf"
        self.assertTrue(gguf.is_file())
        self.assertEqual(gguf.read_bytes()[:4], b"GGUF")

        # Sample via spclip (both entropy-gated and prior-weighted)
        samp1 = mgr.sample_chain(
            run.run_id,
            prompt="Omega ",
            n_tokens=24,
            temp_milli=800,
            seed=42,
            spclip_mode=1,
            strand_weights={"strand-code": 700, "strand-docs": 300},
        )
        self.assertTrue(samp1["ok"])
        self.assertEqual(samp1["mode_used"], "spclip")
        self.assertGreater(len(samp1["text"]), 0)
        self.assertEqual(samp1["telemetry"]["strand_weights"]["strand-code"], 700)

        samp0 = mgr.sample_chain(run.run_id, prompt="Omega ", n_tokens=16, temp_milli=800, seed=42, spclip_mode=0)
        self.assertTrue(samp0["ok"])
        self.assertEqual(samp0["mode_used"], "spclip")

    # ------------------------------------------------------------------ 5. Full Universal Qwen3 Forge

    def test_full_universal_qwen3_chain_forge_and_sample(self) -> None:
        runs_dir = self.tmp_root / "runs_univ"
        mgr = runner.ChainManager(runs_dir=runs_dir, bin_dir=self.bin_dir)

        spec = {
            "name": "test-qwen3-univ",
            "mode": "universal",
            "arch": "qwen3",
            "tokenizer": "byte",
            "V": 256, "T": 16, "D": 32, "H": 4, "H_KV": 2, "F": 64, "L": 2, "B": 2,
            "base_steps": 10, "lr_micro": 3000,
            "base_dataset": {"kind": "preset", "value": "omega-docs"},
            "strands": [],
            "export_hf": True,
            "hf_dtype": "f32",
            "export_gguf": True,
        }
        run = mgr.start_chain(spec, wait=True)
        self.assertEqual(run.status, "done", f"Universal Qwen3 failed: {run.error}")
        self.assertTrue((run.run_dir / "openwaldo-helix-bom.json").is_file())
        self.assertTrue((run.run_dir / "universal" / "model.ck").is_file())
        self.assertTrue((run.run_dir / "universal" / "export-hf" / "model.safetensors").is_file())
        self.assertTrue((run.run_dir / "universal" / "model.gguf").is_file())
        self.assertGreaterEqual(len(run.metrics), 2)

        samp = mgr.sample_chain(run.run_id, prompt="Omega ", n_tokens=12, temp_milli=0, seed=7)
        self.assertTrue(samp["ok"])
        self.assertEqual(samp["mode_used"], "ollm-univ")
        self.assertEqual(len(samp["telemetry"]["tokens"]), 12)

    # ------------------------------------------------------------------ 6. Live HTTP + SSE + /sample Server

    def test_http_server_api_sse_and_sample(self) -> None:
        runs_dir = self.tmp_root / "runs_http"
        srv = studio_server.create_server(host="127.0.0.1", port=0, workspace_dir=runs_dir, bin_dir=self.bin_dir)
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        th.start()

        def _req(method: str, path: str, body: dict | None = None):
            data = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if data is not None else {}
            r = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), json.loads(resp.read().decode("utf-8"))

        try:
            # Static index.html
            with urllib.request.urlopen(f"{base}/", timeout=10) as r_idx:
                self.assertEqual(r_idx.status, 200)
                html = r_idx.read().decode("utf-8")
                self.assertIn("<html", html.lower())

            # /api/system
            st, _, sys_doc = _req("GET", "/api/system")
            self.assertEqual(st, 200)
            self.assertTrue(sys_doc["ok"])
            self.assertTrue(sys_doc["binaries"]["ollm"])

            # /api/catalog
            st, _, cat_doc = _req("GET", "/api/catalog?q=shakespeare&cat=core")
            self.assertEqual(st, 200)
            self.assertEqual(cat_doc["total"], 73)
            self.assertGreaterEqual(cat_doc["count"], 1)

            # /api/dataset/inspect
            st, _, insp = _req("POST", "/api/dataset/inspect", {"preset": "omega-starter"})
            self.assertEqual(st, 200)
            self.assertTrue(insp["ok"])
            self.assertGreater(insp["total_files"], 5)

            # /api/chains/estimate
            spec = {
                "name": "http-smoke",
                "mode": "dna",
                "arch": "gpt",
                "tokenizer": "byte",
                "V": 256, "T": 16, "D": 16, "H": 2, "F": 32, "L": 2, "B": 2,
                "base_steps": 6, "lr_micro": 3000,
                "base_dataset": {"kind": "preset", "value": "omega-docs"},
                "strands": [
                    {"id": "s1", "name": "Native", "add_layers": 1, "steps": 4, "dataset": {"kind": "preset", "value": "omega-native"}}
                ],
                "export_hf": False, "export_gguf": False,
            }
            st, _, est = _req("POST", "/api/chains/estimate", spec)
            self.assertEqual(st, 200)
            self.assertTrue(est["valid"])

            # /api/chains/start
            st, _, started = _req("POST", "/api/chains/start", spec)
            self.assertEqual(st, 200)
            self.assertTrue(started["ok"])
            run_id = started["run_id"]

            # /api/chains/<run_id>/events (SSE)
            sse_events = []
            with urllib.request.urlopen(f"{base}/api/chains/{run_id}/events", timeout=60) as sse_resp:
                self.assertIn("text/event-stream", sse_resp.headers.get("Content-Type", ""))
                for raw_line in sse_resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data: "):
                        ev = json.loads(line[6:])
                        sse_events.append(ev)
                        if ev.get("kind") in ("done", "error"):
                            break

            kinds = {e["kind"] for e in sse_events}
            self.assertIn("stage", kinds)
            self.assertIn("metric", kinds)
            self.assertIn("artifact", kinds)
            self.assertIn("done", kinds)

            # /api/chains & /api/chains/<run_id>
            st, _, runs_list = _req("GET", "/api/chains")
            self.assertEqual(st, 200)
            self.assertTrue(any(r["run_id"] == run_id for r in runs_list["runs"]))

            st, _, detail = _req("GET", f"/api/chains/{run_id}")
            self.assertEqual(st, 200)
            self.assertEqual(detail["run"]["status"], "done")

            # /api/chains/<run_id>/sample
            st, _, samp = _req("POST", f"/api/chains/{run_id}/sample", {
                "prompt": "Omega ", "n_tokens": 16, "temp_milli": 800, "seed": 42,
            })
            self.assertEqual(st, 200)
            self.assertTrue(samp["ok"])
            self.assertEqual(samp["mode_used"], "spsamp")
            self.assertGreater(len(samp["text"]), 0)
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=3.0)

    # ------------------------------------------------------------------ 7. CLI Checks

    def test_cli_doctor_and_catalog_json(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(TOOLS_DIR)
        env["OMEGA_HELIX_BIN_DIR"] = str(self.bin_dir)

        p1 = subprocess.run(
            [sys.executable, "-m", "studio.cli", "doctor", "--json"],
            capture_output=True, text=True, env=env, check=False,
        )
        self.assertEqual(p1.returncode, 0, p1.stderr)
        doc = json.loads(p1.stdout)
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["catalog_count"], 73)

        p2 = subprocess.run(
            [sys.executable, "-m", "studio.cli", "catalog", "--search", "mathlib", "--json"],
            capture_output=True, text=True, env=env, check=False,
        )
        self.assertEqual(p2.returncode, 0, p2.stderr)
        cat = json.loads(p2.stdout)
        self.assertTrue(cat["ok"])
        self.assertGreaterEqual(cat["count"], 1)
        self.assertEqual(cat["corpora"][0]["name"], "mathlib")


if __name__ == "__main__":
    unittest.main()
