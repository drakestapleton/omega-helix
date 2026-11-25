#!/usr/bin/env python3
"""Verification suite for Omega Helix Phase 7 Sovereign Runtime Capabilities."""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
STUDIO_DIR = TOOLS_DIR / "studio"
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
for _p in (str(TOOLS_DIR), str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chain_runner  # noqa: E402
import server as studio_server  # noqa: E402
from runtime.anti_dna import (  # noqa: E402
    forge_anti_strand,
    run_firewall_decode,
    verify_firewall_receipt,
)
from runtime.compile_green import run_compile_until_green  # noqa: E402
from runtime.hf_dna import run_open_trunk_dna_demo  # noqa: E402
from runtime.sleep_cycle import run_sleep_cycle  # noqa: E402


def _bins() -> dict[str, Path]:
    return chain_runner.ensure_binaries(
        ["ollm", "ollm_univ", "ollm_interchange"],
        bin_dir=os.environ.get("OMEGA_HELIX_BIN_DIR"),
        root=ROOT,
    )


def _write_mini_hf(hf_dir: Path, V: int = 256, D: int = 16, H: int = 2, F: int = 32, L: int = 2) -> None:
    hf_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": V,
        "max_position_embeddings": 16,
        "hidden_size": D,
        "num_attention_heads": H,
        "num_key_value_heads": H,
        "intermediate_size": F,
        "num_hidden_layers": L,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    (hf_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tensors = {}

    def add_t(name: str, shape: list[int]) -> None:
        n = 1
        for s in shape:
            n *= s
        data = struct.pack(f"<{n}f", *(((i % 13) - 6) * 0.01 for i in range(n)))
        tensors[name] = (shape, data)

    add_t("model.embed_tokens.weight", [V, D])
    for li in range(L):
        add_t(f"model.layers.{li}.input_layernorm.weight", [D])
        add_t(f"model.layers.{li}.self_attn.q_proj.weight", [D, D])
        add_t(f"model.layers.{li}.self_attn.k_proj.weight", [D, D])
        add_t(f"model.layers.{li}.self_attn.v_proj.weight", [D, D])
        add_t(f"model.layers.{li}.self_attn.o_proj.weight", [D, D])
        add_t(f"model.layers.{li}.post_attention_layernorm.weight", [D])
        add_t(f"model.layers.{li}.mlp.gate_proj.weight", [F, D])
        add_t(f"model.layers.{li}.mlp.up_proj.weight", [F, D])
        add_t(f"model.layers.{li}.mlp.down_proj.weight", [D, F])
    add_t("model.norm.weight", [D])

    hdr = {}
    off = 0
    blobs = []
    for name, (shape, data) in tensors.items():
        hdr[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    while len(hb) % 8 != 0:
        hb += b" "
    (hf_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


class TestSovereignRuntimePhase7(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bins = _bins()
        cls.ollm = str(cls.bins["ollm"])
        cls.univ = str(cls.bins["ollm_univ"])
        cls.bin_dir = str(Path(cls.ollm).parent)

    def _init_dna_base(self, ck: Path, loop: bool = False) -> None:
        argv = [self.ollm, "init", str(ck), "256", "16", "16", "2", "32", "2", "2", "42", "6", "3000"]
        if loop:
            argv += ["1", "0", "2", "2"]  # L=2, LP=0, LC=2, NP=2 (OLLMCK03)
        subprocess.run(argv, check=True, capture_output=True)

    def test_anti_dna_spclip_and_spguard_ltp0_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bck = tdp / "base.ck"
            self._init_dna_base(bck, loop=False)

            s_clean = forge_anti_strand(self.ollm, bck, "fn clean_code ret 0\n" * 6, tdp / "sc", name="clean")
            s_audit = forge_anti_strand(self.ollm, bck, "assert verify hash strict\n" * 6, tdp / "sa", name="audit")
            s_anti = forge_anti_strand(self.ollm, bck, "hallucinate pii secret\n" * 6, tdp / "anti", name="anti")

            # Negative-weight MODE=0 spclip must succeed
            r_clip = subprocess.run(
                [
                    self.ollm, "spclip", str(bck), "0", "fn ", "6", "0", "42", "-",
                    "2", "1000", "1", s_clean["sp_path"], "-500", "1", s_anti["sp_path"],
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(r_clip.returncode, 0, r_clip.stderr)
            self.assertEqual(len(r_clip.stdout), 6)

            # Spguard with JSD threshold=0 triggers disagreement LTP0 refinement + anti_active=1
            res = run_firewall_decode(
                self.ollm, bck,
                [
                    {"weight_milli": 1000, "packs": [s_clean["sp_path"]]},
                    {"weight_milli": 1000, "packs": [s_audit["sp_path"]]},
                    {"weight_milli": -500, "packs": [s_anti["sp_path"]]},
                ],
                prompt="fn ", n_tokens=6, temp_milli=0, seed=42, disagree_milli=0, max_extra=2,
            )
            self.assertEqual(res["summary"]["anti_active"], 1)
            self.assertGreater(res["summary"]["total_extra_passes"], 0)
            chk = verify_firewall_receipt(res)
            self.assertTrue(chk["verified"], chk)

            # Tamper negative control: flip 1 token -> verification MUST fail
            tampered = json.loads(json.dumps(res))
            tampered["records"][0]["token"] ^= 1
            chk_bad = verify_firewall_receipt(tampered)
            self.assertFalse(chk_bad["verified"])
            self.assertIn(0, chk_bad["token_mismatches"])

    def test_grammar_constrained_gsamp_all_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ck = tdp / "u.ck"
            subprocess.run(
                [self.univ, "init", str(ck), "qwen3", "256", "32", "16", "2", "2", "32", "2", "1", "42", "2", "3000"],
                check=True, capture_output=True,
            )
            # Mode 1: 10/10 seeds produce strictly valid JSON objects
            for sd in range(1, 11):
                r = subprocess.run([self.univ, "gsamp", str(ck), str(sd), "24", "950", "1"], capture_output=True, text=True, check=True)
                gline = [ln for ln in r.stdout.splitlines() if ln.startswith("G ")][0][2:]
                parsed = json.loads(gline)
                self.assertIsInstance(parsed, dict)
            # Mode 2: ascii_ident
            r2 = subprocess.run([self.univ, "gsamp", str(ck), "7", "16", "900", "2"], capture_output=True, text=True, check=True)
            g2 = [ln for ln in r2.stdout.splitlines() if ln.startswith("G ")][0][2:]
            self.assertRegex(g2, r"^[A-Za-z_][A-Za-z0-9_ ]+$")
            # Mode 3: csv_num
            r3 = subprocess.run([self.univ, "gsamp", str(ck), "9", "16", "900", "3"], capture_output=True, text=True, check=True)
            g3 = [ln for ln in r3.stdout.splitlines() if ln.startswith("G ")][0][2:]
            self.assertRegex(g3, r"^[0-9]+(,[0-9]+)*$")

    def test_sleep_cycle_pass_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bck = tdp / "base.ck"
            self._init_dna_base(bck, loop=False)
            held = tdp / "held.bin"
            held.write_bytes(b"heldout regression anchor sequence 0123456789\n" * 6)

            # Accepted run with generous budget
            pass_rcpt = run_sleep_cycle(
                self.ollm, bck,
                ["verified memory trace alpha", "", "verified trace beta"],
                held, tdp / "sleep_ok",
                verifiers=("nonempty",), max_regression_nats=2.0, steps=4, date_tag="2026-11-25",
            )
            self.assertTrue(pass_rcpt["accepted"])
            self.assertTrue(Path(pass_rcpt["sp_path"]).is_file())
            self.assertEqual(pass_rcpt["verify_stats"]["rejected"], 1)

            # Reject & quarantine run when regression budget is impossibly negative (-100 nats improvement required)
            fail_rcpt = run_sleep_cycle(
                self.ollm, bck,
                ["another memory trace"],
                held, tdp / "sleep_fail",
                verifiers=("nonempty",), max_regression_nats=-100.0, steps=4, date_tag="2026-11-26",
            )
            self.assertFalse(fail_rcpt["accepted"])
            self.assertEqual(fail_rcpt["sp_path"], "")
            self.assertTrue(Path(fail_rcpt["quarantined_path"]).is_file())

    def test_compile_until_green_retry_and_crystallize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            ck = tdp / "base.ck"
            self._init_dna_base(ck, loop=False)

            def _cand(attempt: int, last_err: str) -> str:
                if attempt == 1:
                    return '{"incomplete": '  # fails json_schema -> triggers tttsamp
                return '{"ok": true, "fixed": true}'

            res = run_compile_until_green(
                self.ollm, ck, "emit json",
                verifier="json_schema", max_attempts=3, ttt_steps=1,
                out_dir=tdp / "green", candidate_override_fn=_cand,
            )
            self.assertTrue(res["passed"])
            self.assertEqual(len(res["attempts"]), 2)
            self.assertTrue(res["attempts"][0]["ttt_applied"])
            self.assertTrue(Path(res["crystallized"]["sp_path"]).is_file())

    def test_open_trunk_hf_to_dna_demo_http_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            hf_dir = tdp / "hf"
            _write_mini_hf(hf_dir)
            demo = run_open_trunk_dna_demo(
                self.bins, hf_dir,
                "fn a x ulong to ulong ret x end\n" * 6,
                "verify merkle hash digest check\n" * 6,
                "bad invalid hallucinate trap\n" * 6,
                tdp / "hf_demo",
            )
            self.assertTrue(demo["receipt_verification"]["verified"])
            self.assertEqual(demo["spguard"]["summary"]["anti_active"], 1)

            # Studio /api/runtime/* HTTP endpoints
            srv = studio_server.create_server(host="127.0.0.1", port=0, workspace_dir=tdp / "ws", bin_dir=self.bin_dir)
            t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            t.start()
            host, port = srv.server_address[:2]
            base = f"http://{host}:{port}"
            try:
                u_ck = tdp / "u.ck"
                subprocess.run([self.univ, "init", str(u_ck), "qwen3", "256", "32", "16", "2", "2", "32", "2", "1", "42", "2", "3000"], check=True, capture_output=True)
                req = urllib.request.Request(
                    f"{base}/api/runtime/gsamp",
                    data=json.dumps({"ck_path": str(u_ck), "seed": 3, "n_tokens": 20, "mode": 1}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    r_js = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(r_js["ok"])
                    self.assertIsInstance(json.loads(r_js["text"]), dict)
            finally:
                srv.shutdown()
                srv.server_close()

            # CLI gsamp --json
            env = dict(os.environ, OMEGA_HELIX_BIN_DIR=self.bin_dir)
            c_gs = subprocess.run(
                [str(ROOT / "bin" / "omega-helix"), "gsamp", str(u_ck), "--seed", "5", "--n-tokens", "20", "--json"],
                env=env, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(c_gs.returncode, 0, c_gs.stderr)
            self.assertIsInstance(json.loads(json.loads(c_gs.stdout)["text"]), dict)


if __name__ == "__main__":
    unittest.main()
