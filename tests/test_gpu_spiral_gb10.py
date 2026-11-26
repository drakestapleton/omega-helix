#!/usr/bin/env python3
"""Verification tests for Spiral-2048 GB10 SM121 GPU scaling, parity, and gfcache.

Always safe on macOS / Linux x86_64 / Spark without touching /dev/nvidia0:
  - Test 1: Pure-Omega CPU dynamic-pool scaling for D=2048 (direct-oww) and
    looped Spiral L_phys=8 x NP=4 (LV=32, dynamic opab overflow).
  - Test 2: SM121 nvdisasm disassembly gate (skipped if nvdisasm unavailable)
    checking all 78 kernel slots at default, D=1536, and D=2048 T=512 F=8192
    with bad=0, RMSF/RMSB/CE_BF16 <= 72 GPRs, vectorized LDG/STG.E.64 epilogues,
    and OMMA.SF.16864 NVFP4 Tensor Core codegen (slot 84).
  - Test 3 (opt-in via OLLM_GB10_LIVE=1 inside spark-window): live physical
    GB10 gbfrag, gacmp, gbcmp2 (L_phys=8 x NP=4 and D=2048), live nvfp4
    OMMA.SF.16864, and gfcache OLLMFH01 parity against CPU fcache.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _resolve_cpu_bin() -> Path:
    sys_name = platform.system()
    mach = platform.machine()
    if sys_name == "Darwin":
        cands = [
            ROOT / "build" / "verify-om" / "ollm-macos",
            ROOT / "build" / "ollm-macos",
        ]
    elif mach in ("x86_64", "amd64"):
        cands = [
            ROOT / "build" / "verify-om" / "ollm-x86linux",
            ROOT / "build" / "ollm-x86linux",
        ]
    else:
        cands = [
            ROOT / "build" / "verify-om" / "ollm",
            ROOT / "build" / "ollm",
        ]
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    target = "macos" if sys_name == "Darwin" else ("x86linux" if mach in ("x86_64", "amd64") else "aarch64")
    subprocess.run(
        ["sh", str(ROOT / "scripts" / "build.sh")],
        cwd=str(ROOT),
        env={**os.environ, "TARGET": target},
        check=True,
    )
    return cands[-1]


def _resolve_gb10_bin() -> Path | None:
    for c in (
        ROOT / "build" / "ollm-gb10",
        ROOT / "build" / "verify-om" / "ollm-gb10",
        Path.home() / ".local" / "opt" / "omega-helix" / "bin" / "ollm-gb10",
    ):
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def _find_nvdisasm() -> str | None:
    p = shutil.which("nvdisasm")
    if p:
        return p
    cand = Path("/usr/local/cuda/bin/nvdisasm")
    return str(cand) if cand.is_file() and os.access(cand, os.X_OK) else None


class TestGpuSpiralGB10(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="omega-gpu-spiral-"))
        self.corp = self.tmp / "corp.bin"
        parts = []
        for rel in ("README.md", "docs/LOOP.md"):
            p = ROOT / rel
            if p.is_file():
                parts.append(p.read_bytes())
        raw = b"\n".join(parts)
        while len(raw) < 32768:
            raw += raw
        self.corp.write_bytes(raw[:32768])

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cpu_dynamic_pool_d2048_and_loop8x4(self) -> None:
        if platform.machine() in ("x86_64", "amd64"):
            self.skipTest("x86linux uses static CONTRACT-V3 pools")
        ollm = _resolve_cpu_bin()

        # 1. Looped Spiral L_phys=8 x NP=4 (LV=32) exercises dynamic opab overflow
        loop_ck = self.tmp / "loop8x4.ck"
        r = subprocess.run(
            [str(ollm), "init", str(loop_ck), "256", "32", "64", "4", "128", "8", "2", "1", "10", "3000", "1", "0", "8", "4"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertTrue(r.stdout.strip().startswith("I "), r.stdout)
        ev = subprocess.run(
            [str(ollm), "eval", str(loop_ck), str(self.corp), "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertTrue(ev.stdout.strip().startswith("E 0 2 "), ev.stdout)

        # 2. D=2048 exercises direct-oww mode (LSTR > OWC - LAY0 - D)
        d2048_ck = self.tmp / "d2048.ck"
        r2 = subprocess.run(
            [str(ollm), "init", str(d2048_ck), "256", "32", "2048", "32", "4096", "1", "1", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertTrue(r2.stdout.strip().startswith("I "), r2.stdout)
        ev2 = subprocess.run(
            [str(ollm), "eval", str(d2048_ck), str(self.corp), "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertTrue(ev2.stdout.strip().startswith("E 0 1 "), ev2.stdout)

    def test_sm121_disasm_d2048_and_swizzle(self) -> None:
        nvdisasm = _find_nvdisasm()
        if not nvdisasm or platform.system() != "Linux" or platform.machine() != "aarch64":
            self.skipTest("requires Spark Linux aarch64 + nvdisasm -b SM121")

        for shape in ("256 32 64 2 128 2", "4096 256 1536 24 4096 16", "4096 512 2048 32 8192 8"):
            out_dir = self.tmp / f"disasm_{shape.replace(' ', '_')}"
            r = subprocess.run(
                ["sh", str(ROOT / "tests" / "gpu-disasm.sh"), "--all"],
                cwd=str(ROOT),
                env={**os.environ, "SHAPE": shape, "OUT": str(out_dir), "NVDISASM": nvdisasm},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("GPU_DISASM_OK slots=78 bad=0", r.stdout)

        # Verify D=2048 RMSF/RMSB/RMSF_BF16/RMSB_BF16/CE_BF16 stay inside 72 GPRs,
        # slot 31 uses LDG/STG.E.64, and slot 84 emits SM121 OMMA.SF.16864 NVFP4.
        out_dir = self.tmp / "disasm_detail"
        r_rms = subprocess.run(
            ["sh", str(ROOT / "tests" / "gpu-disasm.sh"), "10", "11", "31", "75", "76", "84", "87"],
            cwd=str(ROOT),
            env={
                **os.environ,
                "SHAPE": "4096 512 2048 32 8192 8",
                "OUT": str(out_dir),
                "NVDISASM": nvdisasm,
                "VERBOSE": "1",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("GPR 10 max R66 need 69 declared 72 OK", r_rms.stdout)
        self.assertIn("GPR 11 max R67 need 70 declared 72 OK", r_rms.stdout)
        self.assertIn("GPR 75 max R66 need 69 declared 72 OK", r_rms.stdout)
        self.assertIn("GPR 76 max R67 need 70 declared 72 OK", r_rms.stdout)
        self.assertIn("GPR 87 max R68 need 71 declared 72 OK", r_rms.stdout)
        self.assertIn("LDG.E.64", r_rms.stdout)
        self.assertIn("STG.E.64", r_rms.stdout)
        self.assertIn("OMMA.SF.16864", r_rms.stdout)

    def test_live_gb10_silicon_parity_and_gfcache(self) -> None:
        if os.environ.get("OLLM_GB10_LIVE") != "1":
            self.skipTest("set OLLM_GB10_LIVE=1 inside an exclusive spark-window to run live GB10 checks")
        gb10 = _resolve_gb10_bin()
        self.assertIsNotNone(gb10, "ollm-gb10 binary not found")
        assert gb10 is not None

        # 1. GBFRAG exact 0-error check across all 10 64x64 + 10 128x128 BF16 HMMA kernels
        frag_ck = self.tmp / "frag.ck"
        subprocess.run([str(gb10), "init", str(frag_ck), "4096", "256", "256", "4", "512", "2", "8", "1", "10", "3000", "1"], check=True, capture_output=True)
        r = subprocess.run([str(gb10), "gbfrag", str(frag_ck)], capture_output=True, text=True, check=True)
        self.assertIn("GBFRAG PASS", r.stdout)

        # 1b. GACMP AdamW parity (256-CTA 128-bit SQN + 256-thr 128-bit ADAM)
        ac_ck = self.tmp / "ac_gpu.ck"
        subprocess.run([str(gb10), "init", str(ac_ck), "256", "64", "256", "8", "512", "2", "2", "1", "10", "3000", "1"], check=True, capture_output=True)
        ra = subprocess.run([str(gb10), "gacmp", str(ac_ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
        self.assertIn("GACMP PASS", ra.stdout)

        # 1c. Live SM121 OMMA.SF.16864 NVFP4 Tensor Core execution
        rnv = subprocess.run([str(gb10), "nvfp4", "4096", "4096", "8192", "5"], capture_output=True, text=True, check=True)
        self.assertIn("NVFP4 4096 4096 8192 5 ", rnv.stdout)
        self.assertIn(" PASS", rnv.stdout)

        # 2. GBCMP2 on Looped Spiral L_phys=8 x NP=4 (LV=32)
        loop_ck = self.tmp / "loop8x4_gpu.ck"
        subprocess.run([str(gb10), "init", str(loop_ck), "256", "64", "512", "8", "1024", "8", "2", "1", "10", "3000", "1", "0", "8", "4"], check=True, capture_output=True)
        r2 = subprocess.run([str(gb10), "gbcmp2", str(loop_ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
        self.assertIn("GBCMP2 PASS", r2.stdout)

        # 3. GBCMP2 on D=2048 (n=64 tiled RMS + direct-oww + batched)
        d2048_ck = self.tmp / "d2048_gpu.ck"
        subprocess.run([str(gb10), "init", str(d2048_ck), "256", "64", "2048", "32", "4096", "1", "2", "1", "10", "3000", "1"], check=True, capture_output=True)
        r3 = subprocess.run([str(gb10), "gbcmp2", str(d2048_ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
        self.assertIn("GBCMP2 PASS", r3.stdout)

        # 4. gfcache OLLMFH01 parity vs CPU fcache on looped checkpoint
        fc_ck = self.tmp / "fc_loop.ck"
        cpu_hc = self.tmp / "cpu.hc"
        gpu_hc = self.tmp / "gpu.hc"
        subprocess.run([str(gb10), "init", str(fc_ck), "256", "64", "256", "4", "512", "4", "4", "1", "10", "3000", "1", "0", "4", "2"], check=True, capture_output=True)
        subprocess.run([str(gb10), "fcache", str(fc_ck), str(self.corp), "8", str(cpu_hc)], check=True, capture_output=True)
        r4 = subprocess.run([str(gb10), "gfcache", str(fc_ck), str(self.corp), "8", str(gpu_hc), "0"], capture_output=True, text=True, check=True)
        self.assertIn("GFCACHE 8 64 256 4 ", r4.stdout)

        a = cpu_hc.read_bytes()
        b = gpu_hc.read_bytes()
        self.assertEqual(len(a), len(b))
        self.assertEqual(a[:48], b[:48])
        W, T, D, _L = struct.unpack("<4Q", a[8:40])
        rec = (T + T * D) * 8
        max_diff = 0.0
        for w in range(W):
            off = 48 + w * rec
            self.assertEqual(a[off : off + T * 8], b[off : off + T * 8])
            h0 = struct.unpack(f"<{T * D}d", a[off + T * 8 : off + rec])
            h1 = struct.unpack(f"<{T * D}d", b[off + T * 8 : off + rec])
            md = max(abs(x - y) for x, y in zip(h0, h1))
            if md > max_diff:
                max_diff = md
        self.assertLess(max_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
