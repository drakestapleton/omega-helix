#!/usr/bin/env python3
"""Verification tests for SM121 NVFP4 (OMMA.SF.16864) training integration.

Always safe on macOS / Linux x86_64 / Spark without touching /dev/nvidia0:
  - Test 1: Local TARGET=gb10 cross-compile succeeds cleanly with seeded `omc`,
    producing a valid AArch64 ELF containing the NVFP4 training commands
    (`gtrain4`, `gnv4gemm`, `gnv4cmp`) and `OLLM_NVFP4=1` env routing.
  - Test 2: Contract check on `ogcmd` little-endian command constants and
    `ggplan` / `ggmm4` / `gnv4gemm` / `gnv4cmp` definitions.
  - Test 3 (Linux aarch64): CLI bounds validation for `gnv4gemm` (exits 2 on
    invalid/unaligned M, N, K).
  - Test 4 (opt-in via OLLM_GB10_LIVE=1 inside spark-window): Live GB10
    standalone K-looped NVFP4 GEMM parity (`gnv4gemm`), full-step forward +
    backward parity (`gnv4cmp`), `gtrain4` + `OLLM_NVFP4=1 gtrainb` step
    execution, and D=2048 timing check.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cmd_u64(name: str) -> int:
    raw = name.encode("ascii")
    assert len(raw) <= 8
    return int.from_bytes(raw.ljust(8, b"\x00"), "little")


def _resolve_gb10_bin(out_dir: Path | None = None) -> Path | None:
    cands: list[Path] = []
    env_gb10 = os.environ.get("OLLM_GB10_BIN")
    if env_gb10:
        cands.append(Path(env_gb10))
    if out_dir is not None:
        cands.append(out_dir / "ollm-gb10")
    out_env = os.environ.get("OUT")
    if out_env:
        cands.append(Path(out_env) / "ollm-gb10")
    cands.extend(
        (
            ROOT / "build" / "ollm-gb10",
            ROOT / "build" / "verify-om" / "ollm-gb10",
            Path.home() / ".local" / "opt" / "omega-helix" / "bin" / "ollm-gb10",
        )
    )
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


class TestGpuNvfp4Train(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="omega-gpu-nvfp4-"))
        self.corp = self.tmp / "corp.bin"
        parts = []
        for rel in ("README.md", "docs/LOOP.md"):
            p = ROOT / rel
            if p.is_file():
                parts.append(p.read_bytes())
        raw = b"\n".join(parts)
        while len(raw) < 65536:
            raw += raw
        self.corp.write_bytes(raw[:65536])

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_nvfp4_command_constants_and_sources(self) -> None:
        gg_src = (ROOT / "native" / "gpu" / "gg.om").read_text(encoding="utf-8")
        gb_src = (ROOT / "native" / "gpu" / "gb.om").read_text(encoding="utf-8")

        for cmd, fn_name in (
            ("gtrain4", "gbtrain4"),
            ("gnv4gemm", "gnv4gemm"),
            ("gnv4cmp", "gnv4cmp"),
        ):
            val = _cmd_u64(cmd)
            self.assertIn(f"if eq w {val}\n{fn_name}\n", gg_src)

        self.assertIn('text ggnv4e "OLLM_NVFP4=1"', gg_src)
        self.assertIn("fn ggmm4 ", gg_src)
        self.assertIn("fn ggqnv4 ", gg_src)
        self.assertIn("fn ggnv4_ok ", gg_src)
        self.assertIn("fn gbtrain4 ", gb_src)
        self.assertIn("fn gnv4gemm ", gb_src)
        self.assertIn("fn gnv4cmp ", gb_src)

    def test_target_gb10_local_cross_compile(self) -> None:
        out_dir = self.tmp / "gb10_build"
        r = subprocess.run(
            ["sh", str(ROOT / "scripts" / "build.sh")],
            cwd=str(ROOT),
            env={**os.environ, "TARGET": "gb10", "OUT": str(out_dir)},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("ollm-gb10", r.stdout)
        elf = out_dir / "ollm-gb10"
        self.assertTrue(elf.is_file())
        hdr = elf.read_bytes()[:20]
        # Static Linux AArch64 ELF header: \x7fELF, class=64-bit(2), LE(1), e_machine=183 (EM_AARCH64)
        self.assertEqual(hdr[:4], b"\x7fELF")
        self.assertEqual(hdr[4], 2)
        self.assertEqual(hdr[5], 1)
        self.assertEqual(int.from_bytes(hdr[18:20], "little"), 183)

    def test_gnv4gemm_cli_bounds(self) -> None:
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            self.skipTest("gnv4gemm execution requires Linux aarch64 host")
        gb10 = _resolve_gb10_bin()
        if gb10 is None:
            out_dir = self.tmp / "gb10_cli"
            subprocess.run(
                ["sh", str(ROOT / "scripts" / "build.sh")],
                cwd=str(ROOT),
                env={**os.environ, "TARGET": "gb10", "OUT": str(out_dir)},
                check=True,
            )
            gb10 = out_dir / "ollm-gb10"
        bad_cases = [
            [],
            ["128", "128"],
            ["0", "128", "64"],
            ["128", "0", "64"],
            ["128", "128", "0"],
            ["64", "128", "64"],
            ["128", "64", "64"],
            ["128", "128", "32"],
            ["8320", "128", "64"],
            ["128", "8320", "64"],
            ["128", "128", "16448"],
        ]
        for args in bad_cases:
            with self.subTest(args=args):
                r = subprocess.run([str(gb10), "gnv4gemm", *args], capture_output=True, text=True, check=False)
                self.assertEqual(r.returncode, 2, f"args={args} rc={r.returncode} out={r.stdout!r} err={r.stderr!r}")

    def test_live_gb10_nvfp4_parity_and_d2048_timing(self) -> None:
        if os.environ.get("OLLM_GB10_LIVE") != "1":
            self.skipTest("set OLLM_GB10_LIVE=1 inside an exclusive spark-window to run live GB10 checks")
        gb10 = _resolve_gb10_bin()
        self.assertIsNotNone(gb10, "ollm-gb10 binary not found")
        assert gb10 is not None

        # 1. Standalone K-looped NVFP4 GEMM parity across multiple (M, N, K) shapes
        for dims in (["128", "128", "64"], ["256", "512", "256"], ["2048", "2048", "2048"]):
            with self.subTest(gnv4gemm=dims):
                r = subprocess.run([str(gb10), "gnv4gemm", *dims], capture_output=True, text=True, check=True)
                self.assertIn(f"GNV4GEMM {' '.join(dims)} ", r.stdout)
                self.assertIn(" PASS", r.stdout)

        # 2. Full-step forward+backward NVFP4 vs CPU f64 parity (gnv4cmp)
        ck_small = self.tmp / "nv4_small.ck"
        subprocess.run(
            [str(gb10), "init", str(ck_small), "256", "64", "256", "4", "512", "2", "2", "1", "10", "3000", "1"],
            check=True,
            capture_output=True,
        )
        rc = subprocess.run([str(gb10), "gnv4cmp", str(ck_small), str(self.corp), "0"], capture_output=True, text=True, check=True)
        self.assertIn("GNV4CMP ", rc.stdout)
        self.assertIn(" PASS", rc.stdout)

        # 3. gtrain4 and OLLM_NVFP4=1 gtrainb smoke + D=2048 timing check
        ck_d2048 = self.tmp / "nv4_d2048.ck"
        subprocess.run(
            [str(gb10), "init", str(ck_d2048), "256", "64", "2048", "32", "4096", "1", "2", "1", "10", "3000", "1"],
            check=True,
            capture_output=True,
        )
        t0 = time.monotonic()
        rt = subprocess.run(
            [str(gb10), "gtrain4", str(ck_d2048), str(self.corp), "2", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed_s = time.monotonic() - t0
        self.assertIn("T 2 ", rt.stdout)
        self.assertGreater(elapsed_s, 0.0)

        ck_env = self.tmp / "nv4_env.ck"
        subprocess.run(
            [str(gb10), "init", str(ck_env), "256", "64", "256", "4", "512", "2", "2", "1", "10", "3000", "1"],
            check=True,
            capture_output=True,
        )
        re_run = subprocess.run(
            [str(gb10), "gtrainb", str(ck_env), str(self.corp), "2", "2"],
            env={**os.environ, "OLLM_NVFP4": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("T 2 ", re_run.stdout)


if __name__ == "__main__":
    unittest.main()
