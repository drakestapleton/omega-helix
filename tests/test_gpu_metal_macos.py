#!/usr/bin/env python3
"""Verification tests for Apple Silicon Metal GPU backend (native/gpu/metal_macos.om).

Safe on all CI runners (Linux x86_64, Linux aarch64, headless macOS VMs, Apple Silicon):
  - Test 1 (all platforms): pure-Omega source audit under native/gpu/ (zero C/ObjC/Xcode).
  - Test 2 (macOS arm64): Mach-O 64-bit arm64 linkage (libSystem.B.dylib only) & codesign.
  - Test 3 (Apple Silicon live Metal GPU; skips cleanly on headless VMs with rc=75):
      mdev device + 17 pipeline compilation, mops (GEMM0/1/2, RMSNORM, ATTN, ADAMW),
      mgemm throughput & accuracy + OLLM_METAL_MUTANT=2 negative control (rc=45).
  - Test 4 (Apple Silicon live Metal GPU): end-to-end transformer parity against CPU
      ollm-macos across small, medium, and looped OLLMCK03 geometries (gforward, gcmp,
      gacmp, geval vs eval, gsample vs sample) + kernel mutant (rc=45) and tampered-weight
      (rc=14 corrupted FNV & recomputed-FNV diverged loss) negative controls.
  - Test 5 (Apple Silicon live Metal GPU): 10-step resident Metal gtrain vs CPU train
      convergence parity + live Metal GPU vs CPU throughput benchmark printout.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fnv1a64(data: bytes) -> int:
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _resolve_cpu_bin() -> Path:
    env_bin = os.environ.get("OLLM_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    sys_name = platform.system()
    mach = platform.machine()
    if sys_name == "Darwin":
        cands = [ROOT / "build" / "verify-om" / "ollm-macos", ROOT / "build" / "ollm-macos"]
        target = "macos"
    elif mach in ("x86_64", "amd64"):
        cands = [ROOT / "build" / "verify-om" / "ollm-x86linux", ROOT / "build" / "ollm-x86linux"]
        target = "x86linux"
    else:
        cands = [ROOT / "build" / "verify-om" / "ollm", ROOT / "build" / "ollm"]
        target = "aarch64"
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    subprocess.run(
        ["sh", str(ROOT / "scripts" / "build.sh")],
        cwd=str(ROOT),
        env={**os.environ, "TARGET": target},
        check=True,
    )
    return cands[-1]


def _resolve_metal_bin() -> Path | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    env_bin = os.environ.get("OLLM_METAL_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    for c in (
        ROOT / "build" / "ollm-metal",
        ROOT / "build" / "verify-om" / "ollm-metal",
        ROOT / "build" / "verify" / "ollm-metal",
    ):
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


class TestGpuMetalMacOS(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="omega-gpu-metal-"))
        self.corp = self.tmp / "corp.txt"
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

    def _require_live_metal(self) -> tuple[Path, Path]:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.skipTest("requires Apple Silicon macOS (Darwin arm64)")
        mtl = _resolve_metal_bin()
        if mtl is None:
            self.skipTest("ollm-metal binary not built (requires ng / OMEGA_MV)")
        assert mtl is not None
        probe = subprocess.run([str(mtl), "mdev"], capture_output=True, text=True)
        if probe.returncode == 75:
            self.skipTest(f"no Metal GPU device on this macOS host: {probe.stderr.strip()}")
        self.assertEqual(probe.returncode, 0, f"mdev failed ({probe.returncode}): {probe.stderr}")
        self.assertIn("MDEV ", probe.stdout)
        self.assertIn("PASS", probe.stdout)
        cpu = _resolve_cpu_bin()
        return mtl, cpu

    def test_pure_omega_source_audit(self) -> None:
        gpu_dir = ROOT / "native" / "gpu"
        for name in ("mpool.om", "metal_io.om", "metal_msl.om", "metal_macos.om"):
            p = gpu_dir / name
            self.assertTrue(p.is_file(), f"missing pure-Omega Metal unit: {p}")
        disallowed_exts = {".c", ".cc", ".cpp", ".h", ".m", ".mm", ".swift", ".xcodeproj"}
        for p in gpu_dir.rglob("*"):
            self.assertNotIn(p.suffix.lower(), disallowed_exts, f"disallowed non-Omega file: {p}")
        build_sh = (ROOT / "scripts" / "build.sh").read_text(encoding="utf-8")
        self.assertIn("macos-metal", build_sh)
        self.assertIn("@ng a64 macho", build_sh)
        metal_src = (gpu_dir / "metal_macos.om").read_text(encoding="utf-8")
        self.assertIn("/System/Library/Frameworks/Metal.framework/Metal", metal_src)
        self.assertIn("MTLCreateSystemDefaultDevice", metal_src)

    def test_macho_binary_and_linkage(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.skipTest("requires Darwin arm64")
        mtl = _resolve_metal_bin()
        if mtl is None:
            self.skipTest("ollm-metal binary not built")
        assert mtl is not None

        f_out = subprocess.run(["file", str(mtl)], capture_output=True, text=True, check=True)
        self.assertIn("Mach-O 64-bit executable arm64", f_out.stdout)

        ot = subprocess.run(["otool", "-L", str(mtl)], capture_output=True, text=True, check=True)
        deps = [line.strip().split()[0] for line in ot.stdout.splitlines()[1:] if line.strip()]
        self.assertEqual(deps, ["/usr/lib/libSystem.B.dylib"], f"unexpected dynamic deps: {deps}")

        cs = subprocess.run(["codesign", "-v", str(mtl)], capture_output=True, text=True)
        self.assertEqual(cs.returncode, 0, cs.stderr)

    def test_live_metal_device_and_kernels(self) -> None:
        mtl, _ = self._require_live_metal()

        r_dev = subprocess.run([str(mtl), "mdev"], capture_output=True, text=True, check=True)
        self.assertIn("unified=1", r_dev.stdout)
        self.assertIn("pipelines=17", r_dev.stdout)
        self.assertIn("PASS", r_dev.stdout)

        r_mops = subprocess.run([str(mtl), "mops"], capture_output=True, text=True, check=True)
        for op in ("GEMM0", "GEMM1", "GEMM2", "RMSNORM", "ATTN", "ADAMW"):
            self.assertIn(f"MOPS {op} ", r_mops.stdout)
        self.assertIn("MOPS  PASS", r_mops.stdout)

        r_gemm = subprocess.run([str(mtl), "mgemm", "512", "512", "512", "10"], capture_output=True, text=True, check=True)
        self.assertIn("MGEMM 512 512 512 10 ", r_gemm.stdout)
        self.assertIn(" PASS", r_gemm.stdout)

        # Negative control: tampered GEMM mutant adds +1.0f to C[0] and must fail with rc=45
        r_mut = subprocess.run(
            [str(mtl), "mgemm", "128", "128", "128", "1"],
            env={**os.environ, "OLLM_METAL_MUTANT": "2"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_mut.returncode, 45, f"expected rc=45 on GEMM mutant, got {r_mut.returncode}: {r_mut.stdout}")
        self.assertIn(" FAIL", r_mut.stdout)

    def test_live_metal_transformer_parity_and_negative_controls(self) -> None:
        mtl, cpu = self._require_live_metal()

        shapes = [
            ("small", ["256", "16", "16", "2", "32", "2", "2", "1", "10", "3000", "1"]),
            ("medium", ["256", "64", "128", "4", "256", "2", "2", "1", "10", "3000", "1"]),
            ("looped", ["256", "32", "64", "4", "128", "2", "2", "1", "10", "3000", "1", "0", "2", "2"]),
        ]

        for label, init_args in shapes:
            ck = self.tmp / f"{label}.ck"
            subprocess.run([str(mtl), "init", str(ck), *init_args], capture_output=True, text=True, check=True)

            rf = subprocess.run([str(mtl), "gforward", str(ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
            self.assertIn("GFORWARD PASS", rf.stdout, f"{label} gforward failed: {rf.stdout}")

            rg = subprocess.run([str(mtl), "gcmp", str(ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
            self.assertIn("GCMP PASS", rg.stdout, f"{label} gcmp failed: {rg.stdout}")
            gc_line = [ln for ln in rg.stdout.splitlines() if ln.startswith("GC ")][0].split()
            cos_e9 = int(gc_line[9])
            self.assertGreaterEqual(cos_e9, 999900000, f"{label} gradient cosine too low: {gc_line}")

            ra = subprocess.run([str(mtl), "gacmp", str(ck), str(self.corp), "0"], capture_output=True, text=True, check=True)
            self.assertIn("GACMP PASS", ra.stdout, f"{label} gacmp failed: {ra.stdout}")

            ev_cpu = subprocess.run([str(cpu), "eval", str(ck), str(self.corp), "4"], capture_output=True, text=True, check=True)
            ev_mtl = subprocess.run([str(mtl), "geval", str(ck), str(self.corp), "4"], capture_output=True, text=True, check=True)
            c_loss = int([ln for ln in ev_cpu.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
            m_loss = int([ln for ln in ev_mtl.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
            self.assertLess(abs(m_loss - c_loss), 1000, f"{label} eval loss mismatch: cpu={c_loss} mtl={m_loss}")

            # Negative control A: dK sign-flip mutant in k_attn_bwd_dk_dv must fail gcmp with rc=45
            rg_mut = subprocess.run(
                [str(mtl), "gcmp", str(ck), str(self.corp), "0"],
                env={**os.environ, "OLLM_METAL_MUTANT": "1"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(rg_mut.returncode, 45, f"{label} mutant expected rc=45, got {rg_mut.returncode}: {rg_mut.stdout}")
            self.assertIn("GCMP FAIL", rg_mut.stdout)

        # Sample parity on small checkpoint
        sm_ck = self.tmp / "small.ck"
        s_cpu = subprocess.run([str(cpu), "sample", str(sm_ck), "42", "16", "0", "Once upon"], capture_output=True, check=True)
        s_mtl = subprocess.run([str(mtl), "gsample", str(sm_ck), "42", "16", "0", "Once upon"], capture_output=True, check=True)
        self.assertEqual(s_cpu.stdout, s_mtl.stdout)

        # Negative control B1: 1-bit weight corruption without updating FNV-1a header trips rc=14
        bad_ck = bytearray(sm_ck.read_bytes())
        bad_ck[120] ^= 0xFF
        bad_path = self.tmp / "bad_fnv.ck"
        bad_path.write_bytes(bad_ck)
        r_bad = subprocess.run([str(mtl), "geval", str(bad_path), str(self.corp), "4"], capture_output=True, text=True)
        self.assertEqual(r_bad.returncode, 14)

        # Negative control B2: tampered weights with valid recomputed FNV-1a diverge from clean oracle
        tamp_ck = bytearray(sm_ck.read_bytes())
        magic = bytes(tamp_ck[:8])
        hdr_len = {b"OLLMCK01": 112, b"OLLMCK02": 120, b"OLLMCK03": 144}[magic]
        p_count = struct.unpack_from("<Q", tamp_ck, 96)[0]
        fnv_off = hdr_len - 8
        for i in range(min(512, p_count)):
            off = hdr_len + i * 8
            val = struct.unpack_from("<d", tamp_ck, off)[0]
            struct.pack_into("<d", tamp_ck, off, val + 2.5)
        new_fnv = _fnv1a64(bytes(tamp_ck[hdr_len:]))
        struct.pack_into("<Q", tamp_ck, fnv_off, new_fnv)
        tamp_path = self.tmp / "tampered_weights.ck"
        tamp_path.write_bytes(tamp_ck)

        ev_clean = subprocess.run([str(cpu), "eval", str(sm_ck), str(self.corp), "4"], capture_output=True, text=True, check=True)
        ev_tamp = subprocess.run([str(mtl), "geval", str(tamp_path), str(self.corp), "4"], capture_output=True, text=True, check=True)
        clean_loss = int([ln for ln in ev_clean.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
        tamp_loss = int([ln for ln in ev_tamp.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
        self.assertGreater(abs(tamp_loss - clean_loss), 100000, f"tampered weights did not diverge: clean={clean_loss} tamp={tamp_loss}")

    def test_live_metal_training_convergence_and_benchmark(self) -> None:
        mtl, cpu = self._require_live_metal()

        ck_cpu = self.tmp / "bench_cpu.ck"
        ck_mtl = self.tmp / "bench_mtl.ck"
        subprocess.run(
            [str(mtl), "init", str(ck_cpu), "256", "64", "128", "4", "256", "4", "4", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        shutil.copy2(ck_cpu, ck_mtl)

        t0 = time.perf_counter_ns()
        r_cpu = subprocess.run([str(cpu), "train", str(ck_cpu), str(self.corp), "10", "5"], capture_output=True, text=True, check=True)
        cpu_ns = time.perf_counter_ns() - t0

        r_mtl = subprocess.run([str(mtl), "gtrain", str(ck_mtl), str(self.corp), "10", "5"], capture_output=True, text=True, check=True)

        cpu_t = [ln.split() for ln in r_cpu.stdout.splitlines() if ln.startswith("T ")]
        mtl_t = [ln.split() for ln in r_mtl.stdout.splitlines() if ln.startswith("T ")]
        self.assertEqual(len(cpu_t), 2)
        self.assertEqual(len(mtl_t), 2)
        cpu_l5, cpu_l10 = int(cpu_t[0][2]), int(cpu_t[1][2])
        mtl_l5, mtl_l10 = int(mtl_t[0][2]), int(mtl_t[1][2])
        self.assertLess(cpu_l10, cpu_l5, "CPU training loss failed to decrease")
        self.assertLess(mtl_l10, mtl_l5, "Metal GPU training loss failed to decrease")
        self.assertLess(abs(mtl_l10 - cpu_l10), 5000, f"Metal vs CPU step-10 loss diverged: mtl={mtl_l10} cpu={cpu_l10}")

        ev_c = subprocess.run([str(cpu), "eval", str(ck_cpu), str(self.corp), "4"], capture_output=True, text=True, check=True)
        ev_m = subprocess.run([str(cpu), "eval", str(ck_mtl), str(self.corp), "4"], capture_output=True, text=True, check=True)
        el_c = int([ln for ln in ev_c.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
        el_m = int([ln for ln in ev_m.stdout.splitlines() if ln.startswith("E ")][0].split()[3])
        self.assertLess(abs(el_m - el_c), 100, f"post-train eval diverged: mtl={el_m} cpu={el_c}")

        mt_line = [ln.split() for ln in r_mtl.stdout.splitlines() if ln.startswith("MT ")][0]
        mtl_ns = int(mt_line[2])
        ms_step_mtl = float(mt_line[3]) / 100.0
        tok_s_mtl = float(mt_line[4]) / 100.0
        ms_step_cpu = cpu_ns / (10 * 1e6)
        tok_s_cpu = (10 * 4 * 64) / (cpu_ns / 1e9)

        rgemm = subprocess.run([str(mtl), "mgemm", "1024", "1024", "1024", "20"], capture_output=True, text=True, check=True)
        gm = [ln.split() for ln in rgemm.stdout.splitlines() if ln.startswith("MGEMM ")][0]
        gflops = float(gm[6]) / 100.0

        print(
            f"\nMETAL_BENCH P=566400 (D=128,L=4,B=4,T=64): "
            f"Metal={ms_step_mtl:.2f} ms/step ({tok_s_mtl:.1f} tok/s) vs "
            f"CPU={ms_step_cpu:.2f} ms/step ({tok_s_cpu:.1f} tok/s) | "
            f"Speedup={cpu_ns / max(mtl_ns, 1):.2f}x | "
            f"SGEMM_1024={gflops:.2f} GFLOP/s"
        )

    def test_live_metal_freeze_layers_and_spext(self) -> None:
        mtl, cpu = self._require_live_metal()

        base_ck = self.tmp / "base.ck"
        spec_ck = self.tmp / "spec.ck"
        pack_sp = self.tmp / "pack.sp"
        app_ck = self.tmp / "app.ck"
        subprocess.run(
            [str(mtl), "init", str(base_ck), "256", "16", "16", "2", "32", "1", "2", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [str(mtl), "growpro", str(base_ck), str(spec_ck), "2", "10", "3000"],
            capture_output=True,
            text=True,
            check=True,
        )

        rg = subprocess.run([str(mtl), "gcmp", str(spec_ck), str(self.corp), "0", "1"], capture_output=True, text=True, check=True)
        self.assertIn("GCF 1 ", rg.stdout)
        gcf_line = [ln for ln in rg.stdout.splitlines() if ln.startswith("GCF ")][0].split()
        self.assertEqual(int(gcf_line[3]), 0, f"frozen gradients non-zero: {gcf_line}")
        gc_line = [ln for ln in rg.stdout.splitlines() if ln.startswith("GC ")][0].split()
        self.assertGreaterEqual(int(gc_line[9]), 999900000, f"frozen gcmp cos too low: {gc_line}")
        self.assertIn("GCMP PASS", rg.stdout)

        ra = subprocess.run([str(mtl), "gacmp", str(spec_ck), str(self.corp), "0", "1"], capture_output=True, text=True, check=True)
        self.assertIn("GAF 1 ", ra.stdout)
        gaf_line = [ln for ln in ra.stdout.splitlines() if ln.startswith("GAF ")][0].split()
        self.assertEqual(int(gaf_line[3]), 0, f"frozen Adam weights changed: {gaf_line}")
        self.assertIn("GACMP PASS", ra.stdout)

        subprocess.run([str(mtl), "gtrain", str(spec_ck), str(self.corp), "5", "5", "1"], capture_output=True, text=True, check=True)
        rsp = subprocess.run([str(mtl), "spext", str(base_ck), str(spec_ck), str(pack_sp)], capture_output=True, text=True, check=True)
        self.assertIn("SPEXT 1 2 ", rsp.stdout)

        subprocess.run([str(cpu), "spapp", str(base_ck), str(pack_sp), str(app_ck)], capture_output=True, text=True, check=True)
        ev_spec = subprocess.run([str(cpu), "eval", str(spec_ck), str(self.corp), "4"], capture_output=True, text=True, check=True)
        ev_app = subprocess.run([str(cpu), "eval", str(app_ck), str(self.corp), "4"], capture_output=True, text=True, check=True)
        self.assertEqual(ev_spec.stdout.split()[2:], ev_app.stdout.split()[2:])

    def test_live_metal_nan_fails_closed(self) -> None:
        mtl, _ = self._require_live_metal()

        ck = self.tmp / "nan_small.ck"
        subprocess.run(
            [str(mtl), "init", str(ck), "256", "16", "16", "2", "32", "2", "2", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
            check=True,
        )

        r_mg = subprocess.run(
            [str(mtl), "mgemm", "128", "128", "128", "1"],
            env={**os.environ, "OLLM_METAL_MUTANT": "3"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_mg.returncode, 45, f"mgemm NaN expected rc=45, got {r_mg.returncode}: {r_mg.stdout}")
        self.assertIn(" FAIL", r_mg.stdout)

        r_mo = subprocess.run(
            [str(mtl), "mops"],
            env={**os.environ, "OLLM_METAL_MUTANT": "3"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_mo.returncode, 45, f"mops NaN gemm expected rc=45, got {r_mo.returncode}: {r_mo.stdout}")
        self.assertIn("MOPS GEMM0 ", r_mo.stdout)
        self.assertIn(" FAIL", r_mo.stdout)

        r_gf = subprocess.run(
            [str(mtl), "gforward", str(ck), str(self.corp), "0"],
            env={**os.environ, "OLLM_METAL_MUTANT": "3"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_gf.returncode, 45, f"gforward NaN expected rc=45, got {r_gf.returncode}: {r_gf.stdout}")
        self.assertIn(" FAIL", r_gf.stdout)

        r_gc = subprocess.run(
            [str(mtl), "gcmp", str(ck), str(self.corp), "0"],
            env={**os.environ, "OLLM_METAL_MUTANT": "4"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_gc.returncode, 45, f"gcmp NaN grad expected rc=45, got {r_gc.returncode}: {r_gc.stdout}")
        self.assertIn("GCMP FAIL", r_gc.stdout)
        gc_line = [ln for ln in r_gc.stdout.splitlines() if ln.startswith("GC ")][0].split()
        self.assertEqual(int(gc_line[9]), 0, f"expected cos_e9=0 on NaN gradient, got {gc_line}")
        self.assertLess(abs(int(gc_line[5]) - int(gc_line[6])), 100, f"expected finite forward losses on mutant 4: {gc_line}")

        r_ga = subprocess.run(
            [str(mtl), "gacmp", str(ck), str(self.corp), "0"],
            env={**os.environ, "OLLM_METAL_MUTANT": "5"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_ga.returncode, 46, f"gacmp NaN weight expected rc=46, got {r_ga.returncode}: {r_ga.stdout}")
        self.assertIn("GACMP FAIL", r_ga.stdout)

        r_ma = subprocess.run(
            [str(mtl), "mops"],
            env={**os.environ, "OLLM_METAL_MUTANT": "5"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_ma.returncode, 45, f"mops NaN adamw expected rc=45, got {r_ma.returncode}: {r_ma.stdout}")
        self.assertIn("MOPS ADAMW ", r_ma.stdout)
        self.assertIn(" FAIL", r_ma.stdout)

    def test_metal_io_pool_limit_and_dir_read(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.skipTest("requires Darwin arm64")
        mtl = _resolve_metal_bin()
        if mtl is None:
            self.skipTest("ollm-metal binary not built")
        assert mtl is not None

        big_ck = self.tmp / "big_pool.ck"
        r_big = subprocess.run(
            [str(mtl), "init", str(big_ck), "4096", "256", "384", "16", "1536", "8", "1", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r_big.returncode, 12, f"expected exit 12 when P > 8388608, got {r_big.returncode}: {r_big.stderr}")

        sm_ck = self.tmp / "io_small.ck"
        subprocess.run(
            [str(mtl), "init", str(sm_ck), "256", "16", "16", "2", "32", "2", "2", "1", "10", "3000", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        dir_in = self.tmp / "dir_input"
        dir_in.mkdir()

        r_dir_data = subprocess.run([str(mtl), "eval", str(sm_ck), str(dir_in), "1"], capture_output=True, text=True)
        self.assertEqual(r_dir_data.returncode, 70, f"odread on directory expected exit 70, got {r_dir_data.returncode}")

        r_dir_ck = subprocess.run([str(mtl), "eval", str(dir_in), str(self.corp), "1"], capture_output=True, text=True)
        self.assertEqual(r_dir_ck.returncode, 70, f"oprd on directory expected exit 70, got {r_dir_ck.returncode}")

    def test_find_ng_ignores_fake_path_ng(self) -> None:
        omc = os.environ.get("OMC")
        if not omc or not Path(omc).is_file():
            for cand in (
                shutil.which("omc"),
                str(Path.home() / ".local" / "bin" / "omc"),
                str(Path.home() / "canon" / "omega-mv" / "seed" / "out" / "omc"),
                str(ROOT.parent / "omega-mv" / "seed" / "out" / "omc"),
                str(ROOT.parent / "omega-mv-macos" / "seed" / "out" / "omc"),
            ):
                if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
                    omc = cand
                    break
        if not omc or not Path(omc).is_file():
            self.skipTest("omc not found for find_ng test")

        fake_bin = self.tmp / "fake_bin"
        fake_bin.mkdir()
        hit = self.tmp / "fake_ng_called"
        fake_ng = fake_bin / "ng"
        fake_ng.write_text(
            f"#!/bin/sh\ntouch '{hit}'\necho 'Angular CLI: 19.0.0'\nexit 0\n",
            encoding="utf-8",
        )
        fake_ng.chmod(0o755)

        base_env = {k: v for k, v in os.environ.items() if k not in ("NG", "OMEGA_MV")}
        base_env["OMC"] = omc
        base_env["TARGET"] = "macos-metal"
        base_env["PATH"] = f"{fake_bin}:{base_env.get('PATH', '')}"

        r_bad = subprocess.run(
            ["sh", str(ROOT / "scripts" / "build.sh")],
            cwd=str(ROOT),
            env={**base_env, "NG": str(fake_ng), "OUT": str(self.tmp / "out_bad_ng")},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(r_bad.returncode, 0)
        self.assertIn("not an executable Omega ng compiler", r_bad.stderr)
        hit.unlink(missing_ok=True)

        sibling_omv = (
            (ROOT.parent / "omega-mv" / "seed" / "ng" / "src" / "ir.om").is_file()
            or (ROOT.parent / "omega-mv-macos" / "seed" / "ng" / "src" / "ir.om").is_file()
        )
        if not sibling_omv:
            empty_home = self.tmp / "empty_home"
            empty_home.mkdir()
            out_no_ng = self.tmp / "out_no_ng"
            r_no_ng = subprocess.run(
                ["sh", str(ROOT / "scripts" / "build.sh")],
                cwd=str(ROOT),
                env={**base_env, "HOME": str(empty_home), "OUT": str(out_no_ng)},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r_no_ng.returncode, 0, f"fake PATH ng was accepted: {r_no_ng.stdout}\n{r_no_ng.stderr}")
            self.assertFalse((out_no_ng / "ollm-metal").exists())
            hit.unlink(missing_ok=True)

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return

        real_ng: Path | None = None
        for cand in (ROOT / "build" / "verify-om" / "ng", ROOT / "build" / "verify" / "ng", ROOT / "build" / "ng"):
            if cand.is_file() and os.access(cand, os.X_OK):
                real_ng = cand
                break
        if real_ng is not None:
            out_real = self.tmp / "out_real_ng"
            out_real.mkdir()
            shutil.copy2(real_ng, out_real / "ng")
            subprocess.run(
                ["sh", str(ROOT / "scripts" / "build.sh")],
                cwd=str(ROOT),
                env={**base_env, "OUT": str(out_real)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue((out_real / "ollm-metal").is_file())
            self.assertFalse(hit.exists(), "build.sh invoked fake PATH ng despite $OUT/ng")

        omv_root: Path | None = None
        for cand_str in (
            os.environ.get("OMEGA_MV"),
            str(Path.home() / "workspace" / "omega-mv-macos"),
            str(Path.home() / "canon" / "omega-mv"),
            str(ROOT.parent / "omega-mv"),
            str(ROOT.parent / "omega-mv-macos"),
        ):
            if cand_str:
                p = Path(cand_str)
                if (p / "seed" / "ng" / "src" / "ir.om").is_file() and (p / "seed" / "ng" / "src" / "a64.om").is_file():
                    omv_root = p
                    break
        if omv_root is not None:
            out_omv = self.tmp / "out_omv"
            subprocess.run(
                ["sh", str(ROOT / "scripts" / "build.sh")],
                cwd=str(ROOT),
                env={**base_env, "OMEGA_MV": str(omv_root), "OUT": str(out_omv)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue((out_omv / "ollm-metal").is_file())
            self.assertFalse(hit.exists(), "build.sh invoked fake PATH ng despite OMEGA_MV")

    def test_mgemm_rejects_overflow_and_oversized_dims(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            self.skipTest("requires Darwin arm64")
        mtl = _resolve_metal_bin()
        if mtl is None:
            self.skipTest("ollm-metal binary not built")
        assert mtl is not None

        bad_cases = [
            ("4294967296", "4294967296", "1", "1"),
            ("65537", "1", "1", "1"),
            ("16384", "16384", "1", "1"),
            ("16", "16", "16", "10001"),
        ]
        for dims in bad_cases:
            r = subprocess.run([str(mtl), "mgemm", *dims], capture_output=True, text=True)
            self.assertEqual(
                r.returncode,
                2,
                f"mgemm {' '.join(dims)} expected rc=2, got {r.returncode}: {r.stdout}\n{r.stderr}",
            )

    def test_metal_exit_codes_and_io_casts(self) -> None:
        gpu_dir = ROOT / "native" / "gpu"
        msl_src = (gpu_dir / "metal_msl.om").read_text(encoding="utf-8")
        self.assertNotIn("exit 75", msl_src, "metal_msl.om must not overload exit 75")

        io_src = (gpu_dir / "metal_io.om").read_text(encoding="utf-8")
        self.assertNotIn("long mio_g64", io_src, "metal_io.om must not narrow-cast mio_g64 to long (trap 5 on -1)")

        mtl_src = (gpu_dir / "metal_macos.om").read_text(encoding="utf-8")
        for fn_name in ("fn mtl_load_pso", "fn mtl_alloc_buf"):
            start = mtl_src.index(fn_name)
            end = mtl_src.index("\nfn ", start + 1)
            body = mtl_src[start:end]
            self.assertNotIn("exit 75", body, f"{fn_name} must not overload exit 75")


if __name__ == "__main__":
    unittest.main()
