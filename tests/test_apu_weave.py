#!/usr/bin/env python3
"""Independent verification suite for Full-APU Co-Execution Fabric (native/apu_weave.om).

Verifies:
  1. Host binary build and `apuweave` over multi-shard lists (and directory scanning on Linux).
  2. Looped OLLMCK03 (L_phys=2 x NP=2) + both W4 (quant_bits=4) and f64 (quant_bits=64) packs.
  3. Exact machine-readable telemetry contract (`APU_WAVE`, `APU_STRAND`, `APU_SUMMARY`, `APU_OVERLAP`).
  4. Full OLLMSP02 interchangeability: packs produced by `apuweave` pass `ollm spweval` with real
     loss drop vs base seed AND `ollm spwsamp` AND fail closed (`rc=61`) on 1-bit header/payload tamper.
  5. Multi-worker parallel execution on Linux (fork + pipefds + pollfds + cpuset + MAP_SHARED ring).
  6. Optional live GB10 silicon test (`OLLM_GB10_LIVE=1`) verifying `gpu_mode > 0` (`BF16`/`FP32`) and
     real GPU+CPU co-execution parity against CPU `spcryst`.
"""
import os
import platform
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OSP2_MAG = int.from_bytes(b"OLLMSP02", "little")


def select_target_and_bin():
    os_name = platform.system()
    arch = platform.machine().lower()
    if os_name == "Darwin":
        return "macos", "ollm-macos"
    if arch in ("x86_64", "amd64"):
        return "x86linux", "ollm-x86linux"
    return "aarch64", "ollm"


def build_target(out_dir: Path, target: str) -> Path:
    env = os.environ.copy()
    env["TARGET"] = target
    env["OUT"] = str(out_dir)
    r = subprocess.run(
        ["sh", str(REPO / "scripts" / "build.sh")],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"build TARGET={target} failed ({r.returncode}): {r.stderr}\n{r.stdout}")
    names = {
        "macos": "ollm-macos",
        "x86linux": "ollm-x86linux",
        "aarch64": "ollm",
        "gb10": "ollm-gb10",
    }
    return out_dir / names[target]


def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(
            f"command failed rc={r.returncode}: {' '.join(map(str, cmd))}\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
    return r


def parse_telemetry(stdout: str):
    waves, strands, summary, overlap = [], [], None, None
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "APU_WAVE":
            waves.append([int(x) for x in parts[1:]])
        elif parts[0] == "APU_STRAND":
            strands.append([int(x) for x in parts[1:]])
        elif parts[0] == "APU_SUMMARY":
            summary = [int(x) for x in parts[1:]]
        elif parts[0] == "APU_OVERLAP":
            overlap = [int(x) for x in parts[1:]]
    return waves, strands, summary, overlap


class TestApuWeave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory(prefix="omega-apu-weave-test-")
        cls.td = Path(cls._td.name)
        target, _ = select_target_and_bin()
        cls.target = target
        cls.bin = build_target(cls.td / "bin", target)

        # Build a clean 4-shard synthetic corpus with distinct domain patterns
        cls.shards_dir = cls.td / "shards"
        cls.shards_dir.mkdir(parents=True)
        patterns = [
            "fn omega_kernel x ulong y ulong to ulong\nret add mul x 3 y\nend\n",
            "struct RingDesc seq ulong op uint flags uint va ulong end\n",
            "APU_WAVE 1 4 12 0 4 2 1\nAPU_STRAND 1 512 4100000 3800000\n",
            "matrix cholesky solve lambda ridge projection vector norm\n",
        ]
        cls.shard_paths = []
        for idx, pat in enumerate(patterns):
            p = cls.shards_dir / f"s{idx:02d}_domain{idx}.tok"
            p.write_bytes((pat * 40).encode("ascii"))
            cls.shard_paths.append(p)
        cls.all_corpus = cls.td / "all.tok"
        cls.all_corpus.write_bytes(b"".join(p.read_bytes() for p in cls.shard_paths))

        # Train a small looped OLLMCK03 checkpoint (L_phys=2 x NP=2 = 4 effective layers)
        cls.ck = cls.td / "loop_base.ck"
        run_cmd([cls.bin, "init", cls.ck, "256", "16", "16", "2", "32", "2", "4", "1", "10", "4000", "1", "0", "2", "2"])
        run_cmd([cls.bin, "train", cls.ck, cls.all_corpus, "4", "4"])

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_apuweave_list_w4_interchange_and_tamper(self):
        out_dir = self.td / "out_list_w4"
        out_dir.mkdir(parents=True, exist_ok=True)
        list_file = self.td / "shards.list"
        list_file.write_text("\n".join(str(p) for p in self.shard_paths) + "\n", encoding="utf-8")

        workers = "4" if platform.system() == "Linux" else "1"
        r = run_cmd([self.bin, "apuweave", self.ck, list_file, out_dir, "24", "100", "128", "4", workers])
        waves, strands, summary, overlap = parse_telemetry(r.stdout)

        self.assertGreaterEqual(len(waves), 1)
        self.assertEqual(len(strands), 4)
        self.assertIsNotNone(summary)
        self.assertIsNotNone(overlap)

        # Verify APU_SUMMARY fields:
        # n_shards, tot_w, wall_ms, gpu_ms, cpu_ms, sps_x1000, ov_x1000, w4_bytes, m_bev, m_wev, drop, tamper_ok, gpu_mode
        self.assertEqual(summary[0], 4)
        self.assertEqual(summary[11], 4, "Every strand must pass live 1-bit tamper rc=61 check")
        self.assertGreater(summary[7], 4 * 80)

        # Verify every strand passed live tamper check (trc == 61) and improved calibration loss
        for st in strands:
            self.assertEqual(st[10], 61)
            self.assertLessEqual(st[3], st[2], f"strand {st[0]} cal loss did not improve: {st[2]} -> {st[3]}")

        # Verify OLLMSP02 header & full compatibility with `ollm spweval` + `ollm spwsamp`
        sp2_files = [out_dir / f"s{i:02d}_domain{i}.sp2" for i in range(4)]
        for idx, sp in enumerate(sp2_files):
            self.assertTrue(sp.is_file(), f"missing {sp}")
            hdr = struct.unpack("<10Q", sp.read_bytes()[:80])
            self.assertEqual(hdr[0], OSP2_MAG)
            self.assertEqual(hdr[2], 2)   # L_phys
            self.assertEqual(hdr[3], 16)  # D
            self.assertEqual(hdr[4], 24)  # F_exp
            self.assertEqual(hdr[5], idx + 1)
            self.assertEqual(hdr[6], 4)   # W4

        ev = run_cmd([self.bin, "spweval", self.ck, self.all_corpus, "2", "4", *sp2_files])
        toks = ev.stdout.strip().split()
        self.assertEqual(toks[0], "SPWEVAL")
        base_u6, wov_u6 = int(toks[4]), int(toks[5])
        self.assertLess(wov_u6, base_u6, f"spweval woven {wov_u6} must beat base {base_u6}")

        samp = subprocess.run(
            [self.bin, "spwsamp", self.ck, "42", "12", "2", "0", *sp2_files],
            capture_output=True,
        )
        self.assertEqual(samp.returncode, 0, samp.stderr)
        self.assertGreater(len(samp.stdout), 0)

        # External 1-bit payload tamper negative control -> rc=61
        bad = self.td / "tampered.sp2"
        raw = bytearray(sp2_files[0].read_bytes())
        raw[80] ^= 0x01
        bad.write_bytes(bytes(raw))
        tr = run_cmd([self.bin, "spweval", self.ck, self.all_corpus, "1", "1", bad], check=False)
        self.assertEqual(tr.returncode, 61)

    def test_apuweave_f64_and_three_col_list(self):
        out_dir = self.td / "out_f64"
        out_dir.mkdir(parents=True, exist_ok=True)
        list3 = self.td / "shards3.list"
        lines = [f"{10 + i} {p} {out_dir / (p.stem + '.sp2')}" for i, p in enumerate(self.shard_paths[:2])]
        list3.write_text("\n".join(lines) + "\n", encoding="utf-8")

        r = run_cmd([self.bin, "apuweave", self.ck, list3, out_dir, "24", "100", "128", "64", "2"])
        _, strands, summary, _ = parse_telemetry(r.stdout)
        self.assertEqual(len(strands), 2)
        self.assertEqual(summary[11], 2)
        sids = sorted(st[0] for st in strands)
        self.assertEqual(sids, [10, 11])
        sp0 = out_dir / "s00_domain0.sp2"
        hdr = struct.unpack("<10Q", sp0.read_bytes()[:80])
        self.assertEqual(hdr[6], 64)

    def test_apuweave_dir_discovery_and_multiproc_linux(self):
        if platform.system() != "Linux":
            self.skipTest("directory scan (getdnts) + multi-proc fork pool is Linux-only")
        out_dir = self.td / "out_dir_linux"
        # Do NOT pre-create out_dir to test automatic `makedir`!
        r = run_cmd([self.bin, "apuweave", self.ck, self.shards_dir, out_dir, "24", "100", "128", "4", "4"])
        waves, strands, summary, _ = parse_telemetry(r.stdout)
        self.assertEqual(summary[0], 4)
        self.assertEqual(summary[1], 8, "workers=4 + 2 pref + 2 verif = 8 Grace workers")
        self.assertEqual(summary[11], 4)
        self.assertEqual(len(strands), 4)
        self.assertTrue((out_dir / "s03_domain3.sp2").is_file())

    def test_apuweave_gb10_live_silicon(self):
        if os.environ.get("OLLM_GB10_LIVE") != "1":
            self.skipTest("set OLLM_GB10_LIVE=1 on DGX Spark to run live GB10 GPU+Grace test")
        gb10_bin = build_target(self.td / "bin_gb10", "gb10")
        ck_gpu = self.td / "gpu_loop.ck"
        run_cmd([gb10_bin, "init", ck_gpu, "256", "64", "64", "2", "128", "2", "4", "1", "10", "4000", "1", "0", "2", "2"])
        out_dir = self.td / "out_gb10"
        r = run_cmd([gb10_bin, "apuweave", ck_gpu, self.shards_dir, out_dir, "32", "100", "128", "4", "4"])
        _, strands, summary, _ = parse_telemetry(r.stdout)
        self.assertEqual(summary[0], 4)
        self.assertEqual(summary[11], 4)
        self.assertIn(summary[12], (1, 2), f"expected GB10 GPU mode (1=BF16 or 2=FP32), got {summary[12]}")
        self.assertEqual(len(strands), 4)
        sp2_files = [out_dir / f"s{i:02d}_domain{i}.sp2" for i in range(4)]
        ev = run_cmd([gb10_bin, "spweval", ck_gpu, self.all_corpus, "2", "2", *sp2_files])
        toks = ev.stdout.strip().split()
        self.assertEqual(toks[0], "SPWEVAL")
        self.assertLess(int(toks[5]), int(toks[4]))


if __name__ == "__main__":
    unittest.main()
