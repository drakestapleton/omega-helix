#!/usr/bin/env python3
"""Independent verification suite for Spiral-2048 Pure-Omega Razor MoE (native/spiral.om).

Tests:
  1. Direct analytical crystallization (`spcryst`) in both W4 (4-bit block-32) and exact f64,
     including on a looped `OLLMCK03` core (`LP=1, LC=2, NP=2`).
  2. Sparse-FFN Helix Router (`spweval`) Top-K routing drops cross-entropy loss vs base.
  3. Autoregressive woven sampling (`spsamp` auto-dispatch + `spwsamp`) runs cleanly.
  4. Merkle integrity: 1-bit tamper in header OR payload is rejected with rc=61;
     base checkpoint mismatch is rejected with rc=62.
  5. W4 block-32 compression ratio vs exact f64 payload.
"""
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_bin() -> str:
    env_bin = os.environ.get("OLLM_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    sys_name = platform.system()
    mach = platform.machine()
    if sys_name == "Darwin":
        name = "ollm-macos"
    elif mach in ("x86_64", "amd64"):
        name = "ollm-x86linux"
    else:
        name = "ollm"
    for cand in (
        os.path.join(ROOT, "build", "verify-om", name),
        os.path.join(ROOT, "build", name),
        os.path.expanduser(f"~/.local/opt/omega-helix/bin/{name}"),
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise RuntimeError(f"Could not locate {name}; run sh scripts/verify-om.sh first")


class TestSpiralWeave(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bin = resolve_bin()
        cls.tmp = tempfile.mkdtemp(prefix="omega-spiral-test-")
        cls.corpus = os.path.join(cls.tmp, "corpus.txt")
        with open(os.path.join(ROOT, "README.md"), "rb") as f1, open(
            os.path.join(ROOT, "docs", "LOOP.md"), "rb"
        ) as f2:
            data = f1.read() + b"\n" + f2.read()
        with open(cls.corpus, "wb") as out:
            out.write(data)
        # Base OLLMCK03 looped core: V=256 T=16 D=16 H=2 F=32 L=3 B=4 LP=1 LC=2 NP=2
        cls.base_ck = os.path.join(cls.tmp, "loop_base.ck")
        subprocess.run(
            [
                cls.bin, "init", cls.base_ck,
                "256", "16", "16", "2", "32", "3", "4",
                "1", "20", "4000", "1", "1", "2", "2",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            [cls.bin, "train", cls.base_ck, cls.corpus, "5", "5"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_crystallize_w4_f64_weval_samp_and_tamper(self) -> None:
        s1 = os.path.join(self.tmp, "strand1_w4.sp2")
        s2 = os.path.join(self.tmp, "strand2_f64.sp2")

        r1 = subprocess.run(
            [self.bin, "spcryst", self.base_ck, self.corpus, s1, "32", "100", "256", "1", "4"],
            check=True,
            text=True,
            capture_output=True,
        )
        parts1 = r1.stdout.strip().split()
        self.assertEqual(parts1[0], "SPCRYST")
        self.assertEqual(int(parts1[1]), 3)   # L_phys
        self.assertEqual(int(parts1[2]), 16)  # D
        self.assertEqual(int(parts1[3]), 32)  # F_exp
        self.assertEqual(int(parts1[4]), 1)   # strand_id
        self.assertEqual(int(parts1[5]), 4)   # quant_bits W4
        self.assertLess(int(parts1[8]), int(parts1[7]), "W4 calibration loss must drop vs base")

        r2 = subprocess.run(
            [self.bin, "spcryst", self.base_ck, self.corpus, s2, "32", "100", "256", "2", "64"],
            check=True,
            text=True,
            capture_output=True,
        )
        parts2 = r2.stdout.strip().split()
        self.assertEqual(parts2[0], "SPCRYST")
        self.assertEqual(int(parts2[5]), 64)  # quant_bits f64
        self.assertLess(int(parts2[8]), int(parts2[7]), "f64 calibration loss must drop vs base")

        # Verify OLLMSP02 magic and W4 compression
        with open(s1, "rb") as f:
            hdr1 = f.read(80)
        self.assertEqual(hdr1[:8], b"OLLMSP02")
        w4_size = os.path.getsize(s1)
        f64_size = os.path.getsize(s2)
        self.assertLess(w4_size * 5, f64_size, "W4 pack should be >5x smaller than f64 pack")

        # Sparse-FFN Helix Router eval (Top-2 across both W4 and f64 strands)
        re = subprocess.run(
            [self.bin, "spweval", self.base_ck, self.corpus, "2", "4", s1, s2],
            check=True,
            text=True,
            capture_output=True,
        )
        eparts = re.stdout.strip().split()
        self.assertEqual(eparts[0], "SPWEVAL")
        self.assertEqual(int(eparts[1]), 2)
        self.assertEqual(int(eparts[2]), 2)
        base_nats = int(eparts[4])
        woven_nats = int(eparts[5])
        self.assertLess(woven_nats, base_nats, f"Woven loss {woven_nats} should beat base {base_nats}")

        # Autoregressive sampling via spsamp auto-dispatch
        rs = subprocess.run(
            [self.bin, "spsamp", self.base_ck, "42", "16", "2", "0", s1, s2],
            check=True,
            capture_output=True,
        )
        self.assertGreaterEqual(len(rs.stdout), 16)

        # 1-bit payload tamper -> rc=61
        tamper_pay = os.path.join(self.tmp, "tamper_pay.sp2")
        with open(s1, "rb") as f:
            raw = bytearray(f.read())
        raw[80] ^= 0x01
        with open(tamper_pay, "wb") as f:
            f.write(raw)
        tp = subprocess.run(
            [self.bin, "spweval", self.base_ck, self.corpus, "1", "1", tamper_pay],
            capture_output=True,
        )
        self.assertEqual(tp.returncode, 61)

        # 1-bit header tamper -> rc=61
        tamper_hdr = os.path.join(self.tmp, "tamper_hdr.sp2")
        with open(s1, "rb") as f:
            raw_h = bytearray(f.read())
        raw_h[40] ^= 0x01
        with open(tamper_hdr, "wb") as f:
            f.write(raw_h)
        th = subprocess.run(
            [self.bin, "spweval", self.base_ck, self.corpus, "1", "1", tamper_hdr],
            capture_output=True,
        )
        self.assertEqual(th.returncode, 61)

        # Base checkpoint mismatch -> rc=62
        other_ck = os.path.join(self.tmp, "other.ck")
        subprocess.run(
            [self.bin, "init", other_ck, "256", "16", "16", "2", "32", "3", "4", "99", "20", "4000"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tm = subprocess.run(
            [self.bin, "spweval", other_ck, self.corpus, "1", "1", s1],
            capture_output=True,
        )
        self.assertEqual(tm.returncode, 62)


if __name__ == "__main__":
    unittest.main()
