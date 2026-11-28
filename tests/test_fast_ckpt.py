#!/usr/bin/env python3
"""Verification tests for OLLMCK05 fast binary checkpoints and mmap cold-start."""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OCK5_MAGIC = b"OLLMCK05"
OCK5_MAGIC_U64 = int.from_bytes(OCK5_MAGIC, "little")
OCK5_HDR_BYTES = 256

FLAG_F32_MASTER = 1 << 0
FLAG_BF16_SHADOW = 1 << 1
FLAG_NVFP4_WCACHE = 1 << 2
FLAG_ZERO_MV = 1 << 3
FLAG_LOOP_CK03 = 1 << 4

FNV64_BASIS = 14695981039346656037
FNV64_PRIME = 1099511628211


def _cmd_u64(name: str) -> int:
    raw = name.encode("ascii")
    assert len(raw) <= 8
    return int.from_bytes(raw.ljust(8, b"\x00"), "little")


def _fnv1a_u64_words(data: bytes) -> int:
    assert len(data) % 8 == 0
    h = FNV64_BASIS
    for (w,) in struct.iter_unpack("<Q", data):
        h = ((h ^ w) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _resolve_host_bin(out_dir: Path | None = None) -> Path | None:
    sys_name = platform.system()
    mach = platform.machine()
    if sys_name == "Darwin":
        names = ["ollm-macos"]
    elif mach in ("x86_64", "amd64"):
        names = ["ollm-x86linux"]
    else:
        names = ["ollm-gb10", "ollm"]

    cands: list[Path] = []
    if os.environ.get("OLLM_BIN"):
        cands.append(Path(os.environ["OLLM_BIN"]))
    if out_dir is not None:
        for n in names:
            cands.append(out_dir / n)
    if os.environ.get("OUT"):
        for n in names:
            cands.append(Path(os.environ["OUT"]) / n)
    for n in names:
        cands.append(ROOT / "build" / n)
        cands.append(ROOT / "build" / "verify-om" / n)
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def _resolve_gb10_bin(out_dir: Path | None = None) -> Path | None:
    cands: list[Path] = []
    if os.environ.get("OLLM_GB10_BIN"):
        cands.append(Path(os.environ["OLLM_GB10_BIN"]))
    if out_dir is not None:
        cands.append(out_dir / "ollm-gb10")
    if os.environ.get("OUT"):
        cands.append(Path(os.environ["OUT"]) / "ollm-gb10")
    cands.extend(
        (
            ROOT / "build" / "ollm-gb10",
            ROOT / "build" / "verify-om" / "ollm-gb10",
        )
    )
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def _parse_ck5_header(raw: bytes) -> dict[str, int]:
    assert len(raw) >= OCK5_HDR_BYTES
    vals = struct.unpack_from("<32Q", raw, 0)
    return {
        "magic": vals[0],
        "version": vals[1],
        "flags": vals[2],
        "V": vals[3],
        "T": vals[4],
        "D": vals[5],
        "H": vals[6],
        "F": vals[7],
        "L": vals[8],
        "B": vals[9],
        "seed": vals[10],
        "step": vals[11],
        "TOTAL": vals[12],
        "lr_bits": vals[13],
        "P": vals[14],
        "G": vals[15],
        "LP": vals[16],
        "LC": vals[17],
        "NP": vals[18],
        "w_off": vals[19],
        "w_len": vals[20],
        "mv_off": vals[21],
        "mv_len": vals[22],
        "wbf_off": vals[23],
        "wbf_len": vals[24],
        "wnv_off": vals[25],
        "wnv_len": vals[26],
        "orig_type": vals[27],
        "total_bytes": vals[28],
        "reserved": vals[29],
        "payload_fnv": vals[30],
        "hdr_fnv": vals[31],
    }


def _parse_t_lines(stdout: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "T":
            out.append((int(parts[1]), int(parts[2])))
    return out


class TestFastCkpt(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="omega-fast-ckpt-"))
        self.corp = self.tmp / "corp.txt"
        raw = b"\n".join(
            (ROOT / p).read_bytes()
            for p in ("README.md", "docs/LOOP.md")
            if (ROOT / p).is_file()
        )
        while len(raw) < 65536:
            raw += raw
        self.corp.write_bytes(raw[:65536])

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ensure_host_bin(self) -> Path:
        b = _resolve_host_bin()
        if b is not None:
            return b
        out_dir = self.tmp / "host_build"
        sys_name = platform.system()
        mach = platform.machine()
        if sys_name == "Darwin":
            tgt = "macos"
        elif mach in ("x86_64", "amd64"):
            tgt = "x86linux"
        else:
            tgt = "gb10"
        subprocess.run(
            ["sh", str(ROOT / "scripts" / "build.sh")],
            cwd=str(ROOT),
            env={**os.environ, "TARGET": tgt, "OUT": str(out_dir)},
            check=True,
            capture_output=True,
            text=True,
        )
        b = _resolve_host_bin(out_dir)
        assert b is not None
        return b

    def test_ck05_constants_and_sources(self) -> None:
        ckpt_src = (ROOT / "native" / "ckpt.om").read_text(encoding="utf-8")
        cli_src = (ROOT / "native" / "cli.om").read_text(encoding="utf-8")
        gg_src = (ROOT / "native" / "gpu" / "gg.om").read_text(encoding="utf-8")
        lio_src = (ROOT / "native" / "lio.om").read_text(encoding="utf-8")

        self.assertIn(f"const OCK5MAG ulong {OCK5_MAGIC_U64}", ckpt_src)
        for cmd, fn_name in (
            ("ckfast", "cmdckfast"),
            ("ckf64", "cmdckf64"),
            ("initfast", "cmdinitfast"),
        ):
            self.assertIn(f"if eq w {_cmd_u64(cmd)}\n{fn_name}\n", cli_src)
        self.assertIn("fn ggload_ck5 ", gg_src)
        self.assertIn("fn ggsave_ck5 ", gg_src)
        self.assertIn("mapnew fd 0 ulong sz 1 2", lio_src)

    def test_cross_compile_targets(self) -> None:
        targets = ["gb10", "x86linux"]
        if platform.system() == "Darwin":
            targets.append("macos")
        for tgt in targets:
            with self.subTest(target=tgt):
                out_dir = self.tmp / f"build_{tgt}"
                subprocess.run(
                    ["sh", str(ROOT / "scripts" / "build.sh")],
                    cwd=str(ROOT),
                    env={**os.environ, "TARGET": tgt, "OUT": str(out_dir)},
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if tgt == "gb10":
                    hdr = (out_dir / "ollm-gb10").read_bytes()[:20]
                    self.assertEqual(hdr[:4], b"\x7fELF")
                    self.assertEqual(int.from_bytes(hdr[18:20], "little"), 183)
                elif tgt == "x86linux":
                    hdr = (out_dir / "ollm-x86linux").read_bytes()[:20]
                    self.assertEqual(hdr[:4], b"\x7fELF")
                    self.assertEqual(int.from_bytes(hdr[18:20], "little"), 62)
                elif tgt == "macos":
                    hdr = (out_dir / "ollm-macos").read_bytes()[:4]
                    self.assertEqual(hdr, b"\xcf\xfa\xed\xfe")

    def test_ck05_roundtrip_all_modes_and_ck_types(self) -> None:
        binp = self._ensure_host_bin()

        # 1. init (OLLMCK02, ZERO_MV) -> ckfast (f32/bf16/nvfp4) -> ckf64 + initfast -> ckf64
        ck_init = self.tmp / "init.ck"
        subprocess.run(
            [str(binp), "init", str(ck_init), "256", "32", "64", "4", "128", "2", "4", "1", "10", "3000", "2"],
            check=True,
            capture_output=True,
        )
        ck5_init = self.tmp / "initfast.ck5"
        subprocess.run(
            [str(binp), "initfast", str(ck5_init), "256", "32", "64", "4", "128", "2", "4", "1", "10", "3000", "2"],
            check=True,
            capture_output=True,
        )

        # Read original f64 weights from init.ck (120-byte OLLMCK02 header)
        init_raw = ck_init.read_bytes()
        self.assertEqual(init_raw[:8], b"OLLMCK02")
        p_count = struct.unpack_from("<Q", init_raw, 96)[0]
        orig_w_f64 = struct.unpack_from(f"<{p_count}d", init_raw, 120)
        expected_w_f32 = [struct.unpack("<f", struct.pack("<f", w))[0] for w in orig_w_f64]

        recovered_bytes: list[bytes] = []
        for mode in ("f32", "bf16", "nvfp4"):
            ck5 = self.tmp / f"m_{mode}.ck5"
            rec = self.tmp / f"rec_{mode}.ck"
            subprocess.run([str(binp), "ckfast", str(ck_init), str(ck5), mode], check=True, capture_output=True)
            subprocess.run([str(binp), "ckf64", str(ck5), str(rec)], check=True, capture_output=True)

            raw5 = ck5.read_bytes()
            hdr = _parse_ck5_header(raw5)
            self.assertEqual(hdr["magic"], OCK5_MAGIC_U64)
            self.assertEqual(hdr["version"], 1)
            self.assertEqual(hdr["P"], p_count)
            self.assertEqual(hdr["total_bytes"], len(raw5))
            self.assertEqual(hdr["hdr_fnv"], _fnv1a_u64_words(raw5[:248]))
            self.assertEqual(hdr["payload_fnv"], _fnv1a_u64_words(raw5[256:]))
            self.assertTrue(hdr["flags"] & FLAG_F32_MASTER)
            self.assertTrue(hdr["flags"] & FLAG_ZERO_MV)
            self.assertEqual(hdr["mv_len"], 0)
            if mode == "f32":
                self.assertFalse(hdr["flags"] & FLAG_BF16_SHADOW)
                self.assertEqual(hdr["wbf_len"], 0)
            else:
                self.assertTrue(hdr["flags"] & FLAG_BF16_SHADOW)
                self.assertEqual(hdr["wbf_len"], 2 * p_count)

            rec_raw = rec.read_bytes()
            self.assertEqual(rec_raw[:8], b"OLLMCK02")
            rec_w = struct.unpack_from(f"<{p_count}d", rec_raw, 120)
            self.assertEqual(list(rec_w), expected_w_f32)
            rec_m = struct.unpack_from(f"<{2 * p_count}d", rec_raw, 120 + 8 * p_count)
            self.assertTrue(all(x == 0.0 for x in rec_m))
            recovered_bytes.append(rec_raw)

        rec_init = self.tmp / "rec_initfast.ck"
        subprocess.run([str(binp), "ckf64", str(ck5_init), str(rec_init)], check=True, capture_output=True)
        recovered_bytes.append(rec_init.read_bytes())
        for b in recovered_bytes[1:]:
            self.assertEqual(b, recovered_bytes[0])

        # 2. Trained OLLMCK01 (non-zero M & V) + OLLMCK03 (looped LP=0, LC=2, NP=3)
        ck1 = self.tmp / "ck1.ck"
        subprocess.run(
            [str(binp), "init", str(ck1), "256", "32", "64", "4", "128", "2", "4", "1", "10", "3000"],
            check=True,
            capture_output=True,
        )
        subprocess.run([str(binp), "train", str(ck1), str(self.corp), "2", "2"], check=True, capture_output=True)
        ck1_5 = self.tmp / "ck1.ck5"
        ck1_rec = self.tmp / "ck1_rec.ck"
        ck1_5b = self.tmp / "ck1_b.ck5"
        ck1_rec2 = self.tmp / "ck1_rec2.ck"
        subprocess.run([str(binp), "ckfast", str(ck1), str(ck1_5), "bf16"], check=True, capture_output=True)
        hdr1 = _parse_ck5_header(ck1_5.read_bytes())
        self.assertFalse(hdr1["flags"] & FLAG_ZERO_MV)
        self.assertEqual(hdr1["mv_len"], 8 * p_count)
        self.assertEqual(hdr1["orig_type"], 1)
        subprocess.run([str(binp), "ckf64", str(ck1_5), str(ck1_rec)], check=True, capture_output=True)
        self.assertEqual(ck1_rec.read_bytes()[:8], b"OLLMCK01")
        subprocess.run([str(binp), "ckfast", str(ck1_rec), str(ck1_5b), "bf16"], check=True, capture_output=True)
        subprocess.run([str(binp), "ckf64", str(ck1_5b), str(ck1_rec2)], check=True, capture_output=True)
        self.assertEqual(ck1_5.read_bytes(), ck1_5b.read_bytes())
        self.assertEqual(ck1_rec.read_bytes(), ck1_rec2.read_bytes())

        ck3 = self.tmp / "ck3.ck"
        subprocess.run(
            [str(binp), "init", str(ck3), "256", "32", "64", "4", "128", "2", "4", "1", "10", "3000", "2", "0", "2", "3"],
            check=True,
            capture_output=True,
        )
        subprocess.run([str(binp), "train", str(ck3), str(self.corp), "2", "2"], check=True, capture_output=True)
        ck3_5 = self.tmp / "ck3.ck5"
        ck3_rec = self.tmp / "ck3_rec.ck"
        subprocess.run([str(binp), "ckfast", str(ck3), str(ck3_5)], check=True, capture_output=True)
        hdr3 = _parse_ck5_header(ck3_5.read_bytes())
        self.assertTrue(hdr3["flags"] & FLAG_LOOP_CK03)
        self.assertEqual(hdr3["NP"], 3)
        subprocess.run([str(binp), "ckf64", str(ck3_5), str(ck3_rec)], check=True, capture_output=True)
        self.assertEqual(ck3_rec.read_bytes()[:8], b"OLLMCK03")
        ev5 = subprocess.run([str(binp), "eval", str(ck3_5), str(self.corp), "4"], check=True, capture_output=True, text=True)
        ev6 = subprocess.run([str(binp), "eval", str(ck3_rec), str(self.corp), "4"], check=True, capture_output=True, text=True)
        self.assertEqual(ev5.stdout, ev6.stdout)

        # Non-loop command (`grow`) must refuse looped OLLMCK05 with rc=17
        r_grow = subprocess.run(
            [str(binp), "grow", str(ck3_5), str(self.tmp / "g.ck"), "4", "10", "3000"],
            capture_output=True,
        )
        self.assertEqual(r_grow.returncode, 17)

    def test_ck05_and_legacy_negative_controls(self) -> None:
        binp = self._ensure_host_bin()
        ck = self.tmp / "base.ck"
        ck5 = self.tmp / "base.ck5"
        subprocess.run(
            [str(binp), "init", str(ck), "256", "32", "64", "4", "128", "2", "4", "1", "10", "3000", "2"],
            check=True,
            capture_output=True,
        )
        subprocess.run([str(binp), "ckfast", str(ck), str(ck5), "bf16"], check=True, capture_output=True)
        raw = bytearray(ck5.read_bytes())
        out_p = self.tmp / "unused.ck"

        # 1. Bad magic -> rc=11
        bad_mag = bytearray(raw)
        bad_mag[0] ^= 1
        p_bad = self.tmp / "bad_magic.ck5"
        p_bad.write_bytes(bad_mag)
        r = subprocess.run([str(binp), "ckf64", str(p_bad), str(out_p)], capture_output=True)
        self.assertEqual(r.returncode, 11)

        # 2. Truncated (< 256 and < total_bytes) -> rc=10
        for cut in (0, 128, len(raw) - 8, len(raw) - 1):
            p_tr = self.tmp / f"trunc_{cut}.ck5"
            p_tr.write_bytes(raw[:cut])
            r = subprocess.run([str(binp), "ckf64", str(p_tr), str(out_p)], capture_output=True)
            self.assertEqual(r.returncode, 10, f"cut={cut} rc={r.returncode}")

        # 3. Trailing bytes -> rc=15
        p_over = self.tmp / "over.ck5"
        p_over.write_bytes(bytes(raw) + b"\x00" * 8)
        r = subprocess.run([str(binp), "ckf64", str(p_over), str(out_p)], capture_output=True)
        self.assertEqual(r.returncode, 15)

        # 4. 1-bit tamper in header and payload -> rc=14
        for off in (8, 16, 24, 152, 224, 240, 248, 256, 512, len(raw) - 1):
            t = bytearray(raw)
            t[off] ^= 1
            p_t = self.tmp / f"tamper_{off}.ck5"
            p_t.write_bytes(t)
            r = subprocess.run([str(binp), "ckf64", str(p_t), str(out_p)], capture_output=True)
            self.assertEqual(r.returncode, 14, f"tamper offset={off} rc={r.returncode}")

        # 5. Legacy OLLMCK02 accelerated mmap loader 1-bit payload tamper -> rc=14
        leg = bytearray(ck.read_bytes())
        leg[-1] ^= 1
        p_leg = self.tmp / "leg_tamper.ck"
        p_leg.write_bytes(leg)
        r = subprocess.run([str(binp), "eval", str(p_leg), str(self.corp), "2"], capture_output=True)
        self.assertEqual(r.returncode, 14)

    def test_live_gb10_ck05_parity_and_cold_start(self) -> None:
        if os.environ.get("OLLM_GB10_LIVE") != "1":
            self.skipTest("set OLLM_GB10_LIVE=1 inside an exclusive spark-window to run live GB10 checks")
        gb10 = _resolve_gb10_bin()
        self.assertIsNotNone(gb10, "ollm-gb10 binary not found")
        assert gb10 is not None

        # Deterministic u16 token stream for V=4096 (15.83M H2H shape)
        tok_path = self.tmp / "tokens_4096.tok"
        n_tok = 131072
        tok_bytes = b"".join(struct.pack("<H", (i * 131 + 7) % 4096) for i in range(n_tok))
        tok_path.write_bytes(tok_bytes)

        ck2_base = self.tmp / "h2h_base.ck"
        subprocess.run(
            [str(gb10), "init", str(ck2_base), "4096", "256", "384", "12", "1536", "8", "64", "1", "60000", "3000", "16"],
            check=True,
            capture_output=True,
        )

        # 1. Bit-exact T 1..5 parity for gtrainb (BF16+AdamW) and gtrain4 (NVFP4)
        for cmd, mode in (("gtrainb", "bf16"), ("gtrain4", "nvfp4")):
            with self.subTest(cmd=cmd, mode=mode):
                ref_ck = self.tmp / f"ref_{cmd}.ck"
                fast_ck = self.tmp / f"fast_{cmd}.ck5"
                shutil.copy2(ck2_base, ref_ck)
                subprocess.run([str(gb10), "ckfast", str(ck2_base), str(fast_ck), mode], check=True, capture_output=True)

                r_ref = subprocess.run(
                    [str(gb10), cmd, str(ref_ck), str(tok_path), "5", "1"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                r_fast = subprocess.run(
                    [str(gb10), cmd, str(fast_ck), str(tok_path), "5", "1"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(_parse_t_lines(r_fast.stdout), _parse_t_lines(r_ref.stdout))
                self.assertEqual(len(_parse_t_lines(r_fast.stdout)), 5)
                self.assertEqual(fast_ck.read_bytes()[:8], OCK5_MAGIC)
                if cmd == "gtrain4":
                    hdr = _parse_ck5_header(fast_ck.read_bytes())
                    self.assertTrue(hdr["flags"] & FLAG_NVFP4_WCACHE)
                    # Second gtrain4 invocation loads pre-quantized W_nv4 directly
                    r_step6 = subprocess.run(
                        [str(gb10), cmd, str(fast_ck), str(tok_path), "6", "1"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(len(_parse_t_lines(r_step6.stdout)), 1)

        # 2. geval parity on .ck5 vs .ck
        r_ev2 = subprocess.run([str(gb10), "geval", str(ck2_base), str(tok_path), "8"], capture_output=True, text=True)
        # Note: V > 256 geval requires merges arg (or V=256 works directly without merges); verify init/eval on V=256
        ck_v256 = self.tmp / "v256.ck"
        ck5_v256 = self.tmp / "v256.ck5"
        subprocess.run(
            [str(gb10), "init", str(ck_v256), "256", "64", "256", "4", "512", "2", "8", "1", "10", "3000", "2"],
            check=True,
            capture_output=True,
        )
        subprocess.run([str(gb10), "ckfast", str(ck_v256), str(ck5_v256), "bf16"], check=True, capture_output=True)
        ev_ck = subprocess.run([str(gb10), "geval", str(ck_v256), str(self.corp), "8"], check=True, capture_output=True, text=True)
        ev_ck5 = subprocess.run([str(gb10), "geval", str(ck5_v256), str(self.corp), "8"], check=True, capture_output=True, text=True)
        self.assertEqual(ev_ck.stdout.splitlines()[0], ev_ck5.stdout.splitlines()[0])

        # 3. med5 cold-start timing (< 2.0 s target on 15.83M)
        seed_ck5 = self.tmp / "bench_seed.ck5"
        subprocess.run([str(gb10), "ckfast", str(ck2_base), str(seed_ck5), "bf16"], check=True, capture_output=True)
        timings: list[float] = []
        for idx in range(5):
            run_ck5 = self.tmp / f"bench_{idx}.ck5"
            shutil.copy2(seed_ck5, run_ck5)
            t0 = time.monotonic()
            subprocess.run(
                [str(gb10), "gtrainb", str(run_ck5), str(tok_path), "5", "5"],
                check=True,
                capture_output=True,
                text=True,
            )
            timings.append(time.monotonic() - t0)
        med5_s = statistics.median(timings)
        self.assertLess(med5_s, 2.0, f"med5={med5_s:.4f}s timings={timings}")


if __name__ == "__main__":
    unittest.main()
