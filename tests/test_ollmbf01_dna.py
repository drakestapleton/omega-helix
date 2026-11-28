#!/usr/bin/env python3
"""tests/test_ollmbf01_dna.py: stdlib-only verifier for OLLMBF01, ollm-dna, and Studio BF16 DNA integration.

Covers:
  1. Pure-Omega builds (TARGET=interchange, TARGET=dna, cross-compile TARGET=dna-gb10),
     AArch64 static ELF header verification, and zero libc/CUDA proof via otool -L / ldd.
  2. Multi-shard BF16 + F32->BF16 synthetic HF safetensors import with exact weight FNV-1a
     checksum parity against an independent pure-Python BF16 reference, `info-bf16` parity,
     and `ollm-dna` `info` / `base-eval` / `cryst` / `spinfo` / `cal-cos` / `spguard-eval`.
  3. Bounded streaming memory test (>= 64 MiB total weights, peak child RSS < 64 MiB).
  4. All 3 negative-control classes failing closed with zero finalized output:
     (a) Truncated safetensors (<8B, hdr>size, hdr>cap, data_offsets[1]>rem, doff0>doff1, short pread)
     (b) Header/dtype/shape mismatch (invalid JSON, missing keys, I32/I64/BOOL/F8_E4M3, bad byte len, bad shape)
     (c) Corrupted OLLMBF01 (1-bit hdr flip rc=61, 1-bit payload flip rc=61, truncated rc=60, bad magic rc=60)
         across `ollm-interchange info-bf16` and `ollm-dna info` / `base-eval` / `spguard-eval`.
  5. Omega Studio CLI `--json` (success + error) and HTTP `/api/runtime/import-hf-bf16`
     with clean `srv.shutdown(); srv.server_close()`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
STUDIO_DIR = TOOLS_DIR / "studio"
for _p in (str(TOOLS_DIR), str(STUDIO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import server as studio_server  # noqa: E402
from runtime.hf_dna import (  # noqa: E402
    import_hf_bf16_trunk,
    read_ollmbf01_header,
    run_bf16_dna_demo,
    verify_ollmbf01,
)

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def _fnv1a_bytes(data: bytes, h: int = FNV_OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def _fnv1a_words64(data: bytes, h: int = FNV_OFFSET) -> int:
    assert len(data) % 8 == 0
    for (w,) in struct.iter_unpack("<Q", data):
        h = ((h ^ w) * FNV_PRIME) & MASK64
    return h


def _align256(n: int) -> int:
    return ((n + 255) // 256) * 256


def _f32_to_bf16_u16(val: float) -> int:
    f32_b = struct.pack("<f", val)
    bits = struct.unpack("<I", f32_b)[0]
    exp = (bits >> 23) & 0xFF
    if exp == 0xFF:
        if bits & 0x7FFFFF:
            return (bits >> 16) | 0x40
        return (bits >> 16) & 0xFFFF
    lsb = (bits >> 16) & 1
    return ((bits + 32767 + lsb) >> 16) & 0xFFFF


def _pack_bf16_tensor(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<H", _f32_to_bf16_u16(v)) for v in vals)


def _pack_f32_tensor(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    hdr: dict[str, Any] = {}
    off = 0
    blobs: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        hdr[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [off, off + len(raw)],
        }
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    while len(hb) % 8 != 0:
        hb += b" "
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _make_qwen3_config(
    *,
    V: int = 256,
    D: int = 32,
    H: int = 4,
    H_kv: int = 2,
    HD: int = 8,
    F: int = 64,
    L: int = 2,
    tie: bool = False,
) -> dict[str, Any]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": V,
        "hidden_size": D,
        "num_attention_heads": H,
        "num_key_value_heads": H_kv,
        "head_dim": HD,
        "intermediate_size": F,
        "num_hidden_layers": L,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tie,
    }


def _build_sharded_qwen3_hf(
    hf_dir: Path,
    *,
    V: int = 256,
    D: int = 32,
    H: int = 4,
    H_kv: int = 2,
    HD: int = 8,
    F: int = 64,
    L: int = 2,
    tie: bool = False,
    mixed_f32: bool = True,
) -> tuple[dict[str, Any], bytes]:
    """Build a synthetic multi-shard Qwen3 HF directory and return (config, expected_payload_bytes)."""
    hf_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_qwen3_config(V=V, D=D, H=H, H_kv=H_kv, HD=HD, F=F, L=L, tie=tie)
    (hf_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    qd = H * HD
    kvd = H_kv * HD

    def gen_vals(seed_off: int, n: int, scale: float = 0.02) -> list[float]:
        # Exactly representable f32 values so Python & OM-2 f32->bf16 match bit-for-bit
        return [(((i + seed_off) % 17) - 8) * scale for i in range(n)]

    shard0: dict[str, tuple[str, list[int], bytes]] = {}
    shard1: dict[str, tuple[str, list[int], bytes]] = {}
    weight_map: dict[str, str] = {}
    payload_parts: list[bytes] = []

    def add_tensor(
        shard_idx: int,
        name: str,
        shape: list[int],
        vals: list[float],
        src_dtype: str,
        out_dtype: str,
    ) -> None:
        raw_in = _pack_f32_tensor(vals) if src_dtype == "F32" else _pack_bf16_u16_list(vals)
        raw_out = _pack_f32_tensor(vals) if out_dtype == "F32" else _pack_bf16_u16_list(vals)
        if out_dtype == "F32" and src_dtype == "BF16":
            # Re-unpack BF16 -> f32 for exact reference
            raw_out = b"".join(
                struct.pack("<I", struct.unpack("<H", raw_in[2 * i : 2 * i + 2])[0] << 16)
                for i in range(len(vals))
            )
        pad = _align256(len(raw_out)) - len(raw_out)
        payload_parts.append(raw_out + (b"\x00" * pad))

        fname = f"model-0000{shard_idx + 1}-of-00002.safetensors"
        target = shard0 if shard_idx == 0 else shard1
        target[name] = (src_dtype, shape, raw_in)
        weight_map[name] = fname

    def _pack_bf16_u16_list(vals: list[float]) -> bytes:
        return _pack_bf16_tensor(vals)

    add_tensor(0, "model.embed_tokens.weight", [V, D], gen_vals(1, V * D), "F32" if mixed_f32 else "BF16", "BF16")
    for li in range(L):
        sidx = 0 if li == 0 else 1
        pfx = f"model.layers.{li}"
        add_tensor(sidx, f"{pfx}.input_layernorm.weight", [D], gen_vals(10 + li, D, 0.05), "BF16" if mixed_f32 else "F32", "F32")
        add_tensor(sidx, f"{pfx}.self_attn.q_proj.weight", [qd, D], gen_vals(20 + li, qd * D), "F32" if mixed_f32 else "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.self_attn.k_proj.weight", [kvd, D], gen_vals(30 + li, kvd * D), "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.self_attn.v_proj.weight", [kvd, D], gen_vals(40 + li, kvd * D), "F32" if mixed_f32 else "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.self_attn.o_proj.weight", [D, qd], gen_vals(50 + li, D * qd), "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.self_attn.q_norm.weight", [HD], gen_vals(60 + li, HD, 0.1), "BF16" if mixed_f32 else "F32", "F32")
        add_tensor(sidx, f"{pfx}.self_attn.k_norm.weight", [HD], gen_vals(70 + li, HD, 0.1), "F32", "F32")
        add_tensor(sidx, f"{pfx}.post_attention_layernorm.weight", [D], gen_vals(80 + li, D, 0.05), "F32", "F32")
        add_tensor(sidx, f"{pfx}.mlp.gate_proj.weight", [F, D], gen_vals(90 + li, F * D), "F32" if mixed_f32 else "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.mlp.up_proj.weight", [F, D], gen_vals(100 + li, F * D), "BF16", "BF16")
        add_tensor(sidx, f"{pfx}.mlp.down_proj.weight", [D, F], gen_vals(110 + li, D * F), "BF16", "BF16")

    add_tensor(1, "model.norm.weight", [D], gen_vals(200, D, 0.05), "BF16" if mixed_f32 else "F32", "F32")
    if not tie:
        add_tensor(1, "lm_head.weight", [V, D], gen_vals(210, V * D), "BF16", "BF16")

    _write_safetensors(hf_dir / "model-00001-of-00002.safetensors", shard0)
    _write_safetensors(hf_dir / "model-00002-of-00002.safetensors", shard1)
    idx_doc = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(idx_doc, indent=2), encoding="utf-8")
    return cfg, b"".join(payload_parts)


def _build_single_shard_hf(hf_dir: Path, *, V: int = 256, D: int = 32, H: int = 4, H_kv: int = 2, HD: int = 8, F: int = 64, L: int = 1) -> None:
    hf_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_qwen3_config(V=V, D=D, H=H, H_kv=H_kv, HD=HD, F=F, L=L, tie=True)
    (hf_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    qd = H * HD
    kvd = H_kv * HD
    tensors: dict[str, tuple[str, list[int], bytes]] = {}

    def add_bf(name: str, shape: list[int]) -> None:
        n = 1
        for s in shape:
            n *= s
        tensors[name] = ("BF16", shape, b"\x00\x3c" * n)

    add_bf("model.embed_tokens.weight", [V, D])
    for li in range(L):
        pfx = f"model.layers.{li}"
        add_bf(f"{pfx}.input_layernorm.weight", [D])
        add_bf(f"{pfx}.self_attn.q_proj.weight", [qd, D])
        add_bf(f"{pfx}.self_attn.k_proj.weight", [kvd, D])
        add_bf(f"{pfx}.self_attn.v_proj.weight", [kvd, D])
        add_bf(f"{pfx}.self_attn.o_proj.weight", [D, qd])
        add_bf(f"{pfx}.self_attn.q_norm.weight", [HD])
        add_bf(f"{pfx}.self_attn.k_norm.weight", [HD])
        add_bf(f"{pfx}.post_attention_layernorm.weight", [D])
        add_bf(f"{pfx}.mlp.gate_proj.weight", [F, D])
        add_bf(f"{pfx}.mlp.up_proj.weight", [F, D])
        add_bf(f"{pfx}.mlp.down_proj.weight", [D, F])
    add_bf("model.norm.weight", [D])
    _write_safetensors(hf_dir / "model.safetensors", tensors)


class TestOllmBf01Dna(unittest.TestCase):
    build_dir: Path
    ic_bin: Path
    dna_bin: Path
    gb10_dna_bin: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.build_dir = Path(tempfile.mkdtemp(prefix="ollm-bf01-test-")).resolve()
        build_sh = REPO_ROOT / "scripts" / "build.sh"
        for tgt in ("interchange", "dna", "dna-gb10"):
            env = dict(os.environ, TARGET=tgt, OUT=str(cls.build_dir))
            proc = subprocess.run(
                ["sh", str(build_sh)],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"build.sh TARGET={tgt} failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
                )
        cls.ic_bin = cls.build_dir / "ollm-interchange"
        cls.dna_bin = cls.build_dir / "ollm-dna"
        cls.gb10_dna_bin = cls.build_dir / "ollm-dna-gb10"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.build_dir, ignore_errors=True)

    # ------------------------------------------------------------------ 1. Build & Linkage Proofs

    def test_01_static_elf_cross_compile_and_zero_libc_cuda_linkage(self) -> None:
        # Cross-compiled GB10 AArch64 static ELF header verification
        raw = self.gb10_dna_bin.read_bytes()
        self.assertGreaterEqual(len(raw), 64)
        self.assertEqual(raw[:4], b"\x7fELF")
        ei_class = raw[4]
        ei_data = raw[5]
        e_type, e_machine = struct.unpack("<HH", raw[16:20])
        self.assertEqual(ei_class, 2)       # ELF64
        self.assertEqual(ei_data, 1)        # little-endian
        self.assertEqual(e_type, 2)         # ET_EXEC
        self.assertEqual(e_machine, 183)    # EM_AARCH64
        e_phoff = struct.unpack("<Q", raw[32:40])[0]
        e_phentsize, e_phnum = struct.unpack("<HH", raw[54:58])
        ptypes = [
            struct.unpack("<I", raw[e_phoff + i * e_phentsize : e_phoff + i * e_phentsize + 4])[0]
            for i in range(e_phnum)
        ]
        self.assertNotIn(3, ptypes, "ollm-dna-gb10 must have no PT_INTERP (must be static zero-libc ELF)")

        # Host binary linkage check (zero libc on Linux, only libSystem.B.dylib kernel shim on Darwin, zero CUDA)
        system = platform.system()
        for b in (self.ic_bin, self.dna_bin):
            if system == "Darwin":
                r = subprocess.run(["otool", "-L", str(b)], capture_output=True, text=True, check=True)
                libs = [ln.strip().split(" ")[0] for ln in r.stdout.splitlines()[1:] if ln.strip()]
                self.assertEqual(libs, ["/usr/lib/libSystem.B.dylib"], f"{b} unexpected dylibs: {libs}")
                self.assertNotRegex(r.stdout.lower(), r"cuda|cublas|c\+\+|metal")
            else:
                if shutil.which("ldd"):
                    r = subprocess.run(["ldd", str(b)], capture_output=True, text=True, check=False)
                    self.assertNotRegex((r.stdout + r.stderr).lower(), r"libc\.so|libcuda|cublas")

    # ------------------------------------------------------------------ 2. Multi-shard Parity & End-to-End DNA

    def test_02_multishard_bf16_f32_parity_and_end_to_end_dna(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            hf_dir = tdp / "hf_qwen3"
            _, expected_payload = _build_sharded_qwen3_hf(
                hf_dir, V=256, D=32, H=4, H_kv=2, HD=8, F=64, L=2, tie=False, mixed_f32=True,
            )
            expected_wfnv = _fnv1a_words64(expected_payload)

            bf_path = tdp / "trunk.bf16"
            imp = import_hf_bf16_trunk(self.ic_bin, hf_dir, bf_path)
            hdr = imp["header"]
            self.assertEqual(hdr["payload_bytes"], len(expected_payload))
            self.assertEqual(hdr["wfnv"], f"0x{expected_wfnv:016x}")

            # Verify payload bytes match byte-for-byte
            actual_payload = bf_path.read_bytes()[256:]
            self.assertEqual(actual_payload, expected_payload)

            # Parity across info-bf16 and ollm-dna info
            ver = verify_ollmbf01(self.ic_bin, bf_path, dna_bin=self.dna_bin)
            self.assertTrue(ver["verified"])
            self.assertEqual(ver["info_bf16"]["trunk_hash"], hdr["trunk_hash"])
            self.assertEqual(ver["dna_info"]["trunk_hash"], hdr["trunk_hash"])

            # End-to-end ollm-dna: base-eval -> cryst (--w4 & --f64) -> spinfo -> cal-cos -> spguard-eval
            dom_toks = [((i * 7 + 3) % 256) for i in range(64)]
            gen_toks = [((i * 13 + 11) % 256) for i in range(64)]
            bins = {"ollm_interchange": self.ic_bin, "ollm_dna": self.dna_bin}
            demo_w4 = run_bf16_dna_demo(
                bins, hf_dir, dom_toks, gen_toks, tdp / "demo_w4", fexp=16, nproto=4, epochs=3, w4=True,
            )
            self.assertTrue(demo_w4["ok"])
            self.assertEqual(demo_w4["base_eval"]["trunk_hash"], hdr["trunk_hash"])
            self.assertGreater(demo_w4["base_eval"]["loss"], 0.0)
            self.assertIn("SP_INFO V2", demo_w4["spinfo_stdout"])
            self.assertIn("CAL_COS TAU=", demo_w4["cal_cos_stdout"])
            self.assertEqual(demo_w4["spguard_eval"]["trunk_hash"], hdr["trunk_hash"])
            self.assertEqual(demo_w4["spguard_eval"]["dom_n"], 63)
            self.assertEqual(demo_w4["spguard_eval"]["gen_n"], 63)

            demo_f64 = run_bf16_dna_demo(
                bins, hf_dir, dom_toks, gen_toks, tdp / "demo_f64", fexp=16, nproto=4, epochs=2, w4=False,
            )
            self.assertTrue(demo_f64["ok"])
            self.assertEqual(demo_f64["spguard_eval"]["trunk_hash"], hdr["trunk_hash"])

    # ------------------------------------------------------------------ 3. Bounded Streaming Memory (>= 64 MiB)

    def test_03_bounded_rss_on_64mib_plus_sharded_import(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            hf_dir = tdp / "hf_big"
            hf_dir.mkdir()
            V, D, H, H_kv, HD, F, L = 4096, 512, 8, 4, 64, 1024, 16
            cfg = _make_qwen3_config(V=V, D=D, H=H, H_kv=H_kv, HD=HD, F=F, L=L, tie=True)
            (hf_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            qd = H * HD
            kvd = H_kv * HD
            weight_map: dict[str, str] = {}

            def write_blob_shard(fname: str, specs: list[tuple[str, str, list[int]]]) -> None:
                hdr: dict[str, Any] = {}
                off = 0
                chunks: list[bytes] = []
                for name, dt, shape in specs:
                    n = 1
                    for s in shape:
                        n *= s
                    nb = n * (2 if dt == "BF16" else 4)
                    hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
                    off += nb
                    unit = b"\x00\x3c" if dt == "BF16" else b"\x00\x00\x80\x3f"
                    chunks.append(unit * n)
                    weight_map[name] = fname
                hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
                while len(hb) % 8 != 0:
                    hb += b" "
                with (hf_dir / fname).open("wb") as fh:
                    fh.write(struct.pack("<Q", len(hb)))
                    fh.write(hb)
                    for ch in chunks:
                        fh.write(ch)

            s0_specs: list[tuple[str, str, list[int]]] = [("model.embed_tokens.weight", "BF16", [V, D])]
            s1_specs: list[tuple[str, str, list[int]]] = []
            for li in range(L):
                dst = s0_specs if li < L // 2 else s1_specs
                p = f"model.layers.{li}"
                dst.extend([
                    (f"{p}.input_layernorm.weight", "BF16", [D]),
                    (f"{p}.self_attn.q_proj.weight", "BF16", [qd, D]),
                    (f"{p}.self_attn.k_proj.weight", "BF16", [kvd, D]),
                    (f"{p}.self_attn.v_proj.weight", "BF16", [kvd, D]),
                    (f"{p}.self_attn.o_proj.weight", "BF16", [D, qd]),
                    (f"{p}.self_attn.q_norm.weight", "BF16", [HD]),
                    (f"{p}.self_attn.k_norm.weight", "BF16", [HD]),
                    (f"{p}.post_attention_layernorm.weight", "BF16", [D]),
                    (f"{p}.mlp.gate_proj.weight", "BF16", [F, D]),
                    (f"{p}.mlp.up_proj.weight", "BF16", [F, D]),
                    (f"{p}.mlp.down_proj.weight", "BF16", [D, F]),
                ])
            s1_specs.append(("model.norm.weight", "BF16", [D]))
            write_blob_shard("shard-00001.safetensors", s0_specs)
            write_blob_shard("shard-00002.safetensors", s1_specs)
            (hf_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8",
            )

            out_bf = tdp / "big.bf16"
            meas_code = (
                "import os, platform, resource, subprocess, sys\n"
                "r = subprocess.run(sys.argv[1:], capture_output=True)\n"
                "ru = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
                "rss = ru.ru_maxrss if platform.system() == 'Darwin' else ru.ru_maxrss * 1024\n"
                "sys.stdout.buffer.write(f'{r.returncode} {rss}\\n'.encode() + r.stdout)\n"
                "sys.stderr.buffer.write(r.stderr)\n"
                "sys.exit(r.returncode)\n"
            )
            res = subprocess.run(
                [sys.executable, "-c", meas_code, str(self.ic_bin), "import-hf-bf16", str(hf_dir), str(out_bf)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            first_line, _, rest = res.stdout.partition("\n")
            rc_s, rss_s = first_line.split()
            self.assertEqual(int(rc_s), 0)
            self.assertIn("IBF1 ", rest)

            hdr = read_ollmbf01_header(out_bf)
            self.assertGreaterEqual(hdr["payload_bytes"], 64 * 1024 * 1024)

            peak_rss_bytes = int(rss_s)
            self.assertLess(
                peak_rss_bytes,
                64 * 1024 * 1024,
                f"import-hf-bf16 peak RSS {peak_rss_bytes / (1024 * 1024):.2f} MiB exceeded 64 MiB streaming bound",
            )

    # ------------------------------------------------------------------ 4. Negative Controls

    def _assert_import_fails_closed(self, hf_dir: Path, out_bf: Path, desc: str) -> None:
        if out_bf.exists():
            out_bf.unlink()
        proc = subprocess.run(
            [str(self.ic_bin), "import-hf-bf16", str(hf_dir), str(out_bf)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0, f"{desc}: expected non-zero exit code")
        self.assertFalse(out_bf.exists(), f"{desc}: finalized output file {out_bf} must not exist on failure")

    def test_04_negative_controls_truncated_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            base_hf = tdp / "base"
            _build_single_shard_hf(base_hf)
            orig_st = (base_hf / "model.safetensors").read_bytes()
            orig_hlen = struct.unpack("<Q", orig_st[:8])[0]

            # 1. File < 8 bytes
            d1 = tdp / "lt8"
            shutil.copytree(base_hf, d1)
            (d1 / "model.safetensors").write_bytes(b"\x01\x02\x03\x04")
            self._assert_import_fails_closed(d1, tdp / "o1.bf16", "file < 8 bytes")

            # 2. Declared header length > file size
            d2 = tdp / "hdr_gt_sz"
            shutil.copytree(base_hf, d2)
            (d2 / "model.safetensors").write_bytes(struct.pack("<Q", len(orig_st) + 4096) + orig_st[8:64])
            self._assert_import_fails_closed(d2, tdp / "o2.bf16", "declared header > file size")

            # 3. Declared header length > cap (983040)
            d3 = tdp / "hdr_gt_cap"
            shutil.copytree(base_hf, d3)
            (d3 / "model.safetensors").write_bytes(struct.pack("<Q", 1_048_576) + orig_st[8:])
            self._assert_import_fails_closed(d3, tdp / "o3.bf16", "declared header > cap")

            # 4. data_offsets[1] > remaining bytes (truncate 16 bytes off payload)
            d4 = tdp / "doff1_gt_rem"
            shutil.copytree(base_hf, d4)
            (d4 / "model.safetensors").write_bytes(orig_st[:-16])
            self._assert_import_fails_closed(d4, tdp / "o4.bf16", "data_offsets[1] > remaining bytes")

            # 5. data_offsets[0] > data_offsets[1]
            d5 = tdp / "doff0_gt_doff1"
            shutil.copytree(base_hf, d5)
            hdr_obj = json.loads(orig_st[8 : 8 + orig_hlen].decode("utf-8"))
            first_k = next(iter(hdr_obj))
            b0, b1 = hdr_obj[first_k]["data_offsets"]
            hdr_obj[first_k]["data_offsets"] = [b1, b0]
            hb = json.dumps(hdr_obj, separators=(",", ":")).encode("utf-8")
            (d5 / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + orig_st[8 + orig_hlen :])
            self._assert_import_fails_closed(d5, tdp / "o5.bf16", "data_offsets[0] > data_offsets[1]")

            # 6. Short pread (declared header length matches truncated header bytes at EOF)
            d6 = tdp / "short_pread"
            shutil.copytree(base_hf, d6)
            (d6 / "model.safetensors").write_bytes(orig_st[: 8 + orig_hlen - 4])
            self._assert_import_fails_closed(d6, tdp / "o6.bf16", "short pread")

    def test_05_negative_controls_header_dtype_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            base_hf = tdp / "base"
            _build_single_shard_hf(base_hf)
            orig_st = (base_hf / "model.safetensors").read_bytes()
            orig_hlen = struct.unpack("<Q", orig_st[:8])[0]
            orig_hdr = json.loads(orig_st[8 : 8 + orig_hlen].decode("utf-8"))
            orig_blob = orig_st[8 + orig_hlen :]

            # 1. Invalid JSON (config.json and safetensors header)
            d_bad_cfg = tdp / "bad_cfg_json"
            shutil.copytree(base_hf, d_bad_cfg)
            (d_bad_cfg / "config.json").write_text("{not valid json", encoding="utf-8")
            self._assert_import_fails_closed(d_bad_cfg, tdp / "m1.bf16", "invalid config.json")

            d_bad_st = tdp / "bad_st_json"
            shutil.copytree(base_hf, d_bad_st)
            bad_hb = b"{broken json"
            (d_bad_st / "model.safetensors").write_bytes(struct.pack("<Q", len(bad_hb)) + bad_hb + orig_blob)
            self._assert_import_fails_closed(d_bad_st, tdp / "m2.bf16", "invalid safetensors JSON")

            # 2. Missing required keys (config key, entry key, missing tensor)
            d_miss_cfg = tdp / "miss_cfg"
            shutil.copytree(base_hf, d_miss_cfg)
            cfg_m = json.loads((d_miss_cfg / "config.json").read_text(encoding="utf-8"))
            del cfg_m["hidden_size"]
            (d_miss_cfg / "config.json").write_text(json.dumps(cfg_m), encoding="utf-8")
            self._assert_import_fails_closed(d_miss_cfg, tdp / "m3.bf16", "missing hidden_size in config.json")

            d_miss_ent = tdp / "miss_ent"
            shutil.copytree(base_hf, d_miss_ent)
            hdr_m = json.loads(json.dumps(orig_hdr))
            del hdr_m["model.norm.weight"]["shape"]
            hb_m = json.dumps(hdr_m, separators=(",", ":")).encode("utf-8")
            (d_miss_ent / "model.safetensors").write_bytes(struct.pack("<Q", len(hb_m)) + hb_m + orig_blob)
            self._assert_import_fails_closed(d_miss_ent, tdp / "m4.bf16", "missing shape in safetensors entry")

            d_miss_ten = tdp / "miss_ten"
            shutil.copytree(base_hf, d_miss_ten)
            hdr_t = json.loads(json.dumps(orig_hdr))
            off0, off1 = hdr_t.pop("model.norm.weight")["data_offsets"]
            # Trim matching bytes from blob and update offsets? Simpler: drop final tensor at end of blob
            hb_t = json.dumps(hdr_t, separators=(",", ":")).encode("utf-8")
            (d_miss_ten / "model.safetensors").write_bytes(struct.pack("<Q", len(hb_t)) + hb_t + orig_blob[:off0])
            self._assert_import_fails_closed(d_miss_ten, tdp / "m5.bf16", "missing required tensor model.norm.weight")

            # 3. Unsupported dtypes (I32, I64, BOOL, F8_E4M3)
            for dt in ("I32", "I64", "BOOL", "F8_E4M3"):
                d_dt = tdp / f"dt_{dt}"
                shutil.copytree(base_hf, d_dt)
                hdr_dt = json.loads(json.dumps(orig_hdr))
                hdr_dt["model.norm.weight"]["dtype"] = dt
                hb_dt = json.dumps(hdr_dt, separators=(",", ":")).encode("utf-8")
                (d_dt / "model.safetensors").write_bytes(struct.pack("<Q", len(hb_dt)) + hb_dt + orig_blob)
                self._assert_import_fails_closed(d_dt, tdp / f"dt_{dt}.bf16", f"unsupported dtype {dt}")

            # 4. Byte length disagreeing with numel * elem_bytes
            d_blen = tdp / "bad_blen"
            shutil.copytree(base_hf, d_blen)
            hdr_bl = json.loads(json.dumps(orig_hdr))
            b0, b1 = hdr_bl["model.norm.weight"]["data_offsets"]
            hdr_bl["model.norm.weight"]["data_offsets"] = [b0, b1 - 2]
            hb_bl = json.dumps(hdr_bl, separators=(",", ":")).encode("utf-8")
            (d_blen / "model.safetensors").write_bytes(struct.pack("<Q", len(hb_bl)) + hb_bl + orig_blob[:-2])
            self._assert_import_fails_closed(d_blen, tdp / "m6.bf16", "byte length disagreeing with numel*elem_bytes")

            # 5. Tensor shape disagreeing with config.json (V, D, H/H_kv, F, L)
            for field, bad_val in (("vocab_size", 128), ("hidden_size", 64), ("num_attention_heads", 8), ("num_key_value_heads", 4), ("intermediate_size", 128), ("num_hidden_layers", 2)):
                d_sh = tdp / f"bad_cfg_{field}"
                shutil.copytree(base_hf, d_sh)
                c_sh = json.loads((d_sh / "config.json").read_text(encoding="utf-8"))
                c_sh[field] = bad_val
                (d_sh / "config.json").write_text(json.dumps(c_sh), encoding="utf-8")
                self._assert_import_fails_closed(d_sh, tdp / f"sh_{field}.bf16", f"shape mismatch for {field}={bad_val}")

    def test_06_negative_controls_corrupted_ollmbf01_across_all_readers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            hf_dir = tdp / "hf"
            _build_single_shard_hf(hf_dir)
            good_bf = tdp / "good.bf16"
            import_hf_bf16_trunk(self.ic_bin, hf_dir, good_bf)

            # Create tokens & valid .sp2 for spguard-eval negative control testing
            dom_u32 = tdp / "dom.u32"
            gen_u32 = tdp / "gen.u32"
            dom_u32.write_bytes(struct.pack("<16I", *([1, 2, 3, 4] * 4)))
            gen_u32.write_bytes(struct.pack("<16I", *([5, 6, 7, 8] * 4)))
            sp2 = tdp / "strand.sp2"
            subprocess.run(
                [str(self.dna_bin), "cryst", str(good_bf), str(dom_u32), str(sp2), "--fexp", "8", "--nproto", "2", "--epochs", "1", "--w4"],
                check=True, capture_output=True,
            )

            raw = bytearray(good_bf.read_bytes())

            # Variant 1: 1-bit flip in OLLMBF01 header (offset 24, inside D field) -> rc=61
            hdr_flip = bytearray(raw)
            hdr_flip[24] ^= 1
            p_hdr_flip = tdp / "hdr_flip.bf16"
            p_hdr_flip.write_bytes(bytes(hdr_flip))

            # Variant 2: 1-bit flip in OLLMBF01 payload (offset 256) -> rc=61
            pay_flip = bytearray(raw)
            pay_flip[256] ^= 1
            p_pay_flip = tdp / "pay_flip.bf16"
            p_pay_flip.write_bytes(bytes(pay_flip))

            # Variant 3: Truncated OLLMBF01 -> rc=60
            p_trunc = tdp / "trunc.bf16"
            p_trunc.write_bytes(bytes(raw[:-64]))

            # Variant 4: Bad magic -> rc=60
            bad_mag = bytearray(raw)
            bad_mag[0] = ord("X")
            p_mag = tdp / "bad_magic.bf16"
            p_mag.write_bytes(bytes(bad_mag))

            cases = [
                ("header_1bit_flip", p_hdr_flip, 61),
                ("payload_1bit_flip", p_pay_flip, 61),
                ("truncated", p_trunc, 60),
                ("bad_magic", p_mag, 60),
            ]
            for label, bf_case, want_rc in cases:
                cmds = [
                    ("ollm-interchange info-bf16", [str(self.ic_bin), "info-bf16", str(bf_case)]),
                    ("ollm-dna info", [str(self.dna_bin), "info", str(bf_case)]),
                    ("ollm-dna base-eval", [str(self.dna_bin), "base-eval", str(bf_case), str(dom_u32)]),
                    ("ollm-dna spguard-eval", [str(self.dna_bin), "spguard-eval", str(bf_case), str(sp2), str(dom_u32), str(gen_u32)]),
                ]
                for cname, argv in cmds:
                    r = subprocess.run(argv, capture_output=True, text=True, check=False)
                    self.assertEqual(
                        r.returncode, want_rc,
                        f"{label} / {cname}: expected rc={want_rc}, got rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}",
                    )

    # ------------------------------------------------------------------ 5. Studio CLI & HTTP Integration

    def test_07_studio_cli_json_and_http_import_hf_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            hf_dir = tdp / "hf"
            _build_single_shard_hf(hf_dir)
            out_bf = tdp / "cli_out.bf16"

            env = dict(
                os.environ,
                OMEGA_HELIX_BIN_DIR=str(self.build_dir),
                OLLM_INTERCHANGE_BIN=str(self.ic_bin),
                OLLM_DNA_BIN=str(self.dna_bin),
            )
            cli_bin = str(REPO_ROOT / "bin" / "omega-helix")

            # CLI success --json
            r_ok = subprocess.run(
                [cli_bin, "import-hf-bf16", str(hf_dir), "--out", str(out_bf), "--json"],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(r_ok.returncode, 0, r_ok.stderr)
            doc_ok = json.loads(r_ok.stdout)
            self.assertTrue(doc_ok["ok"])
            self.assertTrue(doc_ok["verification"]["verified"])
            self.assertTrue(out_bf.is_file())

            # CLI failure --json (must emit valid JSON on stdout and leave no output file)
            bad_out = tdp / "should_not_exist.bf16"
            r_err = subprocess.run(
                [cli_bin, "import-hf-bf16", str(tdp / "missing_hf"), "--out", str(bad_out), "--json"],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(r_err.returncode, 0)
            doc_err = json.loads(r_err.stdout)
            self.assertFalse(doc_err["ok"])
            self.assertTrue(doc_err["error"])
            self.assertFalse(bad_out.exists())

            # HTTP endpoint /api/runtime/import-hf-bf16
            srv = studio_server.create_server(
                host="127.0.0.1", port=0, workspace_dir=tdp / "ws", bin_dir=self.build_dir,
            )
            th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            th.start()
            host, port = srv.server_address[:2]
            base = f"http://{host}:{port}"
            try:
                http_out = tdp / "http_out.bf16"
                req_ok = urllib.request.Request(
                    f"{base}/api/runtime/import-hf-bf16",
                    data=json.dumps({"hf_dir": str(hf_dir), "out_path": str(http_out), "verify": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_ok, timeout=20.0) as resp:
                    self.assertEqual(resp.status, 200)
                    h_ok = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(h_ok["ok"])
                self.assertTrue(h_ok["verification"]["verified"])
                self.assertTrue(http_out.is_file())

                # HTTP error case
                req_bad = urllib.request.Request(
                    f"{base}/api/runtime/import-hf-bf16",
                    data=json.dumps({"hf_dir": str(tdp / "missing_hf")}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req_bad, timeout=10.0)
                try:
                    self.assertEqual(ctx.exception.code, 400)
                    h_err = json.loads(ctx.exception.read().decode("utf-8"))
                    self.assertFalse(h_err["ok"])
                    self.assertTrue(h_err["error"])
                finally:
                    ctx.exception.close()
            finally:
                srv.shutdown()
                srv.server_close()
                th.join(timeout=3.0)


if __name__ == "__main__":
    unittest.main()
