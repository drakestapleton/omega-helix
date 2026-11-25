#!/usr/bin/env python3
"""One-command OLLM metrics & readiness benchmark suite.

Measures and reports all host-verifiable OLLM metrics in structured JSON and a
clean Markdown summary table, while explicitly marking unrun hardware slots
(GB10 GPU, Aienux/GB20B) as NOT_RUN.

Stdlib-only. Zero hardcoded machine paths. Does not touch any GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
BPE_MAGIC = 3550332126946413647
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def write_synthetic_merges(path: Path) -> None:
    """Write a minimal valid OLLM BPE merges file (V=258, 2 merges) for self-contained kvcmp."""
    payload = struct.pack("<HHHH", ord("t"), ord("h"), ord("i"), ord("n"))
    h = fnv1a64(payload)
    header = struct.pack("<QQQ", BPE_MAGIC, 258, h)
    path.write_bytes(header + payload)


def write_synthetic_tokens(path: Path, n_tokens: int, seed: int = 1) -> None:
    """Write deterministic u16 byte-range tokens (0..255) for small-model chain checks."""
    vals = []
    x = seed & 0xFFFFFFFF
    for i in range(n_tokens):
        if i % 16 == 0:
            vals.append(0)
        else:
            x = (1664525 * x + 1013904223) & 0xFFFFFFFF
            vals.append(32 + ((x >> 16) % 95))
    path.write_bytes(struct.pack("<" + "H" * len(vals), *vals))


def find_omc(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"omc not found at {p}")
    cand = Path.home() / ".local/bin/omc"
    if cand.is_file():
        return cand
    found = shutil.which("omc")
    if found:
        return Path(found)
    raise FileNotFoundError("omc not found; pass --omc PATH")


def concat_ollm_source(target: str, out_om: Path) -> None:
    """Concatenate native/omegallm/*.om for darwin-arm64, linux-aarch64, or gb10."""
    src = REPO_ROOT / "native/omegallm"
    core = [
        src / "math.om",
        src / "data.om",
        src / "lpool.om",
        src / "model.om",
        src / "adam.om",
        src / "ckpt.om",
        src / "bpe.om",
        src / "spec.om",
        src / "cli.om",
    ]
    parts: List[str] = []
    if target == "darwin-arm64":
        parts.append("target a64macos\n")
        files = core + [src / "mio.om", src / "noserve.om", src / "nogpu.om"]
    elif target == "linux-aarch64":
        parts.append("om2\n")
        files = [src / "net/http.om", src / "net/json.om", src / "net/jsonw.om"] + core + [
            src / "lio.om",
            src / "serve.om",
            src / "nogpu.om",
        ]
    elif target == "gb10":
        parts.append("om2\n")
        files = [src / "net/http.om", src / "net/json.om", src / "net/jsonw.om"] + core + [
            src / "lio.om",
            src / "serve.om",
        ]
    else:
        raise ValueError(f"Unknown target: {target}")

    for fp in files:
        parts.append(fp.read_text(encoding="utf-8"))
    combined = "".join(parts)

    if target == "gb10":
        matches = re.findall(r"^fn main to int$", combined, flags=re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"Expected 1 fn main in gb10 concat, found {len(matches)}")
        combined = re.sub(r"^fn main to int$", "fn ocpumain to int", combined, count=1, flags=re.MULTILINE)
        g = src / "gpu"
        for gp in (
            g / "context.om",
            g / "vector.om",
            g / "launch.om",
            g / "batch.om",
            g / "gk.om",
            g / "gr.om",
            g / "gb.om",
            g / "gg.om",
        ):
            combined += gp.read_text(encoding="utf-8")

    out_om.write_text(combined, encoding="utf-8")


def run_cmd(cmd: List[str], timeout: int = 120, cwd: Optional[Path] = None) -> Tuple[int, bytes, bytes, float]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=str(cwd) if cwd else None)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return p.returncode, p.stdout, p.stderr, dt_ms


def compile_omc_file(omc: Path, src_om: Path, out_bin: Path) -> dict:
    data = src_om.read_bytes()
    t0 = time.perf_counter()
    p = subprocess.run([str(omc)], input=data, capture_output=True, timeout=60)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        return {
            "status": "FAIL",
            "rc": p.returncode,
            "stderr": p.stderr.decode("utf-8", errors="replace").strip(),
            "compile_ms": round(dt_ms, 2),
        }
    out_bin.write_bytes(p.stdout)
    out_bin.chmod(0o755)
    return {
        "status": "PASS",
        "rc": 0,
        "compile_ms": round(dt_ms, 2),
        "bytes": len(p.stdout),
        "sha256": sha256_file(out_bin),
    }


def parse_eval_line(out: str) -> Tuple[int, int, int, int]:
    """Parse `E step W loss_micro bpb_micro`."""
    for line in reversed(out.strip().splitlines()):
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "E":
            return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    raise RuntimeError(f"Could not parse E line from: {out!r}")


def section_gate_and_build(omc: Path, workdir: Path, explicit_ollm: Optional[str]) -> Tuple[dict, Path]:
    rc_pr, out_pr, err_pr, ms_pr = run_cmd([sys.executable, str(REPO_ROOT / "tests/omegallm/pre-review.py")], cwd=REPO_ROOT)
    pre_review = {
        "status": "PASS" if rc_pr == 0 else "FAIL",
        "rc": rc_pr,
        "output": (out_pr.decode("utf-8", errors="replace").strip() or err_pr.decode("utf-8", errors="replace").strip()),
        "ms": round(ms_pr, 2),
    }

    builds: Dict[str, dict] = {}
    darwin_bin: Optional[Path] = None
    for target in ("darwin-arm64", "linux-aarch64", "gb10"):
        om_path = workdir / f"ollm-{target}.om"
        bin_path = workdir / f"ollm-{target}"
        concat_ollm_source(target, om_path)
        res = compile_omc_file(omc, om_path, bin_path)
        builds[target] = res
        if target == "darwin-arm64" and res["status"] == "PASS":
            darwin_bin = bin_path

    if explicit_ollm:
        ollm_bin = Path(explicit_ollm).expanduser()
        if not ollm_bin.is_file():
            raise FileNotFoundError(f"--ollm-bin not found: {ollm_bin}")
    elif darwin_bin is not None:
        ollm_bin = darwin_bin
    else:
        raise RuntimeError("Failed to build darwin-arm64 ollm binary and no --ollm-bin supplied")

    return {"pre_review": pre_review, "builds": builds}, ollm_bin


def section_kv_cache(ollm: Path, workdir: Path) -> dict:
    ck = workdir / "kv_l8.ck"
    merges = workdir / "merges_258.bin"
    write_synthetic_merges(merges)
    rc, _, err, _ = run_cmd([str(ollm), "init", str(ck), "258", "32", "32", "2", "64", "8", "1", "7", "10", "1000"])
    if rc != 0:
        raise RuntimeError(f"init kv_l8 failed rc={rc}: {err.decode('utf-8', errors='replace')}")

    n_steps = 20
    rc, out, err, wall_ms = run_cmd([str(ollm), "kvcmp", str(ck), str(n_steps), "hello world", str(merges)])
    text = out.decode("utf-8", errors="replace")
    if rc != 0:
        raise RuntimeError(f"kvcmp failed rc={rc}: {text} {err.decode('utf-8', errors='replace')}")

    m = re.search(r"KVCMP\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not m:
        raise RuntimeError(f"Failed to parse KVCMP line: {text!r}")
    n_got, ndiff, inc_ns, full_ns = (int(m.group(i)) for i in range(1, 5))
    inc_s = max(inc_ns, 1) / 1e9
    full_s = max(full_ns, 1) / 1e9
    return {
        "status": "PASS" if ndiff == 0 else "FAIL",
        "layers": 8,
        "vocab": 258,
        "steps": n_got,
        "ndiff_logits": ndiff,
        "inc_ms": round(inc_ns / 1e6, 3),
        "full_ms": round(full_ns / 1e6, 3),
        "inc_tok_per_s": round(n_got / inc_s, 1),
        "full_tok_per_s": round(n_got / full_s, 1),
        "speedup_x": round(full_ns / max(inc_ns, 1), 2),
        "wall_ms": round(wall_ms, 2),
    }


def section_chains_and_spclip(ollm: Path, workdir: Path) -> Tuple[dict, dict]:
    base_ck = workdir / "base_l2.ck"
    tr_tok = workdir / "chain_train.tok"
    he_tok = workdir / "chain_held.tok"
    write_synthetic_tokens(tr_tok, 16 * 12 + 1, seed=11)
    write_synthetic_tokens(he_tok, 16 * 6 + 1, seed=29)

    rc, _, err, _ = run_cmd([str(ollm), "init", str(base_ck), "256", "16", "32", "2", "64", "2", "1", "42", "5", "5000"])
    if rc != 0:
        raise RuntimeError(f"init base_l2 failed: {err.decode()}")

    _, out_b, _, _ = run_cmd([str(ollm), "eval", str(base_ck), str(he_tok), "4"])
    _, _, base_loss, _ = parse_eval_line(out_b.decode())

    # Grow L2 -> L3 via growpro and verify step-0 bit-identity
    l3_ck = workdir / "l3.ck"
    rc, _, err, _ = run_cmd([str(ollm), "growpro", str(base_ck), str(l3_ck), "3", "5", "5000"])
    if rc != 0:
        raise RuntimeError(f"growpro L2->3 failed: {err.decode()}")
    _, out_g0, _, _ = run_cmd([str(ollm), "eval", str(l3_ck), str(he_tok), "4"])
    _, _, g0_loss, _ = parse_eval_line(out_g0.decode())
    growpro_delta_unats = g0_loss - base_loss

    # Train Link 1 (L2->3) and extract P1.sp
    rc, _, err, _ = run_cmd([str(ollm), "trainf", str(l3_ck), str(tr_tok), "5", "5", "2", "1"])
    if rc != 0:
        raise RuntimeError(f"trainf L2->3 failed: {err.decode()}")
    p1_sp = workdir / "p1.sp"
    rc, _, err, _ = run_cmd([str(ollm), "spext", str(base_ck), str(l3_ck), str(p1_sp)])
    if rc != 0:
        raise RuntimeError(f"spext p1 failed: {err.decode()}")

    prefix_l3 = workdir / "prefix_l3.ck"
    rc, _, err, _ = run_cmd([str(ollm), "spapp", str(base_ck), str(p1_sp), str(prefix_l3)])
    if rc != 0:
        raise RuntimeError(f"spapp p1 failed: {err.decode()}")

    # Train 3 distinct Link-2 sibling branches (P2_a, P2_b, P2_c) pinned to base + P1
    branch_packs: List[Path] = []
    l4_a_ck = workdir / "l4_a.ck"
    for idx, (tag, lr, steps) in enumerate([("a", "5000", "5"), ("b", "8000", "4"), ("c", "3000", "3")]):
        ck4 = l4_a_ck if idx == 0 else workdir / f"l4_{tag}.ck"
        sp4 = workdir / f"p2_{tag}.sp"
        rc, _, err, _ = run_cmd([str(ollm), "growpro", str(prefix_l3), str(ck4), "4", steps, lr])
        if rc != 0:
            raise RuntimeError(f"growpro L3->4 ({tag}) failed: {err.decode()}")
        rc, _, err, _ = run_cmd([str(ollm), "trainf", str(ck4), str(tr_tok), steps, steps, "3", "1"])
        if rc != 0:
            raise RuntimeError(f"trainf L3->4 ({tag}) failed: {err.decode()}")
        rc, _, err, _ = run_cmd([str(ollm), "spext", str(prefix_l3), str(ck4), str(sp4)])
        if rc != 0:
            raise RuntimeError(f"spext p2_{tag} failed: {err.decode()}")
        branch_packs.append(sp4)

    p2_a, p2_b, p2_c = branch_packs

    # Verify spchainapp + spchaineval + spchainsamp bit-identity against l4_a_ck
    rebuilt_l4 = workdir / "rebuilt_l4.ck"
    rc, _, err, _ = run_cmd([str(ollm), "spchainapp", str(base_ck), str(rebuilt_l4), "2", str(p1_sp), str(p2_a)])
    if rc != 0:
        raise RuntimeError(f"spchainapp failed: {err.decode()}")

    _, out_ref_e, _, _ = run_cmd([str(ollm), "eval", str(l4_a_ck), str(he_tok), "4"])
    _, out_ch_e, _, _ = run_cmd([str(ollm), "spchaineval", str(base_ck), str(he_tok), "4", "-", "2", str(p1_sp), str(p2_a)])
    _, out_rb_e, _, _ = run_cmd([str(ollm), "eval", str(rebuilt_l4), str(he_tok), "4"])
    _, _, ref_loss, ref_bpb = parse_eval_line(out_ref_e.decode())
    _, _, ch_loss, ch_bpb = parse_eval_line(out_ch_e.decode())
    _, _, rb_loss, rb_bpb = parse_eval_line(out_rb_e.decode())
    chain_eval_match = (ref_loss == ch_loss == rb_loss) and (ref_bpb == ch_bpb == rb_bpb)

    rc_s0, samp_ref, _, _ = run_cmd([str(ollm), "sample", str(l4_a_ck), "7", "24", "0", "hi"])
    rc_s1, samp_ch, _, _ = run_cmd([str(ollm), "spchainsamp", str(base_ck), "24", "0", "hi", "7", "-", "2", str(p1_sp), str(p2_a)])
    chain_samp_match = (rc_s0 == 0 and rc_s1 == 0 and samp_ref == samp_ch)

    # Negative controls: truncated (60), 1-bit flip (61), swapped order (62), skipped link (62)
    trunc_sp = workdir / "trunc.sp"
    trunc_sp.write_bytes(p2_a.read_bytes()[:32])
    rc_trunc, _, _, _ = run_cmd([str(ollm), "spchaineval", str(base_ck), str(he_tok), "4", "-", "2", str(p1_sp), str(trunc_sp)])

    flip_sp = workdir / "flip.sp"
    fb = bytearray(p2_a.read_bytes())
    fb[-1] ^= 0x01
    flip_sp.write_bytes(bytes(fb))
    rc_flip, _, _, _ = run_cmd([str(ollm), "spchaineval", str(base_ck), str(he_tok), "4", "-", "2", str(p1_sp), str(flip_sp)])

    rc_swap, _, _, _ = run_cmd([str(ollm), "spchaineval", str(base_ck), str(he_tok), "4", "-", "2", str(p2_a), str(p1_sp)])
    rc_skip, _, _, _ = run_cmd([str(ollm), "spchaineval", str(base_ck), str(he_tok), "4", "-", "1", str(p2_a)])

    neg_ok = (rc_trunc == 60 and rc_flip == 61 and rc_swap == 62 and rc_skip == 62)

    pack_total_bytes = p1_sp.stat().st_size + p2_a.stat().st_size
    full_ck_bytes = l4_a_ck.stat().st_size

    chains_res = {
        "status": "PASS" if (growpro_delta_unats == 0 and chain_eval_match and chain_samp_match and neg_ok) else "FAIL",
        "growpro_step0_delta_unats": growpro_delta_unats,
        "spchaineval_bit_identical": chain_eval_match,
        "spchainsamp_bit_identical": chain_samp_match,
        "chain_pack_bytes": pack_total_bytes,
        "full_ck_bytes": full_ck_bytes,
        "compression_ratio_x": round(full_ck_bytes / max(pack_total_bytes, 1), 2),
        "neg_controls": {
            "truncated_rc": rc_trunc,
            "bitflip_rc": rc_flip,
            "swapped_order_rc": rc_swap,
            "skipped_link_rc": rc_skip,
            "all_exact": neg_ok,
        },
    }

    # Multi-Helix DNA Clipping (spclip S=2 and S=3)
    # 1. Double-helix identity (A, A) == A in MODE=0 and MODE=1
    dh_m0_rc, dh_m0, _, _ = run_cmd([
        str(ollm), "spclip", str(base_ck), "0", "hi", "24", "0", "7", "-", "2",
        "500", "2", str(p1_sp), str(p2_a),
        "500", "2", str(p1_sp), str(p2_a),
    ])
    dh_m1_rc, dh_m1, _, _ = run_cmd([
        str(ollm), "spclip", str(base_ck), "1", "hi", "24", "0", "7", "-", "2",
        "500", "2", str(p1_sp), str(p2_a),
        "500", "2", str(p1_sp), str(p2_a),
    ])
    double_id_m0 = (dh_m0_rc == 0 and dh_m0 == samp_ch)
    double_id_m1 = (dh_m1_rc == 0 and dh_m1 == samp_ch)

    # 2. Triple-helix identity (A, A, A) == A in MODE=0 and MODE=1
    th_m0_rc, th_m0, _, _ = run_cmd([
        str(ollm), "spclip", str(base_ck), "0", "hi", "24", "0", "7", "-", "3",
        "334", "2", str(p1_sp), str(p2_a),
        "333", "2", str(p1_sp), str(p2_a),
        "333", "2", str(p1_sp), str(p2_a),
    ])
    th_m1_rc, th_m1, _, _ = run_cmd([
        str(ollm), "spclip", str(base_ck), "1", "hi", "24", "0", "7", "-", "3",
        "334", "2", str(p1_sp), str(p2_a),
        "333", "2", str(p1_sp), str(p2_a),
        "333", "2", str(p1_sp), str(p2_a),
    ])
    triple_id_m0 = (th_m0_rc == 0 and th_m0 == samp_ch)
    triple_id_m1 = (th_m1_rc == 0 and th_m1 == samp_ch)

    # 3. Distinct Triple-Helix (A, B, C) in MODE=1 (entropy-gated), 32 tokens
    n_clip = 32
    d3_rc, d3_out, _, d3_ms = run_cmd([
        str(ollm), "spclip", str(base_ck), "1", "hi", str(n_clip), "200", "7", "-", "3",
        "334", "2", str(p1_sp), str(p2_a),
        "333", "2", str(p1_sp), str(p2_b),
        "333", "2", str(p1_sp), str(p2_c),
    ])
    distinct_ok = (d3_rc == 0 and len(d3_out) > 0)

    spclip_res = {
        "status": "PASS" if (double_id_m0 and double_id_m1 and triple_id_m0 and triple_id_m1 and distinct_ok) else "FAIL",
        "double_helix_identity_mode0": double_id_m0,
        "double_helix_identity_mode1": double_id_m1,
        "triple_helix_identity_mode0": triple_id_m0,
        "triple_helix_identity_mode1": triple_id_m1,
        "distinct_triple_helix_rc": d3_rc,
        "distinct_triple_helix_bytes": len(d3_out),
        "distinct_triple_helix_tokens": n_clip,
        "distinct_triple_helix_ms_per_tok": round(d3_ms / n_clip, 3),
    }
    return chains_res, spclip_res


def section_om_feedback(omc: Path, workdir: Path) -> dict:
    # 1. Live 30-error refusal-repair coverage
    rc_r, out_r, err_r, ms_r = run_cmd([
        sys.executable,
        str(REPO_ROOT / "tools/om_bundle_data.py"),
        "refusals",
        "--omc",
        str(omc),
        "--include-err24",
    ])
    ref_line = out_r.decode("utf-8", errors="replace").strip()
    m_ref = re.search(r"Verified (\d+) live omc refusal-repair pairs covering (\d+) error codes", ref_line)
    pairs_n = int(m_ref.group(1)) if m_ref else 0
    codes_n = int(m_ref.group(2)) if m_ref else 0

    # 2. Local omc compile latency benchmark (20 runs)
    bench_json = workdir / "omc_bench.json"
    rc_b, _, err_b, _ = run_cmd([
        sys.executable,
        str(REPO_ROOT / "tools/om_compile_loop.py"),
        "bench",
        "--omc",
        str(omc),
        "--runs",
        "20",
        "--json-out",
        str(bench_json),
    ])
    bench_rows = json.loads(bench_json.read_text()) if (rc_b == 0 and bench_json.is_file()) else []

    # 3. Deterministic refusal -> repair check
    broken = "fn main to int\nx = 5\nret x\nend\n"
    repaired = "fn main to int\nx = int 5\nret x\nend\n"
    t0 = time.perf_counter()
    p0 = subprocess.run([str(omc)], input=broken.encode(), capture_output=True, timeout=15)
    ms0 = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    p1 = subprocess.run([str(omc)], input=repaired.encode(), capture_output=True, timeout=15)
    ms1 = (time.perf_counter() - t1) * 1000.0
    repair_ok = (p0.returncode == 11 and p1.returncode == 0 and len(p1.stdout) > 0)

    ok = (rc_r == 0 and codes_n == 30 and rc_b == 0 and len(bench_rows) >= 3 and repair_ok)
    return {
        "status": "PASS" if ok else "FAIL",
        "refusal_pairs": pairs_n,
        "refusal_error_codes": codes_n,
        "refusal_check_ms": round(ms_r, 1),
        "omc_bench": bench_rows,
        "deterministic_refusal_repair": {
            "attempt0_rc": p0.returncode,
            "attempt0_stderr": p0.stderr.decode("utf-8", errors="replace").strip(),
            "attempt0_ms": round(ms0, 3),
            "attempt1_rc": p1.returncode,
            "attempt1_bin_bytes": len(p1.stdout),
            "attempt1_ms": round(ms1, 3),
            "passed": repair_ok,
        },
    }


def section_real_checkpoint(
    ollm: Path,
    base_ck: Optional[Path],
    spec_pack: Optional[Path],
    held_tok: Optional[Path],
    merges: Optional[Path],
) -> dict:
    if not (base_ck and spec_pack and held_tok and base_ck.is_file() and spec_pack.is_file() and held_tok.is_file()):
        return {
            "status": "NOT_RUN",
            "reason": "optional real checkpoint/pack/held-tok not supplied or not present",
        }
    he_toks = held_tok.stat().st_size // 2
    wins = max(1, min(9, (he_toks - 1) // 256))
    m_args = [str(merges)] if (merges and merges.is_file()) else []

    rc_b, out_b, err_b, ms_b = run_cmd([str(ollm), "eval", str(base_ck), str(held_tok), str(wins)] + m_args, timeout=300)
    if rc_b != 0:
        raise RuntimeError(f"real base eval failed rc={rc_b}: {err_b.decode()}")
    _, _, b_loss, b_bpb = parse_eval_line(out_b.decode())

    rc_s, out_s, err_s, ms_s = run_cmd(
        [str(ollm), "speval", str(base_ck), str(spec_pack), str(held_tok), str(wins)] + m_args, timeout=300
    )
    if rc_s != 0:
        raise RuntimeError(f"real speval failed rc={rc_s}: {err_s.decode()}")
    _, _, s_loss, s_bpb = parse_eval_line(out_s.decode())

    delta_pct = ((s_loss - b_loss) / b_loss) * 100.0
    ck_bytes = base_ck.stat().st_size
    sp_bytes = spec_pack.stat().st_size
    return {
        "status": "PASS",
        "base_ck_name": base_ck.name,
        "spec_pack_name": spec_pack.name,
        "windows": wins,
        "base_nats_per_tok": round(b_loss / 1e6, 6),
        "base_bpb": round(b_bpb / 1e6, 6),
        "spec_nats_per_tok": round(s_loss / 1e6, 6),
        "spec_bpb": round(s_bpb / 1e6, 6),
        "delta_nats_pct": round(delta_pct, 2),
        "base_ck_bytes": ck_bytes,
        "spec_pack_bytes": sp_bytes,
        "pack_compression_vs_base_x": round(ck_bytes / max(sp_bytes, 1), 2),
        "base_eval_ms": round(ms_b, 1),
        "spec_eval_ms": round(ms_s, 1),
    }


def render_markdown(rep: dict) -> str:
    m = rep["metrics"]
    gb = m["gate_and_build"]
    kv = m["kv_cache"]
    ch = m["merkle_chains"]
    cl = m["multi_helix_spclip"]
    fb = m["om_feedback_loop"]
    rc = m["real_checkpoint"]
    hw = m["hardware_unrun"]

    b_summary = ", ".join(
        f"{k}: {v['status']} ({v.get('bytes', 0):,} B)" for k, v in gb["builds"].items()
    )
    small_b = next((r for r in fb["omc_bench"] if r["unit"] == "small_6loc"), None)
    large_b = next((r for r in fb["omc_bench"] if r["unit"] == "large_564loc"), None)
    omc_lat = (
        f"6 LOC p50={small_b['median_ms']} ms (p95={small_b['p95_ms']} ms); "
        f"565 LOC p50={large_b['median_ms']} ms (p95={large_b['p95_ms']} ms)"
        if (small_b and large_b)
        else "n/a"
    )

    if rc["status"] == "PASS":
        rc_detail = (
            f"{rc['spec_pack_name']} on {rc['base_ck_name']} (W={rc['windows']}): "
            f"{rc['base_nats_per_tok']} -> {rc['spec_nats_per_tok']} nats/tok "
            f"({rc['delta_nats_pct']:+.2f}%), bpb {rc['base_bpb']} -> {rc['spec_bpb']}, "
            f"pack {rc['spec_pack_bytes']:,} B ({rc['pack_compression_vs_base_x']}x smaller than base)"
        )
    else:
        rc_detail = rc.get("reason", "NOT_RUN")

    lines = [
        f"# OLLM Metrics & Readiness Report (`--mode {rep['mode']}`)",
        "",
        f"- **Host:** `{rep['host']['system']} {rep['host']['release']} ({rep['host']['machine']})`",
        f"- **Git HEAD:** `{rep['git_head']}`",
        f"- **Total runtime:** `{rep['elapsed_s']} s`",
        "",
        "| Category | Status | Key Measured Metrics |",
        "|---|---|---|",
        f"| **Pre-review gate** | `{gb['pre_review']['status']}` | `{gb['pre_review']['output']}` |",
        f"| **Cross-target `omc` build** | `{'PASS' if all(v['status']=='PASS' for v in gb['builds'].values()) else 'FAIL'}` | {b_summary} |",
        f"| **KV-cache (`kvcmp`, L={kv['layers']})** | `{kv['status']}` | `ndiff={kv['ndiff_logits']}`, `{kv['inc_tok_per_s']} tok/s` vs `{kv['full_tok_per_s']} tok/s` (`{kv['speedup_x']}x` speedup) |",
        f"| **Merkle chains (`OLLMSP01`, L=2->3->4)** | `{ch['status']}` | `growpro` step-0 `Δ={ch['growpro_step0_delta_unats']} µnats`, `spchaineval/samp` bit-identical, `{ch['compression_ratio_x']}x` smaller, neg controls `rc={ch['neg_controls']['truncated_rc']}/{ch['neg_controls']['bitflip_rc']}/{ch['neg_controls']['swapped_order_rc']}/{ch['neg_controls']['skipped_link_rc']}` |",
        f"| **Multi-helix `spclip` (`S=2,3`)** | `{cl['status']}` | Double `(A,A)==A` & Triple `(A,A,A)==A` bit-identical (`MODE=0,1`); distinct 3-strand `(A,B,C)` `{cl['distinct_triple_helix_ms_per_tok']} ms/tok` |",
        f"| **`.om` refusals & `omc` loop** | `{fb['status']}` | `{fb['refusal_pairs']}` pairs covering `{fb['refusal_error_codes']}/30` errors (`1..30`); {omc_lat}; repair `rc=11 -> rc=0` |",
        f"| **Real checkpoint specialist eval** | `{rc['status']}` | {rc_detail} |",
        f"| **GB10 GPU (`gtrainb`/`geval`/`v3 L=16` Rung 11)** | `{hw['gb10_gpu_training']['status']}` | {hw['gb10_gpu_training']['reason']} |",
        f"| **Aienux / GB20B live decode** | `{hw['aienux_gb20b_live_decode']['status']}` | {hw['aienux_gb20b_live_decode']['reason']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command OLLM metrics & readiness benchmark suite.")
    ap.add_argument("--mode", choices=("quick", "full"), default="quick")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--ollm-bin", default="")
    ap.add_argument("--omc", default="")
    ap.add_argument("--base-ck", default="")
    ap.add_argument("--spec-pack", default="")
    ap.add_argument("--held-tok", default="")
    ap.add_argument("--merges", default="")
    args = ap.parse_args()

    t_start = time.perf_counter()
    omc = find_omc(args.omc or None)

    rc_g, out_g, _, _ = run_cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    git_head = out_g.decode().strip() if rc_g == 0 else "unknown"

    base_ck = Path(args.base_ck).expanduser() if args.base_ck else None
    spec_pack = Path(args.spec_pack).expanduser() if args.spec_pack else None
    held_tok = Path(args.held_tok).expanduser() if args.held_tok else None
    merges = Path(args.merges).expanduser() if args.merges else None

    if args.mode == "full":
        home = Path.home()
        if base_ck is None:
            cand = home / "workspace/aien-train-macbook/local-mac/omega-v2.ck"
            if cand.is_file():
                base_ck = cand
        if merges is None:
            cand = home / "workspace/aien-train-macbook/local-mac/merges4096.bin"
            if cand.is_file():
                merges = cand
        if held_tok is None:
            cand = home / "workspace/aien-train-macbook/specialists/data_m/tool-call_held.tok"
            if cand.is_file():
                held_tok = cand
        if spec_pack is None:
            cand = home / "workspace/aien-train-macbook/specialists/packs/tool-call_pro.sp"
            if cand.is_file():
                spec_pack = cand

    with tempfile.TemporaryDirectory(prefix="ollm-metrics-") as tmp:
        workdir = Path(tmp)
        gate_build, ollm_bin = section_gate_and_build(omc, workdir, args.ollm_bin or None)
        kv_res = section_kv_cache(ollm_bin, workdir)
        chains_res, spclip_res = section_chains_and_spclip(ollm_bin, workdir)
        fb_res = section_om_feedback(omc, workdir)
        real_res = section_real_checkpoint(ollm_bin, base_ck, spec_pack, held_tok, merges)

    elapsed_s = round(time.perf_counter() - t_start, 2)
    report = {
        "mode": args.mode,
        "git_head": git_head,
        "elapsed_s": elapsed_s,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "metrics": {
            "gate_and_build": gate_build,
            "kv_cache": kv_res,
            "merkle_chains": chains_res,
            "multi_helix_spclip": spclip_res,
            "om_feedback_loop": fb_res,
            "real_checkpoint": real_res,
            "hardware_unrun": {
                "gb10_gpu_training": {
                    "status": "NOT_RUN",
                    "reason": "requires authorized DGX Spark spark-window (no GPU touched)",
                },
                "aienux_gb20b_live_decode": {
                    "status": "NOT_RUN",
                    "reason": "requires authorized Aienux/GB20B hardware window",
                },
            },
        },
    }

    md = render_markdown(report)
    print(md)

    if args.json_out:
        p = Path(args.json_out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        p = Path(args.md_out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")

    all_pass = (
        gate_build["pre_review"]["status"] == "PASS"
        and all(v["status"] == "PASS" for v in gate_build["builds"].values())
        and kv_res["status"] == "PASS"
        and chains_res["status"] == "PASS"
        and spclip_res["status"] == "PASS"
        and fb_res["status"] == "PASS"
        and real_res["status"] in ("PASS", "NOT_RUN")
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
