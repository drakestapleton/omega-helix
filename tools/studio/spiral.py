#!/usr/bin/env python3
"""Omega Helix Studio — Spiral-2048 Pure-Omega Razor MoE planner, crystallizer, evaluator & router.

Python 3 standard library only — zero external dependencies, zero telemetry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

STUDIO_DIR = Path(__file__).resolve().parent

SPIRAL_PRESETS: dict[str, dict[str, Any]] = {
    "mini64": {
        "id": "mini64",
        "label": "Mini-Spiral (64 Razor Strands · ~1B)",
        "short_label": "Mini-Spiral · 64 Strands · ~1.2B",
        "badge": "64 Strands · ~1B · W4",
        "description": "Looped 8x4 (32L eff.) seed trunk + 64 functional OLLMSP02 W4 razor-strands (~1.19B total, ~0.56 GiB W4).",
        "V": 4096,
        "T": 256,
        "D": 1536,
        "H": 24,
        "F": 4096,
        "L_phys": 8,
        "NP": 4,
        "B": 16,
        "n_strands": 64,
        "f_exp": 640,
        "top_k": 4,
        "quant_bits": 4,
        "lambda_milli": 1000,
        "base_steps": 2000,
        "est_sec_per_strand": 2.5,
    },
    "mid256": {
        "id": "mid256",
        "label": "Mid-Spiral (256 Strands · ~4B)",
        "short_label": "Mid-Spiral · 256 Strands · ~4.2B",
        "badge": "256 Strands · ~4B · W4",
        "description": "Looped 8x4 (32L eff.) seed trunk + 256 OLLMSP02 W4 razor-strands (~4.21B total, ~2.0 GiB W4).",
        "V": 4096,
        "T": 256,
        "D": 1536,
        "H": 24,
        "F": 4096,
        "L_phys": 8,
        "NP": 4,
        "B": 16,
        "n_strands": 256,
        "f_exp": 640,
        "top_k": 4,
        "quant_bits": 4,
        "lambda_milli": 1000,
        "base_steps": 5000,
        "est_sec_per_strand": 2.5,
    },
    "flagship2048": {
        "id": "flagship2048",
        "label": "Flagship Spiral-2048 (32.5B · 15 GiB W4)",
        "short_label": "Flagship Spiral-2048 · 32.5B · 15 GiB W4",
        "badge": "2048 Strands · 32.5B · 15 GiB W4",
        "description": "Full 2048-strand Razor MoE over 8x4 looped seed (32L eff., V=32K, 32.44B params, ~15.1 GiB W4, 289M active/tok).",
        "V": 32768,
        "T": 256,
        "D": 1536,
        "H": 24,
        "F": 4096,
        "L_phys": 8,
        "NP": 4,
        "B": 16,
        "n_strands": 2048,
        "f_exp": 640,
        "top_k": 4,
        "quant_bits": 4,
        "lambda_milli": 1000,
        "base_steps": 10000,
        "est_sec_per_strand": 2.5,
    },
}

SPIRAL_PILLARS: list[dict[str, Any]] = [
    {
        "id": "general",
        "index": 1,
        "name": "General Reasoning & English",
        "color": "#22d3ee",
        "waldo_default": "core/fineweb-edu",
        "strands": [
            {"id": "p1-logic", "name": "Logic & Deduction"},
            {"id": "p1-prose", "name": "Technical Prose"},
            {"id": "p1-summary", "name": "Distilled Summaries"},
            {"id": "p1-qa", "name": "Grounded Q&A"},
            {"id": "p1-outline", "name": "Structured Outlines"},
            {"id": "p1-clarity", "name": "Editorial Clarity"},
            {"id": "p1-wordmath", "name": "Quantitative Reasoning"},
            {"id": "p1-instruct", "name": "Instruction Following"},
        ],
    },
    {
        "id": "systems",
        "index": 2,
        "name": "Systems / C / Omega / GPU",
        "color": "#a78bfa",
        "waldo_default": "omega-native",
        "strands": [
            {"id": "p2-om2", "name": "Pure Omega (.om)"},
            {"id": "p2-cposix", "name": "C & POSIX Syscalls"},
            {"id": "p2-aarch64", "name": "AArch64 Assembly"},
            {"id": "p2-sm121", "name": "Blackwell SM121 SASS"},
            {"id": "p2-arenas", "name": "Dynamic Memory Arenas"},
            {"id": "p2-elf", "name": "ELF / Mach-O Codegen"},
            {"id": "p2-io", "name": "Zero-Copy Kernel I/O"},
            {"id": "p2-binfmt", "name": "Binary Formats (CK/SP)"},
        ],
    },
    {
        "id": "web",
        "index": 3,
        "name": "Python / TS / Web",
        "color": "#34d399",
        "waldo_default": "code/python",
        "strands": [
            {"id": "p3-py", "name": "Python Stdlib"},
            {"id": "p3-ts", "name": "Strict TypeScript"},
            {"id": "p3-html", "name": "Accessible HTML"},
            {"id": "p3-css", "name": "Responsive CSS"},
            {"id": "p3-sse", "name": "HTTP & Live SSE"},
            {"id": "p3-async", "name": "Async & Workers"},
            {"id": "p3-json", "name": "JSON API Contracts"},
            {"id": "p3-svg", "name": "SVG & Telemetry Charts"},
        ],
    },
    {
        "id": "backend",
        "index": 4,
        "name": "Rust / Go / Backend",
        "color": "#fbbf24",
        "waldo_default": "code/permissive",
        "strands": [
            {"id": "p4-rust", "name": "Safe Rust"},
            {"id": "p4-go", "name": "Go Concurrency"},
            {"id": "p4-sql", "name": "SQL & Schemas"},
            {"id": "p4-store", "name": "Storage Engines"},
            {"id": "p4-mtls", "name": "Local mTLS & PKI"},
            {"id": "p4-bft", "name": "BFT Consensus"},
            {"id": "p4-cli", "name": "CLI Ergonomics"},
            {"id": "p4-prof", "name": "Latency Profiling"},
        ],
    },
    {
        "id": "agentic",
        "index": 5,
        "name": "Agentic Planning & Tool Calling",
        "color": "#38bdf8",
        "waldo_default": "omega-docs",
        "strands": [
            {"id": "p5-tools", "name": "Tool Call Schemas"},
            {"id": "p5-swarm", "name": "Swarm Handoffs"},
            {"id": "p5-board", "name": "Board Coordination"},
            {"id": "p5-dag", "name": "Multi-Step Task DAGs"},
            {"id": "p5-triage", "name": "Failure Recovery"},
            {"id": "p5-ctx", "name": "Context Hygiene"},
            {"id": "p5-decomp", "name": "Goal Decomposition"},
            {"id": "p5-receipts", "name": "Evidence Receipts"},
        ],
    },
    {
        "id": "verify",
        "index": 6,
        "name": "Debugging / Testing / Verification",
        "color": "#f472b6",
        "waldo_default": "omega-starter",
        "strands": [
            {"id": "p6-unit", "name": "Unit & Integration Tests"},
            {"id": "p6-fuzz", "name": "Fuzz & Boundary Probes"},
            {"id": "p6-tamper", "name": "Merkle Tamper Gates"},
            {"id": "p6-parity", "name": "Bit-Exact Parity Diffs"},
            {"id": "p6-trace", "name": "Crash & Stack Forensics"},
            {"id": "p6-bisect", "name": "Regression Bisect"},
            {"id": "p6-ci", "name": "CI & Release Gates"},
            {"id": "p6-contract", "name": "Fail-Closed Contracts"},
        ],
    },
    {
        "id": "math",
        "index": 7,
        "name": "Math / Algorithms / Lean",
        "color": "#fb923c",
        "waldo_default": "math/mathlib",
        "strands": [
            {"id": "p7-lean", "name": "Lean 4 & Mathlib"},
            {"id": "p7-linalg", "name": "Linear Algebra"},
            {"id": "p7-chol", "name": "Cholesky & Ridge Solvers"},
            {"id": "p7-graph", "name": "Graph & Tree Algos"},
            {"id": "p7-interval", "name": "Interval Arithmetic"},
            {"id": "p7-crypto", "name": "SHA-256 & FNV-1a"},
            {"id": "p7-w4", "name": "W4 Block Quantization"},
            {"id": "p7-num", "name": "Numerical Stability"},
        ],
    },
    {
        "id": "devops",
        "index": 8,
        "name": "Shell / Git / DevOps",
        "color": "#2dd4bf",
        "waldo_default": "code/shell",
        "strands": [
            {"id": "p8-sh", "name": "POSIX Shell Scripts"},
            {"id": "p8-git", "name": "Git Worktrees & Rebase"},
            {"id": "p8-ssh", "name": "SSH Tunnels & Remotes"},
            {"id": "p8-daemon", "name": "Systemd & Detach Daemons"},
            {"id": "p8-make", "name": "Build & Packaging"},
            {"id": "p8-oci", "name": "Containers & Sandboxes"},
            {"id": "p8-logs", "name": "Log Forensics"},
            {"id": "p8-sha", "name": "SHA-256 Release Bundles"},
        ],
    },
]


def _int(val: Any, default: int, lo: int = 1, hi: int = 1_000_000) -> int:
    if val is None or val == "":
        return default
    v = int(val)
    return max(lo, min(hi, v))


def list_presets_and_pillars() -> dict[str, Any]:
    all_strands = sum(len(p["strands"]) for p in SPIRAL_PILLARS)
    return {
        "ok": True,
        "presets": list(SPIRAL_PRESETS.values()),
        "pillars": SPIRAL_PILLARS,
        "pillar_count": len(SPIRAL_PILLARS),
        "strand_catalog_count": all_strands,
    }


def plan_spiral(raw: dict[str, Any]) -> dict[str, Any]:
    """Compute exact Spiral-2048 Razor MoE parameter counts, W4 memory, commands, and omega-router config."""
    preset_id = str(raw.get("preset") or "mini64").strip().lower()
    base_cfg = dict(SPIRAL_PRESETS.get(preset_id, SPIRAL_PRESETS["mini64"]))

    V = _int(raw.get("V"), base_cfg["V"], 256, 65536)
    T = _int(raw.get("T"), base_cfg["T"], 16, 2048)
    D = _int(raw.get("D"), base_cfg["D"], 32, 4096)
    H = _int(raw.get("H"), base_cfg["H"], 2, 64)
    F = _int(raw.get("F"), base_cfg["F"], 64, 16384)
    L_phys = _int(raw.get("L_phys") if raw.get("L_phys") is not None else raw.get("L"), base_cfg["L_phys"], 1, 32)
    NP = _int(raw.get("NP"), base_cfg["NP"], 1, 16)
    B = _int(raw.get("B"), base_cfg["B"], 1, 64)
    n_strands = _int(raw.get("n_strands"), base_cfg["n_strands"], 1, 4096)
    f_exp = _int(raw.get("f_exp"), base_cfg["f_exp"], 32, 4096)
    top_k = _int(raw.get("top_k"), base_cfg["top_k"], 1, min(64, n_strands))
    quant_bits = int(raw.get("quant_bits") or base_cfg["quant_bits"])
    if quant_bits not in (4, 64):
        quant_bits = 4
    lambda_milli = _int(raw.get("lambda_milli"), base_cfg["lambda_milli"], 1, 1_000_000)
    base_steps = _int(raw.get("base_steps"), base_cfg["base_steps"], 1, 500_000)
    sec_per_strand = float(raw.get("est_sec_per_strand") or base_cfg["est_sec_per_strand"])
    local_dir = str(raw.get("local_dir") or "").strip()

    errors: list[str] = []
    if D % H != 0:
        errors.append(f"D={D} must be divisible by H={H}")
    if top_k > n_strands:
        errors.append(f"top_k={top_k} cannot exceed n_strands={n_strands}")

    # Selected pillars/strands from the 8x8 catalog
    selected_pillars = raw.get("selected_pillars")
    if not isinstance(selected_pillars, list) or not selected_pillars:
        selected_pillars = [p["id"] for p in SPIRAL_PILLARS]
    selected_strands = raw.get("selected_strands")
    if not isinstance(selected_strands, list) or not selected_strands:
        selected_strands = [
            s["id"]
            for p in SPIRAL_PILLARS
            if p["id"] in selected_pillars
            for s in p["strands"]
        ]

    # Exact parameter math matching OLLMCK + OLLMSP02 thin FFN razor strands
    layer_p = 4 * D * D + 2 * D * F + 2 * D
    base_params = V * D + T * D + L_phys * layer_p + D
    per_strand_params = L_phys * (2 * D * f_exp)
    strand_total_params = n_strands * per_strand_params
    total_params = base_params + strand_total_params
    active_params_per_token = base_params + top_k * per_strand_params
    effective_layers = L_phys * NP

    bytes_per_param = 0.5 if quant_bits == 4 else 8.0
    w4_bytes = int(base_params * 2 + strand_total_params * 0.5)
    f64_bytes = int(total_params * 8)
    packed_bytes = w4_bytes if quant_bits == 4 else f64_bytes

    est_total_cryst_sec = round(n_strands * sec_per_strand, 1)

    name = str(raw.get("name") or f"spiral-{preset_id}").strip() or f"spiral-{preset_id}"
    cmds = [
        f"ollm-gb10 init seed.ck {V} {T} {D} {H} {F} {L_phys} {B} 1 {base_steps} 1500 1 0 {L_phys} {NP}",
        f"ollm-gb10 gtrainb seed.ck seed_train.tok {base_steps} {min(base_steps, 250)}",
        f"ollm-gb10 apuweave seed.ck shards/ strands/ {f_exp} {lambda_milli} 512 {quant_bits} 16  # Full-APU 48 SMs + 20 Grace cores",
        f"ollm-gb10 spcryst seed.ck strand_01.tok strands/s01.sp2 {f_exp} {lambda_milli} 4096 1 {quant_bits}",
        f"ollm-gb10 spweval seed.ck seed_held.tok {top_k} 16 strands/s01.sp2 strands/s02.sp2 ...",
        f"ollm-gb10 spsamp seed.ck \"fn main\" 96 {top_k} 700 strands/s01.sp2 strands/s02.sp2 ...",
    ]

    omega_router_config = {
        "model_id": name,
        "backend": "ollm-spiral",
        "router_endpoint": "http://127.0.0.1:18090/v1",
        "gpu_backend_port": 18091,
        "base_ck": f"~/.local/share/omega-helix/spiral/{name}/seed.ck",
        "strands_dir": f"~/.local/share/omega-helix/spiral/{name}/strands",
        "n_strands": n_strands,
        "top_k": top_k,
        "quant_bits": quant_bits,
        "serve_cmd": f"omega-router --model {name} --spiral-dir ~/.local/share/omega-helix/spiral/{name} --top-k {top_k}",
    }

    return {
        "ok": True,
        "valid": len(errors) == 0,
        "error": "; ".join(errors),
        "preset": preset_id,
        "name": name,
        "V": V,
        "T": T,
        "D": D,
        "H": H,
        "F": F,
        "L_phys": L_phys,
        "NP": NP,
        "B": B,
        "effective_layers": effective_layers,
        "n_strands": n_strands,
        "f_exp": f_exp,
        "top_k": top_k,
        "quant_bits": quant_bits,
        "lambda_milli": lambda_milli,
        "base_steps": base_steps,
        "local_dir": local_dir,
        "selected_pillars": selected_pillars,
        "selected_strands": selected_strands,
        "base_params": base_params,
        "per_strand_params": per_strand_params,
        "strand_total_params": strand_total_params,
        "total_params": total_params,
        "active_params_per_token": active_params_per_token,
        "w4_gib": round(w4_bytes / (1024 ** 3), 3),
        "f64_gib": round(f64_bytes / (1024 ** 3), 3),
        "packed_gib": round(packed_bytes / (1024 ** 3), 3),
        "bytes_per_param": bytes_per_param,
        "est_sec_per_strand": sec_per_strand,
        "est_total_cryst_sec": est_total_cryst_sec,
        "resolved_cmd_preview": cmds,
        "omega_router_config": omega_router_config,
    }


def _file_sha256_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def read_sp2_header(path: Path) -> dict[str, Any]:
    """Read OLLMSP02 header metadata + SHA-256."""
    raw = path.read_bytes()
    sha, size = _file_sha256_size(path)
    magic = raw[:8].decode("ascii", errors="replace") if len(raw) >= 8 else ""
    info: dict[str, Any] = {
        "magic": magic,
        "path": str(path),
        "sha256": sha,
        "bytes": size,
        "valid_magic": raw[:8] == b"OLLMSP02",
    }
    if len(raw) >= 80 and raw[:8] == b"OLLMSP02":
        vals = struct.unpack("<9Q", raw[8:80])
        info["base_hash"] = f"{vals[0]:016x}"
        info["L_phys"] = vals[1]
        info["D"] = vals[2]
        info["F_exp"] = vals[3]
        info["strand_id"] = vals[4]
        info["quant_bits"] = vals[5]
        info["payload_bytes"] = vals[6]
        info["payload_hash"] = f"{vals[7]:016x}"
        info["header_hash"] = f"{vals[8]:016x}"
    return info


def run_spcryst(
    ollm_bin: Path,
    base_ck: str,
    domain_tok: str,
    out_sp2: str,
    *,
    f_exp: int = 256,
    lambda_milli: int = 1000,
    max_toks: int = 4096,
    strand_id: int = 1,
    quant_bits: int = 4,
) -> dict[str, Any]:
    """Run `ollm spcryst` and measure exact seconds-per-strand crystallization."""
    base_p = Path(base_ck).expanduser().resolve()
    tok_p = Path(domain_tok).expanduser().resolve()
    out_p = Path(out_sp2).expanduser().resolve()
    if not base_p.is_file():
        raise ValueError(f"base_ck not found: {base_p}")
    if not tok_p.is_file():
        raise ValueError(f"domain_tok not found: {tok_p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    eff_max_toks = int(max_toks) if int(max_toks) > 0 else 4096

    argv = [
        str(ollm_bin),
        "spcryst",
        str(base_p),
        str(tok_p),
        str(out_p),
        str(int(f_exp)),
        str(int(lambda_milli)),
        str(eff_max_toks),
        str(int(strand_id)),
        str(int(quant_bits)),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = round(time.monotonic() - t0, 4)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"spcryst exited {proc.returncode}: {detail}")

    hdr = read_sp2_header(out_p) if out_p.is_file() else {}
    base_cal = None
    cryst_cal = None
    wall_ms = None
    for ln in proc.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 11 and parts[0] == "SPCRYST":
            try:
                base_cal = int(parts[7]) / 1e6
                cryst_cal = int(parts[8]) / 1e6
                wall_ms = int(parts[10])
            except ValueError:
                pass
    return {
        "ok": True,
        "cmd": " ".join(argv),
        "elapsed_sec": elapsed,
        "sec_per_strand": round(wall_ms / 1000.0, 4) if wall_ms is not None else elapsed,
        "wall_ms": wall_ms,
        "base_cal_loss": base_cal,
        "cryst_cal_loss": cryst_cal,
        "out_sp2": str(out_p),
        "pack": hdr,
        "stdout": proc.stdout.strip(),
    }


def _parse_routing_lines(stdout: str) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for ln in stdout.splitlines():
        m = re.match(r"^(?:ROUTE|SPW_ROUTE|R)\s+(\S+)\s+([0-9.eE+-]+)", ln.strip())
        if m:
            try:
                routes.append({"strand": m.group(1), "weight": float(m.group(2))})
            except ValueError:
                pass
    return routes


def run_spweval(
    ollm_bin: Path,
    base_ck: str,
    eval_tok: str,
    strands: list[str],
    *,
    top_k: int = 4,
    windows: int = 8,
) -> dict[str, Any]:
    """Run `ollm spweval <base.ck> <eval.tok> <top_k> <W> <s1.sp2> [s2.sp2 ...]`."""
    base_p = Path(base_ck).expanduser().resolve()
    eval_p = Path(eval_tok).expanduser().resolve()
    if not base_p.is_file():
        raise ValueError(f"base_ck not found: {base_p}")
    if not eval_p.is_file():
        raise ValueError(f"eval_tok not found: {eval_p}")
    if not strands:
        raise ValueError("at least one .sp2 strand pack is required")
    sp_paths = [str(Path(s).expanduser().resolve()) for s in strands]

    argv = [
        str(ollm_bin),
        "spweval",
        str(base_p),
        str(eval_p),
        str(int(top_k)),
        str(int(windows)),
        *sp_paths,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = round(time.monotonic() - t0, 4)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"spweval exited {proc.returncode}: {detail}")

    base_loss: float | None = None
    loss_nats: float | None = None
    bpb: float | None = None
    for ln in proc.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 7 and parts[0] == "SPWEVAL":
            try:
                base_loss = int(parts[4]) / 1e6
                loss_nats = int(parts[5]) / 1e6
                bpb = int(parts[6]) / 1e6
            except ValueError:
                pass
        elif len(parts) >= 5 and parts[0] in ("E", "WE"):
            try:
                loss_nats = int(parts[3]) / 1e6 if parts[3].isdigit() else float(parts[3])
                bpb = int(parts[4]) / 1e6 if parts[4].isdigit() else float(parts[4])
            except ValueError:
                pass
    return {
        "ok": True,
        "cmd": " ".join(argv),
        "elapsed_sec": elapsed,
        "top_k": int(top_k),
        "n_strands": len(sp_paths),
        "base_loss": base_loss,
        "loss": loss_nats,
        "bpb": bpb,
        "routing": _parse_routing_lines(proc.stdout),
        "stdout": proc.stdout.strip(),
    }


def run_spiral_sample(
    ollm_bin: Path,
    base_ck: str,
    strands: list[str],
    *,
    prompt: str = "fn main",
    max_tok: int = 64,
    top_k: int = 4,
    temp_milli: int = 700,
    subcmd: str = "spwsamp",
) -> dict[str, Any]:
    """Run Spiral Weave Top-K sampling (`ollm spwsamp` / `spsamp`) and parse Top-K routing readout."""
    base_p = Path(base_ck).expanduser().resolve()
    if not base_p.is_file():
        raise ValueError(f"base_ck not found: {base_p}")
    if not strands:
        raise ValueError("at least one .sp2 strand pack is required")
    sp_paths = [str(Path(s).expanduser().resolve()) for s in strands]

    argv = [
        str(ollm_bin),
        subcmd,
        str(base_p),
        str(prompt),
        str(int(max_tok)),
        str(int(top_k)),
        str(int(temp_milli)),
        *sp_paths,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, check=False)
    elapsed = round(time.monotonic() - t0, 4)
    stdout_txt = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr_txt = (proc.stderr or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        detail = (stderr_txt or stdout_txt or "").strip()
        raise RuntimeError(f"{subcmd} exited {proc.returncode}: {detail}")

    routing = _parse_routing_lines(stdout_txt)
    text_lines = [
        ln for ln in stdout_txt.splitlines()
        if not re.match(r"^(?:ROUTE|SPW_ROUTE|R)\s+\S+\s+[0-9.eE+-]+", ln.strip())
    ]
    return {
        "ok": True,
        "mode_used": f"spiral-{subcmd}",
        "cmd": " ".join(argv),
        "elapsed_sec": elapsed,
        "top_k": int(top_k),
        "n_strands": len(sp_paths),
        "text": "\n".join(text_lines).strip(),
        "routing": routing,
        "stdout": stdout_txt.strip(),
    }


def parse_apuweave_stdout(stdout: str) -> dict[str, Any]:
    """Parse machine-readable APU_WAVE / APU_STRAND / APU_SUMMARY / APU_OVERLAP telemetry."""
    waves: list[list[int]] = []
    strands: list[list[int]] = []
    summary_raw: list[int] | None = None
    overlap_raw: list[int] | None = None
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0]
        if tag in ("APU_WAVE", "APU_STRAND", "APU_SUMMARY", "APU_OVERLAP"):
            try:
                vals = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            if tag == "APU_WAVE":
                waves.append(vals)
            elif tag == "APU_STRAND":
                strands.append(vals)
            elif tag == "APU_SUMMARY":
                summary_raw = vals
            elif tag == "APU_OVERLAP":
                overlap_raw = vals

    out: dict[str, Any] = {
        "waves": waves,
        "strands": strands,
        "summary_raw": summary_raw,
        "overlap_raw": overlap_raw,
        "n_shards": len(strands),
        "tot_w": 0,
        "wall_ms": 0,
        "gpu_ms": 0,
        "cpu_core_ms": 0,
        "strands_per_sec": 0.0,
        "overlap": 0.0,
        "w4_bytes": 0,
        "base_eval_loss": None,
        "woven_eval_loss": None,
        "held_drop": None,
        "tamper_ok": 0,
        "gpu_mode": 0,
    }
    if summary_raw and len(summary_raw) >= 13:
        out["n_shards"] = summary_raw[0]
        out["tot_w"] = summary_raw[1]
        out["wall_ms"] = summary_raw[2]
        out["gpu_ms"] = summary_raw[3]
        out["cpu_core_ms"] = summary_raw[4]
        out["strands_per_sec"] = round(summary_raw[5] / 1000.0, 3)
        out["overlap"] = round(summary_raw[6] / 1000.0, 3)
        out["w4_bytes"] = summary_raw[7]
        out["base_eval_loss"] = round(summary_raw[8] / 1e6, 6)
        out["woven_eval_loss"] = round(summary_raw[9] / 1e6, 6)
        out["held_drop"] = round(summary_raw[10] / 1e6, 6)
        out["tamper_ok"] = summary_raw[11]
        out["gpu_mode"] = summary_raw[12]
    return out


def run_apuweave(
    ollm_bin: Path,
    base_ck: str,
    shards_source: str | list[str],
    out_dir: str,
    *,
    f_exp: int = 128,
    lambda_milli: int = 100,
    max_toks: int = 512,
    quant_bits: int = 4,
    workers: int = 0,
) -> dict[str, Any]:
    """Run Full-APU Simultaneous Co-Execution Fabric (`ollm[-gb10] apuweave`)."""
    import platform as _plat

    base_p = Path(base_ck).expanduser().resolve()
    if not base_p.is_file():
        raise ValueError(f"base_ck not found: {base_p}")
    out_p = Path(out_dir).expanduser().resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    if isinstance(shards_source, list):
        clean_paths = [str(Path(s).expanduser().resolve()) for s in shards_source if str(s).strip()]
        if not clean_paths:
            raise ValueError("shards list cannot be empty")
        list_p = out_p / "shards.list"
        list_p.write_text("\n".join(clean_paths) + "\n", encoding="utf-8")
        src_arg = list_p
    else:
        src_arg = Path(str(shards_source)).expanduser().resolve()
        if not src_arg.exists():
            raise ValueError(f"shards_source not found: {src_arg}")

    eff_workers = int(workers) if int(workers) > 0 else (16 if _plat.system() == "Linux" else 1)
    eff_q = int(quant_bits) if int(quant_bits) in (4, 64) else 4
    argv = [
        str(ollm_bin),
        "apuweave",
        str(base_p),
        str(src_arg),
        str(out_p),
        str(int(f_exp)),
        str(int(lambda_milli)),
        str(int(max_toks)),
        str(eff_q),
        str(eff_workers),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = round(time.monotonic() - t0, 4)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"apuweave exited {proc.returncode}: {detail}")

    tel = parse_apuweave_stdout(proc.stdout)
    out_packs = sorted(str(p) for p in out_p.glob("*.sp2"))
    sec_per_strand = (
        round((tel["wall_ms"] / 1000.0) / tel["n_shards"], 4)
        if tel["n_shards"] > 0 and tel["wall_ms"] > 0
        else (round(elapsed / max(1, len(out_packs)), 4))
    )
    return {
        "ok": True,
        "cmd": " ".join(argv),
        "elapsed_sec": elapsed,
        "sec_per_strand": sec_per_strand,
        "out_dir": str(out_p),
        "out_packs": out_packs,
        "workers": eff_workers,
        **tel,
        "stdout": proc.stdout.strip(),
    }
