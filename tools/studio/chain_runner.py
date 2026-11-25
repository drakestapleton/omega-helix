#!/usr/bin/env python3
"""Omega Helix Studio chain estimator, DNA/Universal/Hybrid executor, and interactive sampler.

Python 3 standard library only — zero external dependencies, zero telemetry.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

STUDIO_DIR = Path(__file__).resolve().parent
TOOLS_DIR = STUDIO_DIR.parent
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"

for _p in (str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import omega_helix_backend as univ_backend  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402

LN2 = math.log(2.0)
STRAND_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24"]


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_runs_dir() -> Path:
    env = os.environ.get("OMEGA_HELIX_RUNS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "omega-helix" / "runs").resolve()


def default_cache_bin_dir() -> Path:
    return (Path.home() / ".cache" / "omega-helix" / "bin").resolve()


def find_omc(root: Path | None = None) -> Path | None:
    r = studio_catalog.repo_root(root)
    cands: list[str | Path] = []
    if os.environ.get("OMC"):
        cands.append(os.environ["OMC"])
    which_omc = shutil.which("omc")
    if which_omc:
        cands.append(which_omc)
    cands.extend([
        Path.home() / ".local" / "bin" / "omc",
        Path.home() / "canon" / "omega-mv" / "seed" / "out" / "omc",
        r.parent / "omega-mv" / "seed" / "out" / "omc",
        r.parent / "omega-mv-macos" / "seed" / "out" / "omc",
    ])
    for c in cands:
        p = Path(c).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    return None


def _platform_ollm_names() -> tuple[list[str], str]:
    sysname = platform.system()
    mach = platform.machine().lower()
    if sysname == "Darwin":
        return ["ollm-macos"], "macos"
    if mach in ("x86_64", "amd64"):
        return ["ollm-x86linux", "ollm"], "x86linux"
    return ["ollm", "ollm-gb10"], "aarch64"


def resolve_binaries(bin_dir: str | Path | None = None, root: Path | None = None) -> dict[str, Path | None]:
    r = studio_catalog.repo_root(root)
    search_dirs: list[Path] = []
    if bin_dir:
        search_dirs.append(Path(bin_dir).expanduser().resolve())
    if os.environ.get("OMEGA_HELIX_BIN_DIR"):
        search_dirs.append(Path(os.environ["OMEGA_HELIX_BIN_DIR"]).expanduser().resolve())
    search_dirs.extend([
        (Path.home() / ".local" / "opt" / "omega-helix" / "bin").resolve(),
        (r / "build").resolve(),
        (r / "build" / "verify").resolve(),
        default_cache_bin_dir(),
    ])
    # Deduplicate search dirs preserving order
    uniq_dirs: list[Path] = []
    for d in search_dirs:
        if d not in uniq_dirs:
            uniq_dirs.append(d)

    ollm_names, _ = _platform_ollm_names()
    specs = {
        "ollm": ("OLLM_BIN", ollm_names),
        "ollm_univ": ("OLLM_UNIV_BIN", ["ollm-univ"]),
        "ollm_interchange": ("OLLM_INTERCHANGE_BIN", ["ollm-interchange"]),
        "ollm_waldo": ("OLLM_WALDO_BIN", ["ollm-waldo"]),
        "ollm_tok": ("OLLM_TOK_BIN", ["ollm-tok"]),
        "ollm_gb10": ("OLLM_GB10_BIN", ["ollm-gb10"]),
    }

    resolved: dict[str, Path | None] = {}
    for key, (env_var, names) in specs.items():
        found: Path | None = None
        env_val = os.environ.get(env_var)
        if env_val:
            ep = Path(env_val).expanduser().resolve()
            if ep.is_file() and os.access(ep, os.X_OK):
                found = ep
        if found is None:
            for d in uniq_dirs:
                for n in names:
                    cand = d / n
                    if cand.is_file() and os.access(cand, os.X_OK):
                        found = cand.resolve()
                        break
                if found is not None:
                    break
        if found is None:
            for n in names:
                w = shutil.which(n)
                if w:
                    found = Path(w).resolve()
                    break
        resolved[key] = found
    return resolved


def ensure_binaries(
    needed: list[str],
    bin_dir: str | Path | None = None,
    root: Path | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Locate required pure-Omega binaries, compiling any missing targets into ~/.cache/omega-helix/bin."""
    r = studio_catalog.repo_root(root)
    bins = resolve_binaries(bin_dir=bin_dir, root=r)
    _, ollm_target = _platform_ollm_names()
    target_map = {
        "ollm": ollm_target,
        "ollm_univ": "univ",
        "ollm_interchange": "interchange",
        "ollm_waldo": "openwaldo",
        "ollm_tok": "tok",
        "ollm_gb10": "gb10",
    }

    missing = [k for k in needed if not bins.get(k)]
    if missing:
        omc = find_omc(r)
        build_sh = r / "scripts" / "build.sh"
        if not omc or not build_sh.is_file():
            raise RuntimeError(
                f"Missing required binaries ({', '.join(missing)}) and cannot auto-build "
                f"(omc={omc}, build.sh={build_sh}). Install omc or run `sh scripts/build.sh`."
            )
        out_dir = default_cache_bin_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in missing:
            tgt = target_map[key]
            if log_cb:
                log_cb(f"Building missing pure-Omega binary {key} (TARGET={tgt}) into {out_dir}...")
            env = dict(os.environ)
            env["TARGET"] = tgt
            env["OUT"] = str(out_dir)
            env["OMC"] = str(omc)
            proc = subprocess.run(
                ["sh", str(build_sh)],
                cwd=str(r),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed building TARGET={tgt}: {proc.stderr.strip() or proc.stdout.strip()}"
                )
            if log_cb and proc.stdout.strip():
                log_cb(proc.stdout.strip())
        bins = resolve_binaries(bin_dir=bin_dir, root=r)

    out: dict[str, Path] = {}
    for k in needed:
        p = bins.get(k)
        if not p:
            raise RuntimeError(f"Required binary {k!r} could not be located after build attempt.")
        out[k] = p
    return out


def detect_system(bin_dir: str | Path | None = None, runs_dir: Path | None = None) -> dict[str, Any]:
    bins = resolve_binaries(bin_dir=bin_dir)
    omc = find_omc()
    has_nvidia = os.path.exists("/dev/nvidiactl")
    has_gb10 = bool(has_nvidia and (bins.get("ollm_gb10") is not None or (platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64"))))
    bin_info: dict[str, str] = {}
    bin_detail: dict[str, Any] = {}
    first_dir = ""
    for k, p in bins.items():
        if p is not None:
            if not first_dir:
                first_dir = str(p.parent)
            bin_info[k] = str(p)
            bin_detail[k] = {
                "path": str(p),
                "bytes": p.stat().st_size,
                "ready": True,
            }
        else:
            bin_info[k] = ""
            bin_detail[k] = {"path": "", "bytes": 0, "ready": False}
    return {
        "ok": True,
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "has_gb10": has_gb10,
        "has_nvidia_dev": has_nvidia,
        "has_waldo_cli": bool(shutil.which("waldo")),
        "omc": str(omc) if omc else "",
        "runs_dir": str(runs_dir or default_runs_dir()),
        "bin_dir": str(bin_dir or os.environ.get("OMEGA_HELIX_BIN_DIR") or first_dir or default_cache_bin_dir()),
        "binaries": bin_info,
        "binaries_detail": bin_detail,
        "starter_presets": studio_catalog.starter_presets(),
    }


# --------------------------------------------------------------------------- spec & estimation


def _as_int(val: Any, default: int, name: str) -> int:
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        raise ValueError(f"{name} must be an integer, got boolean")
    try:
        return int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {val!r}") from exc


def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a user/UI/CLI Helix Studio chain spec with clean defaults."""
    if not isinstance(raw, dict):
        raise ValueError("spec must be a JSON object")

    strands_raw = raw.get("strands") or []
    if not isinstance(strands_raw, list):
        raise ValueError("strands must be a list")

    mode = str(raw.get("mode") or ("dna" if strands_raw else "universal")).strip().lower()
    if mode not in ("dna", "universal", "hybrid"):
        raise ValueError(f"mode must be dna|universal|hybrid, got {mode!r}")

    arch = str(raw.get("arch") or ("gpt" if mode in ("dna", "hybrid") else "qwen3")).strip().lower()
    if arch not in univ_backend.ARCHES:
        raise ValueError(f"arch must be one of {'|'.join(univ_backend.ARCHES)}, got {arch!r}")

    tokenizer = str(raw.get("tokenizer") or "byte").strip().lower()
    if tokenizer not in ("bpe", "byte", "u32"):
        raise ValueError(f"tokenizer must be bpe|byte|u32, got {tokenizer!r}")

    default_v = 256 if tokenizer == "byte" else (70000 if tokenizer == "u32" else 300)
    V = _as_int(raw.get("V") if raw.get("V") is not None else raw.get("vocab"), default_v, "V")
    T = _as_int(raw.get("T"), 32, "T")
    D = _as_int(raw.get("D"), 32, "D")
    H = _as_int(raw.get("H"), 4, "H")
    default_hkv = max(1, H // 2) if (mode == "universal" and not strands_raw and H % 2 == 0) else H
    H_KV = _as_int(raw.get("H_KV"), default_hkv, "H_KV")
    F = _as_int(raw.get("F"), max(32, D * 2), "F")
    L = _as_int(raw.get("L"), 2, "L")
    B = _as_int(raw.get("B"), 2, "B")
    LP = _as_int(raw.get("LP"), 0, "LP")
    LC = _as_int(raw.get("LC"), 0, "LC")
    NP = _as_int(raw.get("NP"), 1, "NP")
    adapt_thresh_micro = _as_int(raw.get("adapt_thresh_micro"), 0, "adapt_thresh_micro")
    msg_gain_micro = _as_int(raw.get("msg_gain_micro"), 0, "msg_gain_micro")
    base_steps = _as_int(raw.get("base_steps") if raw.get("base_steps") is not None else raw.get("steps"), 20, "base_steps")
    lr_micro = _as_int(raw.get("lr_micro"), 3000, "lr_micro")
    seed = _as_int(raw.get("seed"), 1, "seed")
    held_mod = _as_int(raw.get("held_mod"), 10, "held_mod")
    spclip_mode = _as_int(raw.get("spclip_mode"), 1, "spclip_mode")

    base_ds = raw.get("base_dataset") or raw.get("dataset") or {"kind": "preset", "value": "omega-starter"}
    if isinstance(base_ds, str):
        kind = "preset" if base_ds in ("omega-starter", "omega-docs", "omega-native") else "local"
        base_ds = {"kind": kind, "value": base_ds}
    elif isinstance(base_ds, dict):
        base_ds = {
            "kind": str(base_ds.get("kind") or "preset").strip().lower(),
            "value": str(base_ds.get("value") or "omega-starter").strip(),
        }
    else:
        raise ValueError("base_dataset must be a string or {kind, value} object")

    strands: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, sr in enumerate(strands_raw):
        if not isinstance(sr, dict):
            raise ValueError(f"strand[{idx}] must be an object")
        sid_raw = str(sr.get("id") or f"strand-{chr(97 + (idx % 26))}").strip()
        sid = re.sub(r"[^a-zA-Z0-9_-]+", "-", sid_raw).strip("-") or f"strand-{idx + 1}"
        if sid in seen_ids:
            sid = f"{sid}-{idx + 1}"
        seen_ids.add(sid)
        sds = sr.get("dataset") or base_ds
        if isinstance(sds, str):
            sk = "preset" if sds in ("omega-starter", "omega-docs", "omega-native") else "local"
            sds = {"kind": sk, "value": sds}
        elif isinstance(sds, dict):
            sds = {
                "kind": str(sds.get("kind") or base_ds["kind"]).strip().lower(),
                "value": str(sds.get("value") or base_ds["value"]).strip(),
            }
        else:
            sds = dict(base_ds)
        growth = str(sr.get("growth") or "growpro").strip().lower()
        if growth not in ("growpro", "grow"):
            raise ValueError(f"strand {sid}: growth must be growpro|grow, got {growth!r}")
        strands.append({
            "id": sid,
            "name": str(sr.get("name") or sid).strip(),
            "color": str(sr.get("color") or STRAND_COLORS[idx % len(STRAND_COLORS)]).strip(),
            "parent_id": str(sr.get("parent_id") or "").strip(),
            "add_layers": _as_int(sr.get("add_layers"), 1, f"strand[{sid}].add_layers"),
            "growth": growth,
            "use_fcachem": bool(sr.get("use_fcachem", True)),
            "steps": _as_int(sr.get("steps"), 15, f"strand[{sid}].steps"),
            "lr_micro": _as_int(sr.get("lr_micro"), 2000, f"strand[{sid}].lr_micro"),
            "weight": _as_int(sr.get("weight"), 500, f"strand[{sid}].weight"),
            "dataset": sds,
        })

    hf_dtype = str(raw.get("hf_dtype") or "f32").strip().lower()
    if hf_dtype not in ("f32", "bf16"):
        raise ValueError(f"hf_dtype must be f32|bf16, got {hf_dtype!r}")

    flags = raw.get("flags")
    if flags is not None:
        flags = _as_int(flags, 0, "flags")

    name = str(raw.get("name") or "my-helix").strip() or "my-helix"
    return {
        "name": name,
        "mode": mode,
        "arch": arch,
        "flags": flags,
        "tokenizer": tokenizer,
        "V": V,
        "T": T,
        "D": D,
        "H": H,
        "H_KV": H_KV,
        "F": F,
        "L": L,
        "B": B,
        "LP": LP,
        "LC": LC,
        "NP": NP,
        "adapt_thresh_micro": adapt_thresh_micro,
        "msg_gain_micro": msg_gain_micro,
        "use_gpu": bool(raw.get("use_gpu", False)),
        "base_steps": base_steps,
        "lr_micro": lr_micro,
        "seed": seed,
        "held_mod": held_mod,
        "base_dataset": base_ds,
        "strands": strands,
        "spclip_mode": spclip_mode,
        "export_hf": bool(raw.get("export_hf", False)),
        "hf_dtype": hf_dtype,
        "export_gguf": bool(raw.get("export_gguf", False)),
    }


def _legacy_params(V: int, T: int, D: int, F: int, L: int) -> tuple[int, int]:
    """Exact OLLMCK01/02/03 parameter count matching native/model.om ogeom."""
    layer_p = 4 * D * D + 2 * D * F + 2 * D
    base_p = V * D + T * D + L * layer_p + D
    return base_p, layer_p


def _univ_params(V: int, T: int, D: int, H: int, H_KV: int, F: int, L: int, flags: int) -> int:
    """Exact OLLMCK04 parameter count matching native/univ/uarena.om ugeom_params."""
    E = D // H
    KVD = H_KV * E
    has_rope = bool(flags & 1)
    is_gated = bool(flags & (2 | 64))
    has_untied = bool(flags & 4)
    has_qknorm = bool(flags & 8)

    p = V * D
    if not has_rope:
        p += T * D
    per_layer = D + D * D + 2 * D * KVD + D * D + D + D * F + F * D
    if has_qknorm:
        per_layer += 2 * E
    if is_gated:
        per_layer += D * F
    p += L * per_layer + D
    if has_untied:
        p += V * D
    return p


def _is_dna_execution(spec: dict[str, Any]) -> bool:
    return bool(spec["strands"]) or spec["mode"] in ("dna", "hybrid")


def estimate_chain(raw_spec: dict[str, Any]) -> dict[str, Any]:
    """Validate geometry, compute exact param count / memory / token width, and build Merkle DAG."""
    try:
        spec = normalize_spec(raw_spec)
    except ValueError as exc:
        return {
            "ok": False,
            "valid": False,
            "error": str(exc),
            "params": 0,
            "base_params": 0,
            "est_mb": 0.0,
            "disk_mb": 0.0,
            "ram_mb": 0.0,
            "tok_bytes": 1,
            "dag": {"nodes": [], "edges": []},
            "resolved_cmd_preview": [],
        }

    V, T, D, H, H_KV, F, L, B = (
        spec["V"], spec["T"], spec["D"], spec["H"], spec["H_KV"], spec["F"], spec["L"], spec["B"]
    )
    LP, LC, NP = spec["LP"], spec["LC"], spec["NP"]
    strands = spec["strands"]
    is_dna = _is_dna_execution(spec)

    errors: list[str] = []
    for k, val in (("V", V), ("T", T), ("D", D), ("H", H), ("H_KV", H_KV), ("F", F), ("L", L), ("B", B)):
        if val < 1:
            errors.append(f"{k} must be >= 1")
    if spec["base_steps"] < 1:
        errors.append("base_steps must be >= 1")
    if spec["lr_micro"] < 1:
        errors.append("lr_micro must be >= 1")
    if D > 0 and H > 0 and D % H != 0:
        errors.append(f"D={D} must be divisible by H={H}")
    if spec["tokenizer"] == "byte" and V < 256:
        errors.append("tokenizer=byte requires V >= 256")
    if spec["tokenizer"] == "bpe" and not (256 < V <= 65535):
        errors.append("tokenizer=bpe requires 256 < V <= 65535")
    if spec["tokenizer"] == "u32" and V <= 65535:
        errors.append("tokenizer=u32 requires V > 65535")
    if spec["spclip_mode"] not in (0, 1):
        errors.append("spclip_mode must be 0 (prior-weighted) or 1 (entropy-gated)")

    if NP < 1:
        errors.append("NP must be >= 1")
    elif NP > 1:
        if LC < 1 or LP + LC > L:
            errors.append(f"Looped core needs LC >= 1 and LP + LC <= L (got LP={LP}, LC={LC}, L={L})")

    # Resolve strand parent depths
    strand_depth: dict[str, int] = {}
    max_total_L = L
    for s in strands:
        if s["add_layers"] < 1:
            errors.append(f"strand {s['id']}: add_layers must be >= 1")
        if s["steps"] < 1:
            errors.append(f"strand {s['id']}: steps must be >= 1")
        if s["lr_micro"] < 1:
            errors.append(f"strand {s['id']}: lr_micro must be >= 1")
        pid = s["parent_id"]
        if pid:
            if pid not in strand_depth:
                errors.append(f"strand {s['id']}: parent_id {pid!r} must reference an earlier strand")
                base_l = L
            else:
                base_l = strand_depth[pid]
        else:
            base_l = L
        tot_l = base_l + s["add_layers"]
        strand_depth[s["id"]] = tot_l
        if tot_l > max_total_L:
            max_total_L = tot_l

    if is_dna:
        if not (256 <= V <= 4096):
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires 256 <= V <= 4096 (got V={V})")
        if T > 256:
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires T <= 256 (got T={T})")
        if D > 384:
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires D <= 384 (got D={D})")
        if H > 16:
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires H <= 16 (got H={H})")
        if F > 1536:
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires F <= 1536 (got F={F})")
        if B > 64:
            errors.append(f"DNA/Hybrid pure-Omega ollm engine requires B <= 64 (got B={B})")
        if max_total_L > 16:
            errors.append(f"DNA/Hybrid max layer depth (base L + added strand layers) <= 16 (got {max_total_L})")
        if NP > 1 and strands:
            errors.append("Looped core (NP > 1) cannot be combined with DNA grow/spext strands (engine exit 17)")
        if spec["tokenizer"] == "u32":
            errors.append("tokenizer=u32 is only supported in universal mode (0 DNA strands)")
    else:
        if H_KV > H or (H_KV > 0 and H % H_KV != 0):
            errors.append(f"H={H} must be a multiple of H_KV={H_KV}")
        if spec["arch"] == "custom" and spec["flags"] is None:
            errors.append("arch=custom requires explicit flags")
        flags = spec["flags"] if spec["arch"] == "custom" else univ_backend.PRESETS[spec["arch"]]["flags"]
        if (flags & 2) and (flags & 64):
            errors.append(f"invalid flags {flags}: SwiGLU (2) and GeGLU (64) are mutually exclusive")
        if (flags & 66) and (flags & 32):
            errors.append(f"invalid flags {flags}: GELU (32) excludes gated FFN (2/64)")
        if (flags & 1) and H > 0 and (D // H) % 2 != 0:
            errors.append(f"RoPE architecture requires even head_dim D/H (got {D // H})")

    tok_bytes = 4 if V > 65535 else (2 if V > 256 else 1)
    tok_width_label = {1: "u8", 2: "u16", 4: "u32"}[tok_bytes]
    eff_layers = (L + LC * (NP - 1)) if NP > 1 else max_total_L

    if is_dna:
        base_params, layer_p = _legacy_params(V, T, D, F, L)
        strand_params = sum(max(0, s["add_layers"]) * layer_p for s in strands)
        total_params = base_params + strand_params
        disk_bytes = (144 + 24 * base_params) + sum(48 + 8 * (max(0, s["add_layers"]) * layer_p + D) for s in strands)
        if strands:
            disk_bytes += 144 + 24 * (base_params + max(0, max_total_L - L) * layer_p)
        if spec["export_hf"]:
            bpe_w = 2 if spec["hf_dtype"] == "bf16" else 4
            disk_bytes += total_params * bpe_w
        if spec["export_gguf"]:
            disk_bytes += total_params * 4 + 4096
        act_vals = eff_layers * (8 * T * D + 2 * T * F + H * T * T) + 10 * T * D + T * V
        ram_bytes = 32 * max(base_params, total_params) + 8 * act_vals
    else:
        flags = (spec["flags"] or 0) if spec["arch"] == "custom" else univ_backend.PRESETS[spec["arch"]]["flags"]
        base_params = _univ_params(V, T, D, max(1, H), max(1, H_KV), F, L, flags) if (H > 0 and D % H == 0) else 0
        total_params = base_params
        disk_bytes = 208 + 24 * base_params
        if spec["export_hf"]:
            disk_bytes += base_params * (2 if spec["hf_dtype"] == "bf16" else 4)
        if spec["export_gguf"]:
            disk_bytes += base_params * 4 + 4096
        act_vals = eff_layers * (8 * T * D + 2 * T * F + H * T * T) + 6 * T * D + T * V
        ram_bytes = 32 * base_params + 8 * act_vals

    # Build Merkle DAG for SVG visualization
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    ds_label = f"{spec['base_dataset']['kind']}: {spec['base_dataset']['value']}"
    nodes.append({
        "id": "base-data",
        "label": "OpenWALDO Verified Data",
        "sublabel": ds_label,
        "kind": "dataset",
        "color": "#38bdf8",
        "meta": dict(spec["base_dataset"]),
    })
    trunk_title = f"Base Trunk ({spec['arch'].upper()} L={L})" if not is_dna else f"Base Trunk (DNA L={L})"
    if NP > 1:
        trunk_title += f" Loop {LP}:{LC}×{NP}"
    nodes.append({
        "id": "base-model",
        "label": trunk_title,
        "sublabel": f"{base_params:,} params · {tok_width_label}",
        "kind": "base",
        "color": "#22d3ee",
        "meta": {"params": base_params, "L": L, "D": D, "V": V, "NP": NP},
    })
    edges.append({"from": "base-data", "to": "base-model", "label": "ollm-waldo + train"})

    leaf_ids: list[str] = []
    if is_dna and strands:
        nodes.append({
            "id": "merkle-lock",
            "label": "Frozen Core Merkle Lock",
            "sublabel": "FNV-1a 64 bit-identical",
            "kind": "lock",
            "color": "#34d399",
            "meta": {"frozen_layers": L},
        })
        edges.append({"from": "base-model", "to": "merkle-lock", "label": "freeze [0..L)"})
        has_child = {s["parent_id"] for s in strands if s["parent_id"]}
        for s in strands:
            nid = f"strand-{s['id']}"
            nodes.append({
                "id": nid,
                "label": s["name"],
                "sublabel": f"+{s['add_layers']}L · {s['growth']} · OLLMSP01",
                "kind": "strand",
                "color": s["color"],
                "meta": {
                    "strand_id": s["id"],
                    "add_layers": s["add_layers"],
                    "growth": s["growth"],
                    "steps": s["steps"],
                    "weight": s["weight"],
                    "dataset": s["dataset"]["value"],
                },
            })
            src_node = f"strand-{s['parent_id']}" if s["parent_id"] else "merkle-lock"
            edges.append({"from": src_node, "to": nid, "label": f"{s['growth']} -> spext"})
            if s["id"] not in has_child:
                leaf_ids.append(nid)

        if len(strands) >= 2:
            nodes.append({
                "id": "spclip",
                "label": "spclip Multi-Helix Router",
                "sublabel": "Entropy-Gated PoE" if spec["spclip_mode"] == 1 else "Prior-Weighted PoE",
                "kind": "clip",
                "color": "#a78bfa",
                "meta": {"spclip_mode": spec["spclip_mode"], "strands": len(strands)},
            })
            for lid in leaf_ids:
                edges.append({"from": lid, "to": "spclip", "label": "OLLMSP01"})
            export_parent = "spclip"
        else:
            export_parent = leaf_ids[0] if leaf_ids else "base-model"
    else:
        export_parent = "base-model"

    if is_dna and (spec["mode"] == "hybrid" or spec["export_hf"] or spec["export_gguf"]):
        nodes.append({
            "id": "ck04-upgrade",
            "label": "OLLMCK04 Upgrade",
            "sublabel": "ollm-interchange",
            "kind": "upgrade",
            "color": "#fbbf24",
            "meta": {},
        })
        edges.append({"from": export_parent, "to": "ck04-upgrade", "label": "upgrade"})
        export_parent = "ck04-upgrade"

    export_leaves: list[str] = []
    if spec["export_hf"]:
        nodes.append({
            "id": "export-hf",
            "label": f"HF Safetensors ({spec['hf_dtype']})",
            "sublabel": "config.json + model.safetensors",
            "kind": "export",
            "color": "#f472b6",
            "meta": {"dtype": spec["hf_dtype"]},
        })
        edges.append({"from": export_parent, "to": "export-hf", "label": "export-hf"})
        export_leaves.append("export-hf")

    if spec["export_gguf"]:
        nodes.append({
            "id": "export-gguf",
            "label": "GGUF v3",
            "sublabel": "model.gguf",
            "kind": "export",
            "color": "#fb923c",
            "meta": {},
        })
        edges.append({"from": export_parent, "to": "export-gguf", "label": "export-gguf"})
        export_leaves.append("export-gguf")

    nodes.append({
        "id": "bom",
        "label": "OpenWALDO Helix BOM",
        "sublabel": "SHA-256 + FNV-1a Provenance",
        "kind": "bom",
        "color": "#34d399",
        "meta": {},
    })
    for src in (export_leaves or [export_parent]):
        edges.append({"from": src, "to": "bom", "label": "bind"})

    # Command preview
    cmds: list[str] = [f"ollm-waldo verified-corpus.jsonl train.bin held.bin {spec['held_mod']}"]
    if is_dna:
        if spec["tokenizer"] == "bpe":
            cmds.append(f"ollm bpefit train.bin merges{V}.bin {V}")
            cmds.append(f"ollm bpeenc merges{V}.bin {V} train.bin train.tok")
        init_cmd = f"ollm init base.ck {V} {T} {D} {H} {F} {L} {B} {spec['seed']} {spec['base_steps']} {spec['lr_micro']}"
        if NP > 1:
            init_cmd += f" 1 {LP} {LC} {NP}"
        cmds.append(init_cmd)
        train_op = "gtrainf" if spec["use_gpu"] else "train"
        cmds.append(f"ollm {train_op} base.ck {'train.tok' if spec['tokenizer']=='bpe' else 'train.bin'} {spec['base_steps']} {max(1, spec['base_steps']//4)}")
        for s in strands:
            pl = strand_depth.get(s["parent_id"], L) if s["parent_id"] else L
            nl = pl + s["add_layers"]
            cmds.append(f"ollm {s['growth']} parent.ck {s['id']}.ck {nl} {s['steps']} {s['lr_micro']}")
            if s["use_fcachem"] and not spec["use_gpu"]:
                cmds.append(f"ollm fcache parent.ck {s['id']}.tok W {s['id']}.hc && ollm trainfc {s['id']}.ck {s['id']}.hc {s['steps']} {max(1, s['steps']//3)} {pl}")
            else:
                cmds.append(f"ollm {'gtrainf' if spec['use_gpu'] else 'trainf'} {s['id']}.ck {s['id']}.tok {s['steps']} {max(1, s['steps']//3)} {pl}")
            cmds.append(f"ollm spext parent.ck {s['id']}.ck {s['id']}.sp")
        if spec["mode"] == "hybrid" or spec["export_hf"] or spec["export_gguf"]:
            cmds.append("ollm-interchange upgrade merged.ck model_v4.ck")
    else:
        cmds.append(
            f"ollm-univ init model.ck {spec['arch']} {V} {T} {D} {H} {H_KV} {F} {L} {B} {spec['seed']} {spec['base_steps']} {spec['lr_micro']}"
        )
        cmds.append(f"ollm-univ train model.ck train.tok {spec['base_steps']} {max(1, spec['base_steps']//4)}")

    if spec["export_hf"]:
        cmds.append(f"ollm-interchange export-hf model_v4.ck export-hf {spec['hf_dtype']}")
    if spec["export_gguf"]:
        cmds.append("ollm-interchange export-gguf model_v4.ck model.gguf")

    disk_mb = round(disk_bytes / (1024 * 1024), 3)
    ram_mb = round(ram_bytes / (1024 * 1024), 3)
    return {
        "ok": True,
        "valid": len(errors) == 0,
        "error": "; ".join(errors),
        "params": total_params,
        "base_params": base_params,
        "disk_bytes": disk_bytes,
        "ram_bytes": ram_bytes,
        "disk_mb": disk_mb,
        "ram_mb": ram_mb,
        "est_mb": round(disk_mb + ram_mb, 3),
        "tok_bytes": tok_bytes,
        "tok_width": tok_width_label,
        "mode": spec["mode"],
        "arch": spec["arch"],
        "is_dna": is_dna,
        "effective_layers": eff_layers,
        "dag": {"nodes": nodes, "edges": edges},
        "resolved_cmd_preview": cmds,
        "spec": spec,
    }


# --------------------------------------------------------------------------- OLLMSP01 helper


def read_sp_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 48 or raw[:8] != b"OLLMSP01":
        raise ValueError(f"{path}: invalid OLLMSP01 specialist pack")
    base_l, spec_l, base_hash, n_spec_params, pack_hash = struct.unpack("<5Q", raw[8:48])
    sha, size = ow_bridge.file_sha256_and_bytes(path)
    return {
        "magic": "OLLMSP01",
        "path": str(path),
        "sha256": sha,
        "bytes": size,
        "base_L": base_l,
        "spec_L": spec_l,
        "base_hash": f"{base_hash:016x}",
        "n_spec_params": n_spec_params,
        "pack_hash": f"{pack_hash:016x}",
    }


def _file_artifact(path: Path, name: str, kind: str, **extra: Any) -> dict[str, Any]:
    sha, size = ow_bridge.file_sha256_and_bytes(path)
    out = {"name": name, "kind": kind, "path": str(path), "sha256": sha, "bytes": size}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- Run state & Manager


class ChainRun:
    def __init__(self, run_id: str, spec: dict[str, Any], run_dir: Path, estimate: dict[str, Any]) -> None:
        self.run_id = run_id
        self.spec = spec
        self.run_dir = run_dir
        self.status = "queued"
        self.stage = "queued"
        self.stages: list[dict[str, Any]] = []
        self.started_at = utc_now()
        self.finished_at = ""
        self.params = estimate["params"]
        self.base_params = estimate["base_params"]
        self.dag = estimate["dag"]
        self.metrics: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.strands: list[dict[str, Any]] = []
        self.bom: dict[str, Any] | None = None
        self.error = ""
        self.latest_loss: float | None = None
        self.latest_bpb: float | None = None
        # Runtime paths for interactive sampling
        self.sample_meta: dict[str, Any] = {}

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._stop_event = threading.Event()
        self._active_proc: subprocess.Popen[str] | None = None

    def emit(self, event_kind: str, *, _terminal_status: str | None = None, **payload: Any) -> dict[str, Any]:
        if event_kind == "log" and "text" in payload and "message" not in payload:
            payload["message"] = payload["text"]
        with self._cond:
            self._seq += 1
            if event_kind == "artifact":
                art_rec = dict(payload)
                art_rec["kind"] = art_rec.get("kind") or "file"
                ev = {
                    "seq": self._seq,
                    "ts": utc_now(),
                    **{k: v for k, v in art_rec.items() if k != "kind"},
                    "artifact_kind": art_rec["kind"],
                    "kind": "artifact",
                }
                self._events.append(ev)
                self.artifacts.append(art_rec)
                self._cond.notify_all()
                return ev

            ev = {"seq": self._seq, "ts": utc_now(), "kind": event_kind, **payload}
            self._events.append(ev)
            if event_kind == "stage":
                self.stage = str(payload.get("stage", self.stage))
                self.stages.append({"stage": self.stage, "label": payload.get("label", self.stage), "ts": ev["ts"], "status": payload.get("status", "active")})
            elif event_kind == "metric":
                self.metrics.append(payload)
                if payload.get("loss") is not None:
                    self.latest_loss = float(payload["loss"])
                if payload.get("bpb") is not None:
                    self.latest_bpb = float(payload["bpb"])
            elif event_kind == "helix":
                nid = payload.get("node_id")
                for n in self.dag.get("nodes", []):
                    if n["id"] == nid:
                        n["status"] = payload.get("status", "done")
                        if payload.get("sha256"):
                            n.setdefault("meta", {})["sha256"] = payload["sha256"]
                        if payload.get("pack_hash"):
                            n.setdefault("meta", {})["pack_hash"] = payload["pack_hash"]
                        if payload.get("loss") is not None:
                            n.setdefault("meta", {})["loss"] = payload["loss"]
            if _terminal_status is not None:
                self.status = _terminal_status
            self._cond.notify_all()
            return ev

    def iter_events(self, start_seq: int = 0, poll_timeout: float = 10.0):
        """Generator yielding buffered + live events cleanly for SSE."""
        cursor = start_seq
        while True:
            with self._cond:
                while cursor >= len(self._events) and self.status in ("queued", "running"):
                    self._cond.wait(timeout=poll_timeout)
                    if cursor >= len(self._events) and self.status in ("queued", "running"):
                        yield None  # keepalive sentinel
                batch = self._events[cursor:]
                done = self.status not in ("queued", "running")
            for ev in batch:
                cursor = ev["seq"]
                yield ev
            if done and cursor >= len(self._events):
                break

    def request_stop(self) -> bool:
        self._stop_event.set()
        with self._lock:
            proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            return True
        return self.status in ("queued", "running")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "name": self.spec.get("name", self.run_id),
                "mode": self.spec.get("mode", "dna"),
                "arch": self.spec.get("arch", "gpt"),
                "status": self.status,
                "stage": self.stage,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "params": self.params,
                "latest_loss": self.latest_loss,
                "latest_bpb": self.latest_bpb,
                "strands_count": len(self.spec.get("strands", [])),
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "spec": self.spec,
                "status": self.status,
                "stage": self.stage,
                "stages": list(self.stages),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "params": self.params,
                "base_params": self.base_params,
                "latest_loss": self.latest_loss,
                "latest_bpb": self.latest_bpb,
                "metrics": list(self.metrics),
                "artifacts": list(self.artifacts),
                "strands": list(self.strands),
                "dag": self.dag,
                "bom": self.bom,
                "error": self.error,
                "run_dir": str(self.run_dir),
                "sample_meta": dict(self.sample_meta),
            }

    def save_state(self, override_status: str | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        if override_status is not None:
            doc["status"] = override_status
        state_path = self.run_dir / "run_state.json"
        tmp = self.run_dir / ".run_state.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        tmp.replace(state_path)

    @classmethod
    def load_from_dir(cls, run_dir: Path) -> "ChainRun | None":
        state_path = run_dir / "run_state.json"
        if not state_path.is_file():
            return None
        try:
            doc = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        est = {
            "params": doc.get("params", 0),
            "base_params": doc.get("base_params", 0),
            "dag": doc.get("dag", {"nodes": [], "edges": []}),
        }
        run = cls(doc["run_id"], doc.get("spec", {}), run_dir, est)
        run.status = doc.get("status", "done")
        if run.status in ("queued", "running"):
            run.status = "interrupted"
        run.stage = doc.get("stage", "done")
        run.stages = doc.get("stages", [])
        run.started_at = doc.get("started_at", "")
        run.finished_at = doc.get("finished_at", "")
        run.latest_loss = doc.get("latest_loss")
        run.latest_bpb = doc.get("latest_bpb")
        run.metrics = doc.get("metrics", [])
        run.artifacts = doc.get("artifacts", [])
        run.strands = doc.get("strands", [])
        run.bom = doc.get("bom")
        run.error = doc.get("error", "")
        run.sample_meta = doc.get("sample_meta", {})
        return run


class ChainManager:
    def __init__(
        self,
        runs_dir: str | Path | None = None,
        bin_dir: str | Path | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).expanduser().resolve() if runs_dir else default_runs_dir()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(bin_dir).expanduser().resolve() if bin_dir else None
        self.root = studio_catalog.repo_root(root)
        self._lock = threading.Lock()
        self._runs: dict[str, ChainRun] = {}
        self._load_existing_runs()

    def _load_existing_runs(self) -> None:
        if not self.runs_dir.is_dir():
            return
        for sub in sorted(self.runs_dir.iterdir()):
            if sub.is_dir():
                r = ChainRun.load_from_dir(sub)
                if r is not None:
                    self._runs[r.run_id] = r

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return [r.summary() for r in runs]

    def get_run(self, run_id: str) -> ChainRun | None:
        with self._lock:
            r = self._runs.get(run_id)
        if r is not None:
            return r
        # Check if user passed a directory path or persisted run_id
        cand = Path(run_id)
        if cand.is_dir() and (cand / "run_state.json").is_file():
            loaded = ChainRun.load_from_dir(cand.resolve())
            if loaded is not None:
                with self._lock:
                    self._runs[loaded.run_id] = loaded
                return loaded
        cand2 = self.runs_dir / run_id
        if cand2.is_dir() and (cand2 / "run_state.json").is_file():
            loaded = ChainRun.load_from_dir(cand2.resolve())
            if loaded is not None:
                with self._lock:
                    self._runs[loaded.run_id] = loaded
                return loaded
        return None

    def start_chain(self, raw_spec: dict[str, Any], wait: bool = False) -> ChainRun:
        est = estimate_chain(raw_spec)
        if not est["valid"]:
            raise ValueError(est["error"] or "invalid chain spec")
        spec = est["spec"]
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", spec["name"]).strip("-").lower()[:24] or "helix"
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}-{slug}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

        run = ChainRun(run_id, spec, run_dir, est)
        with self._lock:
            self._runs[run_id] = run
        run.save_state()

        if wait:
            self._execute_run(run)
        else:
            t = threading.Thread(target=self._execute_run, args=(run,), name=f"helix-{run_id}", daemon=True)
            t.start()
        return run

    def stop_chain(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if not run:
            return False
        return run.request_stop()

    # ------------------------------------------------------------------ subprocess helpers

    def _run_stream(
        self,
        run: ChainRun,
        argv: list[str],
        stage: str,
        line_cb: Callable[[str], None] | None = None,
    ) -> str:
        if run._stop_event.is_set():
            raise InterruptedError("stopped by user")
        cmd_preview = " ".join(Path(argv[0]).name if i == 0 else str(a) for i, a in enumerate(argv))
        run.emit("log", stage=stage, text=f"$ {cmd_preview}")
        lines: list[str] = []
        with subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            with run._lock:
                run._active_proc = proc
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\r\n")
                    lines.append(line)
                    if line:
                        run.emit("log", stage=stage, text=line)
                        if line_cb:
                            line_cb(line)
                    if run._stop_event.is_set():
                        proc.terminate()
                        break
                rc = proc.wait()
            finally:
                with run._lock:
                    run._active_proc = None
        if run._stop_event.is_set():
            raise InterruptedError("stopped by user")
        if rc != 0:
            tail = "\n".join(lines[-8:])
            raise RuntimeError(f"{Path(argv[0]).name} exited {rc}: {tail}")
        return "\n".join(lines)

    def _prepare_dataset_spec(
        self,
        run: ChainRun,
        ds: dict[str, str],
        out_dir: Path,
        label: str,
        ollm_waldo: Path,
    ) -> dict[str, Any]:
        kind = ds.get("kind", "preset")
        val = ds.get("value", "omega-starter")
        if kind == "waldo":
            res = studio_catalog.materialize_waldo(val, out_dir)
            if not res.get("ok"):
                guidance = " ".join(res.get("guidance", []))
                raise RuntimeError(f"{res.get('error', 'waldo export failed')} {guidance}".strip())
            return res
        return studio_catalog.prepare_local_dataset(
            val,
            out_dir,
            name=label,
            ollm_waldo=ollm_waldo,
            held_mod=run.spec["held_mod"],
            root=self.root,
        )

    # ------------------------------------------------------------------ main executor

    def _execute_run(self, run: ChainRun) -> None:
        run.status = "running"
        try:
            if _is_dna_execution(run.spec):
                self._run_dna_or_hybrid(run)
            else:
                self._run_universal(run)
            if run._stop_event.is_set():
                run.finished_at = utc_now()
                run.save_state(override_status="stopped")
                run.emit("error", _terminal_status="stopped", status="stopped", error="Run stopped by user")
            else:
                run.stage = "done"
                run.finished_at = utc_now()
                run.save_state(override_status="done")
                run.emit(
                    "done",
                    _terminal_status="done",
                    status="done",
                    run_id=run.run_id,
                    latest_loss=run.latest_loss,
                    latest_bpb=run.latest_bpb,
                    artifacts_count=len(run.artifacts),
                )
        except InterruptedError:
            run.error = "Stopped by user"
            run.finished_at = utc_now()
            run.save_state(override_status="stopped")
            run.emit("error", _terminal_status="stopped", status="stopped", error=run.error)
        except Exception as exc:
            run.error = str(exc)
            run.finished_at = utc_now()
            run.save_state(override_status="failed")
            run.emit("error", _terminal_status="failed", status="failed", stage=run.stage, error=run.error)

    def _run_dna_or_hybrid(self, run: ChainRun) -> None:
        spec = run.spec
        run_dir = run.run_dir
        V, T, D, H, F, L, B = spec["V"], spec["T"], spec["D"], spec["H"], spec["F"], spec["L"], spec["B"]
        LP, LC, NP = spec["LP"], spec["LC"], spec["NP"]
        strands = spec["strands"]
        need_interchange = spec["mode"] == "hybrid" or spec["export_hf"] or spec["export_gguf"]

        run.emit("stage", stage="resolve", label="Resolving Pure-Omega Binaries")
        needed = ["ollm", "ollm_waldo"]
        if need_interchange:
            needed.append("ollm_interchange")
        bins = ensure_binaries(
            needed,
            bin_dir=self.bin_dir,
            root=self.root,
            log_cb=lambda msg: run.emit("log", stage="resolve", text=msg),
        )
        all_bins = resolve_binaries(bin_dir=self.bin_dir, root=self.root)
        use_gb10 = bool(spec["use_gpu"] and os.path.exists("/dev/nvidiactl") and all_bins.get("ollm_gb10"))
        ollm_bin = all_bins["ollm_gb10"] if use_gb10 else bins["ollm"]
        assert ollm_bin is not None
        waldo_bin = bins["ollm_waldo"]

        # 1. OpenWALDO dataset verification + ingest
        run.emit("stage", stage="dataset", label="OpenWALDO Verification & Ingest")
        base_export_dir = run_dir / "datasets" / "base_export"
        prep = self._prepare_dataset_spec(run, spec["base_dataset"], base_export_dir, "base", waldo_bin)
        verified_export_dir = Path(prep["export_dir"])

        base_data_dir = run_dir / "data" / "base"
        bridge_argv = [
            sys.executable,
            str(Path(ow_bridge.__file__).resolve()),
            "prepare",
            str(verified_export_dir),
            str(base_data_dir),
            "--ollm-waldo",
            str(waldo_bin),
            "--held-mod",
            str(spec["held_mod"]),
        ]
        if spec["tokenizer"] == "bpe" and V > 256:
            bridge_argv += ["--ollm", str(bins["ollm"]), "--vocab", str(V)]
        self._run_stream(run, bridge_argv, "dataset")

        base_bom = json.loads((base_data_dir / "openwaldo-helix-bom.json").read_text(encoding="utf-8"))
        tok_width = 2 if V > 256 else 1
        merges_path: Path | None = None
        if spec["tokenizer"] == "bpe" and V > 256:
            train_data = base_data_dir / "train.tok"
            held_data = base_data_dir / "held.tok"
            merges_path = base_data_dir / f"merges{V}.bin"
        else:
            train_data = base_data_dir / "train.bin"
            held_data = base_data_dir / "held.bin"

        n_train_toks = train_data.stat().st_size // tok_width
        n_held_toks = held_data.stat().st_size // tok_width if held_data.is_file() else 0
        if n_train_toks <= T:
            raise RuntimeError(f"Base corpus too small ({n_train_toks} tokens <= T={T}); add more data or lower T.")
        eval_data = held_data if (held_data.is_file() and n_held_toks > T) else train_data
        eval_toks = n_held_toks if eval_data == held_data else n_train_toks
        eval_w = max(1, min(8, (eval_toks - 1) // T))

        run.emit("stage", stage="dataset_verified", label="OPENWALDO VERIFIED", status="done")
        run.emit("helix", node_id="base-data", status="done", sha256=base_bom["openwaldo_export"]["export_json_sha256"])

        # 2. Base model init & training
        run.emit("stage", stage="base_train", label="Training Base Trunk")
        base_ck = run_dir / "base.ck"
        init_argv = [
            str(bins["ollm"]), "init", str(base_ck),
            str(V), str(T), str(D), str(H), str(F), str(L), str(B),
            str(spec["seed"]), str(spec["base_steps"]), str(spec["lr_micro"]),
        ]
        if NP > 1:
            init_argv += ["1", str(LP), str(LC), str(NP)]
        self._run_stream(run, init_argv, "base_train")

        def _parse_eval_line(text: str) -> tuple[float, float] | None:
            for ln in text.splitlines():
                parts = ln.split()
                if len(parts) == 5 and parts[0] == "E":
                    return int(parts[3]) / 1e6, int(parts[4]) / 1e6
            return None

        ev_before_out = self._run_stream(run, [str(bins["ollm"]), "eval", str(base_ck), str(eval_data), str(eval_w)], "base_train")
        ev_before = _parse_eval_line(ev_before_out)
        if ev_before:
            run.emit("metric", stage="base_train", step=0, loss=ev_before[0], bpb=ev_before[1], tok_s=0.0)

        base_steps = spec["base_steps"]
        every = max(1, min(base_steps, max(1, base_steps // 5)))
        t_start = time.time()
        last_step = [0]

        def _on_train_line(line: str) -> None:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "T":
                try:
                    step_n = int(parts[1])
                    loss_val = int(parts[2]) / 1e6
                    elapsed = max(1e-4, time.time() - t_start)
                    tok_s = round((step_n * B * T) / elapsed, 1)
                    last_step[0] = step_n
                    run.emit("metric", stage="base_train", step=step_n, loss=loss_val, bpb=round(loss_val / LN2, 6), tok_s=tok_s)
                except ValueError:
                    pass

        train_cmd = "gtrainf" if use_gb10 else "train"
        self._run_stream(
            run,
            [str(ollm_bin), train_cmd, str(base_ck), str(train_data), str(base_steps), str(every)],
            "base_train",
            line_cb=_on_train_line,
        )

        ev_after_out = self._run_stream(run, [str(bins["ollm"]), "eval", str(base_ck), str(eval_data), str(eval_w)], "base_train")
        ev_after = _parse_eval_line(ev_after_out)
        base_ck_hdr = ow_bridge.read_ollm_checkpoint_header(base_ck)
        if ev_after:
            run.emit("metric", stage="base_eval", step=base_steps, loss=ev_after[0], bpb=ev_after[1], tok_s=0.0)
        run.emit("artifact", **_file_artifact(base_ck, "base.ck", "checkpoint", magic=base_ck_hdr["magic"]))
        run.emit("helix", node_id="base-model", status="done", sha256=base_ck_hdr["sha256"], loss=ev_after[0] if ev_after else None)

        # 3. DNA Merkle Specialist Strands
        strand_cks: dict[str, Path] = {}
        strand_depths: dict[str, int] = {}
        strand_records: list[dict[str, Any]] = []
        strand_packs: dict[str, Path] = {}
        strand_boms: list[dict[str, Any]] = []

        if strands:
            run.emit("helix", node_id="merkle-lock", status="locked", sha256=base_ck_hdr["sha256"])

        global_step_offset = base_steps
        for s in strands:
            sid = s["id"]
            run.emit("stage", stage=f"strand:{sid}", label=f"Forging Strand: {s['name']}")
            run.emit("helix", node_id=f"strand-{sid}", status="running")

            # Prepare strand dataset if distinct from base
            if s["dataset"] != spec["base_dataset"]:
                s_exp_dir = run_dir / "datasets" / f"strand_{sid}_export"
                s_prep = self._prepare_dataset_spec(run, s["dataset"], s_exp_dir, f"strand-{sid}", waldo_bin)
                s_data_dir = run_dir / "data" / f"strand_{sid}"
                self._run_stream(
                    run,
                    [
                        sys.executable, str(Path(ow_bridge.__file__).resolve()), "prepare",
                        str(s_prep["export_dir"]), str(s_data_dir),
                        "--ollm-waldo", str(waldo_bin), "--held-mod", str(spec["held_mod"]),
                    ],
                    f"strand:{sid}",
                )
                strand_boms.append({
                    "strand_id": sid,
                    "bom": json.loads((s_data_dir / "openwaldo-helix-bom.json").read_text(encoding="utf-8")),
                })
                if merges_path is not None:
                    s_train = s_data_dir / "train.tok"
                    s_held = s_data_dir / "held.tok"
                    self._run_stream(
                        run,
                        [str(bins["ollm"]), "bpeenc", str(merges_path), str(V), str(s_data_dir / "train.bin"), str(s_train)],
                        f"strand:{sid}",
                    )
                    if (s_data_dir / "held.bin").is_file() and (s_data_dir / "held.bin").stat().st_size > 0:
                        self._run_stream(
                            run,
                            [str(bins["ollm"]), "bpeenc", str(merges_path), str(V), str(s_data_dir / "held.bin"), str(s_held)],
                            f"strand:{sid}",
                        )
                else:
                    s_train = s_data_dir / "train.bin"
                    s_held = s_data_dir / "held.bin"
            else:
                s_train = train_data
                s_held = held_data

            s_train_toks = s_train.stat().st_size // tok_width
            if s_train_toks <= T:
                raise RuntimeError(f"Strand {sid} corpus too small ({s_train_toks} tokens <= T={T}).")
            s_held_toks = s_held.stat().st_size // tok_width if s_held.is_file() else 0
            s_eval = s_held if (s_held.is_file() and s_held_toks > T) else s_train
            s_eval_w = max(1, min(8, ((s_held_toks if s_eval == s_held else s_train_toks) - 1) // T))

            pid = s["parent_id"]
            parent_ck = strand_cks[pid] if pid and pid in strand_cks else base_ck
            parent_L = strand_depths[pid] if pid and pid in strand_depths else L
            new_L = parent_L + s["add_layers"]

            strand_ck = run_dir / f"strand_{sid}.ck"
            strand_sp = run_dir / f"strand_{sid}.sp"
            self._run_stream(
                run,
                [str(bins["ollm"]), s["growth"], str(parent_ck), str(strand_ck), str(new_L), str(s["steps"]), str(s["lr_micro"])],
                f"strand:{sid}",
            )

            s_every = max(1, min(s["steps"], max(1, s["steps"] // 3)))
            st_t0 = time.time()

            def _on_strand_line(line: str, _sid: str = sid, _off: int = global_step_offset, _t0: float = st_t0) -> None:
                parts = line.split()
                if len(parts) == 4 and parts[0] in ("F", "FC"):
                    try:
                        st_n = int(parts[1])
                        loss_v = int(parts[3]) / 1e6
                        elapsed = max(1e-4, time.time() - _t0)
                        run.emit(
                            "metric",
                            stage=f"strand:{_sid}",
                            strand_id=_sid,
                            step=_off + st_n,
                            strand_step=st_n,
                            loss=loss_v,
                            bpb=round(loss_v / LN2, 6),
                            tok_s=round((st_n * B * T) / elapsed, 1),
                        )
                    except ValueError:
                        pass
                elif len(parts) == 3 and parts[0] == "T":
                    try:
                        st_n = int(parts[1])
                        loss_v = int(parts[2]) / 1e6
                        elapsed = max(1e-4, time.time() - _t0)
                        run.emit(
                            "metric",
                            stage=f"strand:{_sid}",
                            strand_id=_sid,
                            step=_off + st_n,
                            strand_step=st_n,
                            loss=loss_v,
                            bpb=round(loss_v / LN2, 6),
                            tok_s=round((st_n * B * T) / elapsed, 1),
                        )
                    except ValueError:
                        pass

            if use_gb10:
                self._run_stream(
                    run,
                    [str(ollm_bin), "gtrainf", str(strand_ck), str(s_train), str(s["steps"]), str(s_every), str(parent_L)],
                    f"strand:{sid}",
                    line_cb=_on_strand_line,
                )
            elif s["use_fcachem"]:
                hc_path = run_dir / f"strand_{sid}.hc"
                fc_w = max(1, min(32, (s_train_toks - 1) // T))
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "fcache", str(parent_ck), str(s_train), str(fc_w), str(hc_path)],
                    f"strand:{sid}",
                )
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "trainfc", str(strand_ck), str(hc_path), str(s["steps"]), str(s_every), str(parent_L)],
                    f"strand:{sid}",
                    line_cb=_on_strand_line,
                )
            else:
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "trainf", str(strand_ck), str(s_train), str(s["steps"]), str(s_every), str(parent_L)],
                    f"strand:{sid}",
                    line_cb=_on_strand_line,
                )

            global_step_offset += s["steps"]

            # Extract Merkle-verified OLLMSP01 specialist pack
            self._run_stream(
                run,
                [str(bins["ollm"]), "spext", str(parent_ck), str(strand_ck), str(strand_sp)],
                f"strand:{sid}",
            )
            sp_hdr = read_sp_header(strand_sp)
            s_ev_out = self._run_stream(
                run,
                [str(bins["ollm"]), "eval", str(strand_ck), str(s_eval), str(s_eval_w)],
                f"strand:{sid}",
            )
            s_ev = _parse_eval_line(s_ev_out)

            strand_cks[sid] = strand_ck
            strand_depths[sid] = new_L
            strand_packs[sid] = strand_sp
            rec = {
                "id": sid,
                "name": s["name"],
                "color": s["color"],
                "parent_id": pid,
                "add_layers": s["add_layers"],
                "growth": s["growth"],
                "weight": s["weight"],
                "pack_path": str(strand_sp),
                "pack_sha256": sp_hdr["sha256"],
                "pack_hash": sp_hdr["pack_hash"],
                "base_hash": sp_hdr["base_hash"],
                "base_L": sp_hdr["base_L"],
                "spec_L": sp_hdr["spec_L"],
                "n_spec_params": sp_hdr["n_spec_params"],
                "eval_loss": s_ev[0] if s_ev else None,
                "eval_bpb": s_ev[1] if s_ev else None,
                "status": "locked",
            }
            strand_records.append(rec)
            run.strands = list(strand_records)
            run.emit("artifact", **_file_artifact(strand_sp, strand_sp.name, "specialist_pack", pack_hash=sp_hdr["pack_hash"], base_hash=sp_hdr["base_hash"]))
            run.emit("helix", node_id=f"strand-{sid}", status="locked", sha256=sp_hdr["sha256"], pack_hash=sp_hdr["pack_hash"], loss=s_ev[0] if s_ev else None)

        # 4. Merge / apply packs into merged.ck
        merged_ck = base_ck
        if strands:
            run.emit("stage", stage="merge", label="Applying & Linking Merkle Packs")
            merged_ck = run_dir / "merged.ck"
            root_strands = [r for r in strand_records if not r["parent_id"]]
            is_linear_chain = len(root_strands) == 1 and len(strand_records) >= 1 and all(
                (i == 0 and not r["parent_id"]) or (i > 0 and r["parent_id"] == strand_records[i - 1]["id"])
                for i, r in enumerate(strand_records)
            )
            if len(strand_records) == 1:
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "spapp", str(base_ck), strand_records[0]["pack_path"], str(merged_ck)],
                    "merge",
                )
            elif is_linear_chain:
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "spchainapp", str(base_ck), str(merged_ck), str(len(strand_records))]
                    + [r["pack_path"] for r in strand_records],
                    "merge",
                )
            elif len(root_strands) == len(strand_records) and 2 <= len(root_strands) <= 4 and len({r["spec_L"] for r in root_strands}) == 1:
                merged_sp = run_dir / "merged.sp"
                merge_argv = [str(bins["ollm"]), "spmerge", str(base_ck), "0", str(merged_sp)]
                for r in root_strands:
                    merge_argv += [str(max(1, int(r["weight"]))), r["pack_path"]]
                self._run_stream(run, merge_argv, "merge")
                self._run_stream(
                    run,
                    [str(bins["ollm"]), "spapp", str(base_ck), str(merged_sp), str(merged_ck)],
                    "merge",
                )
                run.emit("artifact", **_file_artifact(merged_sp, "merged.sp", "specialist_pack"))
            else:
                # Apply the first root chain into merged.ck; individual packs stay intact for spclip
                first_chain = self._build_strand_chains(strand_records)[0]
                if len(first_chain) == 1:
                    self._run_stream(
                        run,
                        [str(bins["ollm"]), "spapp", str(base_ck), first_chain[0]["pack_path"], str(merged_ck)],
                        "merge",
                    )
                else:
                    self._run_stream(
                        run,
                        [str(bins["ollm"]), "spchainapp", str(base_ck), str(merged_ck), str(len(first_chain))]
                        + [r["pack_path"] for r in first_chain],
                        "merge",
                    )

            merged_hdr = ow_bridge.read_ollm_checkpoint_header(merged_ck)
            run.emit("artifact", **_file_artifact(merged_ck, "merged.ck", "checkpoint", magic=merged_hdr["magic"]))
            if len(strand_records) >= 2:
                run.emit("helix", node_id="spclip", status="done", sha256=merged_hdr["sha256"])

        # 5. Interchange Upgrade -> OLLMCK04 & HF / GGUF exports
        exports: list[dict[str, Any]] = []
        v4_ck: Path | None = None
        if need_interchange:
            run.emit("stage", stage="export", label="Interchange Upgrade & Open Exports")
            v4_ck = run_dir / "model_v4.ck"
            self._run_stream(
                run,
                [str(bins["ollm_interchange"]), "upgrade", str(merged_ck), str(v4_ck)],
                "export",
            )
            v4_hdr = univ_backend.read_ck4_header(v4_ck)
            run.emit("artifact", **_file_artifact(v4_ck, "model_v4.ck", "checkpoint_v4", magic="OLLMCK04"))
            run.emit("helix", node_id="ck04-upgrade", status="done", sha256=v4_hdr["sha256"])

            if spec["export_hf"]:
                hf_dir = run_dir / "export-hf"
                hf_dir.mkdir(parents=True, exist_ok=True)
                self._run_stream(
                    run,
                    [str(bins["ollm_interchange"]), "export-hf", str(v4_ck), str(hf_dir), spec["hf_dtype"]],
                    "export",
                )
                hf_files = sorted(p for p in hf_dir.iterdir() if p.is_file())
                for hf_f in hf_files:
                    run.emit("artifact", **_file_artifact(hf_f, f"export-hf/{hf_f.name}", "hf_export"))
                exports.append({
                    "format": "hf-safetensors",
                    "dtype": spec["hf_dtype"],
                    "dir": str(hf_dir),
                    "files": [_file_artifact(p, p.name, "hf_export") for p in hf_files],
                })
                st_p = hf_dir / "model.safetensors"
                run.emit("helix", node_id="export-hf", status="done", sha256=ow_bridge.file_sha256_and_bytes(st_p)[0] if st_p.is_file() else "")

            if spec["export_gguf"]:
                gguf_p = run_dir / "model.gguf"
                self._run_stream(
                    run,
                    [str(bins["ollm_interchange"]), "export-gguf", str(v4_ck), str(gguf_p)],
                    "export",
                )
                gguf_art = _file_artifact(gguf_p, "model.gguf", "gguf_export")
                run.emit("artifact", **gguf_art)
                exports.append({"format": "gguf", "version": 3, **gguf_art})
                run.emit("helix", node_id="export-gguf", status="done", sha256=gguf_art["sha256"])

        # 6. Unified OpenWALDO Helix BOM
        run.emit("stage", stage="bom", label="Writing OpenWALDO Helix BOM")
        unified_bom = dict(base_bom)
        unified_bom["generated"] = utc_now()
        unified_bom["studio"] = {
            "run_id": run.run_id,
            "mode": spec["mode"],
            "spec": spec,
            "engine": str(ollm_bin),
            "gpu_used": use_gb10,
        }
        ck_entries = [dict(base_ck_hdr, role="base")]
        if merged_ck != base_ck and merged_ck.is_file():
            ck_entries.append(dict(ow_bridge.read_ollm_checkpoint_header(merged_ck), role="merged"))
        if v4_ck is not None and v4_ck.is_file():
            ck_entries.append(dict(univ_backend.read_ck4_header(v4_ck), role="upgraded_ck04"))
        unified_bom["checkpoints"] = ck_entries
        unified_bom["specialist_packs"] = strand_records
        if strand_boms:
            unified_bom["strand_datasets"] = strand_boms
        unified_bom["exports"] = exports

        bom_path = run_dir / "openwaldo-helix-bom.json"
        tmp_bom = run_dir / ".openwaldo-helix-bom.json.tmp"
        tmp_bom.write_text(json.dumps(unified_bom, indent=2) + "\n", encoding="utf-8")
        tmp_bom.replace(bom_path)
        run.bom = unified_bom
        bom_art = _file_artifact(bom_path, "openwaldo-helix-bom.json", "bom")
        run.emit("artifact", **bom_art)
        run.emit("helix", node_id="bom", status="done", sha256=bom_art["sha256"])

        run.sample_meta = {
            "engine": "dna",
            "ollm_bin": str(bins["ollm"]),
            "base_ck": str(base_ck),
            "merged_ck": str(merged_ck),
            "v4_ck": str(v4_ck) if v4_ck else "",
            "merges_bin": str(merges_path) if merges_path else "",
            "V": V,
            "NP": NP,
            "strands": strand_records,
        }

    def _run_universal(self, run: ChainRun) -> None:
        spec = run.spec
        run_dir = run.run_dir

        run.emit("stage", stage="resolve", label="Resolving Universal Engine Binaries")
        needed = ["ollm_waldo", "ollm_univ"]
        if spec["tokenizer"] == "bpe":
            needed.append("ollm")
        if spec["export_hf"] or spec["export_gguf"]:
            needed.append("ollm_interchange")
        bins = ensure_binaries(
            needed,
            bin_dir=self.bin_dir,
            root=self.root,
            log_cb=lambda msg: run.emit("log", stage="resolve", text=msg),
        )

        run.emit("stage", stage="dataset", label="OpenWALDO Verification & Ingest")
        exp_dir = run_dir / "datasets" / "base_export"
        prep = self._prepare_dataset_spec(run, spec["base_dataset"], exp_dir, "base", bins["ollm_waldo"])
        verified_export_dir = Path(prep["export_dir"])
        run.emit("stage", stage="dataset_verified", label="OPENWALDO VERIFIED", status="done")
        run.emit("helix", node_id="base-data", status="done", sha256=prep["report"]["export_json_sha256"])

        run.emit("stage", stage="train", label=f"Training Universal {spec['arch'].upper()} Engine")
        backend_spec: dict[str, Any] = {
            "export_dir": str(verified_export_dir),
            "workdir": str(run_dir / "universal"),
            "arch": spec["arch"],
            "tokenizer": spec["tokenizer"],
            "vocab": spec["V"],
            "held_mod": spec["held_mod"],
            "T": spec["T"],
            "D": spec["D"],
            "H": spec["H"],
            "H_KV": spec["H_KV"],
            "F": spec["F"],
            "L": spec["L"],
            "B": spec["B"],
            "seed": spec["seed"],
            "steps": spec["base_steps"],
            "lr_micro": spec["lr_micro"],
            "export_hf": spec["export_hf"],
            "hf_dtype": spec["hf_dtype"],
            "export_gguf": spec["export_gguf"],
            "ollm_waldo": str(bins["ollm_waldo"]),
            "ollm_univ": str(bins["ollm_univ"]),
        }
        if spec["arch"] == "custom" and spec["flags"] is not None:
            backend_spec["flags"] = spec["flags"]
        if spec["NP"] > 1:
            backend_spec["LP"] = spec["LP"]
            backend_spec["LC"] = spec["LC"]
            backend_spec["NP"] = spec["NP"]
        if "ollm" in bins:
            backend_spec["ollm"] = str(bins["ollm"])
        if "ollm_interchange" in bins:
            backend_spec["ollm_interchange"] = str(bins["ollm_interchange"])

        run.emit("log", stage="train", text=f"Delegating to omega_helix_backend.py (arch={spec['arch']}, V={spec['V']}, L={spec['L']}, D={spec['D']})")
        t0 = time.time()
        rc, obs = univ_backend.run_backend(backend_spec)
        elapsed = max(1e-4, time.time() - t0)
        if rc != 0:
            raise RuntimeError(f"omega_helix_backend failed at stage {obs.get('stage')}: {obs.get('error')}")

        # Emit metrics from eval_before, train_log, eval_after
        bp = obs["eval_before"]["primary_split"]
        ev_b = obs["eval_before"][bp]
        run.emit("metric", stage="eval_before", step=0, loss=ev_b["loss"], bpb=ev_b["bits_per_token"], tok_s=0.0)
        total_steps = max(1, spec["base_steps"])
        for row in obs.get("train_log", []):
            st = int(row["step"])
            lv = float(row["loss"])
            run.emit("metric", stage="train", step=st, loss=lv, bpb=round(lv / LN2, 6), tok_s=round((st * spec["B"] * spec["T"]) / elapsed, 1))
        ev_a = obs["eval_after"][bp]
        run.emit("metric", stage="eval_after", step=total_steps, loss=ev_a["loss"], bpb=ev_a["bits_per_token"], tok_s=round((total_steps * spec["B"] * spec["T"]) / elapsed, 1))

        # Copy BOM to run_dir root so all runs share identical path conventions
        univ_dir = run_dir / "universal"
        src_bom = univ_dir / "openwaldo-helix-bom.json"
        dst_bom = run_dir / "openwaldo-helix-bom.json"
        bom_doc = json.loads(src_bom.read_text(encoding="utf-8"))
        dst_bom.write_text(json.dumps(bom_doc, indent=2) + "\n", encoding="utf-8")
        run.bom = bom_doc

        ck_p = univ_dir / "model.ck"
        ck_art = _file_artifact(ck_p, "model.ck", "checkpoint_v4", magic="OLLMCK04")
        run.emit("artifact", **ck_art)
        run.emit("helix", node_id="base-model", status="done", sha256=ck_art["sha256"], loss=ev_a["loss"])

        for exp in obs.get("exports", []):
            if exp["format"] == "hf-safetensors":
                for f in exp.get("files", []):
                    run.emit("artifact", name=f"export-hf/{f['name']}", kind="hf_export", path=f["path"], sha256=f["sha256"], bytes=f["bytes"])
                run.emit("helix", node_id="export-hf", status="done", sha256=exp["files"][0]["sha256"] if exp.get("files") else "")
            elif exp["format"] == "gguf":
                run.emit("artifact", name="model.gguf", kind="gguf_export", path=exp["path"], sha256=exp["sha256"], bytes=exp["bytes"])
                run.emit("helix", node_id="export-gguf", status="done", sha256=exp["sha256"])

        bom_art = _file_artifact(dst_bom, "openwaldo-helix-bom.json", "bom")
        run.emit("artifact", **bom_art)
        run.emit("helix", node_id="bom", status="done", sha256=bom_art["sha256"])

        merges_info = obs.get("tokenization", {}).get("merges") or {}
        run.sample_meta = {
            "engine": "universal",
            "ollm_univ_bin": str(bins["ollm_univ"]),
            "ollm_bin": str(bins.get("ollm") or ""),
            "model_ck": str(ck_p),
            "merges_bin": str(merges_info.get("path", "")),
            "V": spec["V"],
            "tokenizer": spec["tokenizer"],
        }

    @staticmethod
    def _build_strand_chains(strand_records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        by_id = {r["id"]: r for r in strand_records}
        has_child = {r["parent_id"] for r in strand_records if r.get("parent_id")}
        leaves = [r for r in strand_records if r["id"] not in has_child]
        chains: list[list[dict[str, Any]]] = []
        for leaf in leaves:
            chain: list[dict[str, Any]] = []
            cur: dict[str, Any] | None = leaf
            seen: set[str] = set()
            while cur is not None and cur["id"] not in seen:
                seen.add(cur["id"])
                chain.append(cur)
                pid = cur.get("parent_id") or ""
                cur = by_id.get(pid) if pid else None
            chain.reverse()
            chains.append(chain)
        return chains

    # ------------------------------------------------------------------ interactive sample

    def sample_chain(
        self,
        run_id: str,
        prompt: str = "Omega ",
        n_tokens: int = 48,
        temp_milli: int = 800,
        seed: int = 42,
        spclip_mode: int = 1,
        strand_weights: dict[str, Any] | list[Any] | None = None,
        adapt_thresh_micro: int = 0,
        msg_gain_micro: int = 0,
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id!r}")
        if run.status not in ("done",):
            raise ValueError(f"run {run_id} is not complete (status={run.status})")

        n_tokens = max(1, min(512, int(n_tokens)))
        temp_milli = max(0, min(5000, int(temp_milli)))
        seed = max(0, int(seed))
        spclip_mode = 1 if int(spclip_mode) else 0
        adapt_thresh_micro = max(0, int(adapt_thresh_micro))
        msg_gain_micro = max(0, int(msg_gain_micro))

        meta = run.sample_meta
        bins = resolve_binaries(bin_dir=self.bin_dir, root=self.root)

        if meta.get("engine") == "universal":
            univ_bin = Path(meta.get("ollm_univ_bin") or "")
            if not univ_bin.is_file():
                univ_bin = ensure_binaries(["ollm_univ"], bin_dir=self.bin_dir, root=self.root)["ollm_univ"]
            ck_p = Path(meta["model_ck"])
            proc = subprocess.run(
                [str(univ_bin), "sample", str(ck_p), str(seed), str(n_tokens), str(temp_milli)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollm-univ sample exited {proc.returncode}: {proc.stderr or proc.stdout}")
            tok_ids = [int(x) for x in proc.stdout.strip().split() if x.isdigit()]
            text = self._decode_tokens(tok_ids, meta.get("merges_bin", ""), bins.get("ollm"), run.run_dir)
            return {
                "ok": True,
                "text": text,
                "stdout": proc.stdout,
                "mode_used": "ollm-univ",
                "telemetry": {
                    "tokens": tok_ids,
                    "exit_passes": [],
                    "exit_histogram": [],
                    "msg_norms_milli": [],
                    "strand_weights": {},
                    "spclip_mode": 0,
                },
            }

        # DNA / Hybrid sampling
        ollm_bin = Path(meta.get("ollm_bin") or "")
        if not ollm_bin.is_file():
            ollm_bin = ensure_binaries(["ollm"], bin_dir=self.bin_dir, root=self.root)["ollm"]
        base_ck = Path(meta["base_ck"])
        merges_bin = meta.get("merges_bin") or ""
        merges_arg = merges_bin if (merges_bin and Path(merges_bin).is_file()) else "-"
        strands = meta.get("strands") or []
        V = int(meta.get("V", 256))
        NP = int(meta.get("NP", 1))

        # Adaptive looped sampling when requested and supported (V > 256 + merges)
        if (adapt_thresh_micro > 0 or msg_gain_micro > 0) and V > 256 and merges_arg != "-":
            ck_target = Path(meta.get("merged_ck") or base_ck)
            proc = subprocess.run(
                [
                    str(ollm_bin), "adaptsamp", str(ck_target),
                    str(seed), str(n_tokens), str(temp_milli),
                    prompt, merges_arg, str(adapt_thresh_micro), str(msg_gain_micro),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollm adaptsamp exited {proc.returncode}: {proc.stderr or proc.stdout}")
            tok_ids, exit_passes, msg_norms, hist = [], [], [], []
            for ln in proc.stdout.splitlines():
                p = ln.split()
                if len(p) >= 6 and p[0] == "AS":
                    tok_ids.append(int(p[3]))
                    exit_passes.append(int(p[4]))
                    msg_norms.append(int(p[5]))
                elif len(p) >= 2 and p[0] == "AH":
                    hist = [int(x) for x in p[1:] if x.lstrip("-").isdigit()]
            text = self._decode_tokens(tok_ids, merges_bin, ollm_bin, run.run_dir)
            return {
                "ok": True,
                "text": text,
                "stdout": proc.stdout,
                "mode_used": "adaptsamp",
                "telemetry": {
                    "tokens": tok_ids,
                    "exit_passes": exit_passes,
                    "exit_histogram": hist,
                    "msg_norms_milli": msg_norms,
                    "strand_weights": {},
                    "spclip_mode": spclip_mode,
                    "adapt_thresh_micro": adapt_thresh_micro,
                    "msg_gain_micro": msg_gain_micro,
                },
            }

        chains = self._build_strand_chains(strands)
        if len(chains) >= 2:
            active_chains = chains[:4]
            resolved_weights: dict[str, int] = {}
            argv = [
                str(ollm_bin), "spclip", str(base_ck), str(spclip_mode),
                prompt, str(n_tokens), str(temp_milli), str(seed), merges_arg,
                str(len(active_chains)),
            ]
            for idx, ch in enumerate(active_chains):
                leaf = ch[-1]
                w = int(leaf.get("weight", 500))
                if isinstance(strand_weights, dict):
                    if leaf["id"] in strand_weights:
                        w = int(strand_weights[leaf["id"]])
                    elif leaf["name"] in strand_weights:
                        w = int(strand_weights[leaf["name"]])
                elif isinstance(strand_weights, list) and idx < len(strand_weights):
                    w = int(strand_weights[idx])
                w = max(1, min(10000, w))
                resolved_weights[leaf["id"]] = w
                argv += [str(w), str(len(ch))] + [r["pack_path"] for r in ch]

            proc = subprocess.run(argv, capture_output=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).decode("utf-8", errors="replace")
                raise RuntimeError(f"ollm spclip exited {proc.returncode}: {err}")
            text = proc.stdout.decode("utf-8", errors="replace")
            return {
                "ok": True,
                "text": text,
                "stdout": text,
                "mode_used": "spclip",
                "telemetry": {
                    "tokens": [],
                    "exit_passes": [],
                    "exit_histogram": [],
                    "msg_norms_milli": [],
                    "strand_weights": resolved_weights,
                    "spclip_mode": spclip_mode,
                },
            }

        if len(chains) == 1 and len(chains[0]) == 1:
            pack_p = chains[0][0]["pack_path"]
            argv = [str(ollm_bin), "spsamp", str(base_ck), pack_p, str(n_tokens), str(temp_milli), prompt, str(seed)]
            if V > 256 and merges_arg != "-":
                argv.append(merges_arg)
            proc = subprocess.run(argv, capture_output=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).decode("utf-8", errors="replace")
                raise RuntimeError(f"ollm spsamp exited {proc.returncode}: {err}")
            text = proc.stdout.decode("utf-8", errors="replace")
            return {
                "ok": True,
                "text": text,
                "stdout": text,
                "mode_used": "spsamp",
                "telemetry": {
                    "tokens": [],
                    "exit_passes": [],
                    "exit_histogram": [],
                    "msg_norms_milli": [],
                    "strand_weights": {chains[0][0]["id"]: 1000},
                    "spclip_mode": spclip_mode,
                },
            }

        if len(chains) == 1 and len(chains[0]) > 1:
            ch = chains[0]
            argv = [
                str(ollm_bin), "spchainsamp", str(base_ck),
                str(n_tokens), str(temp_milli), prompt, str(seed), merges_arg, str(len(ch)),
            ] + [r["pack_path"] for r in ch]
            proc = subprocess.run(argv, capture_output=True, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).decode("utf-8", errors="replace")
                raise RuntimeError(f"ollm spchainsamp exited {proc.returncode}: {err}")
            text = proc.stdout.decode("utf-8", errors="replace")
            return {
                "ok": True,
                "text": text,
                "stdout": text,
                "mode_used": "spchainsamp",
                "telemetry": {
                    "tokens": [],
                    "exit_passes": [],
                    "exit_histogram": [],
                    "msg_norms_milli": [],
                    "strand_weights": {r["id"]: int(r.get("weight", 500)) for r in ch},
                    "spclip_mode": spclip_mode,
                },
            }

        # Base checkpoint only
        argv = [str(ollm_bin), "sample", str(base_ck), str(seed), str(n_tokens), str(temp_milli), prompt]
        if V > 256 and merges_arg != "-":
            argv.append(merges_arg)
        proc = subprocess.run(argv, capture_output=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(f"ollm sample exited {proc.returncode}: {err}")
        text = proc.stdout.decode("utf-8", errors="replace")
        return {
            "ok": True,
            "text": text,
            "stdout": text,
            "mode_used": "sample",
            "telemetry": {
                "tokens": [],
                "exit_passes": [],
                "exit_histogram": [],
                "msg_norms_milli": [],
                "strand_weights": {},
                "spclip_mode": 0,
            },
        }

    def _decode_tokens(
        self,
        tok_ids: list[int],
        merges_bin: str,
        ollm_bin: Path | None,
        tmp_parent: Path,
    ) -> str:
        if not tok_ids:
            return ""
        if merges_bin and Path(merges_bin).is_file() and ollm_bin and ollm_bin.is_file():
            tok_tmp = tmp_parent / f".samp_{uuid.uuid4().hex[:8]}.tok"
            dec_tmp = tmp_parent / f".samp_{uuid.uuid4().hex[:8]}.txt"
            try:
                tok_tmp.write_bytes(struct.pack(f"<{len(tok_ids)}H", *(min(65535, max(0, t)) for t in tok_ids)))
                proc = subprocess.run(
                    [str(ollm_bin), "bpedec", str(merges_bin), str(tok_tmp), str(dec_tmp)],
                    capture_output=True,
                    check=False,
                )
                if proc.returncode == 0 and dec_tmp.is_file():
                    return dec_tmp.read_bytes().decode("utf-8", errors="replace")
            finally:
                tok_tmp.unlink(missing_ok=True)
                dec_tmp.unlink(missing_ok=True)
        # Byte fallback
        raw = bytes(t for t in tok_ids if 0 <= t < 256)
        if raw:
            return raw.decode("utf-8", errors="replace")
        return " ".join(str(t) for t in tok_ids)
