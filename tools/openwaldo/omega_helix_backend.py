#!/usr/bin/env python3
"""OpenWALDO Universal Training Backend Adapter for Omega Helix (Phase 4).

Connects a verified OpenWALDO ``EXPORT.json`` corpus to ANY model architecture the Helix
universal engine (``ollm-univ``, OLLMCK04) supports, fail-closed end to end:

  1. validate the training spec (JSON ``--spec`` and/or CLI flags) -- unknown keys, bad
     shapes, bad arch/tokenizer/vocab combinations are refused before any work is done;
  2. verify the OpenWALDO export (shard sizes, SHA-256, per-record sha256(text)) and ingest
     it through the pure-Omega ``ollm-waldo`` (via ``openwaldo_bridge.py prepare``);
  3. tokenize: ``bpe`` (``ollm bpefit``/``bpeenc``), ``byte`` (bytes widened to the engine's
     token width for V) or ``u32`` (4-byte tokens, V > 65535);
  4. build the model: ``ollm-univ init`` for presets llama|qwen2|qwen3|mistral|gemma|gpt|gelu
     or ``custom`` (explicit flag bits), or ``ollm-interchange import-hf`` when ``hf_dir``
     (config.json + model.safetensors) is given;
  5. evaluate (held-out and train splits) before training, train with ``ollm-univ train``,
     evaluate again, and integrity-check the final OLLMCK04 checkpoint with ``ollm-univ info``;
  6. optionally export HF Safetensors (``export-hf``) and/or GGUF v3 (``export-gguf``) via
     ``ollm-interchange``;
  7. write ``openwaldo-helix-bom.json`` (dataset BOM + artifacts SHA-256 + checkpoint + exports)
     and ``openwaldo-helix-observations.json`` (structured observations incl. eval before/after).

On any failure no checkpoint is bound into the BOM; a ``status: failed`` observations file
records the stage and error and the process exits non-zero.

Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import array
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openwaldo_bridge as bridge  # noqa: E402

SCHEMA = 1
OBS_KIND = "omega-helix-backend-observations"
BOM_NAME = "openwaldo-helix-bom.json"
OBS_NAME = "openwaldo-helix-observations.json"

# preset: (flags, rope_theta_milli, eps(label), wd_micro, warmup) -- mirrors native/univ/ucli.om upreset.
PRESETS: dict[str, dict[str, Any]] = {
    "llama": {"flags": 23, "rope_theta_milli": 500000000, "wd_micro": 100000, "warmup": 100},
    "mistral": {"flags": 23, "rope_theta_milli": 500000000, "wd_micro": 100000, "warmup": 100},
    "qwen2": {"flags": 23, "rope_theta_milli": 1000000000, "wd_micro": 100000, "warmup": 100},
    "qwen3": {"flags": 31, "rope_theta_milli": 1000000000, "wd_micro": 100000, "warmup": 100},
    "gemma": {"flags": 81, "rope_theta_milli": 10000000, "wd_micro": 100000, "warmup": 100},
    "gpt": {"flags": 0, "rope_theta_milli": 10000000, "wd_micro": 0, "warmup": 100},
    "gelu": {"flags": 32, "rope_theta_milli": 10000000, "wd_micro": 0, "warmup": 100},
}
CUSTOM_DEFAULTS = {"rope_theta_milli": 10000000, "wd_micro": 0, "warmup": 100}
ARCHES = tuple(PRESETS) + ("custom",)
TOKENIZERS = ("bpe", "byte", "u32")

DEFAULTS: dict[str, Any] = {
    "export_dir": None,
    "workdir": None,
    "arch": None,
    "hf_dir": "",
    "tokenizer": "byte",
    "vocab": None,
    "held_mod": 10,
    "T": None,
    "D": 32,
    "H": 4,
    "H_KV": None,
    "F": 64,
    "L": 2,
    "B": 2,
    "seed": 1,
    "steps": 20,
    "lr_micro": 3000,
    "eval_every": 0,
    "eval_windows": 4,
    "flags": None,
    "rope_theta_milli": None,
    "wd_micro": None,
    "warmup": None,
    "LP": None,
    "LC": None,
    "NP": None,
    "export_hf": False,
    "hf_dtype": "f32",
    "export_gguf": False,
    "ollm_waldo": "",
    "ollm": "",
    "ollm_univ": "",
    "ollm_interchange": "",
}
INT_KEYS = (
    "vocab", "held_mod", "T", "D", "H", "H_KV", "F", "L", "B", "seed", "steps", "lr_micro",
    "eval_every", "eval_windows", "flags", "rope_theta_milli", "wd_micro", "warmup", "LP", "LC", "NP",
)
MAX_U64 = (1 << 64) - 1

EXIT_HINTS = {
    2: "usage/argument refusal",
    10: "checkpoint truncated", 11: "bad checkpoint magic", 12: "invalid geometry/flags",
    13: "checkpoint parameter count mismatch", 15: "checkpoint has trailing bytes",
    20: "data file shorter than T+1 tokens", 21: "data file not a multiple of token width",
    22: "offset past data", 23: "eval window count out of range", 25: "token id >= V",
    70: "I/O error",
}


class BackendError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, doc: Any) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def art(path: Path, **extra: Any) -> dict[str, Any]:
    sha, size = bridge.file_sha256_and_bytes(path)
    out: dict[str, Any] = {"path": str(path), "sha256": sha, "bytes": size}
    out.update(extra)
    return out


def token_width(vocab: int) -> int:
    if vocab > 65535:
        return 4
    if vocab > 256:
        return 2
    return 1


# --------------------------------------------------------------------------- spec


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized, fully defaulted spec or raise BackendError('spec', ...)."""
    unknown = sorted(set(spec) - set(DEFAULTS))
    if unknown:
        raise BackendError("spec", f"unknown spec keys: {', '.join(unknown)}")
    s = dict(DEFAULTS)
    s.update({k: v for k, v in spec.items() if v is not None})

    for key in ("export_dir", "workdir"):
        if not isinstance(s[key], str) or not s[key]:
            raise BackendError("spec", f"missing required spec key {key!r}")
    for key in ("hf_dir", "ollm_waldo", "ollm", "ollm_univ", "ollm_interchange", "tokenizer", "hf_dtype"):
        if not isinstance(s[key], str):
            raise BackendError("spec", f"{key} must be a string")
    for key in INT_KEYS:
        v = s[key]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > MAX_U64:
            raise BackendError("spec", f"{key} must be a non-negative integer, got {v!r}")
    for key in ("export_hf", "export_gguf"):
        if not isinstance(s[key], bool):
            raise BackendError("spec", f"{key} must be boolean")

    arch = s["arch"]
    if arch not in ARCHES:
        raise BackendError("spec", f"arch must be one of {'|'.join(ARCHES)}, got {arch!r}")
    if s["tokenizer"] not in TOKENIZERS:
        raise BackendError("spec", f"tokenizer must be one of {'|'.join(TOKENIZERS)}, got {s['tokenizer']!r}")
    if s["hf_dtype"] not in ("f32", "bf16"):
        raise BackendError("spec", "hf_dtype must be f32 or bf16")
    if arch != "custom" and s["flags"] is not None:
        raise BackendError("spec", "flags is only valid with arch=custom")
    if arch == "custom" and s["flags"] is None:
        raise BackendError("spec", "arch=custom requires explicit flags (Helix architecture flag bits)")

    for key in ("steps", "B"):
        if s[key] < 1:
            raise BackendError("spec", f"{key} must be >= 1")
    if s["lr_micro"] < 1:
        raise BackendError("spec", "lr_micro must be >= 1")
    if s["eval_windows"] < 1:
        raise BackendError("spec", "eval_windows must be >= 1")

    # vocabulary / tokenizer
    tok = s["tokenizer"]
    if tok == "byte":
        if s["vocab"] is None:
            s["vocab"] = 256
        if s["vocab"] < 256:
            raise BackendError("spec", "tokenizer=byte requires vocab >= 256")
    elif tok == "u32":
        if s["vocab"] is None:
            s["vocab"] = 70000
        if s["vocab"] <= 65535:
            raise BackendError("spec", "tokenizer=u32 requires vocab > 65535 (4-byte tokens)")
    else:  # bpe
        if s["vocab"] is None:
            s["vocab"] = 280
        if not 256 < s["vocab"] <= 65535:
            raise BackendError("spec", "tokenizer=bpe requires 256 < vocab <= 65535")
    if s["held_mod"] < 0:
        raise BackendError("spec", "held_mod must be >= 0")

    if s["hf_dir"]:
        # shape comes from the HF config; only T/B/steps/lr are applied by import-hf
        if s["arch"] not in ARCHES:
            raise BackendError("spec", "arch invalid")
        if any(s[k] is not None for k in ("LP", "LC", "NP", "flags")):
            raise BackendError("spec", "LP/LC/NP/flags cannot be combined with hf_dir (taken from the HF config)")
        return s

    if s["T"] is None:
        s["T"] = 32
    for key in ("T", "D", "H", "F", "L"):
        if s[key] < 1:
            raise BackendError("spec", f"{key} must be >= 1")
    if s["H_KV"] is None:
        s["H_KV"] = max(1, s["H"] // 2) if s["H"] % 2 == 0 else s["H"]
    if s["H_KV"] < 1:
        raise BackendError("spec", "H_KV must be >= 1")
    if s["D"] % s["H"] != 0:
        raise BackendError("spec", f"D={s['D']} must be divisible by H={s['H']}")
    if s["H_KV"] > s["H"] or s["H"] % s["H_KV"] != 0:
        raise BackendError("spec", f"H={s['H']} must be a multiple of H_KV={s['H_KV']} (and H_KV <= H)")
    flags = s["flags"] if arch == "custom" else PRESETS[arch]["flags"]
    head_dim = s["D"] // s["H"]
    if (flags & 2) and (flags & 64):
        raise BackendError("spec", f"invalid architecture flags {flags}: bits 2 and 64 are exclusive")
    if (flags & 66) and (flags & 32):
        raise BackendError("spec", f"invalid architecture flags {flags}: bit 32 excludes bits 2/64")
    if (flags & 1) and head_dim % 2 != 0:
        raise BackendError("spec", f"RoPE requires an even head dimension, got D/H={head_dim}")
    s["_flags"] = flags

    loop_given = [k for k in ("LP", "LC", "NP") if s[k] is not None]
    if loop_given:
        lp = s["LP"] if s["LP"] is not None else 0
        lc = s["LC"] if s["LC"] is not None else 1
        npv = s["NP"] if s["NP"] is not None else 1
        if npv < 1:
            raise BackendError("spec", "NP must be >= 1")
        if npv > 1 and (lc < 1 or lp + lc > s["L"]):
            raise BackendError("spec", f"looped core needs LC >= 1 and LP+LC <= L (LP={lp} LC={lc} L={s['L']})")
        s["LP"], s["LC"], s["NP"] = lp, lc, npv
    return s


def spec_from_args(ns: argparse.Namespace) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    if ns.spec:
        p = Path(ns.spec)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BackendError("spec", f"cannot read spec JSON {p}: {exc}") from exc
        if not isinstance(doc, dict):
            raise BackendError("spec", "spec JSON root must be an object")
        spec.update(doc)
    for key in DEFAULTS:
        val = getattr(ns, key, None)
        if val is not None and val is not False:
            spec[key] = val
    return spec


# --------------------------------------------------------------------------- tools


def find_tool(explicit: str, env: str, names: list[str]) -> Path | None:
    cands: list[str] = []
    if explicit:
        cands.append(explicit)
    if os.environ.get(env):
        cands.append(os.environ[env])
    here = Path(__file__).resolve().parent.parent.parent
    for n in names:
        cands.append(str(here / "build" / n))
    for c in cands:
        p = Path(c).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return p
        if c in (explicit, os.environ.get(env, "\0")):
            raise BackendError("tools", f"{env.lower()} not executable: {c}")
    return None


class Runner:
    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []

    def run(self, stage: str, argv: list[str], ok_codes: tuple[int, ...] = (0,)) -> str:
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        self.log.append(
            {
                "stage": stage,
                "argv": [Path(argv[0]).name] + [str(a) for a in argv[1:]],
                "rc": proc.returncode,
                "seconds": round(time.time() - t0, 3),
            }
        )
        if proc.returncode not in ok_codes:
            hint = EXIT_HINTS.get(proc.returncode, "")
            raise BackendError(
                stage,
                f"{Path(argv[0]).name} {argv[1] if len(argv) > 1 else ''} exited {proc.returncode}"
                f"{' (' + hint + ')' if hint else ''}: {proc.stderr.strip() or proc.stdout.strip()}",
            )
        return proc.stdout


# --------------------------------------------------------------------------- tokens


def read_tokens(path: Path, width: int) -> array.array:
    raw = path.read_bytes()
    if len(raw) % width:
        raise BackendError("tokenize", f"{path.name}: size {len(raw)} not a multiple of token width {width}")
    if width == 1:
        return array.array("B", raw)
    code = "H" if width == 2 else "I"
    arr = array.array(code)
    if arr.itemsize != width:
        raise BackendError("tokenize", "platform array itemsize mismatch")
    arr.frombytes(raw)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def widen_bytes(src: Path, dst: Path, width: int) -> None:
    raw = src.read_bytes()
    if width == 1:
        dst.write_bytes(raw)
        return
    out = bytearray(len(raw) * width)
    for i, b in enumerate(raw):
        out[i * width] = b
    dst.write_bytes(bytes(out))


def check_tokens(path: Path, width: int, vocab: int, label: str) -> int:
    toks = read_tokens(path, width)
    if toks and max(toks) >= vocab:
        raise BackendError("tokenize", f"{label}: token id {max(toks)} >= vocab {vocab}")
    return len(toks)


# --------------------------------------------------------------------------- checkpoint


def read_ck4_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 208:
        raise BackendError("checkpoint", f"{path.name}: shorter than the 208-byte OLLMCK04 header")
    if raw[:8] != b"OLLMCK04":
        raise BackendError("checkpoint", f"{path.name}: bad magic {raw[:8]!r}, expected OLLMCK04")
    q = [int.from_bytes(raw[o : o + 8], "little") for o in range(8, 208, 8)]
    # offsets/8-1: 0..7 V T D H HKV F L B | 8 seed | 9 step | 10 total | 11 lr_bits | 12 P | 13 flags
    # 14 rope_milli | 15 eps_bits | 16 wd_bits | 17 warmup | 18 LP | 19 LC | 20 NP | 24 fnv(offset 200)
    names = ("V", "T", "D", "H", "H_KV", "F", "L", "B")
    return {
        "magic": "OLLMCK04",
        "shape": dict(zip(names, q[:8])),
        "seed": q[8],
        "step": q[9],
        "total_steps": q[10],
        "params": q[12],
        "flags": q[13],
        "rope_theta_milli": q[14],
        "warmup": q[17],
        "loop": {"LP": q[18], "LC": q[19], "NP": q[20]},
        "fnv1a64": f"{q[24]:016x}",
        "sha256": bridge.file_sha256_and_bytes(path)[0],
        "bytes": len(raw),
        "path": str(path),
    }


def parse_eval(out: str) -> dict[str, Any]:
    for line in out.splitlines():
        p = line.split()
        if len(p) == 5 and p[0] == "E":
            return {
                "step": int(p[1]),
                "windows": int(p[2]),
                "loss": int(p[3]) / 1e6,
                "bits_per_token": int(p[4]) / 1e6,
                "loss_micro": int(p[3]),
            }
    raise BackendError("eval", f"unparseable ollm-univ eval output: {out!r}")


# --------------------------------------------------------------------------- pipeline


def run_backend(raw_spec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    started = now_utc()
    obs: dict[str, Any] = {
        "kind": OBS_KIND,
        "schema": SCHEMA,
        "started": started,
        "host": f"{platform.system()}-{platform.machine()}",
        "status": "failed",
        "stage": "spec",
    }
    workdir: Path | None = None
    runner = Runner()
    try:
        spec = validate_spec(raw_spec)
        obs["spec"] = {k: v for k, v in spec.items() if not k.startswith("_")}
        workdir = Path(spec["workdir"]).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        for stale in (BOM_NAME, OBS_NAME):
            (workdir / stale).unlink(missing_ok=True)

        hf = bool(spec["hf_dir"])
        hf_config: dict[str, Any] = {}
        if hf:
            hfd = Path(spec["hf_dir"]).resolve()
            cfg_p, st_p = hfd / "config.json", hfd / "model.safetensors"
            if not cfg_p.is_file() or not st_p.is_file():
                raise BackendError("spec", f"hf_dir must contain config.json and model.safetensors: {hfd}")
            try:
                hf_config = json.loads(cfg_p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise BackendError("spec", f"hf config.json invalid JSON: {exc}") from exc
            if not isinstance(hf_config, dict):
                raise BackendError("spec", "hf config.json root must be an object")

        # ---- tools (resolved up front so we fail before doing work)
        obs["stage"] = "tools"
        waldo = find_tool(spec["ollm_waldo"], "OLLM_WALDO_BIN", ["ollm-waldo"])
        univ = find_tool(spec["ollm_univ"], "OLLM_UNIV_BIN", ["ollm-univ"])
        ollm = find_tool(spec["ollm"], "OLLM_BIN", ["ollm-macos", "ollm"]) if spec["tokenizer"] == "bpe" else None
        inter = find_tool(spec["ollm_interchange"], "OLLM_INTERCHANGE_BIN", ["ollm-interchange"])
        for need, path, name in (
            (True, waldo, "ollm-waldo (TARGET=openwaldo)"),
            (True, univ, "ollm-univ (TARGET=univ)"),
            (spec["tokenizer"] == "bpe", ollm, "ollm (TARGET=macos|aarch64) for BPE"),
            (hf or spec["export_hf"] or spec["export_gguf"], inter, "ollm-interchange (TARGET=interchange)"),
        ):
            if need and path is None:
                raise BackendError("tools", f"required binary not found: {name}")
        obs["tools"] = {
            n: {"path": str(p), "sha256": bridge.file_sha256_and_bytes(p)[0]}
            for n, p in (("ollm-waldo", waldo), ("ollm-univ", univ), ("ollm", ollm), ("ollm-interchange", inter))
            if p is not None and (n != "ollm-interchange" or hf or spec["export_hf"] or spec["export_gguf"])
        }

        # ---- 1. OpenWALDO verification (fail closed) + pure-Omega ingest
        obs["stage"] = "openwaldo-verify"
        try:
            report = bridge.verify_export(Path(spec["export_dir"]), check_records=True)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise BackendError("openwaldo-verify", str(exc)) from exc
        if report["format"] != "jsonl":
            raise BackendError("openwaldo-verify", "export format must be canonical jsonl (re-export with --format jsonl)")

        obs["stage"] = "ingest"
        data_dir = workdir / "data"
        bridge_argv = [
            sys.executable, str(Path(bridge.__file__).resolve()), "prepare", spec["export_dir"], str(data_dir),
            "--ollm-waldo", str(waldo), "--held-mod", str(spec["held_mod"]),
        ]
        tok = spec["tokenizer"]
        if tok == "bpe":
            bridge_argv += ["--ollm", str(ollm), "--vocab", str(spec["vocab"])]
        out = runner.run("ingest", bridge_argv)
        helix_bom = json.loads((data_dir / BOM_NAME).read_text(encoding="utf-8"))
        arts = helix_bom["artifacts"]
        obs["dataset"] = {
            "export_json_sha256": report["export_json_sha256"],
            "total_docs": report["total_docs"],
            "total_bytes": report["total_bytes"],
            "licenses": report["licenses"],
            "shards": helix_bom["openwaldo_export"]["shards"],
            "ingestor_summary": helix_bom["ingestor"]["summary"],
            "train_docs": arts["train_bin"]["docs"],
            "held_docs": arts["held_bin"]["docs"],
        }
        train_bin, held_bin = data_dir / "train.bin", data_dir / "held.bin"

        # ---- 2. model: init or HF import (determines V, T)
        ck = workdir / "model.ck"
        obs["stage"] = "model"
        ck.unlink(missing_ok=True)
        steps, eval_every = spec["steps"], spec["eval_every"]
        if eval_every == 0 or eval_every > steps:
            eval_every = max(1, steps // 4)
        if hf:
            t_arg = spec["T"]
            if t_arg is None:
                mp = hf_config.get("max_position_embeddings")
                t_arg = min(mp, 64) if isinstance(mp, int) and mp > 0 else 32
            runner.run(
                "import-hf",
                [str(inter), "import-hf", str(Path(spec["hf_dir"]).resolve() / "config.json"),
                 str(Path(spec["hf_dir"]).resolve() / "model.safetensors"), str(ck),
                 str(t_arg), str(spec["B"]), str(steps), str(spec["lr_micro"])],
            )
            if not ck.is_file():
                raise BackendError("model", "ollm-interchange import-hf produced no checkpoint")
            ck_v = read_ck4_header(ck)["shape"]["V"]
            if tok == "bpe" and ck_v != spec["vocab"]:
                raise BackendError("model", f"BPE vocab {spec['vocab']} != imported checkpoint V={ck_v}")
            if tok == "byte" and ck_v < 256:
                raise BackendError("model", f"byte tokenizer needs V >= 256, HF checkpoint V={ck_v}")
            if tok == "u32" and ck_v <= 65535:
                raise BackendError("model", f"u32 tokenizer needs V > 65535, HF checkpoint V={ck_v}")
            spec["vocab"] = ck_v
        else:
            preset = PRESETS.get(spec["arch"], CUSTOM_DEFAULTS)
            arch_arg = spec["arch"] if spec["arch"] != "custom" else str(spec["flags"])
            argv = [
                str(univ), "init", str(ck), arch_arg,
                *(str(spec[k]) for k in ("vocab", "T", "D", "H", "H_KV", "F", "L", "B", "seed")),
                str(steps), str(spec["lr_micro"]),
            ]
            extras = [spec["rope_theta_milli"], spec["wd_micro"], spec["warmup"]]
            if any(e is not None for e in extras) or spec["NP"] is not None:
                argv += [
                    str(spec[k] if spec[k] is not None else preset[k]) for k in ("rope_theta_milli", "wd_micro", "warmup")
                ]
                if spec["NP"] is not None:
                    argv += [str(spec["LP"]), str(spec["LC"]), str(spec["NP"])]
            runner.run("init", argv)
        vocab = spec["vocab"]
        width = token_width(vocab)

        # ---- 3. tokenization
        obs["stage"] = "tokenize"
        if tok == "bpe":
            if width != 2:
                raise BackendError("tokenize", f"BPE tokens are u16; vocab {vocab} needs width {width}")
            train_tok = data_dir / "train.tok"
            held_tok = data_dir / "held.tok"
            if not train_tok.is_file():
                raise BackendError("tokenize", "BPE encoding produced no train.tok")
            if not held_tok.is_file():
                held_tok = None
        else:
            if tok == "u32" and width != 4:
                raise BackendError("tokenize", f"u32 tokenizer requires width 4, vocab {vocab} gives {width}")
            train_tok = data_dir / f"train.w{width}.tok"
            widen_bytes(train_bin, train_tok, width)
            held_tok = None
            if held_bin.stat().st_size > 0:
                held_tok = data_dir / f"held.w{width}.tok"
                widen_bytes(held_bin, held_tok, width)
        shape_T = read_ck4_header(ck)["shape"]["T"]
        n_train = check_tokens(train_tok, width, vocab, "train tokens")
        n_held = check_tokens(held_tok, width, vocab, "held tokens") if held_tok else 0
        if n_train <= shape_T:
            raise BackendError(
                "tokenize", f"training corpus too small: {n_train} tokens <= T={shape_T}; need at least T+1"
            )
        tokenization = {
            "mode": tok, "vocab": vocab, "token_width": width, "train_tokens": n_train, "held_tokens": n_held,
            "train": art(train_tok), "held": art(held_tok) if held_tok else None,
        }
        if tok == "bpe":
            tokenization["merges"] = arts["merges_bin"]
        obs["tokenization"] = tokenization

        # ---- 4. eval before / train / eval after
        eval_splits: dict[str, Path] = {"train": train_tok}
        if held_tok and (n_held - 1) // shape_T >= 1:
            eval_splits["held"] = held_tok
        n_by_split = {"train": n_train, "held": n_held}

        def do_eval(label: str) -> dict[str, Any]:
            res: dict[str, Any] = {}
            for split, path in eval_splits.items():
                w = min(spec["eval_windows"], (n_by_split[split] - 1) // shape_T)
                res[split] = parse_eval(runner.run(f"eval-{label}-{split}", [str(univ), "eval", str(ck), str(path), str(w)]))
            res["primary_split"] = "held" if "held" in res else "train"
            return res

        obs["stage"] = "eval-before"
        before = do_eval("before")
        before_ck = read_ck4_header(ck)
        obs["stage"] = "train"
        train_out = runner.run(
            "train", [str(univ), "train", str(ck), str(train_tok), str(steps), str(eval_every)]
        )
        train_log = []
        for line in train_out.splitlines():
            p = line.split()
            if len(p) == 3 and p[0] == "T":
                train_log.append({"step": int(p[1]), "loss": int(p[2]) / 1e6})
        obs["stage"] = "eval-after"
        after = do_eval("after")

        # ---- 5. checkpoint integrity (ollm-univ info verifies FNV-1a and geometry)
        obs["stage"] = "checkpoint"
        info_out = runner.run("info", [str(univ), "info", str(ck)]).split()
        if len(info_out) < 2 or info_out[0] != "U4":
            raise BackendError("checkpoint", f"unexpected ollm-univ info output: {info_out!r}")
        ck_info = read_ck4_header(ck)
        if ck_info["step"] != steps:
            raise BackendError("checkpoint", f"final checkpoint step {ck_info['step']} != requested steps {steps}")
        if ck_info["sha256"] == before_ck["sha256"]:
            raise BackendError("checkpoint", "training did not modify the checkpoint")

        # ---- 6. optional exports
        obs["stage"] = "export"
        exports: list[dict[str, Any]] = []
        if spec["export_hf"]:
            hf_out = workdir / "export-hf"
            hf_out.mkdir(parents=True, exist_ok=True)  # ollm-interchange requires OUT_DIR to exist
            runner.run("export-hf", [str(inter), "export-hf", str(ck), str(hf_out), spec["hf_dtype"]])
            files = sorted(p for p in hf_out.rglob("*") if p.is_file()) if hf_out.is_dir() else []
            if not any(p.name == "model.safetensors" for p in files) or not any(p.name == "config.json" for p in files):
                raise BackendError("export", "export-hf did not produce config.json and model.safetensors")
            exports.append(
                {"format": "hf-safetensors", "dtype": spec["hf_dtype"], "dir": str(hf_out),
                 "files": [art(p, name=str(p.relative_to(hf_out))) for p in files]}
            )
        if spec["export_gguf"]:
            gguf = workdir / "model.gguf"
            gguf.unlink(missing_ok=True)
            runner.run("export-gguf", [str(inter), "export-gguf", str(ck), str(gguf)])
            if not gguf.is_file() or gguf.read_bytes()[:4] != b"GGUF":
                raise BackendError("export", "export-gguf did not produce a GGUF file")
            head = gguf.read_bytes()[:8]
            exports.append(
                {"format": "gguf", "version": int.from_bytes(head[4:8], "little"), **art(gguf)}
            )

        # ---- 7. BOM + observations
        obs["stage"] = "bom"
        ck_record = dict(ck_info)
        ck_record.update({"role": "final", "bound_at": now_utc(), "engine": "ollm-univ",
                          "arch": spec["arch"], "eval_before": before, "eval_after": after})
        helix_bom["generated"] = now_utc()
        helix_bom["training"] = {
            "backend": "omega_helix_backend.py", "schema": SCHEMA,
            "spec": obs["spec"], "tokenization": tokenization,
            "engine": obs["tools"]["ollm-univ"], "source": "hf-import" if hf else "init",
        }
        helix_bom["checkpoints"] = [ck_record]
        helix_bom["exports"] = exports
        bom_path = workdir / BOM_NAME
        atomic_write_json(bom_path, helix_bom)

        obs.update(
            {
                "status": "ok", "stage": "done", "finished": now_utc(),
                "checkpoint": ck_record, "eval_before": before, "eval_after": after,
                "improved": after[after["primary_split"]]["loss_micro"] < before[before["primary_split"]]["loss_micro"],
                "train_log": train_log, "exports": exports,
                "artifacts": {
                    "bom": art(bom_path), "train_bin": art(train_bin), "held_bin": art(held_bin),
                    "checkpoint": art(ck),
                },
                "commands": runner.log,
            }
        )
        atomic_write_json(workdir / OBS_NAME, obs)
        return 0, obs
    except BackendError as exc:
        obs.update({"status": "failed", "stage": exc.stage, "error": exc.message})
    except Exception as exc:  # fail closed on anything unexpected
        obs.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    obs["finished"] = now_utc()
    obs["commands"] = runner.log
    if workdir is None and isinstance(raw_spec.get("workdir"), str) and raw_spec["workdir"]:
        try:
            workdir = Path(raw_spec["workdir"]).resolve()
            workdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            workdir = None
    if workdir is not None and workdir.is_dir():
        # never leave a BOM that could be mistaken for a bound, verified result
        (workdir / BOM_NAME).unlink(missing_ok=True)
        atomic_write_json(workdir / OBS_NAME, obs)
    return 1, obs


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenWALDO -> Omega Helix universal training backend")
    p.add_argument("--spec", help="training spec JSON (CLI flags override)")
    p.add_argument("--export-dir", dest="export_dir")
    p.add_argument("--workdir")
    p.add_argument("--arch", help="|".join(ARCHES))
    p.add_argument("--hf-dir", dest="hf_dir", help="HF dir with config.json + model.safetensors (import instead of init)")
    p.add_argument("--tokenizer", choices=TOKENIZERS)
    p.add_argument("--vocab", type=int)
    p.add_argument("--held-mod", dest="held_mod", type=int)
    for k in ("T", "D", "H", "F", "L", "B", "LP", "LC", "NP"):
        p.add_argument(f"--{k}", dest=k, type=int)
    p.add_argument("--H-KV", dest="H_KV", type=int)
    for k in ("seed", "steps", "lr_micro", "eval_every", "eval_windows", "flags", "rope_theta_milli", "wd_micro", "warmup"):
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)
    p.add_argument("--export-hf", dest="export_hf", action="store_true")
    p.add_argument("--hf-dtype", dest="hf_dtype", choices=("f32", "bf16"))
    p.add_argument("--export-gguf", dest="export_gguf", action="store_true")
    for k in ("ollm_waldo", "ollm", "ollm_univ", "ollm_interchange"):
        p.add_argument(f"--{k.replace('_', '-')}", dest=k)
    p.add_argument("--print-observations", action="store_true", help="print the full observations JSON on stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        spec = spec_from_args(ns)
    except BackendError as exc:
        print(f"ERROR[{exc.stage}]: {exc.message}", file=sys.stderr)
        return 1
    rc, obs = run_backend(spec)
    if rc != 0:
        print(f"ERROR[{obs.get('stage')}]: {obs.get('error')}", file=sys.stderr)
        return rc
    if ns.print_observations:
        print(json.dumps(obs, indent=2))
    bp = obs["eval_before"]["primary_split"]
    print(
        f"OPENWALDO_BACKEND_OK arch={obs['spec']['arch']} tokenizer={obs['spec']['tokenizer']} "
        f"vocab={obs['tokenization']['vocab']} steps={obs['checkpoint']['step']} "
        f"{bp}_loss_before={obs['eval_before'][bp]['loss']:.6f} "
        f"{bp}_loss_after={obs['eval_after'][bp]['loss']:.6f} "
        f"ck_sha256={obs['checkpoint']['sha256']} "
        f"bom={obs['artifacts']['bom']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
