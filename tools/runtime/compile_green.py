"""Latent-Space 'Compile-Until-Green' Reasoning + Ephemeral Micro-TTT Crystallization.

Flow:
  1. Generate candidate with `ollm adaptsamp` (if looped) or `ollm sample` (or deterministic override).
  2. Check candidate via a deterministic verifier (`omc_compile`, `python_ast`, `json_schema`, `regex`).
  3. On failure, run ephemeral `ollm tttsamp` fast-weight update on the error feedback before retrying.
  4. Once green, crystallize the verified trace into a reusable `OLLMSP01` `.sp` specialist + receipt.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

TOOLS_DIR = Path(__file__).resolve().parents[1]
STUDIO_DIR = TOOLS_DIR / "studio"
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
for _p in (str(TOOLS_DIR), str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openwaldo_bridge as ow_bridge  # noqa: E402
from distill.verifiers import verify_record  # noqa: E402

from .anti_dna import read_sp_pack_hash  # noqa: E402


def _sample_candidate(
    ollm: str,
    ck: Path,
    prompt: str,
    *,
    is_loop: bool,
    n_tokens: int,
    temp_milli: int,
    seed: int,
    merges: str,
    adapt_thresh_micro: int,
    msg_gain_micro: int,
) -> str:
    if is_loop:
        argv = [
            ollm, "adaptsamp", str(ck), str(int(seed)), str(int(n_tokens)),
            str(int(temp_milli)), prompt, merges, str(int(adapt_thresh_micro)), str(int(msg_gain_micro)),
        ]
    else:
        argv = [ollm, "sample", str(ck), str(int(seed)), str(int(n_tokens)), str(int(temp_milli)), prompt, merges]
    res = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"sampling command failed (rc={res.returncode}): {res.stderr.strip()}")
    if not is_loop:
        return res.stdout
    toks: list[int] = []
    for ln in res.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 5 and parts[0] == "AS":
            toks.append(int(parts[2]))
    return bytes(t & 0xFF for t in toks).decode("utf-8", errors="replace") if toks else res.stdout


def run_compile_until_green(
    ollm_bin: str | Path,
    ck_path: str | Path,
    prompt: str,
    *,
    verifier: str = "omc_compile",
    max_attempts: int = 3,
    ttt_steps: int = 2,
    ttt_lr_micro: int = 1000,
    adapt_thresh_micro: int = 25000,
    msg_gain_micro: int = 150000,
    add_layers: int = 1,
    cryst_steps: int = 4,
    n_tokens: int = 24,
    temp_milli: int = 0,
    seed: int = 42,
    merges: str = "-",
    out_dir: str | Path,
    candidate_override_fn: Callable[[int, str], str] | None = None,
) -> dict[str, Any]:
    """Iterate LTP0 decode -> deterministic verifier -> micro-TTT on errors -> crystallize .sp."""
    ollm = str(Path(ollm_bin).expanduser().resolve())
    src_ck = Path(ck_path).expanduser().resolve()
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)
    work_ck = od / "work.ck"
    shutil.copy2(src_ck, work_ck)

    hdr = ow_bridge.read_ollm_checkpoint_header(work_ck)
    base_L = int(hdr["shape"]["L"])
    T = int(hdr["shape"]["T"])
    V = int(hdr["shape"]["V"])
    is_loop = hdr["magic"] == "OLLMCK03"
    layer_from = max(0, base_L - 1)

    attempts: list[dict[str, Any]] = []
    last_error = ""
    winning_text = ""
    passed = False

    for att in range(1, max(1, int(max_attempts)) + 1):
        if candidate_override_fn is not None:
            cand = candidate_override_fn(att, last_error)
        else:
            eff_prompt = prompt if not last_error else f"{prompt}\n/* fix: {last_error[:64]} */\n"
            cand = _sample_candidate(
                ollm,
                work_ck,
                eff_prompt,
                is_loop=is_loop,
                n_tokens=n_tokens,
                temp_milli=temp_milli,
                seed=seed + att - 1,
                merges=merges,
                adapt_thresh_micro=adapt_thresh_micro,
                msg_gain_micro=msg_gain_micro,
            )

        v = verify_record(cand, [verifier])
        ok = bool(v["passed"])
        detail = v["checks"][0]["detail"] if v["checks"] else ""
        attempts.append({
            "attempt": att,
            "passed": ok,
            "detail": detail,
            "candidate_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
            "ttt_applied": False,
        })
        if ok:
            passed = True
            winning_text = cand
            break

        last_error = detail or "verification_failed"
        if att < int(max_attempts):
            err_ctx = f"{prompt} // error: {last_error}"
            r_ttt = subprocess.run(
                [
                    ollm, "tttsamp", str(work_ck), str(int(seed + att)),
                    "4", str(int(temp_milli)), err_ctx, merges,
                    str(int(ttt_steps)), str(int(ttt_lr_micro)), str(int(layer_from)),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if r_ttt.returncode == 0:
                attempts[-1]["ttt_applied"] = True

    sp_info: dict[str, Any] | None = None
    if passed:
        trace_bytes = f"{prompt}\n{winning_text}\n".encode("utf-8", errors="replace")
        min_raw = (T + 8) * (4 if V > 256 else 1)
        while len(trace_bytes) <= min_raw:
            trace_bytes = trace_bytes + b"\n" + trace_bytes
        trace_bin = od / "winning_trace.bin"
        trace_bin.write_bytes(trace_bytes)
        train_input = trace_bin
        if V > 256 and merges and merges != "-":
            trace_tok = od / "winning_trace.tok"
            for _ in range(4):
                subprocess.run([ollm, "bpeenc", str(merges), str(V), str(trace_bin), str(trace_tok)], check=True, capture_output=True)
                if (trace_tok.stat().st_size // 2) > T + 2:
                    break
                trace_bytes = trace_bytes + b"\n" + trace_bytes
                trace_bin.write_bytes(trace_bytes)
            train_input = trace_tok
        cryst_ck = od / "crystallized.ck"
        cryst_sp = od / "crystallized.sp"
        new_L = base_L + max(1, int(add_layers))
        subprocess.run([ollm, "growpro", str(src_ck), str(cryst_ck), str(new_L), str(int(cryst_steps)), "2000"], check=True, capture_output=True)
        subprocess.run([ollm, "trainf", str(cryst_ck), str(train_input), str(int(cryst_steps)), str(max(1, int(cryst_steps) // 2)), str(base_L)], check=True, capture_output=True)
        subprocess.run([ollm, "spext", str(src_ck), str(cryst_ck), str(cryst_sp)], check=True, capture_output=True)
        ph = read_sp_pack_hash(cryst_sp)
        sha, sz = ow_bridge.file_sha256_and_bytes(cryst_sp)
        sp_info = {
            "sp_path": str(cryst_sp),
            "pack_hash_hex": f"{ph:016x}",
            "sha256": sha,
            "bytes": sz,
        }

    receipt = {
        "kind": "omega-helix-green-receipt",
        "schema": 1,
        "verifier": verifier,
        "passed": passed,
        "attempts": attempts,
        "crystallized": sp_info,
        "winning_text": winning_text if passed else "",
    }
    rcpt_path = od / "green-receipt.json"
    tmp = od / ".green-receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, rcpt_path)
    receipt["receipt_path"] = str(rcpt_path)
    return receipt
