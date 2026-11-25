"""Verifier-Gated Sleep Cycles for Continual Memory Without Catastrophic Forgetting.

Pipeline:
  1. Filter daily session interactions through VerifierPipeline (only verified interactions pass).
  2. Evaluate `base_ck` on held-out regression data (`ollm eval`).
  3. Consolidate into a daily candidate layer (`growpro` + `fcache` + `trainfc` + `spext`) -> `day-<date>.sp`.
  4. Re-evaluate the grown checkpoint on held-out regression data.
  5. Fail-closed gate: accept iff `eval_after - eval_before <= max_regression_nats`; else reject & quarantine.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
STUDIO_DIR = TOOLS_DIR / "studio"
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
for _p in (str(TOOLS_DIR), str(STUDIO_DIR), str(OPENWALDO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openwaldo_bridge as ow_bridge  # noqa: E402
from distill.verifiers import VerifierPipeline  # noqa: E402

from .anti_dna import read_sp_pack_hash  # noqa: E402


def _run_eval(ollm: str, ck: Path, held_bin: Path, windows: int) -> float:
    r = subprocess.run([ollm, "eval", str(ck), str(held_bin), str(windows)], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ollm eval failed (rc={r.returncode}): {r.stderr.strip()}")
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 5 and parts[0] == "E":
            return int(parts[3]) / 1_000_000.0
    raise RuntimeError(f"unexpected eval stdout: {r.stdout!r}")


def run_sleep_cycle(
    ollm_bin: str | Path,
    base_ck: str | Path,
    session_interactions: list[str | dict[str, Any]],
    held_regression_bin: str | Path,
    out_dir: str | Path,
    *,
    verifiers: list[str] | tuple[str, ...] = ("nonempty",),
    max_regression_nats: float = 0.25,
    steps: int = 8,
    add_layers: int = 1,
    lr_micro: int = 2000,
    date_tag: str | None = None,
) -> dict[str, Any]:
    """Consolidate verifier-passing session traces into a regression-gated daily `.sp` link."""
    ollm = str(Path(ollm_bin).expanduser().resolve())
    bck = Path(base_ck).expanduser().resolve()
    held = Path(held_regression_bin).expanduser().resolve()
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    tag = date_tag or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    hdr = ow_bridge.read_ollm_checkpoint_header(bck)
    base_L = int(hdr["shape"]["L"])
    T = int(hdr["shape"]["T"])
    new_L = base_L + max(1, int(add_layers))

    recs: list[dict[str, Any]] = []
    for item in session_interactions:
        if isinstance(item, dict):
            recs.append({"text": str(item.get("assistant_text") or item.get("text") or "")})
        else:
            recs.append({"text": str(item)})

    vpipe = VerifierPipeline(list(verifiers))
    vres = vpipe.filter_records(recs)
    accepted_recs = vres["accepted"]
    if not accepted_recs:
        raise RuntimeError("sleep cycle rejected all session interactions; zero verified memories to consolidate")

    corpus_bytes = "\n".join(r["text"] for r in accepted_recs).encode("utf-8", errors="replace")
    while len(corpus_bytes) <= T + 8:
        corpus_bytes = corpus_bytes + b"\n" + corpus_bytes
    mem_bin = od / f"memory-{tag}.bin"
    mem_bin.write_bytes(corpus_bytes)

    held_bytes = held.stat().st_size
    if held_bytes <= T:
        raise ValueError(f"held regression corpus {held} shorter than T={T}")
    eval_w = max(1, min(8, (held_bytes - 1) // T))
    eval_before = _run_eval(ollm, bck, held, eval_w)

    cand_ck = od / f"day-{tag}.ck"
    cand_hc = od / f"day-{tag}.hc"
    cand_sp = od / f"day-{tag}.sp"

    r_grow = subprocess.run([ollm, "growpro", str(bck), str(cand_ck), str(new_L), str(int(steps)), str(int(lr_micro))], capture_output=True, text=True, check=False)
    if r_grow.returncode != 0:
        raise RuntimeError(f"growpro failed (rc={r_grow.returncode}): {r_grow.stderr.strip()}")

    fc_w = max(1, min(16, (len(corpus_bytes) - 1) // T))
    r_fc = subprocess.run([ollm, "fcache", str(bck), str(mem_bin), str(fc_w), str(cand_hc)], capture_output=True, text=True, check=False)
    if r_fc.returncode != 0:
        raise RuntimeError(f"fcache failed (rc={r_fc.returncode}): {r_fc.stderr.strip()}")

    every = max(1, int(steps) // 2)
    r_tfc = subprocess.run([ollm, "trainfc", str(cand_ck), str(cand_hc), str(int(steps)), str(every), str(base_L)], capture_output=True, text=True, check=False)
    if r_tfc.returncode != 0:
        raise RuntimeError(f"trainfc failed (rc={r_tfc.returncode}): {r_tfc.stderr.strip()}")

    r_sp = subprocess.run([ollm, "spext", str(bck), str(cand_ck), str(cand_sp)], capture_output=True, text=True, check=False)
    if r_sp.returncode != 0:
        raise RuntimeError(f"spext failed (rc={r_sp.returncode}): {r_sp.stderr.strip()}")

    eval_after = _run_eval(ollm, cand_ck, held, eval_w)
    delta = eval_after - eval_before
    gate_ok = delta <= float(max_regression_nats)

    pack_hash = read_sp_pack_hash(cand_sp)
    sp_sha, sp_bytes = ow_bridge.file_sha256_and_bytes(cand_sp)
    final_sp_path = cand_sp
    if not gate_ok:
        qdir = od / "quarantined"
        qdir.mkdir(parents=True, exist_ok=True)
        qdest = qdir / cand_sp.name
        shutil.move(str(cand_sp), str(qdest))
        final_sp_path = qdest

    receipt = {
        "kind": "omega-helix-sleep-receipt",
        "schema": 1,
        "date_tag": tag,
        "base_ck": str(bck),
        "base_sha256": hdr["sha256"],
        "verify_stats": vres["stats"],
        "memory_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "eval_before_nats": eval_before,
        "eval_after_nats": eval_after,
        "regression_delta_nats": round(delta, 6),
        "max_regression_nats": float(max_regression_nats),
        "accepted": gate_ok,
        "sp_path": str(final_sp_path) if gate_ok else "",
        "quarantined_path": str(final_sp_path) if not gate_ok else "",
        "pack_hash_hex": f"{pack_hash:016x}",
        "pack_sha256": sp_sha,
        "pack_bytes": sp_bytes,
    }
    rcpt_path = od / "sleep-receipt.json"
    tmp = od / ".sleep-receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, rcpt_path)
    receipt["receipt_path"] = str(rcpt_path)
    return receipt
