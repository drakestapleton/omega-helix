"""Anti-DNA Runtime Firewalls, Strand Disagreement (JSD -> LTP0), and Per-Token Receipts.

Wraps pure-Omega `ollm spclip` (negative contrastive steering weights in MODE=0) and
`ollm spguard` (per-token JSD disagreement -> LTP0 extra passes + FNV-1a-64 receipt chain).

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import hashlib
import os
import struct
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


def _mix64(h: int, x: int) -> int:
    for b in struct.pack("<Q", x & 0xFFFFFFFFFFFFFFFF):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def read_sp_pack_hash(sp_path: str | Path) -> int:
    """Read `pack_hash` (bytes 40..48 LE u64) from a 48-byte OLLMSP01 header."""
    p = Path(sp_path).expanduser().resolve()
    raw = p.read_bytes()
    if len(raw) < 48 or raw[:8] != b"OLLMSP01":
        raise ValueError(f"{p}: invalid OLLMSP01 specialist pack")
    return struct.unpack("<Q", raw[40:48])[0]


def forge_anti_strand(
    ollm_bin: str | Path,
    base_ck: str | Path,
    avoid_text: str | bytes,
    out_dir: str | Path,
    *,
    name: str = "anti",
    add_layers: int = 1,
    steps: int = 6,
    lr_micro: int = 2000,
) -> dict[str, Any]:
    """Train a compact OLLMSP01 Anti-DNA pack on undesired patterns for negative steering."""
    ollm = str(Path(ollm_bin).expanduser().resolve())
    bck = Path(base_ck).expanduser().resolve()
    hdr = ow_bridge.read_ollm_checkpoint_header(bck)
    base_L = int(hdr["shape"]["L"])
    T = int(hdr["shape"]["T"])
    new_L = base_L + max(1, int(add_layers))

    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)
    raw_bytes = avoid_text.encode("utf-8", errors="replace") if isinstance(avoid_text, str) else bytes(avoid_text)
    if not raw_bytes:
        raise ValueError("avoid_text must be non-empty")
    # Ensure corpus meets minimum > T tokens for trainf
    while len(raw_bytes) <= T + 8:
        raw_bytes = raw_bytes + b"\n" + raw_bytes

    data_bin = od / f"{name}.bin"
    data_bin.write_bytes(raw_bytes)
    spec_ck = od / f"{name}.ck"
    sp_path = od / f"{name}.sp"

    r1 = subprocess.run([ollm, "growpro", str(bck), str(spec_ck), str(new_L), str(int(steps)), str(int(lr_micro))], capture_output=True, text=True, check=False)
    if r1.returncode != 0:
        raise RuntimeError(f"growpro failed (rc={r1.returncode}): {r1.stderr.strip()}")
    every = max(1, int(steps) // 2)
    r2 = subprocess.run([ollm, "trainf", str(spec_ck), str(data_bin), str(int(steps)), str(every), str(base_L)], capture_output=True, text=True, check=False)
    if r2.returncode != 0:
        raise RuntimeError(f"trainf failed (rc={r2.returncode}): {r2.stderr.strip()}")
    r3 = subprocess.run([ollm, "spext", str(bck), str(spec_ck), str(sp_path)], capture_output=True, text=True, check=False)
    if r3.returncode != 0:
        raise RuntimeError(f"spext failed (rc={r3.returncode}): {r3.stderr.strip()}")

    pack_hash = read_sp_pack_hash(sp_path)
    sha, size = ow_bridge.file_sha256_and_bytes(sp_path)
    return {
        "name": name,
        "sp_path": str(sp_path),
        "spec_ck": str(spec_ck),
        "pack_hash": pack_hash,
        "pack_hash_hex": f"{pack_hash:016x}",
        "sha256": sha,
        "bytes": size,
        "base_L": base_L,
        "spec_L": new_L,
        "avoid_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def run_firewall_decode(
    ollm_bin: str | Path,
    base_ck: str | Path,
    strands: list[dict[str, Any]],
    *,
    prompt: str = "Omega ",
    n_tokens: int = 12,
    temp_milli: int = 0,
    seed: int = 42,
    disagree_milli: int = 50,
    max_extra: int = 2,
    merges: str | Path = "-",
) -> dict[str, Any]:
    """Execute `ollm spguard` with signed strand weights and parse cryptographic receipts."""
    if not (2 <= len(strands) <= 4):
        raise ValueError("spguard requires 2..4 strands")
    ollm = str(Path(ollm_bin).expanduser().resolve())
    bck = str(Path(base_ck).expanduser().resolve())

    argv: list[str] = [
        ollm, "spguard", bck,
        str(int(disagree_milli)), str(int(max_extra)),
        prompt, str(int(n_tokens)), str(int(temp_milli)), str(int(seed)),
        str(merges), str(len(strands)),
    ]
    pack_hash_chains: list[list[int]] = []
    signed_weights: list[int] = []
    for st in strands:
        wm = int(st["weight_milli"])
        packs = [str(Path(p).expanduser().resolve()) for p in st.get("packs") or []]
        if not packs:
            raise ValueError("each strand requires at least one pack path")
        signed_weights.append(wm)
        argv.extend([str(wm), str(len(packs)), *packs])
        pack_hash_chains.append([read_sp_pack_hash(p) for p in packs])

    res = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ollm spguard failed (rc={res.returncode}): {res.stderr.strip()}")

    records: list[dict[str, Any]] = []
    toks: list[int] = []
    summary: dict[str, Any] = {}
    for ln in res.stdout.splitlines():
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "G" and len(parts) == 8:
            step = int(parts[1])
            tok = int(parts[2])
            toks.append(tok)
            records.append({
                "step": step,
                "token": tok,
                "jsd_milli": int(parts[3]),
                "extra_passes": int(parts[4]),
                "contested": int(parts[5]),
                "anti_active": int(parts[6]),
                "receipt": parts[7],
            })
        elif parts[0] == "GR" and len(parts) == 7:
            summary = {
                "drawn": int(parts[1]),
                "contested_count": int(parts[2]),
                "total_extra_passes": int(parts[3]),
                "anti_active": int(parts[4]),
                "base_hash_hex": parts[5],
                "final_receipt_hex": parts[6],
            }

    if not summary:
        raise RuntimeError(f"spguard did not emit GR summary line: {res.stdout!r}")

    text = bytes(t & 0xFF for t in toks).decode("utf-8", errors="replace")
    return {
        "prompt": prompt,
        "text": text,
        "tokens": toks,
        "records": records,
        "summary": summary,
        "signed_weights": signed_weights,
        "pack_hashes": pack_hash_chains,
    }


def verify_firewall_receipt(
    result: dict[str, Any],
    *,
    base_hash_hex: str | None = None,
    pack_hashes: list[list[int]] | None = None,
    signed_weights: list[int] | None = None,
) -> dict[str, Any]:
    """Independently recompute and verify the full FNV-1a-64 spguard receipt chain."""
    summary = result.get("summary") or {}
    bh = base_hash_hex or summary.get("base_hash_hex") or ""
    if not bh:
        return {"verified": False, "error": "missing base_hash_hex"}
    weights = list(signed_weights if signed_weights is not None else result.get("signed_weights", []))
    phashes = list(pack_hashes if pack_hashes is not None else result.get("pack_hashes", []))
    if len(weights) != len(phashes) or not weights:
        return {"verified": False, "error": "weights and pack_hashes length mismatch"}

    h = 14695981039346656037
    h = _mix64(h, int(bh, 16))
    for w, plist in zip(weights, phashes):
        h = _mix64(h, w)
        ph = 14695981039346656037
        for pval in plist:
            ph = _mix64(ph, int(pval))
        h = _mix64(h, ph)

    mismatches: list[int] = []
    for rec in result.get("records") or []:
        for val in (
            int(rec["step"]),
            int(rec["token"]),
            int(rec["jsd_milli"]),
            int(rec["extra_passes"]),
            int(rec["contested"]),
            int(rec["anti_active"]),
        ):
            h = _mix64(h, val)
        if f"{h:016x}" != str(rec.get("receipt", "")).lower():
            mismatches.append(int(rec["step"]))

    expected_final = f"{h:016x}"
    got_final = str(summary.get("final_receipt_hex", "")).lower()
    ok = (len(mismatches) == 0) and (expected_final == got_final)
    return {
        "verified": ok,
        "expected_final": expected_final,
        "got_final": got_final,
        "token_mismatches": mismatches,
        "error": "" if ok else "FNV-1a firewall receipt chain mismatch",
    }
