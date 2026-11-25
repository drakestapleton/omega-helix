"""Trustless Swarm Forging: Content-Addressed Evidence Receipts & BFT Quorum Tallying.

Independent nodes forge/evaluate specialized `.sp` packs locally and emit self-verifying
JSON evidence bundles linking `(node_id, sp_sha256, base_hash, pack_hash, eval_before, eval_after)`.
Any node can re-verify bundles and enforce a quorum threshold without central trust.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .pkg import read_sp_meta


def _bundle_digest(body: dict[str, Any]) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def create_evidence_bundle(
    sp_path: str | Path,
    *,
    node_id: str,
    eval_before: float,
    eval_after: float,
    verify_stats: dict[str, Any] | None = None,
    out_path: str | Path,
) -> dict[str, Any]:
    if not node_id.strip():
        raise ValueError("node_id must be non-empty")
    meta = read_sp_meta(sp_path)
    body = {
        "kind": "omega-helix-swarm-evidence",
        "schema": 1,
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_id": node_id.strip(),
        "sp_sha256": meta["sha256"],
        "sp_bytes": meta["bytes"],
        "base_hash": meta["base_hash"],
        "pack_hash": meta["pack_hash"],
        "base_L": meta["base_L"],
        "spec_L": meta["spec_L"],
        "eval_before_nats": float(eval_before),
        "eval_after_nats": float(eval_after),
        "verify_stats": dict(verify_stats or {}),
    }
    bundle = {**body, "evidence_sha256": _bundle_digest(body)}
    dest = Path(out_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("." + dest.name + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dest)
    return {**bundle, "path": str(dest)}


def verify_evidence_bundle(bundle_path: str | Path, sp_path: str | Path) -> dict[str, Any]:
    bp = Path(bundle_path).expanduser().resolve()
    doc = json.loads(bp.read_text(encoding="utf-8"))
    meta = read_sp_meta(sp_path)
    claimed = str(doc.get("evidence_sha256", ""))
    body = {k: v for k, v in doc.items() if k != "evidence_sha256"}
    recomputed = _bundle_digest(body)
    matches_sp = (
        doc.get("sp_sha256") == meta["sha256"]
        and doc.get("base_hash") == meta["base_hash"]
        and doc.get("pack_hash") == meta["pack_hash"]
    )
    ok = (claimed == recomputed) and matches_sp
    return {
        "verified": ok,
        "node_id": doc.get("node_id", ""),
        "matches_sp": matches_sp,
        "digest_ok": claimed == recomputed,
    }


def tally_quorum(
    bundle_paths: list[str | Path],
    sp_path: str | Path,
    *,
    min_quorum: int = 3,
) -> dict[str, Any]:
    """Enforce distinct-node BFT quorum agreement on an OLLMSP01 `.sp` pack."""
    valid_nodes: set[str] = set()
    invalid: list[str] = []
    for bp in bundle_paths:
        res = verify_evidence_bundle(bp, sp_path)
        if res["verified"] and res["node_id"]:
            valid_nodes.add(res["node_id"])
        else:
            invalid.append(str(bp))
    accepted = len(valid_nodes) >= int(min_quorum)
    return {
        "quorum_met": accepted,
        "valid_node_count": len(valid_nodes),
        "min_quorum": int(min_quorum),
        "valid_nodes": sorted(valid_nodes),
        "invalid_bundles": invalid,
    }
