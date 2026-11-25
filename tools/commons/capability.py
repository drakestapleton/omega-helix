"""4-Wall Capability-Bounded Local Agency + Snapshot/Rollback.

Enforces explicit, revocable capability leases (`allowed_dirs`, `allowed_cmds`,
`allow_network=False`, TTL expiration) around local agent subprocesses, strips outbound
proxy/network hints when offline, and provides instant directory snapshot & rollback.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


def issue_lease(
    lease_path: str | Path,
    *,
    allowed_dirs: list[str | Path],
    allowed_cmds: list[str],
    allow_network: bool = False,
    ttl_seconds: int = 3600,
) -> dict[str, Any]:
    now = time.time()
    doc = {
        "kind": "omega-helix-capability-lease",
        "schema": 1,
        "lease_id": uuid.uuid4().hex,
        "issued_at": now,
        "expires_at": now + max(1, int(ttl_seconds)),
        "revoked": False,
        "allowed_dirs": [str(Path(d).expanduser().resolve()) for d in allowed_dirs],
        "allowed_cmds": [str(c) for c in allowed_cmds],
        "allow_network": bool(allow_network),
    }
    lp = Path(lease_path).expanduser().resolve()
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_name("." + lp.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, lp)
    return doc


def revoke_lease(lease_path: str | Path) -> dict[str, Any]:
    lp = Path(lease_path).expanduser().resolve()
    doc = json.loads(lp.read_text(encoding="utf-8"))
    doc["revoked"] = True
    doc["revoked_at"] = time.time()
    tmp = lp.with_name("." + lp.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, lp)
    return doc


def snapshot_dir(target_dir: str | Path, snap_dir: str | Path) -> dict[str, Any]:
    src = Path(target_dir).expanduser().resolve()
    dst = Path(snap_dir).expanduser().resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    count = sum(1 for _ in dst.rglob("*") if _.is_file())
    return {"target_dir": str(src), "snap_dir": str(dst), "files": count}


def rollback_dir(target_dir: str | Path, snap_dir: str | Path) -> dict[str, Any]:
    tgt = Path(target_dir).expanduser().resolve()
    snap = Path(snap_dir).expanduser().resolve()
    if not snap.is_dir():
        raise FileNotFoundError(f"missing snapshot directory: {snap}")
    if tgt.exists():
        shutil.rmtree(tgt)
    shutil.copytree(snap, tgt)
    count = sum(1 for _ in tgt.rglob("*") if _.is_file())
    return {"restored_dir": str(tgt), "snap_dir": str(snap), "files": count}


def execute_under_lease(
    lease_path: str | Path,
    argv: list[str],
    cwd: str | Path,
    *,
    audit_log_path: str | Path | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Validate lease boundaries fail-closed before executing any command."""
    if not argv:
        raise ValueError("argv must be non-empty")
    lp = Path(lease_path).expanduser().resolve()
    lease = json.loads(lp.read_text(encoding="utf-8"))
    if lease.get("revoked"):
        raise PermissionError("capability lease is revoked")
    if time.time() > float(lease.get("expires_at", 0.0)):
        raise PermissionError("capability lease has expired")

    work = Path(cwd).expanduser().resolve()
    allowed_dirs = [Path(d).resolve() for d in lease.get("allowed_dirs") or []]
    in_allowed = False
    for ad in allowed_dirs:
        try:
            work.relative_to(ad)
            in_allowed = True
            break
        except ValueError:
            continue
    if not in_allowed:
        raise PermissionError(f"cwd {work} outside allowed_dirs {[str(d) for d in allowed_dirs]}")

    cmd0 = Path(argv[0]).name
    allowed_cmds = set(lease.get("allowed_cmds") or [])
    if argv[0] not in allowed_cmds and cmd0 not in allowed_cmds:
        raise PermissionError(f"command {argv[0]!r} not permitted by capability lease")

    env = dict(os.environ)
    if not lease.get("allow_network"):
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(k, None)
        env["OMEGA_OFFLINE"] = "1"

    res = subprocess.run(argv, cwd=str(work), env=env, capture_output=True, text=True, timeout=timeout, check=False)
    rec = {
        "lease_id": lease["lease_id"],
        "ts": time.time(),
        "argv": argv,
        "cwd": str(work),
        "returncode": res.returncode,
        "stdout_sha256": hashlib.sha256(res.stdout.encode("utf-8", errors="replace")).hexdigest(),
    }
    if audit_log_path:
        ap = Path(audit_log_path).expanduser().resolve()
        ap.parent.mkdir(parents=True, exist_ok=True)
        prev = ""
        if ap.is_file():
            lines = [ln for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                prev = json.loads(lines[-1]).get("entry_sha256", "")
        rec["prev_sha256"] = prev
        rec["entry_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()
        with ap.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return {**rec, "stdout": res.stdout, "stderr": res.stderr}
