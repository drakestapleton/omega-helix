"""DNA Commons / Cognitive Package Manager for signed OLLMSP01 `.sp` Merkle packs.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(TOOLS_DIR), str(TOOLS_DIR / "openwaldo"), str(TOOLS_DIR / "distill")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openwaldo_bridge as ow_bridge  # noqa: E402
from distill.harvest import fnv1a64  # noqa: E402


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_sp_meta(sp_path: str | Path) -> dict[str, Any]:
    """Parse and cryptographically verify an OLLMSP01 specialist pack header + payload FNV-1a."""
    p = Path(sp_path).expanduser().resolve()
    raw = p.read_bytes()
    if len(raw) < 48 or raw[:8] != b"OLLMSP01":
        raise ValueError(f"{p}: invalid OLLMSP01 magic/header")
    base_l, spec_l, base_hash, n_spec, pack_hash = struct.unpack("<5Q", raw[8:48])
    want_sz = 48 + 8 * n_spec
    if len(raw) != want_sz:
        raise ValueError(f"{p}: payload length mismatch (got {len(raw)}, expected {want_sz})")
    got_fnv = fnv1a64(raw[48:])
    if got_fnv != pack_hash:
        raise ValueError(f"{p}: OLLMSP01 payload FNV-1a mismatch (got {got_fnv:016x}, expected {pack_hash:016x})")
    sha, size = ow_bridge.file_sha256_and_bytes(p)
    return {
        "magic": "OLLMSP01",
        "path": str(p),
        "sha256": sha,
        "bytes": size,
        "base_L": base_l,
        "spec_L": spec_l,
        "base_hash": f"{base_hash:016x}",
        "n_spec_params": n_spec,
        "pack_hash": f"{pack_hash:016x}",
    }


def _load_registry(registry_dir: Path) -> dict[str, Any]:
    idx_path = registry_dir / "registry.json"
    if idx_path.is_file():
        return json.loads(idx_path.read_text(encoding="utf-8"))
    return {"kind": "omega-helix-dna-commons", "schema": 1, "packages": {}}


def _write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def publish_pack(
    sp_path: str | Path,
    name: str,
    *,
    registry_dir: str | Path,
    version: str = "0.1.0",
    license_id: str = "Apache-2.0",
    description: str = "",
) -> dict[str, Any]:
    """Publish a verified OLLMSP01 `.sp` pack into a content-addressed DNA Commons registry."""
    meta = read_sp_meta(sp_path)
    reg_dir = Path(registry_dir).expanduser().resolve()
    blobs_dir = reg_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blobs_dir / f"{meta['sha256']}.sp"
    shutil.copy2(meta["path"], blob_path)

    reg = _load_registry(reg_dir)
    entry = {
        "name": name,
        "version": version,
        "license": license_id,
        "description": description,
        "sha256": meta["sha256"],
        "bytes": meta["bytes"],
        "base_L": meta["base_L"],
        "spec_L": meta["spec_L"],
        "base_hash": meta["base_hash"],
        "n_spec_params": meta["n_spec_params"],
        "pack_hash": meta["pack_hash"],
        "blob_rel": f"blobs/{meta['sha256']}.sp",
        "published_at": _utc_now(),
    }
    reg["packages"][name] = entry
    _write_json_atomic(reg_dir / "registry.json", reg)
    return entry


def search_packs(
    registry_dir: str | Path,
    *,
    query: str = "",
    base_hash: str = "",
) -> list[dict[str, Any]]:
    """Search packages in a local DNA Commons registry by keyword and/or base_hash pin."""
    reg_dir = Path(registry_dir).expanduser().resolve()
    reg = _load_registry(reg_dir)
    q = query.strip().lower()
    bh = base_hash.strip().lower()
    out: list[dict[str, Any]] = []
    for entry in reg.get("packages", {}).values():
        if bh and str(entry.get("base_hash", "")).lower() != bh:
            continue
        if q:
            hay = f"{entry.get('name', '')} {entry.get('description', '')} {entry.get('license', '')}".lower()
            if q not in hay:
                continue
        out.append(entry)
    out.sort(key=lambda x: str(x.get("name", "")))
    return out


def install_pack(
    name: str,
    env_dir: str | Path,
    *,
    registry_dir: str | Path,
    expected_base_hash: str | None = None,
) -> dict[str, Any]:
    """Install a `.sp` pack from the Commons after fail-closed SHA-256, FNV-1a, and base_hash checks."""
    reg_dir = Path(registry_dir).expanduser().resolve()
    reg = _load_registry(reg_dir)
    entry = reg.get("packages", {}).get(name)
    if not entry:
        raise ValueError(f"package {name!r} not found in registry {reg_dir}")

    blob_path = reg_dir / entry["blob_rel"]
    if not blob_path.is_file():
        raise FileNotFoundError(f"missing registry blob {blob_path}")
    meta = read_sp_meta(blob_path)
    if meta["sha256"] != entry["sha256"] or meta["pack_hash"] != entry["pack_hash"]:
        raise ValueError(f"{name}: blob digest mismatch against registry entry")
    if expected_base_hash is not None and meta["base_hash"].lower() != expected_base_hash.strip().lower():
        raise ValueError(
            f"{name}: base_hash mismatch (pack pinned to {meta['base_hash']}, expected {expected_base_hash})"
        )

    env = Path(env_dir).expanduser().resolve()
    packs_dir = env / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    dest = packs_dir / f"{name}.sp"
    shutil.copy2(blob_path, dest)

    mpath = env / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {"installed": {}}
    manifest["installed"][name] = {**entry, "installed_path": str(dest), "installed_at": _utc_now()}
    _write_json_atomic(mpath, manifest)
    return manifest["installed"][name]


def revoke_pack(
    name: str,
    env_dir: str | Path,
    *,
    reason: str = "right-to-be-forgotten",
) -> dict[str, Any]:
    """Delete an installed `.sp` pack and append an immutable cryptographic revocation record."""
    env = Path(env_dir).expanduser().resolve()
    mpath = env / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {"installed": {}}
    removed = manifest.get("installed", {}).pop(name, None)
    dest = env / "packs" / f"{name}.sp"
    file_deleted = False
    if dest.is_file():
        dest.unlink()
        file_deleted = True
    _write_json_atomic(mpath, manifest)

    rev_rec = {
        "name": name,
        "revoked_at": _utc_now(),
        "reason": reason,
        "removed_entry": removed,
        "file_deleted": file_deleted,
    }
    rev_path = env / "revocations.jsonl"
    rev_path.parent.mkdir(parents=True, exist_ok=True)
    with rev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rev_rec) + "\n")
    return rev_rec


def diff_packs(sp_a: str | Path, sp_b: str | Path) -> dict[str, Any]:
    """Compare two OLLMSP01 specialist packs' Merkle roots, shapes, and hashes."""
    a = read_sp_meta(sp_a)
    b = read_sp_meta(sp_b)
    return {
        "same_base_hash": a["base_hash"] == b["base_hash"],
        "same_pack": a["sha256"] == b["sha256"],
        "same_base_L": a["base_L"] == b["base_L"],
        "same_spec_L": a["spec_L"] == b["spec_L"],
        "a": a,
        "b": b,
    }
