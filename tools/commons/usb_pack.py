"""'USB-Stick Sovereignty': Self-Contained Air-Gap Portable Bundle Packager & Verifier.

Exports static pure-Omega binaries, optional UEFI bare-metal images, Studio UI + CLI tools,
frozen trunk checkpoints, signed OLLMSP01 packs, and personal memory into one relocatable
offline directory with `SHA256SUMS.txt`, `AIRGAP-MANIFEST.json`, and `START-OFFLINE.sh`.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(TOOLS_DIR), str(TOOLS_DIR / "studio"), str(TOOLS_DIR / "openwaldo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import chain_runner  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402


def _copy_tree(src: Path, dst: Path) -> list[Path]:
    copied: list[Path] = []
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
        return copied
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".git", ".pytest_cache"))
        rel_dir = Path(dirpath).relative_to(src)
        for fn in sorted(filenames):
            if fn.endswith(".pyc") or fn.startswith("._"):
                continue
            sp = Path(dirpath) / fn
            dp = dst / rel_dir / fn
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)
            copied.append(dp)
    return copied


def pack_usb_bundle(
    out_dir: str | Path,
    *,
    bin_dir: str | Path | None = None,
    root: str | Path | None = None,
    trunk_ck: str | Path | None = None,
    packs: list[str | Path] | None = None,
    memory_store: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble an air-gap portable directory with SHA256SUMS + START-OFFLINE.sh."""
    r = studio_catalog.repo_root(root)
    od = Path(out_dir).expanduser().resolve()
    if od.exists() and any(od.iterdir()):
        # Ensure clean deterministic manifest without stale files
        shutil.rmtree(od)
    od.mkdir(parents=True, exist_ok=True)

    # 1. Copy binaries (including UEFI .efi / Linux / macOS binaries present in resolved dir)
    bins = chain_runner.resolve_binaries(bin_dir=bin_dir, root=r)
    bdest = od / "bin"
    bdest.mkdir(parents=True, exist_ok=True)
    bin_dirs_seen: set[Path] = set()
    for p in bins.values():
        if p is not None and p.is_file():
            shutil.copy2(p, bdest / p.name)
            bin_dirs_seen.add(p.parent)
    for bd in bin_dirs_seen:
        for extra_name in ("ollm-x86linux", "ollm.efi", "ollm-aa64.efi"):
            ep = bd / extra_name
            if ep.is_file() and not (bdest / extra_name).exists():
                shutil.copy2(ep, bdest / extra_name)

    # 2. Copy Studio, Runtime, Distill, Commons, OpenWALDO tools & launcher
    _copy_tree(r / "bin" / "omega-helix", od / "omega-helix-bin" / "omega-helix")
    for sub in ("studio", "openwaldo", "distill", "runtime", "commons"):
        sp = r / "tools" / sub
        if sp.is_dir():
            _copy_tree(sp, od / "tools" / sub)

    # 3. Copy optional trunk checkpoint, .sp packs, and memory store
    if trunk_ck is not None:
        tp = Path(trunk_ck).expanduser().resolve()
        (od / "models").mkdir(exist_ok=True)
        shutil.copy2(tp, od / "models" / tp.name)
    if packs:
        (od / "packs").mkdir(exist_ok=True)
        for sp_p in packs:
            src_sp = Path(sp_p).expanduser().resolve()
            shutil.copy2(src_sp, od / "packs" / src_sp.name)
    if memory_store is not None:
        mp = Path(memory_store).expanduser().resolve()
        (od / "memory").mkdir(exist_ok=True)
        shutil.copy2(mp, od / "memory" / mp.name)

    start_sh = od / "START-OFFLINE.sh"
    start_sh.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "here=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "export OMEGA_HELIX_HOME=\"$here\"\n"
        "export OMEGA_HELIX_BIN_DIR=\"$here/bin\"\n"
        "export OMEGA_OFFLINE=1\n"
        "exec python3 \"$here/tools/studio/cli.py\" \"$@\"\n",
        encoding="utf-8",
    )
    os.chmod(start_sh, 0o755)

    files_meta: list[dict[str, Any]] = []
    sum_lines: list[str] = []
    for fp in sorted(od.rglob("*")):
        if fp.is_file() and fp.name not in ("SHA256SUMS.txt", "AIRGAP-MANIFEST.json"):
            rel = fp.relative_to(od).as_posix()
            sha, sz = ow_bridge.file_sha256_and_bytes(fp)
            files_meta.append({"path": rel, "sha256": sha, "bytes": sz})
            sum_lines.append(f"{sha}  {rel}")

    sums_path = od / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

    manifest = {
        "kind": "omega-helix-airgap-bundle",
        "schema": 1,
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": len(files_meta),
        "total_bytes": sum(f["bytes"] for f in files_meta),
        "files": files_meta,
    }
    mpath = od / "AIRGAP-MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "bundle_dir": str(od), "manifest_path": str(mpath)}


def verify_usb_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Fail-closed offline integrity check for an air-gap USB bundle."""
    bd = Path(bundle_dir).expanduser().resolve()
    mpath = bd / "AIRGAP-MANIFEST.json"
    if not mpath.is_file():
        return {"verified": False, "error": "missing AIRGAP-MANIFEST.json"}
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    if doc.get("kind") != "omega-helix-airgap-bundle" or doc.get("schema") != 1:
        return {"verified": False, "error": "invalid AIRGAP-MANIFEST header"}
    for entry in doc.get("files") or []:
        rel = entry["path"]
        fp = (bd / rel).resolve()
        try:
            fp.relative_to(bd)
        except ValueError:
            return {"verified": False, "error": f"path escapes bundle: {rel}"}
        if not fp.is_file():
            return {"verified": False, "error": f"missing file: {rel}"}
        got_sha, got_sz = ow_bridge.file_sha256_and_bytes(fp)
        if got_sha != entry["sha256"] or got_sz != entry["bytes"]:
            return {"verified": False, "error": f"digest mismatch on {rel}"}
    return {"verified": True, "file_count": len(doc.get("files") or []), "total_bytes": doc.get("total_bytes", 0)}
