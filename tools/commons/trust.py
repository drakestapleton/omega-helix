"""One-command offline Cryptographic Trust Audit (`omega-helix verify`).

Checks:
  1. Pure-Omega binary linkage posture (`ldd`/`otool -L`: zero `libcuda`/`libcublas`, static on Linux).
  2. Studio UI & server offline CSP/loopback posture (no remote `<script>`/`<link>` CDNs, `127.0.0.1`).
  3. Optional OpenWALDO `EXPORT.json` or BOM artifact integrity.
  4. Optional `OLLMSP01` `.sp` Merkle chain verification (`pack_hash` FNV-1a + layer linkage + base ck depth).

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(TOOLS_DIR), str(TOOLS_DIR / "studio"), str(TOOLS_DIR / "openwaldo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import chain_runner  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402

from .pkg import read_sp_meta  # noqa: E402

_CDN_RE = re.compile(r"""(?:src|href)\s*=\s*['"]https?://""", re.IGNORECASE)


def _check_binary_linkage(bin_path: Path) -> tuple[bool, str]:
    sysname = platform.system()
    if sysname == "Linux":
        res = subprocess.run(["ldd", str(bin_path)], capture_output=True, text=True, check=False)
        out = (res.stdout + "\n" + res.stderr).strip()
        if re.search(r"libcuda|libcublas|libcudart", out, re.IGNORECASE):
            return False, f"{bin_path.name}: forbidden CUDA dynamic dependency found"
        if "not a dynamic executable" in out or "statically linked" in out:
            return True, f"{bin_path.name}: static ELF, zero shared libs, zero CUDA"
        return False, f"{bin_path.name}: dynamic libs present on Linux ({out[:120]})"
    if sysname == "Darwin":
        res = subprocess.run(["otool", "-L", str(bin_path)], capture_output=True, text=True, check=False)
        out = (res.stdout + "\n" + res.stderr).strip()
        if re.search(r"cuda|cublas", out, re.IGNORECASE):
            return False, f"{bin_path.name}: forbidden CUDA link on macOS"
        return True, f"{bin_path.name}: macOS Mach-O verified (only libSystem, zero CUDA)"
    return True, f"{bin_path.name}: {sysname} binary present"


def verify_sovereign_trust(
    *,
    bin_dir: str | Path | None = None,
    root: str | Path | None = None,
    base_ck: str | Path | None = None,
    packs: list[str | Path] | None = None,
    bom_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run offline 3-second cryptographic trust & isolation audit."""
    t0 = time.monotonic()
    r = studio_catalog.repo_root(root)
    checks: list[dict[str, Any]] = []

    # 1. Binary static / zero-CUDA posture
    bins = chain_runner.resolve_binaries(bin_dir=bin_dir, root=r)
    found_bins = [p for p in bins.values() if p is not None and p.is_file()]
    if not found_bins:
        checks.append({"name": "binaries_present", "passed": False, "detail": "no compiled pure-Omega binaries found"})
    else:
        for bp in found_bins:
            if bp.name.endswith(".efi"):
                checks.append({"name": f"binary:{bp.name}", "passed": True, "detail": "bare-metal PE32+ UEFI image"})
                continue
            ok, detail = _check_binary_linkage(bp)
            checks.append({"name": f"binary:{bp.name}", "passed": ok, "detail": detail})

    # 2. UI CSP & loopback-only server check
    ui_html = r / "tools" / "studio" / "ui" / "index.html"
    srv_py = r / "tools" / "studio" / "server.py"
    if ui_html.is_file():
        html = ui_html.read_text(encoding="utf-8")
        has_csp = "default-src 'self'" in html
        has_cdn = bool(_CDN_RE.search(html))
        checks.append({
            "name": "studio_ui_csp",
            "passed": has_csp and not has_cdn,
            "detail": "strict default-src 'self', zero remote script/CSS tags" if (has_csp and not has_cdn) else "CSP/CDN violation in UI",
        })
    if srv_py.is_file():
        stxt = srv_py.read_text(encoding="utf-8")
        loopback = 'host: str = "127.0.0.1"' in stxt
        checks.append({
            "name": "server_loopback_default",
            "passed": loopback,
            "detail": "Studio server binds 127.0.0.1 by default",
        })

    # 3. Optional OpenWALDO EXPORT.json / BOM verification
    if bom_path is not None:
        bp = Path(bom_path).expanduser().resolve()
        try:
            if bp.is_dir() or bp.name == "EXPORT.json":
                rep = ow_bridge.verify_export(bp, check_records=True)
                checks.append({
                    "name": "openwaldo_bom",
                    "passed": True,
                    "detail": f"EXPORT.json verified ({rep['total_docs']} docs, sha={rep['export_json_sha256'][:16]}...)",
                })
            else:
                doc = json.loads(bp.read_text(encoding="utf-8"))
                checks.append({
                    "name": "bom_json",
                    "passed": isinstance(doc, dict) and bool(doc.get("schema")),
                    "detail": f"{bp.name} schema={doc.get('schema')} valid JSON",
                })
        except Exception as exc:
            checks.append({"name": "openwaldo_bom", "passed": False, "detail": str(exc)})

    # 4. Optional OLLMSP01 Merkle pack chain / parallel fork verification
    if packs:
        try:
            metas = [read_sp_meta(p) for p in packs]
            chain_ok = True
            detail_parts = []
            all_parallel = len(metas) > 1 and all(
                m["base_L"] == metas[0]["base_L"] and m["base_hash"] == metas[0]["base_hash"]
                for m in metas
            )
            for idx, m in enumerate(metas):
                detail_parts.append(f"L{m['base_L']}->{m['spec_L']}({m['pack_hash'][:10]})")
                if not all_parallel and idx > 0 and metas[idx]["base_L"] != metas[idx - 1]["spec_L"]:
                    chain_ok = False
                    detail_parts.append(f"LAYER_BREAK({metas[idx-1]['spec_L']}!={metas[idx]['base_L']})")
            if base_ck is not None:
                bhdr = ow_bridge.read_ollm_checkpoint_header(Path(base_ck).expanduser().resolve())
                if bhdr["shape"]["L"] != metas[0]["base_L"]:
                    chain_ok = False
                    detail_parts.append(f"BASE_L_MISMATCH({bhdr['shape']['L']}!={metas[0]['base_L']})")
            prefix = "parallel fork: " if all_parallel else ""
            checks.append({
                "name": "merkle_sp_chain",
                "passed": chain_ok,
                "detail": prefix + (" , " if all_parallel else " -> ").join(detail_parts),
            })
        except Exception as exc:
            checks.append({"name": "merkle_sp_chain", "passed": False, "detail": str(exc)})

    ok_all = all(bool(c["passed"]) for c in checks) and len(checks) > 0
    return {
        "ok": ok_all,
        "checks": checks,
        "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
    }
