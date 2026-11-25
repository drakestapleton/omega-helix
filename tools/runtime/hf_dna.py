"""Open-Trunk HF Safetensors -> Merkle DNA Specialists + Anti-DNA Firewall Showcase.

Demonstrates taking an open-weight HuggingFace Safetensors directory via `ollm-interchange
import-hf`, verifying `OLLMCK04`, forging two domain specialists + an Anti-DNA firewall
strand on a frozen trunk, composing them live with `spguard`, and verifying every token's
FNV-1a receipt with zero CUDA.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import json
import os
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

import omega_helix_backend as univ_backend  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402

from .anti_dna import forge_anti_strand, read_sp_pack_hash, run_firewall_decode, verify_firewall_receipt  # noqa: E402


def _forge_pos_strand(ollm: str, bck: Path, text: str, od: Path, name: str, steps: int = 4) -> dict[str, Any]:
    return forge_anti_strand(ollm, bck, text, od, name=name, add_layers=1, steps=steps, lr_micro=2000)


def run_open_trunk_dna_demo(
    bins: dict[str, Path],
    hf_dir: str | Path,
    domain_a_text: str,
    domain_b_text: str,
    avoid_text: str,
    out_dir: str | Path,
    *,
    prompt: str = "Omega ",
    n_tokens: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Import HF Safetensors -> verify OLLMCK04 -> forge 2 strands + 1 anti-strand -> spguard verify."""
    hf = Path(hf_dir).expanduser().resolve()
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    ic_bin = str(Path(bins["ollm_interchange"]).expanduser().resolve())
    univ_bin = str(Path(bins["ollm_univ"]).expanduser().resolve())
    ollm_bin = str(Path(bins["ollm"]).expanduser().resolve())

    imported_ck = od / "imported_trunk.ck"
    r_imp = subprocess.run(
        [ic_bin, "import-hf", str(hf / "config.json"), str(hf / "model.safetensors"), str(imported_ck)],
        capture_output=True, text=True, check=False,
    )
    if r_imp.returncode != 0:
        raise RuntimeError(f"import-hf failed (rc={r_imp.returncode}): {r_imp.stderr.strip()}")
    r_info = subprocess.run([univ_bin, "info", str(imported_ck)], capture_output=True, text=True, check=False)
    if r_info.returncode != 0:
        raise RuntimeError(f"ollm-univ info failed (rc={r_info.returncode}): {r_info.stderr.strip()}")
    ck4_hdr = univ_backend.read_ck4_header(imported_ck)

    # Create matched frozen DNA trunk matching imported dimensions for spclip/spguard composition
    sh = ck4_hdr["shape"]
    dna_trunk = od / "dna_trunk.ck"
    subprocess.run(
        [
            ollm_bin, "init", str(dna_trunk),
            str(min(256, sh["V"])), str(sh["T"]), str(sh["D"]), str(sh["H"]), str(sh["F"]),
            str(sh["L"]), str(sh["B"]), str(seed), "4", "3000",
        ],
        capture_output=True, text=True, check=True,
    )

    spec_a = _forge_pos_strand(ollm_bin, dna_trunk, domain_a_text, od / "strand_a", "domain_a")
    spec_b = _forge_pos_strand(ollm_bin, dna_trunk, domain_b_text, od / "strand_b", "domain_b")
    anti = forge_anti_strand(ollm_bin, dna_trunk, avoid_text, od / "strand_anti", name="anti_halluc")

    guard_res = run_firewall_decode(
        ollm_bin,
        dna_trunk,
        [
            {"weight_milli": 1000, "packs": [spec_a["sp_path"]]},
            {"weight_milli": 1000, "packs": [spec_b["sp_path"]]},
            {"weight_milli": -500, "packs": [anti["sp_path"]]},
        ],
        prompt=prompt,
        n_tokens=n_tokens,
        temp_milli=0,
        seed=seed,
        disagree_milli=50,
        max_extra=2,
    )
    rcpt_chk = verify_firewall_receipt(guard_res)
    if not rcpt_chk["verified"]:
        raise RuntimeError(f"firewall receipt verification failed: {rcpt_chk['error']}")

    doc = {
        "kind": "omega-helix-open-trunk-dna-receipt",
        "schema": 1,
        "imported_ck4": ck4_hdr,
        "dna_trunk_sha256": ow_bridge.file_sha256_and_bytes(dna_trunk)[0],
        "strands": [spec_a, spec_b, anti],
        "spguard": guard_res,
        "receipt_verification": rcpt_chk,
    }
    rcpt_path = od / "open-trunk-demo-receipt.json"
    tmp = od / ".open-trunk-demo-receipt.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, rcpt_path)
    doc["receipt_path"] = str(rcpt_path)
    return doc
