"""Community Hardware-Porting Kit: Automated Bit-Exact Parity Gate Runner.

Any contributor porting Omega Helix to Metal, Vulkan, AMD, Intel, or consumer RTX runs
`run_port_parity_check(candidate_bin)` to execute `adamck`, `gcheck`, `kvcmp`, and
`tests/oracle.py` parity gates and produce a signed JSON receipt.

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR / "studio") not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR / "studio"))

import catalog as studio_catalog  # noqa: E402


def run_port_parity_check(
    candidate_bin: str | Path,
    *,
    root: str | Path | None = None,
    out_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run all bit-level self-checks + oracle.py gates against a candidate Omega Helix binary."""
    r = studio_catalog.repo_root(root)
    bin_p = Path(candidate_bin).expanduser().resolve()
    oracle_py = r / "tests" / "oracle.py"
    gates: list[dict[str, Any]] = []

    def _run_gate(name: str, argv: list[str]) -> None:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
        gates.append({
            "gate": name,
            "passed": res.returncode == 0,
            "returncode": res.returncode,
            "stdout_tail": (res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""),
        })

    _run_gate("adamck", [str(bin_p), "adamck"])

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ck = tdp / "tiny.ck"
        bpe_ck = tdp / "bpe.ck"
        merges = tdp / "m300.bin"
        raw = tdp / "raw.txt"
        raw.write_text(
            ("Hardware porting parity gate reference text for Omega Helix.\n<|endoftext|>\n") * 30,
            encoding="utf-8",
        )

        _run_gate("init", [str(bin_p), "init", str(ck), "256", "16", "16", "2", "32", "2", "2", "42", "4", "3000"])
        _run_gate("gcheck", [str(bin_p), "gcheck", str(ck), str(raw)])
        subprocess.run([str(bin_p), "bpefit", str(raw), str(merges), "300"], check=True, capture_output=True)
        subprocess.run([str(bin_p), "init", str(bpe_ck), "300", "16", "16", "2", "32", "2", "4", "1", "20", "5000", "1"], check=True, capture_output=True)
        _run_gate("kvcmp", [str(bin_p), "kvcmp", str(bpe_ck), "8", "Omega", str(merges)])
        _run_gate("oracle_xprep", [sys.executable, str(oracle_py), "xprep", "--ollm", str(bin_p), "--data", str(raw), "--outdir", str(tdp)])
        _run_gate("oracle_xinit", [sys.executable, str(oracle_py), "xinit", "--ollm", str(bin_p), "--cfg", "small"])
        _run_gate("oracle_xgrad", [sys.executable, str(oracle_py), "xgrad", "--ollm", str(bin_p), "--data", str(tdp / "ollm-train.bin"), "--cfg", "small"])
        _run_gate("oracle_xstep", [sys.executable, str(oracle_py), "xstep", "--ollm", str(bin_p), "--data", str(tdp / "ollm-train.bin"), "--cfg", "small"])
        _run_gate("oracle_xeval", [sys.executable, str(oracle_py), "xeval", "--ollm", str(bin_p), "--held", str(tdp / "ollm-held.bin"), "--cfg", "small"])

    passed_all = all(g["passed"] for g in gates)
    receipt = {
        "kind": "omega-helix-port-parity-receipt",
        "schema": 1,
        "candidate_bin": str(bin_p),
        "ok": passed_all,
        "passed_count": sum(1 for g in gates if g["passed"]),
        "total_gates": len(gates),
        "gates": gates,
    }
    if out_json:
        dest = Path(out_json).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name("." + dest.name + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, dest)
        receipt["receipt_path"] = str(dest)
    return receipt
