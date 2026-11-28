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

import omega_helix_backend as univ_backend  # noqa: E402
import openwaldo_bridge as ow_bridge  # noqa: E402

from .anti_dna import forge_anti_strand, read_sp_pack_hash, run_firewall_decode, verify_firewall_receipt  # noqa: E402

OLLMBF01_MAGIC = b"OLLMBF01"
OLLMBF01_HEADER_SIZE = 256


def read_ollmbf01_header(bf_path: str | Path) -> dict[str, Any]:
    """Parse and validate a 256-byte OLLMBF01 header without reading the weight payload."""
    p = Path(bf_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"OLLMBF01 file not found: {p}")
    sz = p.stat().st_size
    if sz < OLLMBF01_HEADER_SIZE:
        raise ValueError(f"OLLMBF01 truncated header ({sz} < {OLLMBF01_HEADER_SIZE}): {p}")
    with p.open("rb") as fh:
        hdr = fh.read(OLLMBF01_HEADER_SIZE)
    if len(hdr) < OLLMBF01_HEADER_SIZE:
        raise ValueError(f"OLLMBF01 short header read: {p}")
    if hdr[:8] != OLLMBF01_MAGIC:
        raise ValueError(f"bad OLLMBF01 magic {hdr[:8]!r}: {p}")
    fields = struct.unpack("<16Q", hdr[8:136])
    (
        version,
        V,
        D,
        H,
        H_kv,
        HD,
        F,
        L,
        flags,
        rope_theta_milli,
        rms_eps_bits,
        max_pos,
        P,
        payload_bytes,
        wfnv,
        trunk_hash,
    ) = fields
    if version != 1:
        raise ValueError(f"unsupported OLLMBF01 version {version}: {p}")
    if sz != OLLMBF01_HEADER_SIZE + payload_bytes:
        raise ValueError(f"OLLMBF01 size mismatch (file={sz}, expected={OLLMBF01_HEADER_SIZE + payload_bytes}): {p}")
    hdr_fnv = struct.unpack("<Q", hdr[248:256])[0]
    rms_eps = struct.unpack("<d", struct.pack("<Q", rms_eps_bits))[0]
    return {
        "magic": "OLLMBF01",
        "version": version,
        "V": V,
        "D": D,
        "H": H,
        "H_kv": H_kv,
        "HD": HD,
        "F": F,
        "L": L,
        "flags": flags,
        "rope_theta_milli": rope_theta_milli,
        "rms_eps": rms_eps,
        "max_pos": max_pos,
        "P": P,
        "payload_bytes": payload_bytes,
        "file_bytes": sz,
        "wfnv": f"0x{wfnv:016x}",
        "trunk_hash": f"0x{trunk_hash:016x}",
        "trunk_hash_hex": f"{trunk_hash:016x}",
        "hdr_fnv": f"0x{hdr_fnv:016x}",
    }


def _parse_bf_meta_line(line: str, prefix: str) -> dict[str, Any]:
    parts = line.strip().split()
    if len(parts) != 12 or parts[0] != prefix:
        raise ValueError(f"unexpected {prefix} metadata line: {line!r}")
    return {
        "tag": parts[0],
        "V": int(parts[1]),
        "D": int(parts[2]),
        "H": int(parts[3]),
        "H_kv": int(parts[4]),
        "HD": int(parts[5]),
        "F": int(parts[6]),
        "L": int(parts[7]),
        "P": int(parts[8]),
        "flags": int(parts[9]),
        "trunk_hash_hex": parts[10],
        "trunk_hash": f"0x{parts[10]}",
        "payload_bytes": int(parts[11]),
    }


def _parse_kv_line(line: str, prefix: str) -> dict[str, str]:
    parts = line.strip().split()
    if not parts or parts[0] != prefix:
        raise ValueError(f"unexpected {prefix} output: {line!r}")
    out: dict[str, str] = {}
    for item in parts[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def import_hf_bf16_trunk(
    interchange_bin: str | Path,
    hf_dir: str | Path,
    out_bf_path: str | Path,
) -> dict[str, Any]:
    """Stream HF safetensors directory (single or sharded) -> OLLMBF01 via `ollm-interchange import-hf-bf16`."""
    ic = Path(interchange_bin).expanduser().resolve()
    hf = Path(hf_dir).expanduser().resolve()
    out_p = Path(out_bf_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(ic), "import-hf-bf16", str(hf), str(out_p)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"import-hf-bf16 failed (rc={proc.returncode}): {detail}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("import-hf-bf16 produced empty stdout")
    meta = _parse_bf_meta_line(lines[-1], "IBF1")
    hdr = read_ollmbf01_header(out_p)
    return {
        "ok": True,
        "bf_path": str(out_p),
        "stdout": lines[-1],
        "header": hdr,
        "V": hdr["V"],
        "D": hdr["D"],
        "H": hdr["H"],
        "H_kv": hdr["H_kv"],
        "HD": hdr["HD"],
        "F": hdr["F"],
        "L": hdr["L"],
        "P": hdr["P"],
        "flags": hdr["flags"],
        "payload_bytes": hdr["payload_bytes"],
        "trunk_hash": hdr["trunk_hash"],
        "meta": meta,
    }


def verify_ollmbf01(
    interchange_bin: str | Path,
    bf_path: str | Path,
    *,
    dna_bin: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an OLLMBF01 trunk with `ollm-interchange info-bf16` and optionally `ollm-dna info`."""
    ic = Path(interchange_bin).expanduser().resolve()
    bfp = Path(bf_path).expanduser().resolve()
    r_ic = subprocess.run(
        [str(ic), "info-bf16", str(bfp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if r_ic.returncode != 0:
        detail = r_ic.stderr.strip() or r_ic.stdout.strip() or f"exit {r_ic.returncode}"
        raise RuntimeError(f"info-bf16 failed (rc={r_ic.returncode}): {detail}")
    ic_lines = [ln for ln in r_ic.stdout.splitlines() if ln.strip()]
    if not ic_lines:
        raise RuntimeError("info-bf16 produced empty stdout")
    info_bf16 = _parse_bf_meta_line(ic_lines[-1], "BF1")
    hdr = read_ollmbf01_header(bfp)
    for k in ("V", "D", "H", "H_kv", "HD", "F", "L", "P", "flags", "payload_bytes", "trunk_hash"):
        if info_bf16[k] != hdr[k]:
            raise RuntimeError(f"info-bf16 mismatch on {k}: {info_bf16[k]} != {hdr[k]}")

    dna_info: dict[str, Any] | None = None
    if dna_bin is not None:
        db = Path(dna_bin).expanduser().resolve()
        r_dn = subprocess.run(
            [str(db), "info", str(bfp)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if r_dn.returncode != 0:
            detail = r_dn.stderr.strip() or r_dn.stdout.strip() or f"exit {r_dn.returncode}"
            raise RuntimeError(f"ollm-dna info failed (rc={r_dn.returncode}): {detail}")
        dn_lines = [ln for ln in r_dn.stdout.splitlines() if ln.strip()]
        if not dn_lines:
            raise RuntimeError("ollm-dna info produced empty stdout")
        kv = _parse_kv_line(dn_lines[-1], "INFO_BF16")
        dna_info = {
            "stdout": dn_lines[-1],
            "V": int(kv["V"]),
            "D": int(kv["D"]),
            "H": int(kv["H"]),
            "H_kv": int(kv["HKV"]),
            "HD": int(kv["HD"]),
            "F": int(kv["F"]),
            "L": int(kv["L"]),
            "payload_bytes": int(kv["PAYLOAD"]),
            "trunk_hash": kv["TRUNK_HASH"],
        }
        for k in ("V", "D", "H", "H_kv", "HD", "F", "L", "payload_bytes", "trunk_hash"):
            if dna_info[k] != hdr[k]:
                raise RuntimeError(f"ollm-dna info mismatch on {k}: {dna_info[k]} != {hdr[k]}")

    return {
        "ok": True,
        "verified": True,
        "bf_path": str(bfp),
        "header": hdr,
        "info_bf16": info_bf16,
        "dna_info": dna_info,
    }


def _materialize_u32_tokens(
    spec: str | Path | bytes | list[int] | tuple[int, ...],
    out_path: Path,
    V: int,
) -> Path:
    if isinstance(spec, Path):
        p = spec.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"token file not found: {p}")
        return p
    if isinstance(spec, str) and "\n" not in spec and len(spec) < 512:
        cand = Path(spec).expanduser()
        if cand.is_file():
            return cand.resolve()
    if isinstance(spec, (list, tuple)):
        toks = [int(t) for t in spec]
    else:
        raw = spec.encode("utf-8") if isinstance(spec, str) else bytes(spec)
        if not raw:
            raise ValueError("token input must be non-empty")
        while len(raw) < 4:
            raw = raw + raw
        toks = [b % V for b in raw]
    if len(toks) < 2:
        raise ValueError("need at least 2 tokens for OLLMBF01 evaluation/crystallization")
    for t in toks:
        if t < 0 or t >= V:
            raise ValueError(f"token id {t} out of range [0, {V})")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(struct.pack(f"<{len(toks)}I", *toks))
    os.replace(tmp, out_path)
    return out_path


def run_bf16_dna_demo(
    bins: dict[str, Path],
    hf_dir: str | Path,
    domain_tokens: str | Path | bytes | list[int] | tuple[int, ...],
    general_tokens: str | Path | bytes | list[int] | tuple[int, ...],
    out_dir: str | Path,
    *,
    fexp: int = 16,
    nproto: int = 4,
    epochs: int = 4,
    w4: bool = True,
    cal_tokens: str | Path | bytes | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Import BF16 HF trunk -> verify OLLMBF01 -> ollm-dna base-eval, cryst, cal-cos, spguard-eval."""
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)
    ic_bin = Path(bins["ollm_interchange"]).expanduser().resolve()
    dna_bin = Path(bins.get("ollm_dna_gb10") or bins["ollm_dna"]).expanduser().resolve()

    trunk_bf = od / "trunk.bf16"
    imported = import_hf_bf16_trunk(ic_bin, hf_dir, trunk_bf)
    verified = verify_ollmbf01(ic_bin, trunk_bf, dna_bin=dna_bin)
    V = imported["V"]

    dom_u32 = _materialize_u32_tokens(domain_tokens, od / "domain.u32", V)
    gen_u32 = _materialize_u32_tokens(general_tokens, od / "general.u32", V)
    cal_u32 = _materialize_u32_tokens(cal_tokens, od / "cal.u32", V) if cal_tokens is not None else None

    base_json = od / "base_eval.json"
    r_base = subprocess.run(
        [str(dna_bin), "base-eval", str(trunk_bf), str(dom_u32), str(base_json)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if r_base.returncode != 0:
        raise RuntimeError(f"ollm-dna base-eval failed (rc={r_base.returncode}): {r_base.stderr.strip() or r_base.stdout.strip()}")

    sp2_path = od / "strand.sp2"
    cryst_cmd = [
        str(dna_bin), "cryst", str(trunk_bf), str(dom_u32), str(sp2_path),
        "--fexp", str(int(fexp)), "--nproto", str(int(nproto)), "--epochs", str(int(epochs)),
        "--w4" if w4 else "--f64",
    ]
    if cal_u32 is not None:
        cryst_cmd.extend(["--cal-dom", str(cal_u32)])
    r_cry = subprocess.run(cryst_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if r_cry.returncode != 0:
        raise RuntimeError(f"ollm-dna cryst failed (rc={r_cry.returncode}): {r_cry.stderr.strip() or r_cry.stdout.strip()}")

    r_spi = subprocess.run(
        [str(dna_bin), "sp-info", str(sp2_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if r_spi.returncode != 0:
        raise RuntimeError(f"ollm-dna sp-info failed (rc={r_spi.returncode}): {r_spi.stderr.strip() or r_spi.stdout.strip()}")

    r_cal = subprocess.run(
        [str(dna_bin), "cal-cos", str(trunk_bf), str(sp2_path), str(dom_u32), str(gen_u32)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if r_cal.returncode != 0:
        raise RuntimeError(f"ollm-dna cal-cos failed (rc={r_cal.returncode}): {r_cal.stderr.strip() or r_cal.stdout.strip()}")

    guard_json = od / "spguard_eval.json"
    r_grd = subprocess.run(
        [str(dna_bin), "spguard-eval", str(trunk_bf), str(sp2_path), str(dom_u32), str(gen_u32), str(guard_json)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if r_grd.returncode != 0:
        raise RuntimeError(f"ollm-dna spguard-eval failed (rc={r_grd.returncode}): {r_grd.stderr.strip() or r_grd.stdout.strip()}")

    doc = {
        "kind": "omega-helix-bf16-dna-receipt",
        "schema": 1,
        "ok": True,
        "import": imported,
        "verification": verified,
        "base_eval": json.loads(base_json.read_text(encoding="utf-8")),
        "cryst_stdout": r_cry.stdout.strip(),
        "spinfo_stdout": r_spi.stdout.strip(),
        "cal_cos_stdout": r_cal.stdout.strip(),
        "sp2_path": str(sp2_path),
        "spguard_eval": json.loads(guard_json.read_text(encoding="utf-8")),
    }
    rcpt_path = od / "bf16-dna-demo-receipt.json"
    tmp = od / ".bf16-dna-demo-receipt.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, rcpt_path)
    doc["receipt_path"] = str(rcpt_path)
    return doc


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
