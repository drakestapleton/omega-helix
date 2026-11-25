"""Model-Agnostic Personal Memory & 'Migrate Brain to New Trunk' Workflow.

Stores human-inspectable append-only memories on disk (`memory.jsonl`), supports authenticated
PBKDF2+HMAC encrypted snapshots, and consolidates verified memories into an `OLLMSP01`
specialist pack pinned to any new base trunk (`growpro` + `fcache` + `trainfc` + `spext`).

Python 3 standard library only — zero third-party deps.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
for _p in (str(TOOLS_DIR), str(TOOLS_DIR / "openwaldo"), str(TOOLS_DIR / "distill")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openwaldo_bridge as ow_bridge  # noqa: E402
from distill.verifiers import VerifierPipeline  # noqa: E402

from .pkg import read_sp_meta  # noqa: E402

_ENC_MAGIC = b"OMEMENC1"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _derive_keys(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 100_000, dklen=64)
    return dk[:32], dk[32:]


def _xor_stream(enc_key: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    blk_idx = 0
    pos = 0
    while pos < len(data):
        ks = hashlib.sha256(enc_key + blk_idx.to_bytes(8, "little")).digest()
        take = min(len(ks), len(data) - pos)
        for i in range(take):
            out[pos + i] = data[pos + i] ^ ks[i]
        pos += take
        blk_idx += 1
    return bytes(out)


class MemoryStore:
    """Append-only, SHA-256 deduplicated personal memory ledger."""

    def __init__(self, store_path: str | Path) -> None:
        self.path = Path(store_path).expanduser().resolve()

    def list_memories(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s:
                out.append(json.loads(s))
        return out

    def add(self, text: str, *, tags: list[str] | None = None) -> dict[str, Any]:
        clean = text.strip()
        if not clean:
            raise ValueError("memory text must be non-empty")
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        existing = self.list_memories()
        for rec in existing:
            if rec.get("sha256") == digest:
                return rec
        rec = {
            "id": len(existing) + 1,
            "ts": _utc_now(),
            "text": clean,
            "tags": list(tags or []),
            "sha256": digest,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def export_encrypted(store_path: str | Path, out_path: str | Path, passphrase: str) -> dict[str, Any]:
    if not passphrase:
        raise ValueError("passphrase must be non-empty")
    plain = Path(store_path).expanduser().resolve().read_bytes()
    salt = secrets.token_bytes(16)
    enc_k, mac_k = _derive_keys(passphrase, salt)
    ct = _xor_stream(enc_k, plain)
    body = _ENC_MAGIC + salt + ct
    tag = hmac.new(mac_k, body, hashlib.sha256).digest()
    dest = Path(out_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("." + dest.name + ".tmp")
    tmp.write_bytes(body + tag)
    os.replace(tmp, dest)
    return {"path": str(dest), "bytes": dest.stat().st_size, "sha256": hashlib.sha256(body + tag).hexdigest()}


def import_encrypted(enc_path: str | Path, out_store_path: str | Path, passphrase: str) -> dict[str, Any]:
    raw = Path(enc_path).expanduser().resolve().read_bytes()
    if len(raw) < len(_ENC_MAGIC) + 16 + 32 or not raw.startswith(_ENC_MAGIC):
        raise ValueError("invalid encrypted memory bundle header")
    body = raw[:-32]
    got_tag = raw[-32:]
    salt = raw[len(_ENC_MAGIC) : len(_ENC_MAGIC) + 16]
    ct = raw[len(_ENC_MAGIC) + 16 : -32]
    enc_k, mac_k = _derive_keys(passphrase, salt)
    want_tag = hmac.new(mac_k, body, hashlib.sha256).digest()
    if not hmac.compare_digest(got_tag, want_tag):
        raise ValueError("passphrase incorrect or memory bundle tampered")
    plain = _xor_stream(enc_k, ct)
    dest = Path(out_store_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(plain)
    return {"path": str(dest), "records": len([ln for ln in plain.decode("utf-8").splitlines() if ln.strip()])}


def migrate_brain_to_trunk(
    ollm_bin: str | Path,
    store_path: str | Path,
    new_trunk_ck: str | Path,
    out_dir: str | Path,
    *,
    steps: int = 6,
    add_layers: int = 1,
    lr_micro: int = 2000,
) -> dict[str, Any]:
    """Replay verified personal memories onto a new base trunk and mint a fresh `.sp` pack."""
    ollm = str(Path(ollm_bin).expanduser().resolve())
    bck = Path(new_trunk_ck).expanduser().resolve()
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    mems = MemoryStore(store_path).list_memories()
    if not mems:
        raise ValueError("memory store is empty")
    vpipe = VerifierPipeline(["nonempty"])
    vres = vpipe.filter_records([{"text": m["text"]} for m in mems])
    accepted = vres["accepted"]
    if not accepted:
        raise RuntimeError("all memories failed verification")

    hdr = ow_bridge.read_ollm_checkpoint_header(bck)
    base_L = int(hdr["shape"]["L"])
    T = int(hdr["shape"]["T"])
    new_L = base_L + max(1, int(add_layers))

    raw_bytes = "\n".join(r["text"] for r in accepted).encode("utf-8", errors="replace")
    while len(raw_bytes) <= T + 8:
        raw_bytes = raw_bytes + b"\n" + raw_bytes
    mem_bin = od / "migrated_memory.bin"
    mem_bin.write_bytes(raw_bytes)

    cand_ck = od / "migrated_brain.ck"
    cand_hc = od / "migrated_brain.hc"
    cand_sp = od / "migrated_brain.sp"
    fc_w = max(1, min(16, (len(raw_bytes) - 1) // T))
    every = max(1, int(steps) // 2)

    subprocess.run([ollm, "growpro", str(bck), str(cand_ck), str(new_L), str(int(steps)), str(int(lr_micro))], check=True, capture_output=True)
    subprocess.run([ollm, "fcache", str(bck), str(mem_bin), str(fc_w), str(cand_hc)], check=True, capture_output=True)
    subprocess.run([ollm, "trainfc", str(cand_ck), str(cand_hc), str(int(steps)), str(every), str(base_L)], check=True, capture_output=True)
    subprocess.run([ollm, "spext", str(bck), str(cand_ck), str(cand_sp)], check=True, capture_output=True)

    sp_meta = read_sp_meta(cand_sp)
    receipt = {
        "kind": "omega-helix-memory-migration",
        "schema": 1,
        "migrated_at": _utc_now(),
        "new_trunk_ck": str(bck),
        "new_trunk_sha256": hdr["sha256"],
        "memories_count": len(accepted),
        "memory_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "sp_pack": sp_meta,
    }
    rpath = od / "migration-receipt.json"
    rpath.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(rpath)
    return receipt
