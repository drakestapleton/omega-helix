#!/usr/bin/env python3
"""OpenWALDO <-> Omega Helix fail-closed dataset verifier and BOM provenance bridge.

Verifies OpenWALDO corpus exports (`EXPORT.json`, kind `waldo-corpus-export` schema 1),
checks every exported shard's byte size and SHA-256 against `EXPORT.json` and `bom.shards`,
verifies canonical JSONL record `sha256 == sha256(text)` integrity, invokes the pure-Omega
`ollm-waldo` binary to materialize NUL-delimited `train.bin`/`held.bin`, optionally fits
and encodes BPE tokens via `ollm`, and writes an `openwaldo-helix-bom.json` provenance
record binding the OpenWALDO corpus BOM to the prepared training files and checkpoints.

Uses only the Python 3 standard library.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import posixpath
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


HEX64 = set("0123456789abcdef")


def is_sha256(val: Any) -> bool:
    return isinstance(val, str) and len(val) == 64 and all(c in HEX64 for c in val)


def file_sha256_and_bytes(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def validate_export_path(rel: Any) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/"):
        raise ValueError(f"invalid export file path: {rel!r}")
    cleaned = posixpath.normpath(rel)
    if cleaned != rel or cleaned == ".." or cleaned.startswith("../") or not cleaned.startswith("data/"):
        raise ValueError(f"unsafe or non-canonical export file path: {rel!r}")
    return cleaned


def verify_jsonl_records(path: Path, expected_docs: int) -> tuple[int, int]:
    docs = 0
    tokens = 0
    with path.open("rb") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid UTF-8/JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{lineno}: record must be a JSON object")
            sha = rec.get("sha256")
            kind = rec.get("kind")
            text = rec.get("text")
            source = rec.get("source")
            lic = rec.get("license")
            if not is_sha256(sha):
                raise ValueError(f"{path}:{lineno}: invalid record sha256")
            if kind not in ("pretrain", "conversation"):
                raise ValueError(f"{path}:{lineno}: unsupported record kind {kind!r}")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{lineno}: missing or empty text")
            actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if actual_sha != sha:
                raise ValueError(
                    f"{path}:{lineno}: record sha256 mismatch (got {actual_sha}, declared {sha})"
                )
            if not isinstance(source, str) or not source:
                raise ValueError(f"{path}:{lineno}: missing source")
            if not isinstance(lic, str) or not lic:
                raise ValueError(f"{path}:{lineno}: missing license")
            if kind == "conversation":
                try:
                    conv = json.loads(text)
                except Exception as exc:
                    raise ValueError(f"{path}:{lineno}: invalid conversation JSON: {exc}") from exc
                msgs = conv.get("messages") if isinstance(conv, dict) else None
                if not isinstance(msgs, list) or not msgs:
                    raise ValueError(f"{path}:{lineno}: conversation missing messages")
                for m_idx, msg in enumerate(msgs):
                    if not isinstance(msg, dict):
                        raise ValueError(f"{path}:{lineno}: message {m_idx} not an object")
                    role = msg.get("role")
                    content = msg.get("content")
                    if role not in ("system", "user", "assistant", "tool"):
                        raise ValueError(f"{path}:{lineno}: message {m_idx} bad role {role!r}")
                    if not isinstance(content, str) or not content:
                        raise ValueError(f"{path}:{lineno}: message {m_idx} empty content")
            docs += 1
            rec_tok = rec.get("tokens", 0)
            if isinstance(rec_tok, int) and rec_tok >= 0:
                tokens += rec_tok
    if expected_docs >= 0 and docs != expected_docs:
        raise ValueError(f"{path}: record count {docs} != EXPORT.json docs {expected_docs}")
    return docs, tokens


def verify_export(export_location: Path, check_records: bool = True) -> dict[str, Any]:
    export_json = export_location / "EXPORT.json" if export_location.is_dir() else export_location
    if not export_json.is_file():
        raise ValueError(f"missing OpenWALDO EXPORT.json at {export_json}")
    export_root = export_json.parent.resolve()
    export_sha, export_bytes = file_sha256_and_bytes(export_json)
    doc = json.loads(export_json.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("EXPORT.json root must be an object")
    if doc.get("kind") != "waldo-corpus-export" or doc.get("schema") != 1:
        raise ValueError(
            f"unsupported OpenWALDO export identity {doc.get('kind')!r} schema {doc.get('schema')!r}"
        )
    fmt = doc.get("format")
    if fmt not in ("jsonl", "native"):
        raise ValueError(f"unsupported export format {fmt!r}")
    bom = doc.get("bom")
    if not isinstance(bom, dict) or bom.get("kind") != "openwaldo-bom" or bom.get("schema") != 1:
        raise ValueError("EXPORT.json missing valid openwaldo-bom schema 1")
    files = doc.get("files")
    shards = bom.get("shards")
    if not isinstance(files, list) or not isinstance(shards, list) or not files:
        raise ValueError("EXPORT.json must contain non-empty files and bom.shards lists")
    if len(files) != len(shards):
        raise ValueError(f"EXPORT.json files ({len(files)}) != bom.shards ({len(shards)})")

    seen: set[str] = set()
    verified_files: list[dict[str, Any]] = []
    total_docs = 0
    total_bytes = 0
    licenses: set[str] = set()

    for idx, (fentry, shard) in enumerate(zip(files, shards)):
        if not isinstance(fentry, dict) or not isinstance(shard, dict):
            raise ValueError(f"entry {idx} must be an object")
        rel = validate_export_path(fentry.get("path"))
        if rel in seen:
            raise ValueError(f"duplicate export path {rel!r}")
        seen.add(rel)

        target = (export_root / rel).resolve()
        try:
            target.relative_to(export_root)
        except ValueError as exc:
            raise ValueError(f"export path escapes root: {rel!r}") from exc
        if not target.is_file():
            raise ValueError(f"missing exported shard file: {target}")

        if fentry.get("manifest") != shard.get("manifest"):
            raise ValueError(f"{rel}: manifest mismatch against bom.shards[{idx}]")
        if fentry.get("object_sha256") != shard.get("sha256") or not is_sha256(fentry.get("object_sha256")):
            raise ValueError(f"{rel}: object_sha256 mismatch against bom.shards[{idx}]")
        if fentry.get("object_bytes") != shard.get("bytes"):
            raise ValueError(f"{rel}: object_bytes mismatch against bom.shards[{idx}]")
        expected_sha = fentry.get("sha256")
        expected_bytes = fentry.get("bytes")
        if not is_sha256(expected_sha) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"{rel}: invalid sha256 or bytes declaration")
        if fmt == "jsonl" and fentry.get("format") != "jsonl":
            raise ValueError(f"{rel}: expected format 'jsonl', got {fentry.get('format')!r}")

        got_sha, got_bytes = file_sha256_and_bytes(target)
        if got_bytes != expected_bytes:
            raise ValueError(f"{rel}: size mismatch (got {got_bytes}, expected {expected_bytes})")
        if got_sha != expected_sha:
            raise ValueError(f"{rel}: sha256 mismatch (got {got_sha}, expected {expected_sha})")

        expected_docs = fentry.get("docs", -1)
        if fmt == "jsonl" and check_records:
            docs, _ = verify_jsonl_records(target, expected_docs)
        else:
            docs = expected_docs if isinstance(expected_docs, int) else 0

        lic = fentry.get("license") or ""
        if lic:
            licenses.add(lic)
        for extra_lic in fentry.get("licenses") or []:
            if isinstance(extra_lic, str) and extra_lic:
                licenses.add(extra_lic)

        total_docs += docs
        total_bytes += got_bytes
        verified_files.append(
            {
                "path": rel,
                "abs_path": str(target),
                "manifest": fentry["manifest"],
                "format": fentry.get("format", fmt),
                "object_sha256": fentry["object_sha256"],
                "sha256": got_sha,
                "bytes": got_bytes,
                "docs": docs,
                "license": lic,
                "licenses": sorted(set(fentry.get("licenses") or ([lic] if lic else []))),
            }
        )

    return {
        "export_json": str(export_json),
        "export_json_sha256": export_sha,
        "export_json_bytes": export_bytes,
        "format": fmt,
        "generated": doc.get("generated", ""),
        "index": bom.get("index", {}),
        "paths": bom.get("paths", []),
        "files": verified_files,
        "total_docs": total_docs,
        "total_bytes": total_bytes,
        "licenses": sorted(licenses),
    }


def read_ollm_checkpoint_header(ck_path: Path) -> dict[str, Any]:
    raw = ck_path.read_bytes()
    if len(raw) < 112:
        raise ValueError(f"{ck_path}: file too short for OLLMCK header")
    magic = raw[:8].decode("ascii", errors="replace")
    if magic not in ("OLLMCK01", "OLLMCK02", "OLLMCK03"):
        raise ValueError(f"{ck_path}: unexpected checkpoint magic {magic!r}")
    fields = struct.unpack("<12Q", raw[8:104])
    V, T, D, H, F, L, B, seed, step, total, lr_bits, P = fields
    sha, size = file_sha256_and_bytes(ck_path)
    out: dict[str, Any] = {
        "path": str(ck_path),
        "sha256": sha,
        "bytes": size,
        "magic": magic,
        "shape": {"V": V, "T": T, "D": D, "H": H, "F": F, "L": L, "B": B, "P": P},
        "seed": seed,
        "step": step,
        "total_steps": total,
    }
    if magic in ("OLLMCK02", "OLLMCK03") and len(raw) >= 120:
        out["G"] = struct.unpack("<Q", raw[104:112])[0]
    if magic == "OLLMCK03" and len(raw) >= 144:
        lp, lc, np_val = struct.unpack("<3Q", raw[112:136])
        out["loop"] = {"LP": lp, "LC": lc, "NP": np_val}
    return out


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_export(Path(args.export_dir), check_records=not args.skip_records)
    print(
        f"OPENWALDO_VERIFY_OK format={report['format']} files={len(report['files'])} "
        f"docs={report['total_docs']} bytes={report['total_bytes']} "
        f"export_sha256={report['export_json_sha256']}"
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    report = verify_export(Path(args.export_dir), check_records=True)
    if report["format"] != "jsonl":
        raise ValueError(
            "OpenWALDO export is in 'native' (Parquet) format; re-export canonical JSONL with "
            "`waldo index export --format jsonl <corpus> <dir>` so Omega Helix's zero-dependency "
            "pure-Omega ingestor (`ollm-waldo`) can verify and stream every record."
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ollm_waldo = Path(args.ollm_waldo).resolve()
    if not ollm_waldo.is_file() or not os.access(ollm_waldo, os.X_OK):
        raise ValueError(f"pure-Omega ingestor not executable at {ollm_waldo} (build with TARGET=openwaldo sh scripts/build.sh)")

    combined_jsonl = out_dir / "verified-corpus.jsonl"
    with combined_jsonl.open("wb") as dst:
        for fentry in report["files"]:
            with Path(fentry["abs_path"]).open("rb") as src:
                shutil.copyfileobj(src, dst, length=1 << 20)

    train_bin = out_dir / "train.bin"
    held_bin = out_dir / "held.bin"
    proc = subprocess.run(
        [str(ollm_waldo), str(combined_jsonl), str(train_bin), str(held_bin), str(args.held_mod)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ollm-waldo exited {proc.returncode}: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    waldo_line = proc.stdout.strip()
    if not waldo_line.startswith("W "):
        raise RuntimeError(f"unexpected ollm-waldo output: {waldo_line!r}")
    parts = waldo_line.split()
    if len(parts) != 6:
        raise RuntimeError(f"malformed ollm-waldo summary: {waldo_line!r}")
    w_total, w_tr_docs, w_he_docs, w_tr_bytes, w_he_bytes = (int(x) for x in parts[1:])
    if w_total != report["total_docs"]:
        raise RuntimeError(f"ollm-waldo processed {w_total} docs != verified {report['total_docs']}")

    combined_jsonl.unlink()

    tr_sha, tr_bytes = file_sha256_and_bytes(train_bin)
    he_sha, he_bytes = file_sha256_and_bytes(held_bin)
    waldo_bin_sha, _ = file_sha256_and_bytes(ollm_waldo)

    artifacts: dict[str, Any] = {
        "train_bin": {"path": str(train_bin), "sha256": tr_sha, "bytes": tr_bytes, "docs": w_tr_docs},
        "held_bin": {"path": str(held_bin), "sha256": he_sha, "bytes": he_bytes, "docs": w_he_docs},
    }

    if args.ollm:
        ollm_bin = Path(args.ollm).resolve()
        if not ollm_bin.is_file() or not os.access(ollm_bin, os.X_OK):
            raise ValueError(f"ollm binary not executable at {ollm_bin}")
        if args.vocab > 256:
            merges = out_dir / f"merges{args.vocab}.bin"
            tr_tok = out_dir / "train.tok"
            he_tok = out_dir / "held.tok"
            subprocess.run(
                [str(ollm_bin), "bpefit", str(train_bin), str(merges), str(args.vocab)],
                check=True,
            )
            subprocess.run(
                [str(ollm_bin), "bpeenc", str(merges), str(args.vocab), str(train_bin), str(tr_tok)],
                check=True,
            )
            m_sha, m_bytes = file_sha256_and_bytes(merges)
            tt_sha, tt_bytes = file_sha256_and_bytes(tr_tok)
            artifacts["merges_bin"] = {"path": str(merges), "sha256": m_sha, "bytes": m_bytes, "vocab": args.vocab}
            artifacts["train_tok"] = {"path": str(tr_tok), "sha256": tt_sha, "bytes": tt_bytes}
            if he_bytes > 0:
                subprocess.run(
                    [str(ollm_bin), "bpeenc", str(merges), str(args.vocab), str(held_bin), str(he_tok)],
                    check=True,
                )
                ht_sha, ht_bytes = file_sha256_and_bytes(he_tok)
                artifacts["held_tok"] = {"path": str(he_tok), "sha256": ht_sha, "bytes": ht_bytes}

    helix_bom = {
        "kind": "omega-helix-training-bom",
        "schema": 1,
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "openwaldo_export": {
            "export_json_sha256": report["export_json_sha256"],
            "export_json_bytes": report["export_json_bytes"],
            "format": report["format"],
            "generated": report["generated"],
            "index": report["index"],
            "paths": report["paths"],
            "licenses": report["licenses"],
            "total_docs": report["total_docs"],
            "shards": [
                {
                    "path": f["path"],
                    "manifest": f["manifest"],
                    "object_sha256": f["object_sha256"],
                    "sha256": f["sha256"],
                    "bytes": f["bytes"],
                    "docs": f["docs"],
                    "licenses": f["licenses"],
                }
                for f in report["files"]
            ],
        },
        "ingestor": {
            "name": "ollm-waldo",
            "sha256": waldo_bin_sha,
            "held_mod": args.held_mod,
            "summary": waldo_line,
        },
        "artifacts": artifacts,
        "checkpoints": [],
    }
    bom_path = out_dir / "openwaldo-helix-bom.json"
    tmp_path = out_dir / ".openwaldo-helix-bom.json.tmp"
    tmp_path.write_text(json.dumps(helix_bom, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(bom_path)
    print(f"OPENWALDO_PREPARE_OK bom={bom_path} {waldo_line}")
    return 0


def cmd_bind_checkpoint(args: argparse.Namespace) -> int:
    bom_path = Path(args.bom).resolve()
    bom = json.loads(bom_path.read_text(encoding="utf-8"))
    if bom.get("kind") != "omega-helix-training-bom" or bom.get("schema") != 1:
        raise ValueError(f"{bom_path}: not an omega-helix-training-bom schema 1 document")
    ck_info = read_ollm_checkpoint_header(Path(args.checkpoint).resolve())
    ck_info["bound_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bom.setdefault("checkpoints", []).append(ck_info)
    tmp_path = bom_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(bom_path)
    print(
        f"OPENWALDO_BIND_OK bom={bom_path} magic={ck_info['magic']} "
        f"step={ck_info['step']} sha256={ck_info['sha256']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenWALDO <-> Omega Helix dataset & BOM bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify", help="Verify an OpenWALDO EXPORT.json directory fail-closed")
    p_ver.add_argument("export_dir", help="Path to OpenWALDO export directory (containing EXPORT.json)")
    p_ver.add_argument("--skip-records", action="store_true", help="Skip per-record sha256(text) scan")
    p_ver.set_defaults(func=cmd_verify)

    p_prep = sub.add_parser("prepare", help="Verify OpenWALDO export and ingest into Omega Helix")
    p_prep.add_argument("export_dir", help="Path to OpenWALDO JSONL export directory")
    p_prep.add_argument("out_dir", help="Output directory for train.bin/held.bin/tokens/BOM")
    p_prep.add_argument("--ollm-waldo", required=True, help="Path to pure-Omega ollm-waldo binary")
    p_prep.add_argument("--ollm", default="", help="Optional path to ollm binary for BPE fit/encode")
    p_prep.add_argument("--vocab", type=int, default=4096, help="BPE vocab size (>256 to tokenize)")
    p_prep.add_argument("--held-mod", type=int, default=10, help="Held-out split modulus (default 10; 0=none)")
    p_prep.set_defaults(func=cmd_prepare)

    p_bind = sub.add_parser("bind-checkpoint", help="Bind a trained OLLMCK checkpoint into the training BOM")
    p_bind.add_argument("bom", help="Path to openwaldo-helix-bom.json")
    p_bind.add_argument("checkpoint", help="Path to trained OLLMCK01/02/03 checkpoint")
    p_bind.set_defaults(func=cmd_bind_checkpoint)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
