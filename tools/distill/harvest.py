"""Universal trace & corpus harvester + OpenWALDO & OLLMDK01 exporters.

Harvests OpenAI/Anthropic chat JSONL, ShareGPT conversations, instruction/prompt-completion
JSONL, canonical OpenWALDO JSONL/EXPORT.json, and raw code/text directories into deduplicated
training records, writes verifiable OpenWALDO schema-1 exports, and packs sparse top-K teacher
distributions into OLLMDK01 binary shards.

Python 3 standard library only — zero external packages.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
if str(OPENWALDO_DIR) not in sys.path:
    sys.path.insert(0, str(OPENWALDO_DIR))

import openwaldo_bridge as ow_bridge  # noqa: E402

UMAGIC_DK1 = 3544415663467285583
UDKHDR = 64
UDK_MAXK = 64

TEXT_EXTS = {
    ".txt", ".md", ".om", ".py", ".c", ".h", ".rs", ".go", ".ts", ".tsx",
    ".js", ".jsx", ".sh", ".json", ".yaml", ".yml",
}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fnv1a64(data: bytes) -> int:
    """Standard 64-bit FNV-1a digest matching native/univ/umath.om & udistill.om."""
    h = 14695981039346656037
    mask = 0xFFFFFFFFFFFFFFFF
    for b in data:
        h = ((h ^ b) * 1099511628211) & mask
    return h


def _flatten_content(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts: list[str] = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(val, dict) and isinstance(val.get("text"), str):
        return val["text"]
    return str(val) if val is not None else ""


def _norm_role(role: Any) -> str:
    r = str(role or "").strip().lower()
    if r in ("system", "developer"):
        return "system"
    if r in ("user", "human", "prompter"):
        return "user"
    if r in ("assistant", "gpt", "model", "bot", "completion"):
        return "assistant"
    if r == "tool":
        return "tool"
    return "user"


def _compute_chat_spans(messages: list[dict[str, str]]) -> tuple[str, str, list[list[int]]]:
    """Render deterministic chat transcript plus exact assistant UTF-8 byte spans."""
    rendered_parts: list[str] = []
    spans: list[list[int]] = []
    assistant_chunks: list[str] = []
    byte_off = 0
    for idx, msg in enumerate(messages):
        prefix = f"{msg['role']}: "
        body = msg["content"]
        suffix = "\n" if idx + 1 < len(messages) else ""
        prefix_b = prefix.encode("utf-8")
        body_b = body.encode("utf-8")
        if msg["role"] == "assistant" and body_b:
            spans.append([byte_off + len(prefix_b), byte_off + len(prefix_b) + len(body_b)])
            assistant_chunks.append(body)
        chunk = prefix + body + suffix
        rendered_parts.append(chunk)
        byte_off += len(chunk.encode("utf-8"))
    rendered = "".join(rendered_parts)
    assistant_text = "\n\n".join(assistant_chunks) if assistant_chunks else rendered
    return rendered, assistant_text, spans


def _normalize_record(
    obj: dict[str, Any],
    *,
    default_source: str,
    default_license: str,
) -> dict[str, Any] | None:
    src = str(obj.get("source") or default_source)
    lic = str(obj.get("license") or default_license)
    logprobs = obj.get("logprobs")
    teacher_model = str(obj.get("teacher_model") or obj.get("model") or "")

    # 1. OpenWALDO canonical record
    if obj.get("kind") in ("conversation", "pretrain") and isinstance(obj.get("text"), str):
        raw_text = obj["text"]
        if obj["kind"] == "conversation":
            conv = json.loads(raw_text)
            msgs = [
                {"role": _norm_role(m.get("role")), "content": _flatten_content(m.get("content"))}
                for m in conv.get("messages", [])
                if _flatten_content(m.get("content")).strip()
            ]
            if not msgs:
                return None
            rendered, assistant_text, spans = _compute_chat_spans(msgs)
            return {
                "kind": "conversation",
                "messages": msgs,
                "text": json.dumps({"messages": msgs}, ensure_ascii=False),
                "rendered_text": rendered,
                "assistant_text": assistant_text,
                "assistant_spans": spans,
                "source": src,
                "license": lic,
                "teacher_model": teacher_model,
                "logprobs": logprobs,
            }
        if not raw_text.strip():
            return None
        return {
            "kind": "pretrain",
            "messages": None,
            "text": raw_text,
            "rendered_text": raw_text,
            "assistant_text": raw_text,
            "assistant_spans": [[0, len(raw_text.encode("utf-8"))]],
            "source": src,
            "license": lic,
            "teacher_model": teacher_model,
            "logprobs": logprobs,
        }

    # 2. Chat messages / ShareGPT conversations
    raw_msgs = obj.get("messages") or obj.get("conversations")
    if isinstance(raw_msgs, list) and raw_msgs:
        msgs: list[dict[str, str]] = []
        for m in raw_msgs:
            if not isinstance(m, dict):
                continue
            role = _norm_role(m.get("role") if "role" in m else m.get("from"))
            content = _flatten_content(m.get("content") if "content" in m else m.get("value")).strip()
            if content:
                msgs.append({"role": role, "content": content})
        if not msgs:
            return None
        rendered, assistant_text, spans = _compute_chat_spans(msgs)
        return {
            "kind": "conversation",
            "messages": msgs,
            "text": json.dumps({"messages": msgs}, ensure_ascii=False),
            "rendered_text": rendered,
            "assistant_text": assistant_text,
            "assistant_spans": spans,
            "source": src,
            "license": lic,
            "teacher_model": teacher_model,
            "logprobs": logprobs,
        }

    # 3. Prompt/response or instruction/input/output or question/answer
    prompt = ""
    response = ""
    if "prompt" in obj or "response" in obj or "completion" in obj:
        prompt = _flatten_content(obj.get("prompt")).strip()
        response = _flatten_content(obj.get("response") or obj.get("completion")).strip()
    elif "instruction" in obj or "output" in obj:
        inst = _flatten_content(obj.get("instruction")).strip()
        inp = _flatten_content(obj.get("input")).strip()
        prompt = f"{inst}\n\n{inp}".strip() if inp else inst
        response = _flatten_content(obj.get("output")).strip()
    elif "question" in obj or "answer" in obj:
        prompt = _flatten_content(obj.get("question")).strip()
        response = _flatten_content(obj.get("answer")).strip()

    if prompt and response:
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        rendered, assistant_text, spans = _compute_chat_spans(msgs)
        return {
            "kind": "conversation",
            "messages": msgs,
            "text": json.dumps({"messages": msgs}, ensure_ascii=False),
            "rendered_text": rendered,
            "assistant_text": assistant_text,
            "assistant_spans": spans,
            "source": src,
            "license": lic,
            "teacher_model": teacher_model,
            "logprobs": logprobs,
        }
    if response or isinstance(obj.get("text"), str):
        txt = response or str(obj.get("text") or "")
        if not txt.strip():
            return None
        return {
            "kind": "pretrain",
            "messages": None,
            "text": txt,
            "rendered_text": txt,
            "assistant_text": txt,
            "assistant_spans": [[0, len(txt.encode("utf-8"))]],
            "source": src,
            "license": lic,
            "teacher_model": teacher_model,
            "logprobs": logprobs,
        }
    return None


def _harvest_jsonl(
    path: Path,
    out: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    source_tag: str,
    license_id: str,
) -> None:
    with path.open("rb") as f:
        for raw in f:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            stats["raw_records"] += 1
            obj = json.loads(line.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: JSONL line must decode to object")
            rec = _normalize_record(obj, default_source=source_tag, default_license=license_id)
            if rec is None:
                continue
            digest = hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
            if digest in seen:
                stats["duplicates_skipped"] += 1
                continue
            seen.add(digest)
            rec["sha256"] = digest
            out.append(rec)


def harvest_paths(
    inputs: list[str | Path],
    *,
    source_tag: str = "distill-harvest",
    license_id: str = "Apache-2.0",
) -> dict[str, Any]:
    """Harvest records from JSONL traces, OpenWALDO EXPORT.json dirs, or code/text files."""
    if not inputs:
        raise ValueError("harvest_paths requires at least one input path")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"files_scanned": 0, "raw_records": 0, "duplicates_skipped": 0}
    input_shas: list[dict[str, Any]] = []

    for raw_in in inputs:
        p = Path(raw_in).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"harvest input not found: {p}")

        if p.is_dir():
            export_json = p / "EXPORT.json"
            if export_json.is_file():
                rep = ow_bridge.verify_export(p, check_records=True)
                input_shas.append({"path": str(export_json), "sha256": rep["export_json_sha256"]})
                for fentry in rep["files"]:
                    fp = Path(fentry["abs_path"])
                    stats["files_scanned"] += 1
                    _harvest_jsonl(fp, records, seen, stats, source_tag=source_tag, license_id=license_id)
                continue

            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for fn in sorted(filenames):
                    fp = Path(dirpath) / fn
                    if fp.suffix.lower() == ".jsonl":
                        stats["files_scanned"] += 1
                        sha, _ = ow_bridge.file_sha256_and_bytes(fp)
                        input_shas.append({"path": str(fp), "sha256": sha})
                        _harvest_jsonl(fp, records, seen, stats, source_tag=source_tag, license_id=license_id)
                    elif fp.suffix.lower() in TEXT_EXTS:
                        stats["files_scanned"] += 1
                        txt = fp.read_text(encoding="utf-8", errors="replace")
                        if not txt.strip():
                            continue
                        stats["raw_records"] += 1
                        digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                        if digest in seen:
                            stats["duplicates_skipped"] += 1
                            continue
                        seen.add(digest)
                        records.append({
                            "kind": "pretrain",
                            "messages": None,
                            "text": txt,
                            "rendered_text": txt,
                            "assistant_text": txt,
                            "assistant_spans": [[0, len(txt.encode("utf-8"))]],
                            "source": f"{source_tag}:{fp.name}",
                            "license": license_id,
                            "teacher_model": "",
                            "logprobs": None,
                            "sha256": digest,
                        })
        elif p.suffix.lower() == ".json" and p.name == "EXPORT.json":
            rep = ow_bridge.verify_export(p, check_records=True)
            input_shas.append({"path": str(p), "sha256": rep["export_json_sha256"]})
            for fentry in rep["files"]:
                fp = Path(fentry["abs_path"])
                stats["files_scanned"] += 1
                _harvest_jsonl(fp, records, seen, stats, source_tag=source_tag, license_id=license_id)
        elif p.suffix.lower() == ".jsonl":
            stats["files_scanned"] += 1
            sha, _ = ow_bridge.file_sha256_and_bytes(p)
            input_shas.append({"path": str(p), "sha256": sha})
            _harvest_jsonl(p, records, seen, stats, source_tag=source_tag, license_id=license_id)
        else:
            stats["files_scanned"] += 1
            txt = p.read_text(encoding="utf-8", errors="replace")
            if txt.strip():
                stats["raw_records"] += 1
                digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                if digest in seen:
                    stats["duplicates_skipped"] += 1
                else:
                    seen.add(digest)
                    sha, _ = ow_bridge.file_sha256_and_bytes(p)
                    input_shas.append({"path": str(p), "sha256": sha})
                    records.append({
                        "kind": "pretrain",
                        "messages": None,
                        "text": txt,
                        "rendered_text": txt,
                        "assistant_text": txt,
                        "assistant_spans": [[0, len(txt.encode("utf-8"))]],
                        "source": f"{source_tag}:{p.name}",
                        "license": license_id,
                        "teacher_model": "",
                        "logprobs": None,
                        "sha256": digest,
                    })

    stats["unique_records"] = len(records)
    return {"records": records, "stats": stats, "inputs": input_shas}


def write_openwaldo_export(
    records: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    corpus_name: str = "distilled-traces",
    license_id: str = "Apache-2.0",
    source_tag: str = "distill-harvest",
) -> dict[str, Any]:
    """Materialize canonical OpenWALDO EXPORT.json + shard-00000.jsonl + MASKS.jsonl."""
    if not records:
        raise ValueError("cannot write empty OpenWALDO export")
    root = Path(out_dir).expanduser().resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shard_rel = "data/shard-00000.jsonl"
    shard_path = root / shard_rel
    masks_path = root / "MASKS.jsonl"

    lines: list[bytes] = []
    mask_lines: list[bytes] = []
    for idx, rec in enumerate(records):
        text = str(rec["text"])
        kind = str(rec.get("kind") or "pretrain")
        if kind not in ("pretrain", "conversation"):
            kind = "pretrain"
        rec_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        src = str(rec.get("source") or source_tag)
        lic = str(rec.get("license") or license_id)
        entry = {
            "sha256": rec_sha,
            "kind": kind,
            "text": text,
            "source": src,
            "license": lic,
        }
        lines.append((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
        mask_entry = {
            "index": idx,
            "sha256": rec_sha,
            "kind": kind,
            "assistant_spans": rec.get("assistant_spans") or [],
            "teacher_model": rec.get("teacher_model") or "",
        }
        mask_lines.append((json.dumps(mask_entry, ensure_ascii=False) + "\n").encode("utf-8"))

    raw_shard = b"".join(lines)
    tmp_shard = shard_path.with_suffix(".jsonl.tmp")
    with tmp_shard.open("wb") as f:
        f.write(raw_shard)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_shard, shard_path)

    tmp_masks = masks_path.with_suffix(".jsonl.tmp")
    with tmp_masks.open("wb") as f:
        f.write(b"".join(mask_lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_masks, masks_path)

    shard_sha, shard_bytes = ow_bridge.file_sha256_and_bytes(shard_path)
    export_doc = {
        "kind": "waldo-corpus-export",
        "schema": 1,
        "format": "jsonl",
        "generated": utc_now(),
        "files": [
            {
                "path": shard_rel,
                "manifest": corpus_name,
                "format": "jsonl",
                "object_sha256": shard_sha,
                "object_bytes": shard_bytes,
                "sha256": shard_sha,
                "bytes": shard_bytes,
                "docs": len(records),
                "license": license_id,
                "licenses": [license_id],
            }
        ],
        "bom": {
            "kind": "openwaldo-bom",
            "schema": 1,
            "index": {"name": corpus_name, "source": source_tag},
            "paths": [shard_rel],
            "shards": [
                {
                    "manifest": corpus_name,
                    "sha256": shard_sha,
                    "bytes": shard_bytes,
                    "license": license_id,
                }
            ],
        },
    }
    export_path = root / "EXPORT.json"
    tmp_export = export_path.with_suffix(".json.tmp")
    tmp_export.write_text(json.dumps(export_doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_export, export_path)

    verified = ow_bridge.verify_export(root, check_records=True)
    return {
        "export_dir": str(root),
        "export_json": verified["export_json"],
        "export_json_sha256": verified["export_json_sha256"],
        "masks_jsonl": str(masks_path),
        "docs": verified["total_docs"],
        "bytes": verified["total_bytes"],
        "licenses": verified["licenses"],
    }


def write_ollmdk01(
    records_dk: list[dict[str, Any]],
    out_path: str | Path,
    *,
    V: int,
    K: int,
    T: int,
) -> dict[str, Any]:
    """Write a verified OLLMDK01 sparse top-K distillation shard atomically."""
    if V < 2:
        raise ValueError(f"V must be >= 2, got {V}")
    if K < 1 or K > UDK_MAXK or K > V:
        raise ValueError(f"K={K} must satisfy 1 <= K <= min({UDK_MAXK}, V={V})")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if not records_dk:
        raise ValueError("records_dk must be non-empty")

    body_chunks: list[bytes] = []
    for idx, rec in enumerate(records_dk):
        toks = list(rec["tokens"])
        mask = list(rec["mask"])
        top_ids = list(rec["top_ids"])
        top_probs = list(rec["top_probs"])
        if len(toks) != T + 1:
            raise ValueError(f"record {idx}: expected {T + 1} tokens, got {len(toks)}")
        if len(mask) != T:
            raise ValueError(f"record {idx}: expected {T} mask bytes, got {len(mask)}")
        if len(top_ids) != T * K or len(top_probs) != T * K:
            raise ValueError(f"record {idx}: expected {T * K} sparse entries")
        for tok in toks:
            if not isinstance(tok, int) or tok < 0 or tok >= V:
                raise ValueError(f"record {idx}: token {tok!r} out of range [0, {V})")
        for m in mask:
            if m not in (0, 1):
                raise ValueError(f"record {idx}: invalid mask byte {m!r}")
        for pos in range(T):
            s = 0.0
            base = pos * K
            for j in range(K):
                tid = top_ids[base + j]
                pr = float(top_probs[base + j])
                if not isinstance(tid, int) or tid < 0 or tid >= V:
                    raise ValueError(f"record {idx} pos {pos}: top_id {tid!r} out of range")
                if not math.isfinite(pr) or pr < 0.0:
                    raise ValueError(f"record {idx} pos {pos}: invalid prob {pr!r}")
                s += pr
            if s > 1.0001:
                raise ValueError(f"record {idx} pos {pos}: prob sum {s} > 1.0001")

        body_chunks.append(struct.pack(f"<{T + 1}I", *toks))
        body_chunks.append(bytes(mask))
        body_chunks.append(struct.pack(f"<{T * K}I", *top_ids))
        body_chunks.append(struct.pack(f"<{T * K}f", *(float(x) for x in top_probs)))

    body = b"".join(body_chunks)
    fnv = fnv1a64(body)
    hdr = struct.pack("<8Q", UMAGIC_DK1, 1, V, K, T, len(records_dk), 0, fnv)
    dest = Path(out_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(hdr)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    sha, total_bytes = ow_bridge.file_sha256_and_bytes(dest)
    return {
        "path": str(dest),
        "magic": "OLLMDK01",
        "version": 1,
        "V": V,
        "K": K,
        "T": T,
        "N": len(records_dk),
        "flags": 0,
        "fnv_body": fnv,
        "fnv_hex": f"{fnv:016x}",
        "sha256": sha,
        "bytes": total_bytes,
    }


def read_ollmdk01_header(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    raw = p.read_bytes()
    if len(raw) < UDKHDR:
        raise ValueError(f"{p}: truncated OLLMDK01 header ({len(raw)} bytes)")
    magic, ver, V, K, T, N, flags, want_fnv = struct.unpack("<8Q", raw[:UDKHDR])
    if magic != UMAGIC_DK1 or ver != 1 or flags != 0:
        raise ValueError(f"{p}: bad OLLMDK01 magic/version/flags ({magic}, {ver}, {flags})")
    rec_sz = 4 * (T + 1) + T + 8 * T * K
    want_sz = UDKHDR + N * rec_sz
    if len(raw) != want_sz:
        raise ValueError(f"{p}: size mismatch (got {len(raw)}, expected {want_sz})")
    got_fnv = fnv1a64(raw[UDKHDR:])
    if got_fnv != want_fnv:
        raise ValueError(f"{p}: body FNV-1a mismatch (got {got_fnv:016x}, expected {want_fnv:016x})")
    sha, size = ow_bridge.file_sha256_and_bytes(p)
    return {
        "path": str(p),
        "magic": "OLLMDK01",
        "version": ver,
        "V": V,
        "K": K,
        "T": T,
        "N": N,
        "flags": flags,
        "fnv_body": want_fnv,
        "fnv_hex": f"{want_fnv:016x}",
        "sha256": sha,
        "bytes": size,
    }
