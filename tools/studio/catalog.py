#!/usr/bin/env python3
"""OpenWALDO corpus catalog, SPDX license normalizer, and local dataset builder.

Python 3 standard library only — zero external packages or trackers.
"""

from __future__ import annotations

import collections
import contextlib
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STUDIO_DIR = Path(__file__).resolve().parent
TOOLS_DIR = STUDIO_DIR.parent
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
DEFAULT_CACHE_PATH = STUDIO_DIR / "waldo_catalog_cache.json"

if str(OPENWALDO_DIR) not in sys.path:
    sys.path.insert(0, str(OPENWALDO_DIR))

import openwaldo_bridge as ow_bridge  # noqa: E402

SPDX_SPLIT_RE = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)
SUPPORTED_TEXT_EXTS = {
    ".txt", ".md", ".jsonl", ".om", ".py", ".go", ".rs", ".ts", ".tsx",
    ".js", ".jsx", ".c", ".h", ".cc", ".cpp", ".sh", ".json", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "build", "dist", "target", ".cache", ".pytest_cache",
}


def repo_root(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    env_home = os.environ.get("OMEGA_HELIX_HOME")
    if env_home and Path(env_home).is_dir():
        return Path(env_home).resolve()
    return TOOLS_DIR.parent.resolve()


def normalize_spdx_licenses(raw_licenses: Any) -> dict[str, Any]:
    """Normalize raw SPDX strings into clean individual tokens + concise summary."""
    raw_list: list[str] = []
    if isinstance(raw_licenses, str):
        raw_list = [raw_licenses]
    elif isinstance(raw_licenses, list):
        raw_list = [str(x).strip() for x in raw_licenses if isinstance(x, str) and x.strip()]

    tokens: set[str] = set()
    for expr in raw_list:
        cleaned = expr.replace("(", " ").replace(")", " ").strip()
        if not cleaned:
            continue
        parts = SPDX_SPLIT_RE.split(cleaned)
        for p in parts:
            tok = p.strip().strip(",")
            if tok:
                tokens.add(tok)

    sorted_tags = sorted(tokens, key=lambda s: (s.lower().startswith("licenseref-"), s.lower()))
    count = len(sorted_tags)
    if count == 0:
        summary = "See manifest"
    elif count <= 4:
        summary = ", ".join(sorted_tags)
    else:
        summary = f"{', '.join(sorted_tags[:4])} +{count - 4} more ({count} SPDX)"

    return {
        "licenses": raw_list,
        "license_tags": sorted_tags,
        "license_count": count,
        "raw_license_count": len(raw_list),
        "license_summary": summary,
    }


def _enrich_corpus(entry: dict[str, Any]) -> dict[str, Any]:
    item = dict(entry)
    item.setdefault("path", item.get("name", ""))
    item.setdefault("manifest", "")
    item.setdefault("category", "other")
    item.setdefault("name", item["path"].split("/")[-1] if item["path"] else "unnamed")
    item.setdefault("title", item["name"])
    item.setdefault("description", "")
    item.setdefault("languages", [])
    item.setdefault("programming_languages", [])
    item.setdefault("sources_count", 0)
    item.setdefault("url", f"https://github.com/openwaldo/waldo-index/blob/main/{item['path']}")
    lic_info = normalize_spdx_licenses(item.get("licenses", []))
    item.update(lic_info)
    return item


def load_catalog(cache_path: Path | str | None = None) -> dict[str, Any]:
    """Load and normalize the OpenWALDO catalog cache."""
    p = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not p.is_file():
        return {"repo": "openwaldo/waldo-index", "count": 0, "corpora": []}
    doc = json.loads(p.read_text(encoding="utf-8"))
    raw_corpora = doc.get("corpora", []) if isinstance(doc, dict) else (doc if isinstance(doc, list) else [])
    enriched = [_enrich_corpus(c) for c in raw_corpora if isinstance(c, dict)]
    return {
        "repo": doc.get("repo", "openwaldo/waldo-index") if isinstance(doc, dict) else "openwaldo/waldo-index",
        "synced_at": doc.get("synced_at", "") if isinstance(doc, dict) else "",
        "count": len(enriched),
        "corpora": enriched,
    }


def starter_presets(root: Path | str | None = None) -> list[dict[str, Any]]:
    r = repo_root(root)
    readme = [r / "README.md"] if (r / "README.md").is_file() else []
    docs = sorted((r / "docs").glob("*.md")) if (r / "docs").is_dir() else []
    native_om = sorted((r / "native").glob("*.om")) if (r / "native").is_dir() else []
    univ_om = sorted((r / "native" / "univ").glob("*.om")) if (r / "native" / "univ").is_dir() else []

    def _size(files: list[Path]) -> int:
        return sum(f.stat().st_size for f in files if f.is_file())

    starter_files = readme + docs + native_om
    docs_files = readme + docs
    code_files = native_om + univ_om
    return [
        {
            "id": "omega-starter",
            "name": "Omega Helix Starter Corpus",
            "description": "Local repo README.md, docs/*.md, and native/*.om packaged as a verified Apache-2.0 OpenWALDO export.",
            "files_count": len(starter_files),
            "bytes": _size(starter_files),
            "license": "Apache-2.0",
        },
        {
            "id": "omega-docs",
            "name": "Omega Architecture Docs",
            "description": "Repository README.md and docs/*.md technical specifications.",
            "files_count": len(docs_files),
            "bytes": _size(docs_files),
            "license": "Apache-2.0",
        },
        {
            "id": "omega-native",
            "name": "Pure-Omega (.om) Engine Sources",
            "description": "Pure seeded OM-2 sources from native/*.om and native/univ/*.om (great for code specialist strands).",
            "files_count": len(code_files),
            "bytes": _size(code_files),
            "license": "Apache-2.0",
        },
    ]


def _compute_facets(corpora: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cat_c: collections.Counter[str] = collections.Counter()
    lic_c: collections.Counter[str] = collections.Counter()
    prog_c: collections.Counter[str] = collections.Counter()
    lang_c: collections.Counter[str] = collections.Counter()

    for c in corpora:
        cat = c.get("category")
        if cat:
            cat_c[cat] += 1
        for t in c.get("license_tags", []):
            lic_c[t] += 1
        for p in c.get("programming_languages", []):
            if p:
                prog_c[p] += 1
        for l in c.get("languages", []):
            if l:
                lang_c[l] += 1

    def _fmt(counter: collections.Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
        rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        if limit is not None:
            rows = rows[:limit]
        return [{"name": k, "count": v} for k, v in rows]

    return {
        "categories": _fmt(cat_c),
        "licenses": _fmt(lic_c, limit=40),
        "prog_languages": _fmt(prog_c, limit=40),
        "languages": _fmt(lang_c, limit=30),
    }


def search_catalog(
    q: str = "",
    category: str = "",
    license: str = "",
    prog: str = "",
    lang: str = "",
    include_local: bool = True,
    cache_path: Path | str | None = None,
) -> dict[str, Any]:
    """Search and filter the OpenWALDO catalog with relevance ranking and facets."""
    cat_doc = load_catalog(cache_path)
    all_corpora = cat_doc["corpora"]
    facets = _compute_facets(all_corpora)

    q_norm = (q or "").strip().lower()
    q_words = [w for w in re.split(r"\s+", q_norm) if w]
    cat_norm = (category or "").strip().lower()
    lic_norm = (license or "").strip().lower()
    prog_norm = (prog or "").strip().lower()
    lang_norm = (lang or "").strip().lower()

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for c in all_corpora:
        if cat_norm and cat_norm != "all" and c.get("category", "").lower() != cat_norm:
            continue
        if lic_norm:
            tag_match = any(t.lower() == lic_norm or lic_norm in t.lower() for t in c.get("license_tags", []))
            raw_match = any(lic_norm in r.lower() for r in c.get("licenses", []))
            if not (tag_match or raw_match):
                continue
        if prog_norm:
            if not any(p.lower() == prog_norm or prog_norm in p.lower() for p in c.get("programming_languages", [])):
                continue
        if lang_norm:
            if not any(l.lower() == lang_norm for l in c.get("languages", [])):
                continue

        score = 0
        name_l = c.get("name", "").lower()
        path_l = c.get("path", "").lower()
        title_l = c.get("title", "").lower()
        desc_l = c.get("description", "").lower()
        cat_l = c.get("category", "").lower()
        progs_l = " ".join(c.get("programming_languages", [])).lower()
        lics_l = " ".join(c.get("license_tags", [])).lower()
        blob = f"{name_l} {path_l} {title_l} {desc_l} {cat_l} {progs_l} {lics_l}"

        if q_norm:
            if q_norm == name_l or q_norm == path_l:
                score += 120
            elif name_l.startswith(q_norm) or title_l.startswith(q_norm):
                score += 75
            elif q_norm in name_l or q_norm in title_l:
                score += 50
            elif q_norm in path_l or q_norm in progs_l or q_norm in lics_l:
                score += 30
            elif q_norm in desc_l:
                score += 18

            word_hits = 0
            for w in q_words:
                if w in name_l or w in title_l:
                    score += 20
                    word_hits += 1
                elif w in blob:
                    score += 10
                    word_hits += 1
            if word_hits < len(q_words):
                continue
        else:
            score = 1

        scored.append((score, name_l, c))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results = [item[2] for item in scored]

    out: dict[str, Any] = {
        "ok": True,
        "repo": cat_doc.get("repo", "openwaldo/waldo-index"),
        "synced_at": cat_doc.get("synced_at", ""),
        "total": len(all_corpora),
        "count": len(results),
        "corpora": results,
        "facets": facets,
    }
    if include_local:
        out["local_presets"] = starter_presets()
    return out


# --------------------------------------------------------------------------- sync


def _parse_simple_yaml_manifest(text: str, rel_manifest: str) -> dict[str, Any]:
    """Minimal zero-dep YAML parser tailored to openwaldo/waldo-index manifests."""
    data: dict[str, Any] = {
        "languages": [],
        "programming_languages": [],
        "licenses": [],
        "sources_count": 0,
    }
    cur_list_key: str | None = None
    in_sources = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            cur_list_key = None
            in_sources = False
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("name", "title", "description", "category"):
                    data[k] = v
                elif k in ("languages", "programming_languages", "licenses"):
                    if v.startswith("[") and v.endswith("]"):
                        inner = v[1:-1].strip()
                        if inner:
                            data[k] = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
                    else:
                        cur_list_key = k
                elif k == "sources":
                    in_sources = True
        else:
            if cur_list_key and stripped.startswith("- "):
                val = stripped[2:].strip().strip('"').strip("'")
                if val:
                    data[cur_list_key].append(val)
            elif in_sources:
                if stripped.startswith("- ") and indent <= 4:
                    data["sources_count"] += 1
                elif stripped.startswith("license:") or stripped.startswith("spdx:"):
                    _, _, lv = stripped.partition(":")
                    lv = lv.strip().strip('"').strip("'")
                    if lv and lv not in data["licenses"]:
                        data["licenses"].append(lv)

    corpus_dir = str(Path(rel_manifest).parent).replace("\\", "/")
    parts = corpus_dir.split("/")
    category = data.get("category") or (parts[0] if len(parts) >= 2 else "other")
    name = data.get("name") or (parts[-1] if parts else Path(rel_manifest).stem)
    return _enrich_corpus({
        "path": corpus_dir,
        "manifest": rel_manifest,
        "category": category,
        "name": name,
        "title": data.get("title") or name,
        "description": data.get("description") or "",
        "languages": data["languages"],
        "programming_languages": data["programming_languages"],
        "licenses": sorted(set(data["licenses"])),
        "sources_count": data["sources_count"],
        "url": f"https://github.com/openwaldo/waldo-index/blob/main/{corpus_dir}",
    })


def sync_catalog(cache_path: Path | str | None = None, timeout: int = 12) -> dict[str, Any]:
    """Refresh catalog via `waldo index list --json` (if installed) and GitHub waldo-index."""
    target = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    existing = load_catalog(target)
    by_path: dict[str, dict[str, Any]] = {c["path"]: c for c in existing["corpora"]}

    waldo_cli = shutil.which("waldo")
    waldo_cli_used = False
    synced_from: list[str] = []
    errors: list[str] = []

    if waldo_cli:
        try:
            proc = subprocess.run(
                [waldo_cli, "index", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parsed = json.loads(proc.stdout)
                items = parsed if isinstance(parsed, list) else parsed.get("corpora", [])
                for row in items:
                    if isinstance(row, dict) and (row.get("path") or row.get("name")):
                        enr = _enrich_corpus(row)
                        by_path[enr["path"]] = enr
                waldo_cli_used = True
                synced_from.append("waldo-cli")
            else:
                errors.append(f"waldo cli rc={proc.returncode}")
        except Exception as exc:
            errors.append(f"waldo cli: {exc}")

    # Fetch tree metadata from GitHub openwaldo/waldo-index
    headers = {"User-Agent": "omega-helix-studio/0.1 (stdlib; no-telemetry)", "Accept": "application/vnd.github+json"}
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/openwaldo/waldo-index/git/trees/main?recursive=1",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tree_doc = json.loads(resp.read().decode("utf-8"))
        yaml_paths = [
            e["path"]
            for e in tree_doc.get("tree", [])
            if e.get("type") == "blob"
            and e.get("path", "").endswith(".yaml")
            and "/" in e.get("path", "")
            and not e.get("path", "").startswith(".")
        ]
        fetched = 0
        for rel in yaml_paths:
            corpus_dir = str(Path(rel).parent).replace("\\", "/")
            # Fetch new manifests or refresh up to 25 per sync to stay polite to GitHub rate limits
            if corpus_dir not in by_path or fetched < 25:
                raw_url = f"https://raw.githubusercontent.com/openwaldo/waldo-index/main/{rel}"
                try:
                    r2 = urllib.request.Request(raw_url, headers={"User-Agent": headers["User-Agent"]})
                    with urllib.request.urlopen(r2, timeout=timeout) as resp2:
                        yml = resp2.read().decode("utf-8", errors="replace")
                    parsed_c = _parse_simple_yaml_manifest(yml, rel)
                    # Preserve richer cached metadata if remote parser had empty fields
                    if corpus_dir in by_path:
                        prev = by_path[corpus_dir]
                        for k in ("title", "description", "languages", "programming_languages", "licenses"):
                            if not parsed_c.get(k) and prev.get(k):
                                parsed_c[k] = prev[k]
                        parsed_c = _enrich_corpus(parsed_c)
                    by_path[corpus_dir] = parsed_c
                    fetched += 1
                except Exception:
                    continue
        if yaml_paths:
            synced_from.append(f"github:openwaldo/waldo-index({len(yaml_paths)} manifests)")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"github sync: {exc}")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    corpora_list = sorted(by_path.values(), key=lambda c: (c.get("category", ""), c.get("path", "")))

    if synced_from and corpora_list:
        clean_store = []
        for c in corpora_list:
            clean_store.append({
                "path": c["path"],
                "manifest": c.get("manifest", ""),
                "category": c.get("category", "other"),
                "name": c["name"],
                "title": c.get("title", c["name"]),
                "description": c.get("description", ""),
                "languages": c.get("languages", []),
                "programming_languages": c.get("programming_languages", []),
                "licenses": c.get("licenses", []),
                "sources_count": c.get("sources_count", 0),
                "url": c.get("url", ""),
            })
        payload = {
            "repo": "openwaldo/waldo-index",
            "synced_at": now,
            "count": len(clean_store),
            "corpora": clean_store,
        }
        tmp = target.with_name("." + target.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        tmp.replace(target)

    return {
        "ok": bool(synced_from) or len(corpora_list) > 0,
        "total": len(corpora_list),
        "synced_from": synced_from or ["local-cache"],
        "waldo_cli_used": waldo_cli_used,
        "updated_at": now,
        "warnings": errors,
    }


# --------------------------------------------------------------------------- local dataset prep


def _collect_preset_files(preset: str, root: Path) -> list[tuple[Path, str]]:
    files: list[Path] = []
    if preset == "omega-starter":
        if (root / "README.md").is_file():
            files.append(root / "README.md")
        if (root / "docs").is_dir():
            files.extend(sorted((root / "docs").glob("*.md")))
        if (root / "native").is_dir():
            files.extend(sorted((root / "native").glob("*.om")))
    elif preset == "omega-docs":
        if (root / "README.md").is_file():
            files.append(root / "README.md")
        if (root / "docs").is_dir():
            files.extend(sorted((root / "docs").glob("*.md")))
    elif preset == "omega-native":
        if (root / "native").is_dir():
            files.extend(sorted((root / "native").glob("*.om")))
        if (root / "native" / "univ").is_dir():
            files.extend(sorted((root / "native" / "univ").glob("*.om")))
    else:
        raise ValueError(f"unknown dataset preset: {preset!r} (expected omega-starter, omega-docs, omega-native)")

    out: list[tuple[Path, str]] = []
    for f in files:
        if f.is_file() and f.stat().st_size > 0:
            out.append((f, str(f.relative_to(root)).replace("\\", "/")))
    if not out:
        raise ValueError(f"preset {preset!r} resolved 0 files under {root}")
    return out


def _collect_local_files(source: Path) -> list[tuple[Path, str]]:
    if source.is_file():
        return [(source, source.name)]
    if not source.is_dir():
        raise ValueError(f"dataset path does not exist: {source}")
    out: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            p = Path(dirpath) / fname
            if p.suffix.lower() in SUPPORTED_TEXT_EXTS and p.is_file() and p.stat().st_size > 0:
                rel = str(p.relative_to(source)).replace("\\", "/")
                out.append((p, rel))
    if not out:
        raise ValueError(f"no supported text/code files found in {source}")
    return out


def _chunk_text(text: str, max_chars: int = 2048) -> list[str]:
    text = text.replace("\x00", "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if len(p) > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_len = [], 0
            for i in range(0, len(p), max_chars):
                sub = p[i : i + max_chars].strip()
                if sub:
                    chunks.append(sub)
        elif cur_len + len(p) + 2 > max_chars:
            if cur:
                chunks.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def inspect_dataset(
    path: str | None = None,
    preset: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect a preset, local file/folder, or existing OpenWALDO EXPORT.json directory."""
    r = repo_root(root)
    if preset:
        try:
            collected = _collect_preset_files(preset, r)
            total_bytes = sum(p.stat().st_size for p, _ in collected)
            return {
                "ok": True,
                "kind": "preset",
                "preset": preset,
                "path": str(r),
                "name": preset,
                "total_files": len(collected),
                "total_bytes": total_bytes,
                "total_docs": len(collected),
                "licenses": ["Apache-2.0"],
                "sample_files": [{"path": rel, "bytes": p.stat().st_size} for p, rel in collected[:12]],
                "export_verified": True,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    if not path:
        return {"ok": False, "error": "provide either 'path' or 'preset'"}

    src = Path(path).expanduser().resolve()
    if not src.exists():
        return {"ok": False, "error": f"path not found: {src}"}

    # Check if this is an OpenWALDO EXPORT.json dir or file
    export_json = src / "EXPORT.json" if src.is_dir() else (src if src.name == "EXPORT.json" else None)
    if export_json and export_json.is_file():
        try:
            rep = ow_bridge.verify_export(export_json.parent, check_records=True)
            return {
                "ok": True,
                "kind": "export_dir",
                "path": str(export_json.parent),
                "name": export_json.parent.name,
                "format": rep["format"],
                "total_files": len(rep["files"]),
                "total_bytes": rep["total_bytes"],
                "total_docs": rep["total_docs"],
                "licenses": rep["licenses"] or ["Open-Source"],
                "export_json_sha256": rep["export_json_sha256"],
                "sample_files": [
                    {"path": f["path"], "bytes": f["bytes"], "docs": f["docs"], "sha256": f["sha256"]}
                    for f in rep["files"][:12]
                ],
                "export_verified": True,
            }
        except Exception as exc:
            return {"ok": False, "kind": "export_dir", "path": str(src), "export_verified": False, "error": str(exc)}

    try:
        collected = _collect_local_files(src)
        total_bytes = sum(p.stat().st_size for p, _ in collected)
        return {
            "ok": True,
            "kind": "local",
            "path": str(src),
            "name": src.name,
            "total_files": len(collected),
            "total_bytes": total_bytes,
            "total_docs": len(collected),
            "licenses": ["Apache-2.0"],
            "sample_files": [{"path": rel, "bytes": p.stat().st_size} for p, rel in collected[:12]],
            "export_verified": False,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def prepare_local_dataset(
    source_path_or_preset: str | Path,
    out_dir: str | Path,
    name: str = "custom",
    ollm_waldo: str | Path | None = None,
    license_str: str = "Apache-2.0",
    held_mod: int = 10,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Prepare a verified schema-1 OpenWALDO EXPORT.json + JSONL shard and verify via openwaldo_bridge (+ ollm-waldo)."""
    r = repo_root(root)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "custom").strip()).strip("-") or "custom"
    src_str = str(source_path_or_preset).strip()

    # Case 1: Existing OpenWALDO EXPORT.json directory
    src_path = Path(src_str).expanduser()
    if src_str not in ("omega-starter", "omega-docs", "omega-native"):
        if src_path.is_dir() and (src_path / "EXPORT.json").is_file():
            export_dir = src_path.resolve()
            report = ow_bridge.verify_export(export_dir, check_records=True)
            waldo_summary = _verify_with_ollm_waldo(export_dir, out / "_waldo_check", ollm_waldo, held_mod)
            return {
                "ok": True,
                "kind": "export_dir",
                "export_dir": str(export_dir),
                "export_json": str(export_dir / "EXPORT.json"),
                "report": report,
                "waldo_summary": waldo_summary,
                "total_docs": report["total_docs"],
                "total_bytes": report["total_bytes"],
                "licenses": report["licenses"],
            }
        if src_path.is_file() and src_path.name == "EXPORT.json":
            export_dir = src_path.parent.resolve()
            report = ow_bridge.verify_export(export_dir, check_records=True)
            waldo_summary = _verify_with_ollm_waldo(export_dir, out / "_waldo_check", ollm_waldo, held_mod)
            return {
                "ok": True,
                "kind": "export_dir",
                "export_dir": str(export_dir),
                "export_json": str(export_dir / "EXPORT.json"),
                "report": report,
                "waldo_summary": waldo_summary,
                "total_docs": report["total_docs"],
                "total_bytes": report["total_bytes"],
                "licenses": report["licenses"],
            }

    # Case 2: Preset or raw local files/folder -> build deterministic schema-1 OpenWALDO export
    if src_str in ("omega-starter", "omega-docs", "omega-native"):
        collected = _collect_preset_files(src_str, r)
        corpus_id = f"local/{src_str}"
        origin_kind = "preset"
    else:
        collected = _collect_local_files(src_path.resolve())
        corpus_id = f"local/{safe_name}"
        origin_kind = "local"

    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shard_rel = f"data/{safe_name}-00000.jsonl"
    shard_path = out / shard_rel

    records: list[dict[str, Any]] = []
    for fpath, rel in collected:
        if fpath.suffix.lower() == ".jsonl":
            raw_text = fpath.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(raw_text.splitlines(), start=1):
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
                        txt = obj["text"].replace("\x00", "")
                        rec_lic = str(obj.get("license") or license_str)
                        rec_src = str(obj.get("source") or f"{rel}#L{lineno}")
                        rec_sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                        records.append({
                            "sha256": rec_sha,
                            "kind": "pretrain",
                            "text": txt,
                            "source": rec_src,
                            "license": rec_lic,
                            "tokens": max(1, len(txt) // 4),
                        })
                        continue
                except json.JSONDecodeError:
                    pass
                # Fallback: treat raw line as text
                txt = line_s.replace("\x00", "")
                if txt:
                    records.append({
                        "sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                        "kind": "pretrain",
                        "text": txt,
                        "source": f"{rel}#L{lineno}",
                        "license": license_str,
                        "tokens": max(1, len(txt) // 4),
                    })
        else:
            raw_text = fpath.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk_text(raw_text, max_chars=2048)
            for idx, ch in enumerate(chunks):
                records.append({
                    "sha256": hashlib.sha256(ch.encode("utf-8")).hexdigest(),
                    "kind": "pretrain",
                    "text": ch,
                    "source": f"{rel}#chunk-{idx}",
                    "license": license_str,
                    "tokens": max(1, len(ch) // 4),
                })

    if not records:
        raise ValueError(f"no non-empty records extracted from {source_path_or_preset}")

    # Ensure deterministic line-delimited canonical JSONL
    lines = [json.dumps(rec, ensure_ascii=False, separators=(",", ":")) for rec in records]
    shard_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    shard_path.write_bytes(shard_bytes)

    shard_sha = hashlib.sha256(shard_bytes).hexdigest()
    shard_len = len(shard_bytes)
    unique_licenses = sorted({r["license"] for r in records if r.get("license")})
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_name = f"{corpus_id}/{safe_name}.yaml"

    export_doc = {
        "kind": "waldo-corpus-export",
        "schema": 1,
        "format": "jsonl",
        "generated": now,
        "files": [
            {
                "path": shard_rel,
                "manifest": manifest_name,
                "format": "jsonl",
                "object_sha256": shard_sha,
                "object_bytes": shard_len,
                "sha256": shard_sha,
                "bytes": shard_len,
                "docs": len(records),
                "license": unique_licenses[0] if unique_licenses else license_str,
                "licenses": unique_licenses or [license_str],
            }
        ],
        "bom": {
            "kind": "openwaldo-bom",
            "schema": 1,
            "index": {
                "repo": "local/omega-helix-studio",
                "origin": origin_kind,
                "source": str(source_path_or_preset),
            },
            "paths": [corpus_id],
            "shards": [
                {
                    "manifest": manifest_name,
                    "sha256": shard_sha,
                    "bytes": shard_len,
                }
            ],
        },
    }
    export_json = out / "EXPORT.json"
    tmp_json = out / ".EXPORT.json.tmp"
    tmp_json.write_text(json.dumps(export_doc, indent=2) + "\n", encoding="utf-8")
    tmp_json.replace(export_json)

    # Fail-closed verification via openwaldo_bridge
    report = ow_bridge.verify_export(out, check_records=True)
    waldo_summary = _verify_with_ollm_waldo(out, out / "_waldo_check", ollm_waldo, held_mod)

    return {
        "ok": True,
        "kind": origin_kind,
        "export_dir": str(out),
        "export_json": str(export_json),
        "report": report,
        "waldo_summary": waldo_summary,
        "total_docs": report["total_docs"],
        "total_bytes": report["total_bytes"],
        "licenses": report["licenses"],
    }


def _verify_with_ollm_waldo(
    export_dir: Path,
    check_dir: Path,
    ollm_waldo: str | Path | None,
    held_mod: int = 10,
) -> str:
    """If an executable ollm-waldo binary is supplied, verify ingestion end-to-end."""
    if not ollm_waldo:
        return ""
    wb = Path(ollm_waldo).resolve()
    if not wb.is_file() or not os.access(wb, os.X_OK):
        raise ValueError(f"ollm-waldo not executable: {wb}")
    check_dir.mkdir(parents=True, exist_ok=True)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ow_bridge.main([
                "prepare", str(export_dir), str(check_dir),
                "--ollm-waldo", str(wb), "--held-mod", str(held_mod),
            ])
        if rc != 0:
            raise RuntimeError(f"openwaldo_bridge prepare exited {rc}")
        bom = json.loads((check_dir / "openwaldo-helix-bom.json").read_text(encoding="utf-8"))
        return str(bom.get("ingestor", {}).get("summary", ""))
    finally:
        shutil.rmtree(check_dir, ignore_errors=True)


def materialize_waldo(corpus_path: str, out_dir: str | Path) -> dict[str, Any]:
    """Export an OpenWALDO waldo-index corpus to verified canonical JSONL via `waldo` CLI."""
    out = Path(out_dir).resolve()
    waldo_cli = shutil.which("waldo")
    if not waldo_cli:
        return {
            "ok": False,
            "installed": False,
            "corpus_path": corpus_path,
            "error": "OpenWALDO CLI ('waldo') is not installed on PATH.",
            "guidance": [
                "Install the OpenWALDO CLI (https://github.com/openwaldo/waldo) to pull public corpora directly.",
                f"Then run: waldo index export --format jsonl {corpus_path} {out}",
                "Alternatively, point Omega Helix Studio at an existing EXPORT.json directory, any local folder, or the built-in 'omega-starter' preset.",
            ],
            "commands": [
                f"waldo index export --format jsonl {corpus_path} {out}",
            ],
        }

    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [waldo_cli, "index", "export", "--format", "jsonl", corpus_path, str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout or "").strip()
        needs_lookaside = any(k in err_text.lower() for k in ("lookaside", "mirror", "cache", "auth", "token", "offline"))
        guidance = [
            f"`waldo index export --format jsonl {corpus_path} {out}` exited {proc.returncode}.",
        ]
        if needs_lookaside:
            guidance.append("This corpus may require a configured OpenWALDO lookaside store or local cache mirror.")
        guidance.append("Verify corpus name with `waldo index list` or select a local folder / 'omega-starter' preset.")
        return {
            "ok": False,
            "installed": True,
            "corpus_path": corpus_path,
            "returncode": proc.returncode,
            "needs_lookaside": needs_lookaside,
            "error": err_text or f"waldo exited {proc.returncode}",
            "guidance": guidance,
        }

    try:
        report = ow_bridge.verify_export(out, check_records=True)
    except Exception as exc:
        return {
            "ok": False,
            "installed": True,
            "corpus_path": corpus_path,
            "error": f"exported corpus failed fail-closed verification: {exc}",
            "guidance": ["Ensure the export completed in canonical JSONL format (`--format jsonl`)."],
        }

    return {
        "ok": True,
        "installed": True,
        "corpus_path": corpus_path,
        "export_dir": str(out),
        "report": report,
        "total_docs": report["total_docs"],
        "total_bytes": report["total_bytes"],
        "licenses": report["licenses"],
    }
