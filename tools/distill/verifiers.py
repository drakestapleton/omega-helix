"""Fail-closed verifiers for distilled teacher outputs and harvested traces.

Built-in verifiers:
- ``nonempty``: minimum stripped character count
- ``json_schema``: valid JSON object/array + optional required top-level keys
- ``python_ast``: Python ``ast.parse`` check (NEVER exec/eval)
- ``omc_compile``: seeded ``omc`` compile-only check (stdin -> DEVNULL, NEVER executes)
- ``regex``: regular expression match against text

Python 3 standard library only — zero external packages.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

BUILTIN_VERIFIERS = ("nonempty", "json_schema", "python_ast", "omc_compile", "regex")

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*\n(.*?)```", re.DOTALL)


def find_omc(explicit: str | Path | None = None) -> Path | None:
    cands: list[str | Path] = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("OMC"):
        cands.append(os.environ["OMC"])
    w = shutil.which("omc")
    if w:
        cands.append(w)
    root = Path(__file__).resolve().parents[2]
    cands.extend([
        Path.home() / ".local" / "bin" / "omc",
        Path.home() / "canon" / "omega-mv" / "seed" / "out" / "omc",
        root.parent / "omega-mv" / "seed" / "out" / "omc",
        root.parent / "omega-mv-macos" / "seed" / "out" / "omc",
    ])
    for c in cands:
        p = Path(c).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    return None


def extract_fenced_block(text: str, langs: tuple[str, ...]) -> str:
    """Return first matching markdown fenced code block, or the raw stripped text."""
    wanted = {x.lower() for x in langs}
    for m in _FENCE_RE.finditer(text or ""):
        tag = (m.group(1) or "").strip().lower()
        body = m.group(2)
        if not wanted or tag in wanted:
            return body.strip()
    return (text or "").strip()


def _check_nonempty(text: str, min_chars: int = 1) -> tuple[bool, str]:
    s = (text or "").strip()
    ok = len(s) >= max(1, int(min_chars))
    return ok, f"chars={len(s)}"


def _check_json_schema(text: str, required_keys: list[str] | None = None) -> tuple[bool, str]:
    candidate = extract_fenced_block(text, ("json", ""))
    try:
        obj = json.loads(candidate)
    except Exception as exc:
        return False, f"json parse error: {exc}"
    if not isinstance(obj, (dict, list)):
        return False, f"expected JSON object/array, got {type(obj).__name__}"
    if required_keys:
        if not isinstance(obj, dict):
            return False, "required_keys requires a JSON object"
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return False, f"missing keys: {','.join(missing)}"
    return True, f"valid json ({type(obj).__name__})"


def _check_python_ast(text: str) -> tuple[bool, str]:
    candidate = extract_fenced_block(text, ("python", "py", ""))
    if not candidate:
        return False, "empty python snippet"
    try:
        tree = ast.parse(candidate, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error line {exc.lineno}: {exc.msg}"
    return True, f"ast ok ({len(tree.body)} statements)"


def _check_omc_compile(text: str, omc_path: Path | None = None) -> tuple[bool, str]:
    compiler = Path(omc_path) if omc_path else find_omc()
    if compiler is None or not compiler.is_file():
        return False, "omc compiler not found"
    src = extract_fenced_block(text, ("om", "omega", ""))
    if not src:
        return False, "empty .om source"
    if platform.system() == "Darwin" and not src.lstrip().startswith("target "):
        src = f"target a64macos\n{src}"
    if not src.endswith("\n"):
        src += "\n"
    try:
        res = subprocess.run(
            [str(compiler)],
            input=src.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "omc compile timeout"
    except OSError as exc:
        return False, f"omc invocation error: {exc}"
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        msg = err[-1] if err else f"exit {res.returncode}"
        return False, f"omc error: {msg}"
    return True, f"omc compile ok ({compiler.name})"


def _check_regex(text: str, pattern: str | None = None) -> tuple[bool, str]:
    pat = pattern or r".+"
    try:
        cre = re.compile(pat, re.DOTALL)
    except re.error as exc:
        return False, f"invalid regex: {exc}"
    m = cre.search(text or "")
    if m is None:
        return False, f"pattern {pat!r} not matched"
    return True, f"matched ({len(m.group(0))} chars)"


def verify_record(
    text: str,
    verifiers: list[str] | tuple[str, ...],
    *,
    omc_path: Path | None = None,
    regex_pattern: str | None = None,
    json_required_keys: list[str] | None = None,
    min_chars: int = 1,
) -> dict[str, Any]:
    """Run requested verifiers against a single candidate response text."""
    vlist = [v.strip() for v in verifiers if str(v).strip()]
    if not vlist:
        vlist = ["nonempty"]

    checks: list[dict[str, Any]] = []
    passed_count = 0
    for name in vlist:
        if name == "nonempty":
            ok, detail = _check_nonempty(text, min_chars=min_chars)
        elif name == "json_schema":
            ok, detail = _check_json_schema(text, required_keys=json_required_keys)
        elif name == "python_ast":
            ok, detail = _check_python_ast(text)
        elif name == "omc_compile":
            ok, detail = _check_omc_compile(text, omc_path=omc_path)
        elif name == "regex":
            ok, detail = _check_regex(text, pattern=regex_pattern)
        else:
            raise ValueError(f"unsupported verifier {name!r}; expected one of {BUILTIN_VERIFIERS}")
        if ok:
            passed_count += 1
        checks.append({"name": name, "passed": ok, "detail": detail})

    total = len(checks)
    score = passed_count / total if total else 1.0
    all_passed = passed_count == total
    return {"passed": all_passed, "score": score, "checks": checks}


class VerifierPipeline:
    """Batch verification and aggregate telemetry for distilled trace corpora."""

    def __init__(
        self,
        verifiers: list[str] | tuple[str, ...] = ("nonempty",),
        *,
        omc_path: Path | None = None,
        regex_pattern: str | None = None,
        json_required_keys: list[str] | None = None,
        min_chars: int = 1,
        min_score: float = 1.0,
    ) -> None:
        self.verifiers = [str(v).strip() for v in verifiers if str(v).strip()] or ["nonempty"]
        for v in self.verifiers:
            if v not in BUILTIN_VERIFIERS:
                raise ValueError(f"unsupported verifier {v!r}")
        self.omc_path = Path(omc_path).resolve() if omc_path else find_omc()
        self.regex_pattern = regex_pattern
        self.json_required_keys = list(json_required_keys) if json_required_keys else None
        self.min_chars = max(1, int(min_chars))
        self.min_score = float(min_score)

    def check_one(self, text: str) -> dict[str, Any]:
        res = verify_record(
            text,
            self.verifiers,
            omc_path=self.omc_path,
            regex_pattern=self.regex_pattern,
            json_required_keys=self.json_required_keys,
            min_chars=self.min_chars,
        )
        res["accepted"] = bool(res["score"] >= self.min_score)
        return res

    def filter_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        per_verifier: dict[str, dict[str, int]] = {
            v: {"passed": 0, "failed": 0} for v in self.verifiers
        }

        for rec in records:
            target_text = str(rec.get("assistant_text") or rec.get("text") or "")
            ver = self.check_one(target_text)
            for chk in ver["checks"]:
                bucket = per_verifier.setdefault(chk["name"], {"passed": 0, "failed": 0})
                if chk["passed"]:
                    bucket["passed"] += 1
                else:
                    bucket["failed"] += 1
            enriched = dict(rec)
            enriched["verification"] = ver
            if ver["accepted"]:
                accepted.append(enriched)
            else:
                rejected.append(enriched)

        total = len(records)
        passed_n = len(accepted)
        stats = {
            "total": total,
            "passed": passed_n,
            "rejected": len(rejected),
            "pass_rate": round(passed_n / total, 4) if total else 0.0,
            "verifiers": list(self.verifiers),
            "per_verifier": per_verifier,
        }
        return {"accepted": accepted, "rejected": rejected, "stats": stats}
