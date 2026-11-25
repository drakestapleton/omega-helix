#!/usr/bin/env python3
"""Omega Helix Studio local HTTP & SSE server (Python 3 stdlib only, zero telemetry)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STUDIO_DIR = Path(__file__).resolve().parent
TOOLS_DIR = STUDIO_DIR.parent
DISTILL_DIR = TOOLS_DIR / "distill"
UI_DIR = STUDIO_DIR / "ui"

for _p in (str(STUDIO_DIR), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catalog as studio_catalog  # noqa: E402
import chain_runner as runner  # noqa: E402
from distill import harvest as distill_harvest  # noqa: E402
from distill import pipeline as distill_pipeline  # noqa: E402
from distill import teachers as distill_teachers  # noqa: E402
from distill import verifiers as distill_verifiers  # noqa: E402
import runtime as sovereign_runtime  # noqa: E402
import commons as sovereign_commons  # noqa: E402
import spiral as studio_spiral  # noqa: E402
import ssh_remote  # noqa: E402


_DEFAULT_OMG_SEED = (
    "om2\n"
    "fn scale x ulong k ulong to ulong\n"
    "  res = mul x k\n"
    "  res = add res 1\n"
    "  ret res\n"
    "end\n"
    "fn main to int\n"
    "  a = ulong 7\n"
    "  b = scale a 3\n"
    "  if gt b 10\n"
    "    ret int sub b 10\n"
    "  end\n"
    "  ret 0\n"
    "end\n"
)


_DEFAULT_OMG_SEEDS = [
    _DEFAULT_OMG_SEED,
    (
        "om2\n"
        "fn sum_to n ulong to ulong\n"
        "  acc = ulong 0\n"
        "  for i ulong n\n"
        "    acc = add acc i\n"
        "  end\n"
        "  ret acc\n"
        "end\n"
        "fn main to int\n"
        "  v = sum_to 10\n"
        "  ret int rem v 127\n"
        "end\n"
    ),
    (
        "om2\n"
        "fn clamp_u x ulong hi ulong to ulong\n"
        "  if gt x hi\n"
        "    ret hi\n"
        "  end\n"
        "  ret x\n"
        "end\n"
        "fn main to int\n"
        "  r = clamp_u 42 30\n"
        "  ret int r\n"
        "end\n"
    ),
]


def _build_arboris_omg_corpus(ollm: str, tdp: Path, seed_om: str | None = None) -> Path:
    seeds = [seed_om] if seed_om else _DEFAULT_OMG_SEEDS
    chunks: list[bytes] = []
    for idx, src in enumerate(seeds):
        in_p = tdp / f"seed_{idx}.om"
        tok_p = tdp / f"seed_{idx}.tok"
        in_p.write_text(src, encoding="utf-8")
        r_pk = subprocess.run([ollm, "omg", "pack", str(in_p), str(tok_p)], capture_output=True, text=True, check=False)
        if r_pk.returncode == 0 and tok_p.is_file():
            chunks.append(tok_p.read_bytes())
    if not chunks:
        raise ValueError("omg pack failed for Arboris corpus seeds")
    out_tok = tdp / "arboris_corpus.tok"
    out_tok.write_bytes(b"".join(chunks))
    return out_tok


def _find_omc_bin() -> str | None:
    for cand in (
        os.environ.get("OMC", ""),
        shutil.which("omc") or "",
        str(Path.home() / ".local" / "bin" / "omc"),
        str(Path.home() / "canon" / "omega-mv" / "seed" / "out" / "omc"),
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _resolve_v256_ck(run: Any) -> str | None:
    if run is None:
        return None
    cands: list[Path] = []
    if getattr(run, "run_dir", None):
        cands.append(Path(run.run_dir) / "base.ck")
    for art in getattr(run, "artifacts", []) or []:
        if isinstance(art, dict) and art.get("name") == "base.ck" and art.get("path"):
            cands.append(Path(art["path"]))
    for p in cands:
        if p.is_file() and p.stat().st_size >= 32:
            try:
                hdr = p.read_bytes()[:24]
                if hdr[:6] == b"OLLMCK":
                    v = int.from_bytes(hdr[16:24], "little")
                    if v == 256:
                        return str(p)
            except OSError:
                pass
    return None


def _compile_om_text(om_text: str) -> tuple[bool, int]:
    omc = _find_omc_bin()
    if not omc:
        return False, -1
    hdr = "target a64macos\n" if platform.system() == "Darwin" else ""
    p = subprocess.run([omc], input=(hdr + om_text).encode("utf-8"), capture_output=True, check=False)
    return p.returncode == 0, int(p.returncode)


def _run_eight_pillar_locks(bin_dir: Path | None, root: Path | None, steps: int = 8, lr_milli: int = 10) -> dict[str, Any]:
    bins = runner.ensure_binaries(["ollm"], bin_dir=bin_dir, root=root)
    ollm = str(bins["ollm"])
    r_omg = subprocess.run([ollm, "omg"], capture_output=True, text=True, check=False)
    r_arb = subprocess.run([ollm, "arboris"], capture_output=True, text=True, check=False)
    rev_supported = platform.system() == "Darwin" or platform.machine().lower() in ("aarch64", "arm64")
    rev_ok = True
    act_bytes = 0
    q32_bits = 0
    loss_init = None
    loss_final = None
    rev_out = ""
    if rev_supported:
        r_rev = subprocess.run([ollm, "rev", "train", str(int(steps)), str(int(lr_milli))], capture_output=True, text=True, check=False)
        rev_ok = r_rev.returncode == 0
        rev_out = r_rev.stdout.strip()
        for ln in rev_out.splitlines():
            if ln.startswith("REVTRAIN "):
                parts = ln.split()
                if len(parts) >= 6:
                    act_bytes = int(parts[2])
                    q32_bits = int(parts[3])
                    loss_init = int(parts[4]) / 1_000_000.0
                    loss_final = int(parts[5]) / 1_000_000.0
    omg_ok = r_omg.returncode == 0 and "OMG_SELFTEST PASS" in r_omg.stdout
    arb_ok = r_arb.returncode == 0 and "ARBORIS_SELFTEST PASS" in r_arb.stdout
    return {
        "ok": bool(omg_ok and arb_ok and rev_ok),
        "omg_pass": omg_ok,
        "arboris_pass": arb_ok,
        "rev_supported": rev_supported,
        "rev_pass": rev_ok,
        "act_bytes": act_bytes,
        "q32_inv_bits": q32_bits,
        "rev_loss_init": loss_init,
        "rev_loss_final": loss_final,
        "output": "\n".join(x for x in (r_omg.stdout.strip(), r_arb.stdout.strip(), rev_out) if x),
    }


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: runner.ChainManager,
        ui_dir: Path = UI_DIR,
        distill_manager: distill_pipeline.DistillManager | None = None,
        access_policy: ssh_remote.StudioAccessPolicy | None = None,
        tunnel_manager: ssh_remote.TunnelManager | None = None,
        instance_lock: ssh_remote.InstanceLock | None = None,
    ) -> None:
        self.manager = manager
        self.distill_manager = distill_manager or distill_pipeline.DistillManager(
            runs_dir=manager.runs_dir / "_distill",
            bin_dir=manager.bin_dir,
            root=manager.root,
        )
        self.ui_dir = ui_dir.resolve()
        self.access_policy = access_policy or ssh_remote.StudioAccessPolicy()
        self.tunnel_manager = tunnel_manager or ssh_remote.TunnelManager()
        self.instance_lock = instance_lock
        super().__init__(server_address, StudioHandler)

    def server_close(self) -> None:
        try:
            self.tunnel_manager.close_all()
        finally:
            if self.instance_lock is not None:
                self.instance_lock.release()
            super().server_close()


class StudioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: StudioServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet by default unless OMEGA_STUDIO_VERBOSE=1
        import os
        if os.environ.get("OMEGA_STUDIO_VERBOSE") == "1":
            super().log_message(format, *args)

    def _send_json(self, status: int, payload: Any, extra_headers: dict[str, str] | None = None) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _extract_pairing_token(self, qs: dict[str, list[str]]) -> tuple[str, bool]:
        auth = self.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth[7:].strip(), False
        qtok = (qs.get("pair") or [""])[0].strip()
        if qtok:
            return qtok, True
        raw_cookie = self.headers.get("Cookie") or ""
        for part in raw_cookie.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "omega_pair" and v:
                return v.strip(), False
        return "", False

    def _check_access(self, qs: dict[str, list[str]]) -> tuple[bool, str | None]:
        client_ip = str(self.client_address[0] if self.client_address else "")
        pol = self.server.access_policy
        allowed, reason = pol.ip_allowed(client_ip)
        if not allowed:
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": f"origin rejected ({reason})"})
            return False, None
        tok, from_query = self._extract_pairing_token(qs)
        if pol.needs_token(client_ip):
            if not pol.verify_token(tok):
                self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "pairing token required"})
                return False, None
            if from_query:
                return True, tok
        elif from_query and pol.verify_token(tok):
            return True, tok
        return True, None

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if not raw_len:
            return {}
        try:
            length = int(raw_len)
        except ValueError:
            raise ValueError("invalid Content-Length")
        if length > 8 * 1024 * 1024:
            raise ValueError("request body exceeds 8 MB limit")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw.strip():
            return {}
        doc = json.loads(raw.decode("utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("JSON body must be an object")
        return doc

    def _serve_static(self, req_path: str) -> None:
        rel = "index.html" if req_path in ("", "/") else req_path.lstrip("/")
        target = (self.server.ui_dir / rel).resolve()
        try:
            target.relative_to(self.server.ui_dir)
        except ValueError:
            self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "forbidden"})
            return
        if not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"not found: {req_path}"})
            return
        data = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        if target.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif target.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif target.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif target.suffix == ".svg":
            ctype = "image/svg+xml"
        else:
            ctype = ctype or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        ok_acc, set_pair_tok = self._check_access(qs)
        if not ok_acc:
            return
        pair_hdrs = {"Set-Cookie": f"omega_pair={set_pair_tok}; HttpOnly; SameSite=Lax; Path=/"} if set_pair_tok else None

        if path == "/api/system":
            sys_info = runner.detect_system(bin_dir=self.server.manager.bin_dir, runs_dir=self.server.manager.runs_dir)
            sys_info["bind_port"] = self.server.server_address[1]
            sys_info["pairing_required"] = self.server.access_policy.require_pairing
            sys_info["no_pairing"] = self.server.access_policy.no_pairing
            self._send_json(HTTPStatus.OK, sys_info, extra_headers=pair_hdrs)
            return

        if path == "/api/remotes":
            addr = self.server.access_policy.pairing_address or f"127.0.0.1:{self.server.server_address[1]}"
            code = self.server.access_policy.pairing_code
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                "ssh_hosts": ssh_remote.discover_ssh_hosts(),
                "remotes": ssh_remote.load_remotes(),
                "active_tunnels": self.server.tunnel_manager.list_tunnels(),
                "server_pairing": {
                    "pairing_address": addr,
                    "pairing_url": f"http://{addr}/?pair={code}" if code else f"http://{addr}/",
                    "ssh_forward_hint": f"ssh -N -L 18421:127.0.0.1:{self.server.server_address[1]} <ssh-host>",
                    "headless_cmd": f"omega-helix studio --headless --bind 127.0.0.1 --port {self.server.server_address[1]} --json",
                },
            }, extra_headers=pair_hdrs)
            return

        if path == "/api/catalog":
            q = (qs.get("q") or [""])[0]
            cat = (qs.get("cat") or qs.get("category") or [""])[0]
            lic = (qs.get("lic") or qs.get("license") or [""])[0]
            prog = (qs.get("prog") or [""])[0]
            lang = (qs.get("lang") or [""])[0]
            res = studio_catalog.search_catalog(q=q, category=cat, license=lic, prog=prog, lang=lang)
            self._send_json(HTTPStatus.OK, res)
            return

        if path == "/api/chains":
            self._send_json(HTTPStatus.OK, {"ok": True, "runs": self.server.manager.list_runs()})
            return

        if path.startswith("/api/chains/"):
            rest = path[len("/api/chains/"):].strip("/")
            parts = rest.split("/")
            run_id = urllib.parse.unquote(parts[0])
            run = self.server.manager.get_run(run_id)
            if run is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"run not found: {run_id}"})
                return
            if len(parts) == 1:
                self._send_json(HTTPStatus.OK, {"ok": True, "run": run.to_dict()})
                return
            if len(parts) == 2 and parts[1] == "events":
                self._stream_sse(run)
                return

        if path == "/api/distill/teachers/probe":
            eps = distill_teachers.probe_local_teachers()
            self._send_json(HTTPStatus.OK, {"ok": True, "endpoints": eps})
            return

        if path == "/api/distill/runs":
            self._send_json(HTTPStatus.OK, {"ok": True, "runs": self.server.distill_manager.list_runs()})
            return

        if path.startswith("/api/distill/runs/"):
            rest = path[len("/api/distill/runs/"):].strip("/")
            parts = rest.split("/")
            run_id = urllib.parse.unquote(parts[0])
            drun = self.server.distill_manager.get_run(run_id)
            if drun is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"distill run not found: {run_id}"})
                return
            if len(parts) == 1:
                self._send_json(HTTPStatus.OK, {"ok": True, "run": drun.to_dict()})
                return
            if len(parts) == 2 and parts[1] == "events":
                self._stream_sse(drun)
                return

        if path == "/api/spiral/presets":
            self._send_json(HTTPStatus.OK, studio_spiral.list_presets_and_pillars())
            return

        if path == "/api/runtime/locks":
            try:
                res = _run_eight_pillar_locks(self.server.manager.bin_dir, self.server.manager.root)
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/trust/verify":
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                base_ck = (q.get("base_ck") or [None])[0]
                bom_path = (q.get("bom") or [None])[0]
                packs = q.get("pack") or None
                res = sovereign_commons.verify_sovereign_trust(
                    bin_dir=self.server.manager.bin_dir,
                    root=self.server.manager.root,
                    base_ck=base_ck,
                    packs=packs,
                    bom_path=bom_path,
                )
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if not path.startswith("/api/"):
            self._serve_static(path)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown endpoint: {path}"})

    def _stream_sse(self, run: runner.ChainRun) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in run.iter_events(start_seq=0, poll_timeout=8.0):
                if ev is None:
                    self.wfile.write(b": keepalive\n\n")
                else:
                    line = f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        ok_acc, _ = self._check_access(qs)
        if not ok_acc:
            return

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid JSON body: {exc}"})
            return

        if path == "/api/catalog/sync":
            res = studio_catalog.sync_catalog()
            self._send_json(HTTPStatus.OK, res)
            return

        if path == "/api/dataset/inspect":
            res = studio_catalog.inspect_dataset(
                path=body.get("path"),
                preset=body.get("preset"),
                root=self.server.manager.root,
            )
            status = HTTPStatus.OK if res.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(status, res)
            return

        if path == "/api/chains/estimate":
            est = runner.estimate_chain(body)
            self._send_json(HTTPStatus.OK, est)
            return

        if path == "/api/spiral/plan":
            try:
                plan = studio_spiral.plan_spiral(body)
                self._send_json(HTTPStatus.OK, plan)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/spiral/cryst":
            try:
                if not body.get("base_ck") or not body.get("domain_tok") or not body.get("out_sp2"):
                    raise ValueError("base_ck, domain_tok, and out_sp2 are required")
                all_bins = runner.resolve_binaries(bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                use_gb10 = bool(body.get("use_gpu", True) and os.path.exists("/dev/nvidiactl") and all_bins.get("ollm_gb10"))
                ollm_bin = all_bins["ollm_gb10"] if use_gb10 else runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)["ollm"]
                assert ollm_bin is not None
                res = studio_spiral.run_spcryst(
                    ollm_bin,
                    str(body["base_ck"]),
                    str(body["domain_tok"]),
                    str(body["out_sp2"]),
                    f_exp=int(body.get("f_exp", 256)),
                    lambda_milli=int(body.get("lambda_milli", 1000)),
                    max_toks=int(body.get("max_toks", 0)),
                    strand_id=int(body.get("strand_id", 1)),
                    quant_bits=int(body.get("quant_bits", 4)),
                )
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/spiral/eval":
            try:
                strands = body.get("strands") or []
                if isinstance(strands, str):
                    strands = [s.strip() for s in strands.splitlines() if s.strip()]
                if not body.get("base_ck") or not body.get("eval_tok") or not strands:
                    raise ValueError("base_ck, eval_tok, and strands[] are required")
                all_bins = runner.resolve_binaries(bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                use_gb10 = bool(body.get("use_gpu", True) and os.path.exists("/dev/nvidiactl") and all_bins.get("ollm_gb10"))
                ollm_bin = all_bins["ollm_gb10"] if use_gb10 else runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)["ollm"]
                assert ollm_bin is not None
                res = studio_spiral.run_spweval(
                    ollm_bin,
                    str(body["base_ck"]),
                    str(body["eval_tok"]),
                    strands,
                    top_k=int(body.get("top_k", 4)),
                    windows=int(body.get("windows", 8)),
                )
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/spiral/sample":
            try:
                strands = body.get("strands") or []
                if isinstance(strands, str):
                    strands = [s.strip() for s in strands.splitlines() if s.strip()]
                if not body.get("base_ck") or not strands:
                    raise ValueError("base_ck and strands[] are required")
                all_bins = runner.resolve_binaries(bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                use_gb10 = bool(body.get("use_gpu", True) and os.path.exists("/dev/nvidiactl") and all_bins.get("ollm_gb10"))
                ollm_bin = all_bins["ollm_gb10"] if use_gb10 else runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)["ollm"]
                assert ollm_bin is not None
                res = studio_spiral.run_spiral_sample(
                    ollm_bin,
                    str(body["base_ck"]),
                    strands,
                    prompt=str(body.get("prompt", "fn main")),
                    max_tok=int(body.get("max_tok") or body.get("n_tokens") or 64),
                    top_k=int(body.get("top_k", 4)),
                    temp_milli=int(body.get("temp_milli", 700)),
                    subcmd=str(body.get("subcmd") or "spsamp"),
                )
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/chains/start":
            try:
                run = self.server.manager.start_chain(body, wait=False)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "run_id": run.run_id, "run": run.summary()})
            return

        if path.startswith("/api/chains/"):
            rest = path[len("/api/chains/"):].strip("/")
            parts = rest.split("/")
            if len(parts) == 2:
                run_id = urllib.parse.unquote(parts[0])
                action = parts[1]
                if action == "stop":
                    stopped = self.server.manager.stop_chain(run_id)
                    self._send_json(HTTPStatus.OK, {"ok": True, "stopped": stopped})
                    return
                if action == "sample":
                    try:
                        res = self.server.manager.sample_chain(
                            run_id=run_id,
                            prompt=str(body.get("prompt", "Omega ")),
                            n_tokens=int(body.get("n_tokens", 48)),
                            temp_milli=int(body.get("temp_milli", 800)),
                            seed=int(body.get("seed", 42)),
                            spclip_mode=int(body.get("spclip_mode", 1)),
                            strand_weights=body.get("strand_weights") or body.get("weights"),
                            adapt_thresh_micro=int(body.get("adapt_thresh_micro", 0)),
                            msg_gain_micro=int(body.get("msg_gain_micro", 0)),
                        )
                        self._send_json(HTTPStatus.OK, res)
                    except ValueError as exc:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    except Exception as exc:
                        self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                    return

        if path == "/api/distill/harvest":
            try:
                inputs = body.get("inputs") or body.get("trace_paths") or []
                if isinstance(inputs, str):
                    inputs = [inputs]
                verifiers = body.get("verifiers") or ["nonempty"]
                harv = distill_harvest.harvest_paths(
                    inputs,
                    source_tag=str(body.get("corpus_name") or "distilled-traces"),
                    license_id=str(body.get("license") or "Apache-2.0"),
                )
                vpipe = distill_verifiers.VerifierPipeline(
                    verifiers,
                    regex_pattern=body.get("regex_pattern") or None,
                    json_required_keys=body.get("json_required_keys") or None,
                )
                vres = vpipe.filter_records(harv["records"])
                export_meta = None
                out_dir = body.get("out_dir")
                if out_dir and vres["accepted"]:
                    export_meta = distill_harvest.write_openwaldo_export(
                        vres["accepted"],
                        out_dir,
                        corpus_name=str(body.get("corpus_name") or "distilled-traces"),
                        license_id=str(body.get("license") or "Apache-2.0"),
                    )
                self._send_json(HTTPStatus.OK, {
                    "ok": True,
                    "harvest_stats": harv["stats"],
                    "verify_stats": vres["stats"],
                    "export": export_meta,
                })
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/distill/runs":
            try:
                drun = self.server.distill_manager.start_run(body, wait=False)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "run_id": drun.run_id, "run": drun.summary()})
            return

        if path.startswith("/api/distill/runs/"):
            rest = path[len("/api/distill/runs/"):].strip("/")
            parts = rest.split("/")
            if len(parts) == 2 and parts[1] == "stop":
                run_id = urllib.parse.unquote(parts[0])
                stopped = self.server.distill_manager.stop_run(run_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "stopped": stopped})
                return

        if path == "/api/spiral/presets":
            self._send_json(HTTPStatus.OK, studio_spiral.list_presets_and_pillars())
            return

        if path == "/api/runtime/locks":
            try:
                res = _run_eight_pillar_locks(
                    self.server.manager.bin_dir,
                    self.server.manager.root,
                    steps=int(body.get("steps", 8)),
                    lr_milli=int(body.get("lr_milli", 10)),
                )
                self._send_json(HTTPStatus.OK, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/omg":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                ollm = str(bins["ollm"])
                action = str(body.get("action", "samp")).strip().lower()
                if action == "selftest":
                    r = subprocess.run([ollm, "omg"], capture_output=True, text=True, check=False)
                    self._send_json(HTTPStatus.OK, {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr})
                    return
                with tempfile.TemporaryDirectory(prefix="helix-omg-") as td:
                    tdp = Path(td)
                    if action == "pack":
                        src_om = str(body.get("source_om") or _DEFAULT_OMG_SEED)
                        in_p = tdp / "in.om"
                        out_p = Path(body["out_tok"]).expanduser().resolve() if body.get("out_tok") else (tdp / "out.tok")
                        in_p.write_text(src_om, encoding="utf-8")
                        r = subprocess.run([ollm, "omg", "pack", str(in_p), str(out_p)], capture_output=True, text=True, check=False)
                        if r.returncode != 0:
                            raise ValueError(f"omg pack rc={r.returncode}: {r.stderr.strip()}")
                        parts = r.stdout.strip().split()
                        self._send_json(HTTPStatus.OK, {
                            "ok": True,
                            "action": "pack",
                            "in_bytes": int(parts[1]) if len(parts) > 1 else len(src_om),
                            "ntok": int(parts[2]) if len(parts) > 2 else 0,
                            "sem_id": parts[3] if len(parts) > 3 else "",
                            "tok_path": str(out_p) if body.get("out_tok") else "",
                        })
                        return
                    ck_path = body.get("ck_path")
                    if not ck_path and body.get("run_id"):
                        run = self.server.manager.get_run(str(body["run_id"]))
                        ck_path = _resolve_v256_ck(run)
                    if not ck_path:
                        # Auto-init tiny V=256 OMG model for zero-friction Playground exploration
                        ck_p = tdp / "omg_auto.ck"
                        r_i = subprocess.run([ollm, "init", str(ck_p), "256", "16", "16", "2", "32", "2", "4", "1", "10", "4000"], capture_output=True, text=True, check=False)
                        if r_i.returncode != 0:
                            raise ValueError(f"auto omg init failed rc={r_i.returncode}")
                        ck_path = str(ck_p)
                    out_om = tdp / "samp.om"
                    r = subprocess.run([
                        ollm, "omg", "samp",
                        str(Path(ck_path).expanduser().resolve()),
                        str(int(body.get("seed", 42))),
                        str(int(body.get("budget", 64))),
                        str(int(body.get("temp_milli", 0))),
                        str(out_om),
                    ], capture_output=True, text=True, check=False)
                    if r.returncode != 0:
                        raise ValueError(f"omg samp rc={r.returncode}: {r.stderr.strip()}")
                    parts = r.stdout.strip().split()
                    src_out = out_om.read_text(encoding="utf-8", errors="replace")
                    compiled, omc_rc = _compile_om_text(src_out)
                    self._send_json(HTTPStatus.OK, {
                        "ok": True,
                        "action": "samp",
                        "ntok": int(parts[1]) if len(parts) > 1 else 0,
                        "out_bytes": int(parts[2]) if len(parts) > 2 else len(src_out),
                        "sem_id": parts[3] if len(parts) > 3 else "",
                        "source_om": src_out,
                        "omc_compiled": compiled,
                        "omc_rc": omc_rc,
                    })
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/arbsamp":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                ollm = str(bins["ollm"])
                with tempfile.TemporaryDirectory(prefix="helix-arb-") as td:
                    tdp = Path(td)
                    corpus_tok = body.get("corpus_tok")
                    if not corpus_tok:
                        corpus_tok = str(_build_arboris_omg_corpus(ollm, tdp, body.get("seed_om")))
                    ck_path = body.get("ck_path")
                    if not ck_path and body.get("run_id"):
                        run = self.server.manager.get_run(str(body["run_id"]))
                        ck_path = _resolve_v256_ck(run)
                    if not ck_path:
                        ck_p = tdp / "arb_auto.ck"
                        r_i = subprocess.run([ollm, "init", str(ck_p), "256", "16", "16", "2", "32", "2", "4", "1", "10", "4000"], capture_output=True, text=True, check=False)
                        if r_i.returncode != 0:
                            raise ValueError(f"auto init failed rc={r_i.returncode}")
                        ck_path = str(ck_p)
                    out_om = tdp / "guide.om"
                    r = subprocess.run([
                        ollm, "arboris", "guide",
                        str(Path(ck_path).expanduser().resolve()),
                        str(Path(corpus_tok).expanduser().resolve()),
                        str(int(body.get("seed", 42))),
                        str(int(body.get("budget", 64))),
                        str(int(body.get("temp_milli", 0))),
                        str(out_om),
                    ], capture_output=True, text=True, check=False)
                    if r.returncode != 0:
                        raise ValueError(f"arboris guide rc={r.returncode}: {r.stderr.strip()}")
                    parts = r.stdout.strip().split()
                    ntok = int(parts[1]) if len(parts) > 1 else 0
                    bypass = int(parts[2]) if len(parts) > 2 else 0
                    ltp0 = int(parts[3]) if len(parts) > 3 else 0
                    out_bytes = int(parts[4]) if len(parts) > 4 else 0
                    sem_id = parts[5] if len(parts) > 5 else ""
                    src_out = out_om.read_text(encoding="utf-8", errors="replace")
                    compiled, omc_rc = _compile_om_text(src_out)
                    self._send_json(HTTPStatus.OK, {
                        "ok": True,
                        "ntok": ntok,
                        "bypass_hits": bypass,
                        "ltp0_calls": ltp0,
                        "bypass_rate": round(bypass / ntok, 4) if ntok > 0 else 0.0,
                        "out_bytes": out_bytes,
                        "sem_id": sem_id,
                        "source_om": src_out,
                        "omc_compiled": compiled,
                        "omc_rc": omc_rc,
                    })
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/spforge":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                ollm = str(bins["ollm"])
                base_ck = str(Path(body["base_ck"]).expanduser().resolve())
                corpus = str(Path(body["corpus"]).expanduser().resolve())
                out_dir = Path(body["out_dir"]).expanduser().resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                windows = int(body.get("windows", 8))
                ridge_milli = int(body.get("ridge_milli", 10000))
                gn_iters = int(body.get("gn_iters", 1))
                fc_path = out_dir / "frozen.hc"
                sp_path = out_dir / "closed.sp"
                spec_ck = out_dir / "spec.ck"
                r1 = subprocess.run([ollm, "fcache", base_ck, corpus, str(windows), str(fc_path)], capture_output=True, text=True, check=False)
                if r1.returncode != 0:
                    raise ValueError(f"fcache rc={r1.returncode}: {r1.stderr.strip()}")
                r2 = subprocess.run([ollm, "spforge", base_ck, str(fc_path), str(sp_path), str(ridge_milli), str(gn_iters)], capture_output=True, text=True, check=False)
                if r2.returncode != 0:
                    raise ValueError(f"spforge rc={r2.returncode}: {r2.stderr.strip()}")
                r3 = subprocess.run([ollm, "spapp", base_ck, str(sp_path), str(spec_ck)], capture_output=True, text=True, check=False)
                if r3.returncode != 0:
                    raise ValueError(f"spapp rc={r3.returncode}: {r3.stderr.strip()}")
                r4 = subprocess.run([ollm, "speval", base_ck, str(sp_path), corpus, str(windows)], capture_output=True, text=True, check=False)
                if r4.returncode != 0:
                    raise ValueError(f"speval rc={r4.returncode}: {r4.stderr.strip()}")
                sp_parts = r2.stdout.strip().split()
                ev_parts = r4.stdout.strip().split()
                self._send_json(HTTPStatus.OK, {
                    "ok": True,
                    "sp_path": str(sp_path),
                    "spec_ck": str(spec_ck),
                    "sp_bytes": sp_path.stat().st_size,
                    "loss_before_unats": int(sp_parts[5]) if len(sp_parts) > 5 else 0,
                    "loss_after_unats": int(sp_parts[6]) if len(sp_parts) > 6 else 0,
                    "eval_unats": int(ev_parts[3]) if len(ev_parts) > 3 else 0,
                })
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/gsamp":
            try:
                bins = runner.ensure_binaries(["ollm_univ"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                r = subprocess.run(
                    [
                        str(bins["ollm_univ"]), "gsamp",
                        str(Path(body["ck_path"]).expanduser().resolve()),
                        str(int(body.get("seed", 42))),
                        str(int(body.get("n_tokens", 32))),
                        str(int(body.get("temp_milli", 900))),
                        str(int(body.get("mode", 1))),
                    ],
                    capture_output=True, text=True, check=False,
                )
                if r.returncode != 0:
                    raise ValueError(f"gsamp rc={r.returncode}: {r.stderr.strip()}")
                g_raw = ""
                toks: list[int] = []
                for ln in r.stdout.splitlines():
                    if ln.startswith("G "):
                        g_raw = ln[2:]
                    elif ln.startswith("S "):
                        toks = [int(x) for x in ln[2:].split()]
                self._send_json(HTTPStatus.OK, {"ok": True, "mode": int(body.get("mode", 1)), "text": g_raw, "tokens": toks})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/guard":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                res = sovereign_runtime.run_firewall_decode(
                    bins["ollm"], body["base_ck"], body["strands"],
                    prompt=str(body.get("prompt", "Omega ")),
                    n_tokens=int(body.get("n_tokens", 12)),
                    temp_milli=int(body.get("temp_milli", 0)),
                    seed=int(body.get("seed", 42)),
                    disagree_milli=int(body.get("disagree_milli", 50)),
                    max_extra=int(body.get("max_extra", 2)),
                    merges=str(body.get("merges", "-")),
                )
                chk = sovereign_runtime.verify_firewall_receipt(res)
                self._send_json(HTTPStatus.OK, {"ok": chk["verified"], "spguard": res, "verification": chk})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/sleep":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                res = sovereign_runtime.run_sleep_cycle(
                    bins["ollm"], body["base_ck"], body["interactions"], body["held_bin"], body["out_dir"],
                    verifiers=body.get("verifiers") or ["nonempty"],
                    max_regression_nats=float(body.get("max_regression_nats", 0.25)),
                    steps=int(body.get("steps", 8)),
                    date_tag=body.get("date_tag"),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "receipt": res})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/runtime/green":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                res = sovereign_runtime.run_compile_until_green(
                    bins["ollm"], body["ck_path"], body["prompt"],
                    verifier=str(body.get("verifier", "omc_compile")),
                    max_attempts=int(body.get("max_attempts", 3)),
                    out_dir=body["out_dir"],
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "receipt": res})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/commons/pack-usb":
            try:
                res = sovereign_commons.pack_usb_bundle(
                    body["out_dir"],
                    bin_dir=self.server.manager.bin_dir,
                    root=self.server.manager.root,
                    trunk_ck=body.get("trunk_ck"),
                    packs=body.get("packs"),
                    memory_store=body.get("memory_store"),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "bundle": res})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/commons/memory/migrate":
            try:
                bins = runner.ensure_binaries(["ollm"], bin_dir=self.server.manager.bin_dir, root=self.server.manager.root)
                res = sovereign_commons.migrate_brain_to_trunk(
                    bins["ollm"],
                    body["store_path"],
                    body["new_trunk_ck"],
                    body["out_dir"],
                    steps=int(body.get("steps", 6)),
                    add_layers=int(body.get("add_layers", 1)),
                    lr_micro=int(body.get("lr_micro", 2000)),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "receipt": res})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/remotes/probe":
            try:
                res = ssh_remote.probe_ssh_host(
                    str(body.get("host") or body.get("target") or ""),
                    ssh_port=body.get("ssh_port"),
                    remote_port=int(body.get("remote_port", 18420)),
                )
                status = HTTPStatus.OK if res.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_json(status, res)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/remotes/connect":
            try:
                tun = self.server.tunnel_manager.connect(
                    str(body.get("host") or body.get("target") or ""),
                    remote_port=int(body.get("remote_port", 18420)),
                    local_port=int(body.get("local_port", 0)),
                    ssh_port=body.get("ssh_port"),
                    bootstrap=bool(body.get("bootstrap", True)),
                    repo_dir=body.get("repo_dir"),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "tunnel": tun})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/remotes/disconnect":
            try:
                tid = str(body.get("id") or body.get("host") or "")
                disc = self.server.tunnel_manager.disconnect(tid)
                self._send_json(HTTPStatus.OK, {"ok": True, "disconnected": disc})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/remotes/pair":
            try:
                parsed_p = ssh_remote.parse_pairing_input(str(body.get("pairing") or body.get("url") or ""))
                rid = str(body.get("id") or parsed_p["host"])
                entry = ssh_remote.save_remote({
                    "id": rid,
                    "host": parsed_p["host"],
                    "open_url": parsed_p["open_url"],
                    "url": parsed_p["url"],
                    "source": "paired",
                })
                self._send_json(HTTPStatus.OK, {"ok": True, "remote": entry})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        if path == "/api/remotes/remove":
            try:
                rid = str(body.get("id") or body.get("host") or "")
                self._send_json(HTTPStatus.OK, {"ok": True, "removed": ssh_remote.remove_remote(rid)})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown endpoint: {path}"})


def create_server(
    host: str = "127.0.0.1",
    port: int = 18420,
    workspace_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    ui_dir: Path | None = None,
    access_policy: ssh_remote.StudioAccessPolicy | None = None,
    tunnel_manager: ssh_remote.TunnelManager | None = None,
    instance_lock: ssh_remote.InstanceLock | None = None,
) -> StudioServer:
    mgr = runner.ChainManager(runs_dir=workspace_dir, bin_dir=bin_dir)
    return StudioServer(
        (host, port),
        manager=mgr,
        ui_dir=ui_dir or UI_DIR,
        access_policy=access_policy,
        tunnel_manager=tunnel_manager,
        instance_lock=instance_lock,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Omega Helix Studio server (stdlib-only)")
    parser.add_argument("--host", "--bind", dest="host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18420, help="Port (default: 18420)")
    parser.add_argument("--pairing-address", default=None, help="Advertised host:port for pairing links / SSH forwards")
    parser.add_argument("--allow-public", action="store_true", help="Allow public remote client IPs")
    parser.add_argument("--refuse-public", action="store_true", help="Refuse public remote client IPs")
    parser.add_argument("--no-pairing", action="store_true", help="Disable pairing token requirement")
    parser.add_argument("--require-pairing", action="store_true", help="Require pairing token even on loopback")
    parser.add_argument("--json", action="store_true", help="Print startup metadata as JSON")
    parser.add_argument("--workspace-dir", default=None, help="Directory to store runs & artifacts")
    parser.add_argument("--bin-dir", default=None, help="Override directory for pure-Omega binaries")
    args = parser.parse_args(argv)

    pol = ssh_remote.StudioAccessPolicy(
        allow_public=args.allow_public,
        refuse_public=args.refuse_public,
        require_pairing=args.require_pairing,
        no_pairing=args.no_pairing,
        pairing_address=args.pairing_address,
    )
    srv = create_server(host=args.host, port=args.port, workspace_dir=args.workspace_dir, bin_dir=args.bin_dir, access_policy=pol)
    host, port = srv.server_address[:2]
    adv = args.pairing_address or f"{host}:{port}"
    tok = pol.pairing_code
    url = f"http://{adv}/?pair={tok}" if tok else f"http://{adv}/"
    if args.json:
        print(json.dumps({
            "ok": True,
            "bind": f"{host}:{port}",
            "port": port,
            "pairing_address": adv,
            "pairing_code": tok,
            "url": url,
        }), flush=True)
    else:
        print(f"Omega Helix Studio listening on http://{host}:{port}/ (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nShutting down Omega Helix Studio.", flush=True)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
