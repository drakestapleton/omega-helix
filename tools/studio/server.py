#!/usr/bin/env python3
"""Omega Helix Studio local HTTP & SSE server (Python 3 stdlib only, zero telemetry)."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
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


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: runner.ChainManager,
        ui_dir: Path = UI_DIR,
        distill_manager: distill_pipeline.DistillManager | None = None,
    ) -> None:
        self.manager = manager
        self.distill_manager = distill_manager or distill_pipeline.DistillManager(
            runs_dir=manager.runs_dir / "_distill",
            bin_dir=manager.bin_dir,
            root=manager.root,
        )
        self.ui_dir = ui_dir.resolve()
        super().__init__(server_address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: StudioServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet by default unless OMEGA_STUDIO_VERBOSE=1
        import os
        if os.environ.get("OMEGA_STUDIO_VERBOSE") == "1":
            super().log_message(format, *args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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

        if path == "/api/system":
            sys_info = runner.detect_system(bin_dir=self.server.manager.bin_dir, runs_dir=self.server.manager.runs_dir)
            self._send_json(HTTPStatus.OK, sys_info)
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

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown endpoint: {path}"})


def create_server(
    host: str = "127.0.0.1",
    port: int = 18420,
    workspace_dir: str | Path | None = None,
    bin_dir: str | Path | None = None,
    ui_dir: Path | None = None,
) -> StudioServer:
    mgr = runner.ChainManager(runs_dir=workspace_dir, bin_dir=bin_dir)
    return StudioServer((host, port), manager=mgr, ui_dir=ui_dir or UI_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Omega Helix Studio local server (127.0.0.1, stdlib-only)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18420, help="Port (default: 18420)")
    parser.add_argument("--workspace-dir", default=None, help="Directory to store runs & artifacts")
    parser.add_argument("--bin-dir", default=None, help="Override directory for pure-Omega binaries")
    args = parser.parse_args(argv)

    srv = create_server(host=args.host, port=args.port, workspace_dir=args.workspace_dir, bin_dir=args.bin_dir)
    host, port = srv.server_address[:2]
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
