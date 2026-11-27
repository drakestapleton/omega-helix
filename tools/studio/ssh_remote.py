#!/usr/bin/env python3
"""Orca-inspired SSH remote discovery, tunnel management, pairing & bind policy.

Python 3 standard library only — zero external dependencies, zero telemetry.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import hmac
import ipaddress
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def config_dir() -> Path:
    env = os.environ.get("OMEGA_HELIX_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".config" / "omega-helix"


def state_dir() -> Path:
    env = os.environ.get("OMEGA_HELIX_STATE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".local" / "state" / "omega-helix"


def _write_json_private(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# --------------------------------------------------------------------------- Bind / Origin Policy (Orca serve style)

_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_ULA_V6 = ipaddress.ip_network("fc00::/7")


def classify_client_ip(addr: str) -> str:
    """Classify client IP as 'loopback', 'lan', 'tailscale', 'public', or 'unknown'."""
    host = (addr or "").split("%", 1)[0].strip()
    if not host:
        return "unknown"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "unknown"
    if ip.is_loopback:
        return "loopback"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
        if ip.is_loopback:
            return "loopback"
    if isinstance(ip, ipaddress.IPv4Address) and ip in _TAILSCALE_V4:
        return "tailscale"
    if isinstance(ip, ipaddress.IPv6Address) and ip in _TAILSCALE_V6:
        return "tailscale"
    if ip.is_private or ip.is_link_local:
        return "lan"
    if isinstance(ip, ipaddress.IPv6Address) and ip in _ULA_V6:
        return "lan"
    return "public"


class StudioAccessPolicy:
    """Client address & Orca-style pairing policy."""

    def __init__(
        self,
        *,
        allow_public: bool = False,
        refuse_public: bool = False,
        require_pairing: bool = False,
        no_pairing: bool = False,
        pairing_code: str | None = None,
        pairing_address: str | None = None,
    ) -> None:
        if allow_public and refuse_public:
            raise ValueError("--allow-public and --refuse-public cannot be combined")
        self.allow_public = bool(allow_public)
        self.refuse_public = bool(refuse_public)
        self.no_pairing = bool(no_pairing)
        self.require_pairing = bool(require_pairing) and not self.no_pairing
        self.pairing_code = "" if self.no_pairing else (pairing_code or secrets.token_urlsafe(24))
        self.pairing_address = pairing_address or ""
        self.paired_tokens: set[str] = set()
        if self.pairing_code:
            self.paired_tokens.add(self.pairing_code)

    def ip_allowed(self, client_ip: str) -> tuple[bool, str]:
        kind = classify_client_ip(client_ip)
        if kind in ("loopback", "lan", "tailscale"):
            return True, kind
        if kind == "public":
            if self.refuse_public:
                return False, "public-refused"
            if self.allow_public:
                return True, "public-allowed"
            return False, "public-denied"
        return False, "unknown"

    def needs_token(self, client_ip: str) -> bool:
        if self.no_pairing:
            return False
        kind = classify_client_ip(client_ip)
        if kind == "loopback" and not self.require_pairing:
            return False
        return True

    def verify_token(self, candidate: str | None) -> bool:
        if self.no_pairing:
            return True
        if not candidate:
            return False
        for tok in self.paired_tokens:
            if hmac.compare_digest(candidate.encode("utf-8"), tok.encode("utf-8")):
                return True
        return False


# --------------------------------------------------------------------------- Single-instance Lock (Orca serve style)

def _current_boot_id() -> str:
    p = Path("/proc/sys/kernel/random/boot_id")
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, check=False, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return socket.gethostname()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class InstanceLock:
    def __init__(self, port: int, lock_dir: Path | None = None) -> None:
        self.dir = lock_dir or state_dir()
        self.path = self.dir / f"studio-{port}.lock"
        self.port = port
        self.acquired = False

    def acquire(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = _read_json(self.path, None)
        if isinstance(existing, dict):
            old_pid = int(existing.get("pid") or 0)
            old_host = str(existing.get("host") or "")
            if old_host == socket.gethostname() and _pid_alive(old_pid) and old_pid != os.getpid():
                raise RuntimeError(
                    f"Another Omega Studio already holds {self.path} (pid={old_pid}, port={self.port}). "
                    "Use --reclaim-stale-lock if stale."
                )
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "boot_id": _current_boot_id(),
            "port": self.port,
            "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        _write_json_private(self.path, payload)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            cur = _read_json(self.path, None)
            if isinstance(cur, dict) and int(cur.get("pid") or 0) == os.getpid():
                self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self.acquired = False


def reclaim_stale_lock(port: int, *, yes: bool = False, lock_dir: Path | None = None) -> dict[str, Any]:
    path = (lock_dir or state_dir()) / f"studio-{port}.lock"
    if not path.is_file():
        return {"ok": True, "status": "none", "path": str(path)}
    doc = _read_json(path, {})
    if not isinstance(doc, dict):
        doc = {}
    pid = int(doc.get("pid") or 0)
    host = str(doc.get("host") or "")
    boot_id = str(doc.get("boot_id") or "")
    same_host = host == socket.gethostname()
    alive = same_host and _pid_alive(pid)
    same_boot = boot_id == _current_boot_id()
    if alive and same_boot:
        return {"ok": False, "status": "refused", "reason": f"pid {pid} is still alive", "path": str(path), "lock": doc}
    if not yes:
        return {"ok": True, "status": "eligible", "path": str(path), "lock": doc}
    path.unlink(missing_ok=True)
    return {"ok": True, "status": "reclaimed", "path": str(path), "lock": doc}


# --------------------------------------------------------------------------- SSH Host Discovery & Saved Remotes

_HOST_RE = re.compile(r"^[A-Za-z0-9._:@/-]+$")


def validate_ssh_target(target: str) -> str:
    t = (target or "").strip()
    if not t or t.startswith("-") or not _HOST_RE.match(t):
        raise ValueError(f"invalid SSH target: {target!r}")
    return t


def parse_ssh_config(path: Path | None = None) -> list[dict[str, Any]]:
    cfg_path = path or (Path.home() / ".ssh" / "config")
    if not cfg_path.is_file():
        return []
    hosts: list[dict[str, Any]] = []
    cur_aliases: list[str] = []
    cur_props: dict[str, str] = {}

    def flush() -> None:
        for alias in cur_aliases:
            if not alias or any(ch in alias for ch in "*?!") or alias.startswith("-"):
                continue
            hosts.append({
                "name": alias,
                "target": alias,
                "hostname": cur_props.get("hostname", alias),
                "user": cur_props.get("user", ""),
                "port": int(cur_props["port"]) if cur_props.get("port", "").isdigit() else 22,
                "source": "ssh_config",
            })

    try:
        lines = cfg_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        val = parts[1].strip()
        if key == "host":
            flush()
            cur_aliases = val.split()
            cur_props = {}
        elif key in ("hostname", "user", "port", "identityfile") and cur_aliases:
            cur_props[key] = val
    flush()
    return hosts


def load_saved_hosts() -> list[dict[str, Any]]:
    doc = _read_json(config_dir() / "hosts.json", {"hosts": []})
    rows = doc.get("hosts") if isinstance(doc, dict) else []
    return [r for r in (rows or []) if isinstance(r, dict) and r.get("name")]


def add_saved_host(name: str, target: str | None = None, *, port: int = 22, label: str = "") -> dict[str, Any]:
    clean_name = validate_ssh_target(name)
    clean_target = validate_ssh_target(target or clean_name)
    hosts = [h for h in load_saved_hosts() if h.get("name") != clean_name]
    entry = {
        "name": clean_name,
        "target": clean_target,
        "port": int(port),
        "label": label or clean_name,
        "source": "saved",
    }
    hosts.append(entry)
    _write_json_private(config_dir() / "hosts.json", {"hosts": hosts})
    return entry


def remove_saved_host(name: str) -> bool:
    hosts = load_saved_hosts()
    kept = [h for h in hosts if h.get("name") != name]
    if len(kept) == len(hosts):
        return False
    _write_json_private(config_dir() / "hosts.json", {"hosts": kept})
    return True


def discover_ssh_hosts(ssh_config_path: Path | None = None) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for h in parse_ssh_config(ssh_config_path):
        merged[h["name"]] = h
    for h in load_saved_hosts():
        merged[h["name"]] = h
    return sorted(merged.values(), key=lambda x: str(x["name"]).lower())


def load_remotes() -> list[dict[str, Any]]:
    doc = _read_json(config_dir() / "remotes.json", {"remotes": []})
    rows = doc.get("remotes") if isinstance(doc, dict) else []
    return [r for r in (rows or []) if isinstance(r, dict) and r.get("id")]


def save_remote(entry: dict[str, Any]) -> dict[str, Any]:
    remotes = [r for r in load_remotes() if r.get("id") != entry.get("id")]
    remotes.append(entry)
    _write_json_private(config_dir() / "remotes.json", {"remotes": remotes})
    return entry


def remove_remote(remote_id: str) -> bool:
    remotes = load_remotes()
    kept = [r for r in remotes if r.get("id") != remote_id and r.get("host") != remote_id]
    if len(kept) == len(remotes):
        return False
    _write_json_private(config_dir() / "remotes.json", {"remotes": kept})
    return True


def parse_pairing_input(raw: str) -> dict[str, Any]:
    """Parse a pairing URL (http://host:port/?pair=TOKEN) or host:port#TOKEN string."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty pairing code or URL")
    if s.startswith("http://") or s.startswith("https://"):
        import urllib.parse
        u = urllib.parse.urlsplit(s)
        qs = urllib.parse.parse_qs(u.query)
        tok = (qs.get("pair") or [""])[0]
        base = f"{u.scheme}://{u.netloc}"
        return {"url": base, "token": tok, "open_url": s, "host": u.hostname or u.netloc}
    if "#" in s:
        base, tok = s.split("#", 1)
        if not base.startswith("http"):
            base = f"http://{base}"
        return {"url": base, "token": tok, "open_url": f"{base}/?pair={tok}" if tok else f"{base}/", "host": base}
    raise ValueError("expected pairing URL (http://.../?pair=...) or host:port#token")


# --------------------------------------------------------------------------- SSH Probe & Tunnel Manager

def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ssh_base_cmd(target: str, ssh_port: int | None = None) -> list[str]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if ssh_port and int(ssh_port) != 22:
        cmd.extend(["-p", str(int(ssh_port))])
    cmd.append(validate_ssh_target(target))
    return cmd


def probe_ssh_host(target: str, *, ssh_port: int | None = None, remote_port: int = 18420) -> dict[str, Any]:
    """Probe a remote host over SSH for OS, arch, GB10 GPU, omc, and running Studio."""
    t = validate_ssh_target(target)
    rport = int(remote_port)
    script = (
        f"printf 'HOST=%s\\nOS=%s\\nARCH=%s\\n' \"$(hostname)\" \"$(uname -s)\" \"$(uname -m)\"; "
        f"[ -e /dev/nvidiactl ] && echo 'GB10=1' || echo 'GB10=0'; "
        f"command -v omc 2>/dev/null || [ -x \"$HOME/.local/bin/omc\" ] && echo 'OMC=1' || echo 'OMC=0'; "
        f"for d in \"$HOME/workspace/omega-helix\" \"$HOME/omega-helix\" \"$HOME/.local/opt/omega-helix\"; do "
        f"  if [ -f \"$d/tools/studio/server.py\" ]; then echo \"REPO=$d\"; break; fi; "
        f"done; "
        f"python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{rport}/api/system', timeout=1.5).read(); print('STUDIO=1')\" 2>/dev/null || echo 'STUDIO=0'"
    )
    r = subprocess.run(_ssh_base_cmd(t, ssh_port) + [script], capture_output=True, text=True, check=False, timeout=10)
    if r.returncode != 0:
        return {
            "ok": False,
            "target": t,
            "error": (r.stderr or r.stdout or f"ssh exited {r.returncode}").strip(),
        }
    kv: dict[str, str] = {}
    for ln in r.stdout.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()
    return {
        "ok": True,
        "target": t,
        "hostname": kv.get("HOST", t),
        "os": kv.get("OS", ""),
        "arch": kv.get("ARCH", ""),
        "has_gb10": kv.get("GB10") == "1",
        "has_omc": kv.get("OMC") == "1",
        "repo_dir": kv.get("REPO", ""),
        "studio_running": kv.get("STUDIO") == "1",
        "remote_port": rport,
    }


def _bootstrap_remote_studio(
    target: str,
    *,
    remote_port: int,
    local_port: int,
    ssh_port: int | None = None,
    repo_dir: str | None = None,
) -> dict[str, Any]:
    t = validate_ssh_target(target)
    rport = int(remote_port)
    lport = int(local_port)
    repo_q = shlex.quote(repo_dir) if repo_dir else ""
    remote_cmd = (
        f"set -e; "
        f"BIN=$(command -v omega-helix 2>/dev/null || true); "
        f"if [ -z \"$BIN\" ]; then "
        f"  for d in {repo_q} \"$HOME/workspace/omega-helix\" \"$HOME/omega-helix\" \"$HOME/.local/opt/omega-helix\"; do "
        f"    if [ -n \"$d\" ] && [ -x \"$d/bin/omega-helix\" ]; then BIN=\"$d/bin/omega-helix\"; break; fi; "
        f"  done; "
        f"fi; "
        f"[ -n \"$BIN\" ] || {{ echo 'omega-helix not found on remote' >&2; exit 12; }}; "
        f"LOG=/tmp/omega-helix-studio-{rport}.log; "
        f"if command -v setsid >/dev/null 2>&1; then "
        f"  setsid -f \"$BIN\" studio --headless --no-pairing --bind 127.0.0.1 --port {rport} "
        f"    --pairing-address 127.0.0.1:{lport} --json >\"$LOG\" 2>&1 </dev/null; "
        f"else "
        f"  nohup \"$BIN\" studio --headless --no-pairing --bind 127.0.0.1 --port {rport} "
        f"    --pairing-address 127.0.0.1:{lport} --json >\"$LOG\" 2>&1 </dev/null & "
        f"fi; "
        f"sleep 0.6; head -n 5 \"$LOG\" 2>/dev/null || true"
    )
    r = subprocess.run(_ssh_base_cmd(t, ssh_port) + [remote_cmd], capture_output=True, text=True, check=False, timeout=12)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or f"remote studio bootstrap failed rc={r.returncode}").strip())
    info: dict[str, Any] = {}
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                info = json.loads(ln)
                break
            except Exception:
                pass
    return info


class TunnelManager:
    """Manages active local->remote SSH port-forward tunnels cleanly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tunnels: dict[str, dict[str, Any]] = {}
        atexit.register(self.close_all)

    def list_tunnels(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for tid, rec in list(self._tunnels.items()):
                proc: subprocess.Popen[bytes] = rec["proc"]
                alive = proc.poll() is None
                out.append({
                    "id": tid,
                    "host": rec["host"],
                    "local_port": rec["local_port"],
                    "remote_port": rec["remote_port"],
                    "local_url": rec["local_url"],
                    "open_url": rec["open_url"],
                    "ssh_cmd": rec["ssh_cmd"],
                    "alive": alive,
                    "pid": proc.pid,
                    "remote_system": rec.get("remote_system"),
                    "started_at": rec["started_at"],
                })
            return out

    def connect(
        self,
        target: str,
        *,
        remote_port: int = 18420,
        local_port: int = 0,
        ssh_port: int | None = None,
        bootstrap: bool = True,
        repo_dir: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        t = validate_ssh_target(target)
        rport = int(remote_port)
        lport = int(local_port) if int(local_port) > 0 else _pick_free_port()

        # Check if tunnel for this host:rport already exists & alive
        with self._lock:
            for tid, rec in self._tunnels.items():
                if rec["host"] == t and rec["remote_port"] == rport and rec["proc"].poll() is None:
                    return {
                        "id": tid,
                        "host": t,
                        "local_port": rec["local_port"],
                        "remote_port": rport,
                        "local_url": rec["local_url"],
                        "open_url": rec["open_url"],
                        "ssh_cmd": rec["ssh_cmd"],
                        "alive": True,
                        "pid": rec["proc"].pid,
                        "remote_system": rec.get("remote_system"),
                        "started_at": rec["started_at"],
                    }

        probe = probe_ssh_host(t, ssh_port=ssh_port, remote_port=rport)
        if not probe.get("ok"):
            raise RuntimeError(probe.get("error") or f"cannot reach {t} via ssh")

        pairing_tok = ""
        if bootstrap and not probe.get("studio_running"):
            boot = _bootstrap_remote_studio(
                t,
                remote_port=rport,
                local_port=lport,
                ssh_port=ssh_port,
                repo_dir=repo_dir or probe.get("repo_dir") or None,
            )
            pairing_tok = str(boot.get("pairing_code") or "")

        fwd_spec = f"127.0.0.1:{lport}:127.0.0.1:{rport}"
        cmd = [
            "ssh",
            "-N",
            "-T",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5",
        ]
        if ssh_port and int(ssh_port) != 22:
            cmd.extend(["-p", str(int(ssh_port))])
        cmd.extend(["-L", fwd_spec, t])

        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        local_url = f"http://127.0.0.1:{lport}"
        open_url = f"{local_url}/?pair={pairing_tok}" if pairing_tok else f"{local_url}/"
        remote_sys: dict[str, Any] | None = None

        deadline = time.monotonic() + timeout
        last_err = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err_b = proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(f"ssh tunnel exited rc={proc.returncode}: {err_b.decode('utf-8', 'replace').strip()}")
            req = urllib.request.Request(f"{local_url}/api/system")
            if pairing_tok:
                req.add_header("Authorization", f"Bearer {pairing_tok}")
            try:
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    remote_sys = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.15)

        if remote_sys is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError(f"ssh tunnel timed out waiting for {local_url}/api/system ({last_err})")

        tid = f"{t}:{rport}->{lport}"
        ssh_cmd = " ".join(shlex.quote(c) for c in cmd)
        rec = {
            "id": tid,
            "host": t,
            "local_port": lport,
            "remote_port": rport,
            "local_url": local_url,
            "open_url": open_url,
            "ssh_cmd": ssh_cmd,
            "proc": proc,
            "remote_system": remote_sys,
            "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        with self._lock:
            self._tunnels[tid] = rec
        save_remote({
            "id": tid,
            "host": t,
            "remote_port": rport,
            "local_port": lport,
            "open_url": open_url,
            "hostname": probe.get("hostname", t),
            "os": remote_sys.get("os", probe.get("os", "")),
            "arch": remote_sys.get("arch", probe.get("arch", "")),
            "has_gb10": bool(remote_sys.get("has_gb10", probe.get("has_gb10"))),
            "updated_at": rec["started_at"],
        })
        return {
            "id": tid,
            "host": t,
            "local_port": lport,
            "remote_port": rport,
            "local_url": local_url,
            "open_url": open_url,
            "ssh_cmd": ssh_cmd,
            "alive": True,
            "pid": proc.pid,
            "remote_system": remote_sys,
            "started_at": rec["started_at"],
        }

    def disconnect(self, tid_or_host: str) -> bool:
        with self._lock:
            keys = [
                k for k, v in self._tunnels.items()
                if k == tid_or_host or v["host"] == tid_or_host
            ]
            if not keys:
                return False
            for k in keys:
                proc = self._tunnels.pop(k)["proc"]
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                for pipe in (proc.stdout, proc.stderr):
                    if pipe:
                        try:
                            pipe.close()
                        except OSError:
                            pass
            return True

    def close_all(self) -> None:
        with self._lock:
            items = list(self._tunnels.values())
            self._tunnels.clear()
        for rec in items:
            proc = rec["proc"]
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            for pipe in (proc.stdout, proc.stderr):
                if pipe:
                    try:
                        pipe.close()
                    except OSError:
                        pass
