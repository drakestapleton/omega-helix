#!/usr/bin/env python3
"""Host machine & desktop/headless/WSL 2 topology auto-detection for Omega Helix Studio.

Python 3 standard library only — zero external dependencies, zero telemetry.
Never assumes a MacBook SSHing to Spark: supports Direct Spark/Linux (desktop or
headless), macOS (local native a64macos or optional remote Spark tunnel), and
Windows 10/11 via WSL 2 (static x86linux local or optional remote Spark tunnel).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


def _read_proc_version() -> str:
    p = Path("/proc/version")
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def detect_topology(
    *,
    os_name: str | None = None,
    arch: str | None = None,
    env: Mapping[str, str] | None = None,
    proc_version: str | None = None,
    has_nvidia_dev: bool | None = None,
    has_aien_gpu_dev: bool | None = None,
    is_tty: bool | None = None,
    which: Callable[[str], str | None] | None = None,
    remote_target: str | None = None,
    headless_flag: bool = False,
    no_open_flag: bool = False,
    bind_host: str = "127.0.0.1",
    port: int = 18420,
) -> dict[str, Any]:
    """Detect local host topology, binary target, SSH-tunnel skip policy, and browser opener."""
    env_map = os.environ if env is None else env
    sys_os = platform.system() if os_name is None else os_name
    sys_arch = platform.machine() if arch is None else arch
    arch_l = (sys_arch or "").lower()
    which_fn = shutil.which if which is None else which
    tty = (sys.stdout.isatty() if is_tty is None else bool(is_tty))

    is_linux = sys_os == "Linux"
    is_darwin = sys_os == "Darwin"
    is_windows_native = (
        sys_os == "Windows"
        or sys_os.startswith("MINGW")
        or sys_os.startswith("MSYS")
        or sys_os.startswith("CYGWIN")
    )

    pver = (_read_proc_version() if (proc_version is None and is_linux) else (proc_version or ""))
    is_wsl = bool(
        is_linux
        and (
            env_map.get("WSL_DISTRO_NAME")
            or env_map.get("WSL_INTEROP")
            or ("microsoft" in pver.lower())
        )
    )

    nv_dev = os.path.exists("/dev/nvidiactl") if has_nvidia_dev is None else bool(has_nvidia_dev)
    aien_dev = os.path.exists("/dev/aien-gpu") if has_aien_gpu_dev is None else bool(has_aien_gpu_dev)
    has_gpu_dev = bool(is_linux and not is_wsl and (nv_dev or aien_dev))
    is_arm64 = arch_l in ("aarch64", "arm64")
    is_spark_local = bool(is_linux and not is_wsl and is_arm64 and has_gpu_dev)

    is_ssh_session = bool(
        env_map.get("SSH_CONNECTION")
        or env_map.get("SSH_CLIENT")
        or env_map.get("SSH_TTY")
    )
    has_linux_display = bool(env_map.get("DISPLAY") or env_map.get("WAYLAND_DISPLAY"))

    remote_clean = (remote_target or "").strip()
    url = f"http://{bind_host}:{int(port)}/"

    topology = "linux_headless"
    platform_label = f"{sys_os} {sys_arch}"
    binary_target = "aarch64" if is_arm64 else "x86linux"
    signing_mode = "sha256"
    native_exe_available = False
    has_display = False
    browser_cmd: list[str] | None = None
    browser_kind = "none"
    remote_tunnel_supported = True
    guidance = ""

    if is_wsl:
        topology = "wsl2"
        distro = env_map.get("WSL_DISTRO_NAME") or "WSL 2"
        platform_label = f"Windows 10/11 ({distro}, {sys_arch})"
        binary_target = "aarch64" if is_arm64 else "x86linux"
        signing_mode = "sha256"
        has_display = not headless_flag
        if has_display and not no_open_flag:
            if which_fn("wslview"):
                browser_cmd = ["wslview", url]
                browser_kind = "wslview"
            elif which_fn("explorer.exe"):
                browser_cmd = ["explorer.exe", url]
                browser_kind = "explorer.exe"
            elif which_fn("cmd.exe"):
                browser_cmd = ["cmd.exe", "/c", "start", "", url]
                browser_kind = "cmd.exe"
        guidance = (
            "WSL 2 detected: uses static x86linux pure-Omega binaries inside WSL 2 "
            "(no native Windows .exe). Runs locally or via --remote <host> SSH tunnel."
        )
    elif is_windows_native:
        topology = "windows_powershell"
        platform_label = f"Windows ({sys_os} {sys_arch})"
        binary_target = "x86linux-wsl2"
        signing_mode = "sha256"
        has_display = False
        remote_tunnel_supported = True
        guidance = (
            "Native Windows .exe is not shipped. Install & run Omega Helix inside WSL 2 "
            "(static x86linux binary) or use https://interplane.io/omega/compile/."
        )
    elif is_darwin:
        topology = "macos_arm64" if is_arm64 else "macos_x86_64"
        platform_label = "macOS Apple Silicon (arm64)" if is_arm64 else f"macOS Intel ({sys_arch})"
        binary_target = "macos" if is_arm64 else "remote-only"
        signing_mode = "sha256+adhoc-codesign" if is_arm64 else "none"
        has_display = (not is_ssh_session) and (not headless_flag)
        if has_display and not no_open_flag and which_fn("open"):
            browser_cmd = ["open", url]
            browser_kind = "open"
        if is_arm64:
            guidance = (
                "macOS Apple Silicon: native a64macos Mach-O (SHA-256 verified + ad-hoc codesign -s - -f, "
                "zero Apple notarization fees). Runs locally or via --remote <host> SSH tunnel."
            )
        else:
            guidance = (
                "macOS Intel (x86_64): Studio runs locally and connects to a DGX Spark / Linux host "
                "via --remote <host> SSH tunnel (native Mach-O omc targets Apple Silicon arm64)."
            )
    elif is_linux:
        if is_spark_local:
            platform_label = f"DGX Spark Linux ({sys_arch}, GB10 local)"
            binary_target = "gb10"
        else:
            platform_label = f"Linux ({sys_arch})"
            binary_target = "aarch64" if is_arm64 else "x86linux"
        has_display = bool(has_linux_display and not headless_flag and not is_ssh_session)
        if has_display:
            topology = "spark_desktop" if is_spark_local else "linux_desktop"
            if not no_open_flag and which_fn("xdg-open"):
                browser_cmd = ["xdg-open", url]
                browser_kind = "xdg-open"
            guidance = (
                "Direct local Linux/Spark desktop detected ($DISPLAY/$WAYLAND_DISPLAY): "
                "binding localhost directly (no SSH tunnel) and opening browser via xdg-open."
            )
        else:
            topology = "spark_headless" if is_spark_local else "linux_headless"
            guidance = (
                "Headless Linux/Spark detected (no local display or SSH session): "
                "binding localhost directly (no outbound SSH tunnel) and printing clickable URL."
            )
    else:
        topology = "unknown"
        guidance = f"Unrecognized OS {sys_os}/{sys_arch}; binding localhost and printing URL."

    mode = "remote_tunnel" if remote_clean else "local"
    should_skip_ssh_tunnel = not bool(remote_clean)

    # Open browser only when a concrete browser command was resolved, not headless/no-open,
    # and either TTY is active or WSL/desktop launch is requested cleanly.
    can_open = bool(browser_cmd is not None and not headless_flag and not no_open_flag)
    open_action = "browser" if can_open else "print_url"

    return {
        "topology": topology,
        "platform_label": platform_label,
        "os": sys_os,
        "arch": sys_arch,
        "is_wsl": is_wsl,
        "is_spark_local": is_spark_local,
        "is_ssh_session": is_ssh_session,
        "has_gpu_dev": has_gpu_dev,
        "has_nvidia_dev": nv_dev,
        "has_aien_gpu_dev": aien_dev,
        "has_display": has_display,
        "is_tty": tty,
        "mode": mode,
        "remote_target": remote_clean,
        "should_skip_ssh_tunnel": should_skip_ssh_tunnel,
        "remote_tunnel_supported": remote_tunnel_supported,
        "recommended_bind": bind_host,
        "port": int(port),
        "url": url,
        "binary_target": binary_target,
        "signing_mode": signing_mode,
        "native_exe_available": native_exe_available,
        "browser_kind": browser_kind,
        "browser_cmd": browser_cmd,
        "open_action": open_action,
        "guidance": guidance,
    }


def open_studio_browser(
    url: str,
    topo: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Open the Studio URL using the topology-appropriate launcher (never blindly calling webbrowser)."""
    if topo.get("open_action") != "browser" or not topo.get("browser_cmd"):
        return {"opened": False, "action": "print_url", "cmd": None, "url": url}
    cmd = [url if tok == topo.get("url") else tok for tok in topo["browser_cmd"]]
    if not cmd or cmd[-1] != url:
        cmd = list(topo["browser_cmd"][:-1]) + [url]
    if dry_run:
        return {"opened": False, "action": "dry_run_browser", "cmd": cmd, "url": url}
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"opened": True, "action": "browser", "cmd": cmd, "url": url}
    except OSError as exc:
        return {"opened": False, "action": "print_url", "cmd": cmd, "url": url, "error": str(exc)}
