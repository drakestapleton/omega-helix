#!/usr/bin/env python3
"""Automated unit & dry-run topology detection tests for Omega Helix Studio & install.sh.

Covers all 4 topologies without assuming a MacBook SSHing to Spark:
  1. Direct DGX Spark / Native Linux with monitor attached ($DISPLAY / $WAYLAND_DISPLAY -> xdg-open, skip SSH tunnel)
  2. Headless Linux / DGX Spark (no display or SSH session -> print clickable localhost URL, no browser spawn, skip SSH tunnel)
  3. macOS (Apple Silicon native a64macos + SHA-256 + ad-hoc codesign, Intel remote mode, local vs --remote Spark tunnel -> `open`)
  4. Windows 10/11 via WSL 2 (WSL_DISTRO_NAME & /proc/version microsoft -> static x86linux, never native .exe, local/remote, wslview / explorer.exe / cmd.exe /c start)

Run:
  python3 -W error::ResourceWarning -m unittest tests/test_studio_topology.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_DIR = REPO_ROOT / "tools" / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import cli as studio_cli  # noqa: E402
import topology as studio_topology  # noqa: E402


def _fake_which(present: set[str]):
    return lambda name: (f"/usr/bin/{name}" if name in present else None)


def _run_install_dry_run(extra_env: dict[str, str]) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    env.update(extra_env)
    proc = subprocess.run(
        ["sh", str(REPO_ROOT / "install.sh"), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"install.sh --dry-run rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    out: dict[str, str] = {}
    for ln in proc.stdout.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class TestStudioTopology(unittest.TestCase):
    def test_01_direct_spark_with_x11_and_wayland_monitor(self) -> None:
        # Direct DGX Spark (Linux aarch64 + GB10 + X11 DISPLAY)
        t_x11 = studio_topology.detect_topology(
            os_name="Linux",
            arch="aarch64",
            env={"DISPLAY": ":0"},
            proc_version="Linux version 7.0.0-nvidia (aarch64)",
            has_nvidia_dev=True,
            has_aien_gpu_dev=False,
            is_tty=True,
            which=_fake_which({"xdg-open"}),
        )
        self.assertEqual(t_x11["topology"], "spark_desktop")
        self.assertTrue(t_x11["is_spark_local"])
        self.assertFalse(t_x11["is_wsl"])
        self.assertEqual(t_x11["mode"], "local")
        self.assertTrue(t_x11["should_skip_ssh_tunnel"])
        self.assertEqual(t_x11["recommended_bind"], "127.0.0.1")
        self.assertEqual(t_x11["binary_target"], "gb10")
        self.assertEqual(t_x11["open_action"], "browser")
        self.assertEqual(t_x11["browser_cmd"], ["xdg-open", "http://127.0.0.1:18420/"])

        act = studio_topology.open_studio_browser("http://127.0.0.1:19000/", t_x11, dry_run=True)
        self.assertEqual(act["action"], "dry_run_browser")
        self.assertEqual(act["cmd"], ["xdg-open", "http://127.0.0.1:19000/"])

        # Direct DGX Spark with Wayland + pure-Omega /dev/aien-gpu (zero nvidia.ko)
        t_wl = studio_topology.detect_topology(
            os_name="Linux",
            arch="aarch64",
            env={"WAYLAND_DISPLAY": "wayland-0"},
            proc_version="Linux version 7.0.0-aien",
            has_nvidia_dev=False,
            has_aien_gpu_dev=True,
            is_tty=True,
            which=_fake_which({"xdg-open"}),
        )
        self.assertEqual(t_wl["topology"], "spark_desktop")
        self.assertTrue(t_wl["is_spark_local"])
        self.assertTrue(t_wl["should_skip_ssh_tunnel"])
        self.assertEqual(t_wl["browser_kind"], "xdg-open")

        # Native Linux x86_64 desktop (no GB10)
        t_lin = studio_topology.detect_topology(
            os_name="Linux",
            arch="x86_64",
            env={"DISPLAY": ":1"},
            proc_version="Linux version 6.8.0-generic",
            has_nvidia_dev=False,
            has_aien_gpu_dev=False,
            which=_fake_which({"xdg-open"}),
        )
        self.assertEqual(t_lin["topology"], "linux_desktop")
        self.assertFalse(t_lin["is_spark_local"])
        self.assertEqual(t_lin["binary_target"], "x86linux")
        self.assertTrue(t_lin["should_skip_ssh_tunnel"])
        self.assertEqual(t_lin["browser_cmd"], ["xdg-open", "http://127.0.0.1:18420/"])

    def test_02_headless_linux_and_ssh_spark(self) -> None:
        # Headless Linux (no DISPLAY/WAYLAND_DISPLAY)
        t_hl = studio_topology.detect_topology(
            os_name="Linux",
            arch="x86_64",
            env={},
            proc_version="Linux version 6.8.0-generic",
            has_nvidia_dev=False,
            has_aien_gpu_dev=False,
            which=_fake_which({"xdg-open"}),
        )
        self.assertEqual(t_hl["topology"], "linux_headless")
        self.assertEqual(t_hl["mode"], "local")
        self.assertTrue(t_hl["should_skip_ssh_tunnel"])
        self.assertEqual(t_hl["open_action"], "print_url")
        self.assertIsNone(t_hl["browser_cmd"])

        # Direct Spark over SSH (even if DISPLAY forwarded/stale, SSH_CONNECTION -> headless print_url)
        t_ssh = studio_topology.detect_topology(
            os_name="Linux",
            arch="aarch64",
            env={"DISPLAY": "localhost:10.0", "SSH_CONNECTION": "10.0.0.2 51234 10.0.0.1 22"},
            proc_version="Linux version 7.0.0-nvidia",
            has_nvidia_dev=True,
            has_aien_gpu_dev=False,
            which=_fake_which({"xdg-open"}),
        )
        self.assertEqual(t_ssh["topology"], "spark_headless")
        self.assertTrue(t_ssh["is_spark_local"])
        self.assertTrue(t_ssh["should_skip_ssh_tunnel"])
        self.assertEqual(t_ssh["open_action"], "print_url")
        self.assertIsNone(t_ssh["browser_cmd"])
        act = studio_topology.open_studio_browser("http://127.0.0.1:18420/", t_ssh, dry_run=True)
        self.assertEqual(act["action"], "print_url")
        self.assertIsNone(act["cmd"])

    def test_03_macos_local_and_remote_spark_tunnel(self) -> None:
        # macOS Apple Silicon local mode
        t_mac_loc = studio_topology.detect_topology(
            os_name="Darwin",
            arch="arm64",
            env={},
            has_nvidia_dev=False,
            has_aien_gpu_dev=False,
            which=_fake_which({"open"}),
        )
        self.assertEqual(t_mac_loc["topology"], "macos_arm64")
        self.assertEqual(t_mac_loc["binary_target"], "macos")
        self.assertEqual(t_mac_loc["signing_mode"], "sha256+adhoc-codesign")
        self.assertEqual(t_mac_loc["mode"], "local")
        self.assertTrue(t_mac_loc["should_skip_ssh_tunnel"])
        self.assertTrue(t_mac_loc["remote_tunnel_supported"])
        self.assertEqual(t_mac_loc["open_action"], "browser")
        self.assertEqual(t_mac_loc["browser_cmd"], ["open", "http://127.0.0.1:18420/"])

        # macOS Apple Silicon optional remote Spark SSH tunnel mode
        t_mac_rem = studio_topology.detect_topology(
            os_name="Darwin",
            arch="arm64",
            env={},
            remote_target="spark",
            which=_fake_which({"open"}),
        )
        self.assertEqual(t_mac_rem["topology"], "macos_arm64")
        self.assertEqual(t_mac_rem["mode"], "remote_tunnel")
        self.assertFalse(t_mac_rem["should_skip_ssh_tunnel"])
        self.assertEqual(t_mac_rem["remote_target"], "spark")
        self.assertEqual(t_mac_rem["browser_cmd"], ["open", "http://127.0.0.1:18420/"])

        # macOS Intel (x86_64) remote-friendly
        t_intel = studio_topology.detect_topology(
            os_name="Darwin",
            arch="x86_64",
            env={},
            which=_fake_which({"open"}),
        )
        self.assertEqual(t_intel["topology"], "macos_x86_64")
        self.assertEqual(t_intel["binary_target"], "remote-only")
        self.assertEqual(t_intel["browser_kind"], "open")

    def test_04_windows_wsl2_and_powershell(self) -> None:
        # WSL 2 via WSL_DISTRO_NAME with wslview
        t_wsl1 = studio_topology.detect_topology(
            os_name="Linux",
            arch="x86_64",
            env={"WSL_DISTRO_NAME": "Ubuntu-24.04"},
            proc_version="Linux version 5.15.153.1-microsoft-standard-WSL2",
            has_nvidia_dev=False,
            has_aien_gpu_dev=False,
            which=_fake_which({"wslview", "explorer.exe", "cmd.exe"}),
        )
        self.assertEqual(t_wsl1["topology"], "wsl2")
        self.assertTrue(t_wsl1["is_wsl"])
        self.assertFalse(t_wsl1["native_exe_available"])
        self.assertEqual(t_wsl1["binary_target"], "x86linux")
        self.assertEqual(t_wsl1["mode"], "local")
        self.assertTrue(t_wsl1["should_skip_ssh_tunnel"])
        self.assertTrue(t_wsl1["remote_tunnel_supported"])
        self.assertEqual(t_wsl1["browser_cmd"], ["wslview", "http://127.0.0.1:18420/"])

        # WSL 2 via /proc/version 'microsoft' fallback to explorer.exe
        t_wsl2 = studio_topology.detect_topology(
            os_name="Linux",
            arch="x86_64",
            env={},
            proc_version="Linux version 6.6.36.3-microsoft-standard-WSL2",
            which=_fake_which({"explorer.exe", "cmd.exe"}),
        )
        self.assertEqual(t_wsl2["topology"], "wsl2")
        self.assertEqual(t_wsl2["browser_cmd"], ["explorer.exe", "http://127.0.0.1:18420/"])

        # WSL 2 fallback to cmd.exe /c start + optional remote Spark tunnel
        t_wsl3 = studio_topology.detect_topology(
            os_name="Linux",
            arch="x86_64",
            env={"WSL_INTEROP": "/run/WSL/1_interop"},
            proc_version="Linux",
            remote_target="spark",
            which=_fake_which({"cmd.exe"}),
        )
        self.assertEqual(t_wsl3["topology"], "wsl2")
        self.assertEqual(t_wsl3["mode"], "remote_tunnel")
        self.assertFalse(t_wsl3["should_skip_ssh_tunnel"])
        self.assertEqual(t_wsl3["browser_cmd"], ["cmd.exe", "/c", "start", "", "http://127.0.0.1:18420/"])

        # Native Windows / PowerShell outside WSL 2 -> never claims .exe exists
        t_win = studio_topology.detect_topology(
            os_name="Windows",
            arch="AMD64",
            env={},
            which=_fake_which(set()),
        )
        self.assertEqual(t_win["topology"], "windows_powershell")
        self.assertFalse(t_win["native_exe_available"])
        self.assertIn("WSL 2", t_win["guidance"])
        self.assertIn("x86linux", t_win["guidance"])

    def test_05_install_sh_dry_run_all_4_topologies(self) -> None:
        # Syntax check first
        syn = subprocess.run(["sh", "-n", str(REPO_ROOT / "install.sh")], capture_output=True, text=True, check=False)
        self.assertEqual(syn.returncode, 0, f"install.sh syntax error: {syn.stderr}")

        # 1. Direct DGX Spark with monitor attached
        d_spark = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "Linux",
            "OMEGA_INSTALL_TEST_ARCH": "aarch64",
            "OMEGA_INSTALL_TEST_WSL": "0",
            "OMEGA_INSTALL_TEST_GPU": "1",
            "OMEGA_INSTALL_TEST_DISPLAY": ":0",
            "OMEGA_INSTALL_TEST_WAYLAND": "",
            "OMEGA_INSTALL_TEST_SSH": "",
            "OMEGA_INSTALL_TEST_BROWSER": "xdg-open",
        })
        self.assertEqual(d_spark["TOPOLOGY"], "spark-desktop")
        self.assertEqual(d_spark["GB10"], "1")
        self.assertEqual(d_spark["SKIP_SSH_TUNNEL"], "1")
        self.assertEqual(d_spark["BROWSER_CMD"], "xdg-open")
        self.assertIn("aarch64", d_spark["TARGETS"])

        # 2. Headless Linux
        d_hl = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "Linux",
            "OMEGA_INSTALL_TEST_ARCH": "x86_64",
            "OMEGA_INSTALL_TEST_WSL": "0",
            "OMEGA_INSTALL_TEST_GPU": "0",
            "OMEGA_INSTALL_TEST_DISPLAY": "",
            "OMEGA_INSTALL_TEST_WAYLAND": "",
            "OMEGA_INSTALL_TEST_SSH": "",
        })
        self.assertEqual(d_hl["TOPOLOGY"], "linux-headless")
        self.assertEqual(d_hl["SKIP_SSH_TUNNEL"], "1")
        self.assertEqual(d_hl["BROWSER_CMD"], "print-url")
        self.assertIn("x86linux", d_hl["TARGETS"])

        # 3a. macOS Apple Silicon (native a64macos + SHA-256 + ad-hoc codesign)
        d_mac = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "Darwin",
            "OMEGA_INSTALL_TEST_ARCH": "arm64",
            "OMEGA_INSTALL_TEST_SSH": "",
        })
        self.assertEqual(d_mac["TOPOLOGY"], "macos-arm64")
        self.assertEqual(d_mac["SIGN_MODE"], "sha256+adhoc-codesign")
        self.assertEqual(d_mac["BROWSER_CMD"], "open")
        self.assertIn("macos", d_mac["TARGETS"])
        self.assertEqual(d_mac["SKIP_BUILD"], "0")

        # 3b. macOS Intel (remote tunnel mode)
        d_mac_x = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "Darwin",
            "OMEGA_INSTALL_TEST_ARCH": "x86_64",
            "OMEGA_INSTALL_TEST_SSH": "",
        })
        self.assertEqual(d_mac_x["TOPOLOGY"], "macos-x86_64")
        self.assertEqual(d_mac_x["LAUNCH_MODE"], "remote-tunnel")
        self.assertEqual(d_mac_x["SKIP_BUILD"], "1")
        self.assertEqual(d_mac_x["BROWSER_CMD"], "open")

        # 4a. Windows 10/11 WSL 2 (static x86linux, no .exe)
        d_wsl = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "Linux",
            "OMEGA_INSTALL_TEST_ARCH": "x86_64",
            "OMEGA_INSTALL_TEST_WSL": "1",
            "OMEGA_INSTALL_TEST_GPU": "0",
            "OMEGA_INSTALL_TEST_BROWSER": "wslview",
        })
        self.assertEqual(d_wsl["TOPOLOGY"], "wsl2")
        self.assertEqual(d_wsl["IS_WSL"], "1")
        self.assertEqual(d_wsl["NATIVE_EXE"], "0")
        self.assertIn("x86linux", d_wsl["TARGETS"])
        self.assertEqual(d_wsl["BROWSER_CMD"], "wslview")

        # 4b. Native Windows MSYS/PowerShell refusal (never claims .exe)
        d_win = _run_install_dry_run({
            "OMEGA_INSTALL_TEST_OS": "MINGW64_NT-10.0",
            "OMEGA_INSTALL_TEST_ARCH": "x86_64",
        })
        self.assertEqual(d_win["TOPOLOGY"], "windows-powershell")
        self.assertEqual(d_win["NATIVE_EXE"], "0")
        self.assertEqual(d_win["LAUNCH_MODE"], "wsl2-required")

    def test_06_studio_cli_dry_run_and_doctor_json(self) -> None:
        # Exercise real `studio --dry-run --json` end-to-end on the live Spark host
        proc = subprocess.run(
            [sys.executable, "-m", "studio.cli", "studio", "--dry-run", "--json"],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "tools")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"studio --dry-run failed: {proc.stderr}")
        doc = json.loads(proc.stdout)
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["dry_run"])
        self.assertTrue(doc["should_skip_ssh_tunnel"])
        self.assertEqual(doc["mode"], "local")
        self.assertIn("topology", doc["topology"])

        # Exercise `studio --remote spark --dry-run --json`
        proc_r = subprocess.run(
            [sys.executable, "-m", "studio.cli", "studio", "--remote", "spark", "--dry-run", "--json"],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "tools")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_r.returncode, 0, f"studio --remote --dry-run failed: {proc_r.stderr}")
        doc_r = json.loads(proc_r.stdout)
        self.assertEqual(doc_r["mode"], "remote_tunnel")
        self.assertFalse(doc_r["should_skip_ssh_tunnel"])
        self.assertEqual(doc_r["topology"]["remote_target"], "spark")


if __name__ == "__main__":
    unittest.main()
