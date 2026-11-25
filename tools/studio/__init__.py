"""Omega Helix Studio — zero-dependency pure-stdlib backend, server, and ANSI CLI."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

STUDIO_DIR = Path(__file__).resolve().parent
TOOLS_DIR = STUDIO_DIR.parent
REPO_ROOT = Path(os.environ.get("OMEGA_HELIX_HOME") or TOOLS_DIR.parent).resolve()
OPENWALDO_DIR = TOOLS_DIR / "openwaldo"
UI_DIR = STUDIO_DIR / "ui"
CACHE_PATH = STUDIO_DIR / "waldo_catalog_cache.json"
