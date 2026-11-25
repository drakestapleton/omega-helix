"""Sovereign Runtime Workflows for Omega Helix Phase 7.

Python 3 standard library only — zero external packages, zero telemetry:
- ``anti_dna``: Contrastive Anti-DNA firewall strands + FNV-1a cryptographic receipts + LTP0 disagreement deep-thinking (`spguard`)
- ``sleep_cycle``: Verifier-gated offline memory consolidation (`fcache` + `trainfc` + regression-gated `spext`)
- ``compile_green``: Latent-space + micro-TTT compile-until-green loop -> crystallized `.sp`
- ``hf_dna``: Open-weight HF Safetensors (`import-hf`) -> DNA specialist composition + firewall verification
"""

from __future__ import annotations

from .anti_dna import forge_anti_strand, read_sp_pack_hash, run_firewall_decode, verify_firewall_receipt
from .sleep_cycle import run_sleep_cycle
from .compile_green import run_compile_until_green
from .hf_dna import run_open_trunk_dna_demo

__all__ = [
    "forge_anti_strand",
    "read_sp_pack_hash",
    "run_firewall_decode",
    "verify_firewall_receipt",
    "run_sleep_cycle",
    "run_compile_until_green",
    "run_open_trunk_dna_demo",
]
