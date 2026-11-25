"""Sovereign Distillation Pipeline Network for Omega Helix.

Python 3 standard library only — zero external dependencies, zero telemetry.
"""

from __future__ import annotations

DISTILL_SCHEMA = 1
OLLMDK01_MAGIC = 3544415663467285583

from .verifiers import BUILTIN_VERIFIERS, VerifierPipeline, verify_record
from .harvest import (
    fnv1a64,
    harvest_paths,
    read_ollmdk01_header,
    write_ollmdk01,
    write_openwaldo_export,
)
from .teachers import (
    DEFAULT_LOCAL_ENDPOINTS,
    LocalHelixTeacher,
    OpenAICompatTeacher,
    TeacherAdapter,
    TeacherRouter,
    TraceReplayTeacher,
    probe_local_teachers,
)
from .pipeline import DistillManager, DistillRun, estimate_distill

__all__ = [
    "DISTILL_SCHEMA",
    "OLLMDK01_MAGIC",
    "BUILTIN_VERIFIERS",
    "DEFAULT_LOCAL_ENDPOINTS",
    "VerifierPipeline",
    "verify_record",
    "fnv1a64",
    "harvest_paths",
    "read_ollmdk01_header",
    "write_ollmdk01",
    "write_openwaldo_export",
    "TeacherAdapter",
    "OpenAICompatTeacher",
    "LocalHelixTeacher",
    "TraceReplayTeacher",
    "TeacherRouter",
    "probe_local_teachers",
    "DistillManager",
    "DistillRun",
    "estimate_distill",
]
