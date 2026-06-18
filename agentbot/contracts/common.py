"""Shared primitives for every contract module."""
from __future__ import annotations

import time
import uuid
from enum import Enum


def now_ts() -> float:
    """Unix seconds (float) — the one clock all contracts use."""
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Modality(str, Enum):
    VOICE = "voice"
    TEXT = "text"
    UI = "ui"


class Embodiment(str, Enum):
    SIM = "sim"
    HARDWARE = "hardware"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
