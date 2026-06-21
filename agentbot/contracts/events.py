"""Monitor / Event Bus (block 3) contracts.

Every layer publishes ``Event``s; the Console, the Brain's feedback loop, and the
SafetyWatchdog subscribe. ``payload`` is free-form but the typed helpers below
(``RobotState``, ``SafetyEvent``, ``SystemStatus``) document/validate the common shapes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Severity, new_id, now_ts


class EventType(str, Enum):
    ROBOT_STATE = "robot_state"
    ENVIRONMENT = "environment"
    SAFETY = "safety"
    SYSTEM_STATUS = "system_status"
    VLA_TELEMETRY = "vla_telemetry"
    SKILL_LIFECYCLE = "skill_lifecycle"
    AGENT_LIFECYCLE = "agent_lifecycle"
    USER_IO = "user_io"
    CAMERA = "camera"


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    type: EventType
    source: str                          # producing component, e.g. "vla.worker"
    severity: Severity = Severity.INFO
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    ts: float = Field(default_factory=now_ts)


# --- typed payload helpers (validated views of Event.payload) ----------------
class RobotState(BaseModel):
    mode: str = "idle"                   # running | idle | blocked | failed | done
    detail: str = ""


class SafetyEvent(BaseModel):
    kind: str                            # collision | emergency | violation
    immediate: bool = False              # True => trip the SafetyWatchdog (immediate-stop path)
    detail: str = ""


class EnvironmentState(BaseModel):
    objects: list[str] = Field(default_factory=list)
    persons: int = 0
    scene: str = ""


class SystemStatus(BaseModel):
    battery: Optional[float] = None
    temperature: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    gpu_util: Optional[float] = None
