"""Command contract — the top-level user intent unit (Phase 2 orchestrator).

A ``Command`` travels from UI/console through the command queue to the Brain
orchestrator, which attaches a ``SkillPlan`` and tracks execution to completion.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .common import new_id, now_ts
from .skills import SkillPlan


class CommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CLEARED = "cleared"


class Command(BaseModel):
    command_id: str = Field(default_factory=lambda: new_id("cmd"))
    session_id: str = "console"
    text: str
    status: CommandStatus = CommandStatus.QUEUED
    plan: Optional[SkillPlan] = None
    result_summary: str = ""
    created_ts: float = Field(default_factory=now_ts)
