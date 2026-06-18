"""User Input (block 1) <-> Brain (block 2) contracts."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .common import Modality, new_id, now_ts
from .skills import SkillPlan


class UserMessage(BaseModel):
    session_id: str
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    modality: Modality = Modality.TEXT
    text: str = ""
    image_path: Optional[str] = None     # path or data-uri for VLM input
    ts: float = Field(default_factory=now_ts)


class AgentState(str, Enum):
    THINKING = "thinking"
    PLANNING = "planning"
    ACTING = "acting"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


class AgentResponse(BaseModel):
    session_id: str
    turn_id: str
    state: AgentState = AgentState.DONE
    text: str = ""                       # natural-language reply
    plan: Optional[SkillPlan] = None     # structured plan if any
    ts: float = Field(default_factory=now_ts)
