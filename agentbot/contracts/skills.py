"""Skill Layer (block 4) contracts + the Brain's structured output (block 2).

``SkillPlan`` is literally the "ClawBot Output (Structured)" box in
architecture.png. ``SkillSpec.to_tool_schema()`` exports a skill as an LLM
tool/function schema so the Brain can select skills via tool-calling.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Embodiment, new_id, now_ts


class SkillParamSpec(BaseModel):
    name: str
    type: str = "string"          # string | number | integer | boolean | enum
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[list[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class SkillSpec(BaseModel):
    name: str
    description: str
    params: list[SkillParamSpec] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    embodiments: list[Embodiment] = Field(default_factory=lambda: [Embodiment.SIM])
    sim_runnable: bool = True  # False = skeleton skill with no registered IsaacLab gym id; orchestrator skips it

    def to_tool_schema(self) -> dict:
        """Export as an LLM tool schema (Anthropic/OpenAI ``input_schema`` shape)."""
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.params:
            if p.enum:
                s: dict = {"enum": p.enum}
            else:
                s = {"type": "number" if p.type in ("number", "integer") else p.type}
            s["description"] = p.description
            if p.minimum is not None:
                s["minimum"] = p.minimum
            if p.maximum is not None:
                s["maximum"] = p.maximum
            props[p.name] = s
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", "properties": props, "required": required},
        }


class SkillCall(BaseModel):
    skill_call_id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    rationale: str = ""


class SkillPlan(BaseModel):
    """The Brain's structured output: intent + an ordered list of skill calls."""
    intent: str = ""
    goal: str = ""
    context_summary: str = ""
    calls: list[SkillCall] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)


class SkillStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"


class SkillResult(BaseModel):
    skill_call_id: str
    name: str
    status: SkillStatus
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None
    ts: float = Field(default_factory=now_ts)
