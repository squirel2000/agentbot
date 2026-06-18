"""Skill base class (block 4).

A Skill is a *semantic action tool*: it declares a ``SkillSpec`` (name, params,
constraints — also exported as an LLM tool schema), validates a ``SkillCall``
against that spec, and translates the call into a concrete ``VlaTaskRequest``.

``ctx`` is supplied by the API/deps layer and carries the resolved checkpoint and
embodiment, so skills NEVER hardcode a checkpoint path:
    ctx = {"checkpoint": <resolved path>, "checkpoint_name": <registry key>,
           "embodiment": Embodiment, "gr00t_ver": str}
"""
from __future__ import annotations

import abc
from typing import Any

from agentbot.contracts.skills import SkillCall, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest


class Skill(abc.ABC):
    @property
    @abc.abstractmethod
    def spec(self) -> SkillSpec: ...

    def validate(self, call: SkillCall) -> tuple[bool, str]:
        """Check call.args against the spec's params (required / enum / min / max)."""
        by_name = {p.name: p for p in self.spec.params}
        for p in self.spec.params:
            if p.required and p.name not in call.args:
                return False, f"missing required param '{p.name}'"
        for name, value in call.args.items():
            p = by_name.get(name)
            if p is None:
                return False, f"unknown param '{name}'"
            if p.enum is not None and value not in p.enum:
                return False, f"param '{name}' must be one of {p.enum}, got {value!r}"
            if p.minimum is not None and isinstance(value, (int, float)) and value < p.minimum:
                return False, f"param '{name}' below minimum {p.minimum}"
            if p.maximum is not None and isinstance(value, (int, float)) and value > p.maximum:
                return False, f"param '{name}' above maximum {p.maximum}"
        return True, ""

    @abc.abstractmethod
    def to_vla_request(self, call: SkillCall, ctx: dict[str, Any]) -> VlaTaskRequest:
        """Translate a validated SkillCall into a VlaTaskRequest for the engine."""
        ...
