"""Skill registry — the catalog the Brain selects from (as LLM tools)."""
from __future__ import annotations

from typing import Optional

from agentbot.contracts.skills import SkillSpec
from agentbot.skills.base import Skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.spec.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_specs(self) -> list[SkillSpec]:
        return [s.spec for s in self._skills.values()]

    def as_tool_schemas(self) -> list[dict]:
        """All skills as LLM tool/function schemas (for the Brain's tool-calling)."""
        return [s.spec.to_tool_schema() for s in self._skills.values()]
