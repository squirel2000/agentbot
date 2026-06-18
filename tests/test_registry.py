"""The registry exposes skills as LLM tool schemas for the Brain's tool-calling."""
from agentbot.skills.registry import SkillRegistry
from agentbot.skills.builtin.sort_can import SortCanSkill
from agentbot.skills.builtin.pour_water import PourWaterSkill


def test_registry_lists_tool_schemas():
    reg = SkillRegistry()
    reg.register(SortCanSkill())
    reg.register(PourWaterSkill())
    names = [t["name"] for t in reg.as_tool_schemas()]
    assert "sort_can" in names and "pour_water" in names
    assert reg.get("sort_can") is not None
    assert {s.name for s in reg.list_specs()} == {"sort_can", "pour_water"}


def test_registry_unknown_skill_returns_none():
    assert SkillRegistry().get("nope") is None
