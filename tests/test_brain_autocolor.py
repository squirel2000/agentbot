"""Bare "sort can" resolves the target plate color from the live scene.

The user types "sort can" with no color; the Brain fills the color of the can on the
table (env's randomized target, published to state['environment'] by sim_session) so
the plan/instruction matches the scene. An explicit "...green plate" always wins.
"""
from agentbot.brain.agent import BrainAgent
from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillPlan
from agentbot.skills.builtin.sort_can import SortCanSkill


def _agent(scene_color):
    # llm/registry/conversation are unused by _resolve_colors.
    return BrainAgent(llm=None, registry=None, conversation=None,
                      scene_provider=lambda: scene_color)


def _plan(args):
    return SkillPlan(calls=[SkillCall(name="sort_can", args=args)])


def test_bare_sort_can_uses_scene_color():
    p = _plan({})
    _agent("green")._resolve_colors("sort can", p)
    assert p.calls[0].args["target_color"] == "green"


def test_scene_color_overrides_a_hallucinated_llm_color_for_bare_command():
    p = _plan({"target_color": "orange"})        # LLM guessed orange...
    _agent("green")._resolve_colors("sort can", p)
    assert p.calls[0].args["target_color"] == "green"   # ...but the table can is green


def test_explicit_user_color_wins_over_scene():
    p = _plan({"target_color": "orange"})
    _agent("green")._resolve_colors("sort the can onto the orange plate", p)
    assert p.calls[0].args["target_color"] == "orange"


def test_explicit_user_color_fills_when_llm_omitted():
    p = _plan({})
    _agent("orange")._resolve_colors("put it on the green plate", p)
    assert p.calls[0].args["target_color"] == "green"


def test_scene_unknown_defaults_to_orange():
    p = _plan({})
    _agent(None)._resolve_colors("sort can", p)
    assert p.calls[0].args["target_color"] == "orange"


def test_sort_can_to_vla_request_tolerates_missing_color():
    skill = SortCanSkill()
    ctx = {"checkpoint": "/tmp/ck", "embodiment": Embodiment.SIM}
    assert skill.to_vla_request(SkillCall(name="sort_can", args={}), ctx).instruction \
        == "place the can on the orange plate"
    ctx2 = {"checkpoint": "/tmp/ck", "target_color": "green"}
    assert skill.to_vla_request(SkillCall(name="sort_can", args={}), ctx2).instruction \
        == "place the can on the green plate"
