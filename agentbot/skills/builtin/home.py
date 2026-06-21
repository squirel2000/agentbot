"""home — return the arm to its neutral home/rest pose (utility primitive).

A no-arg reset skill. Useful as the final step of a plan, or for the orchestrator
to issue after an abort to leave the arm in a known-safe pose before re-planning.

Skeleton: the gym id below follows the project's naming convention. In practice this
can map to a lightweight non-policy reset primitive rather than a full VLA rollout;
that backend choice is settled during the MVP phase.
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.skills.base import Skill

# TODO(MVP): confirm whether this maps to a gym task or a direct reset primitive.
TASK_NAME = "Isaac-Home-OpenArm-DexHand-v0"


class HomeSkill(Skill):
    _spec = SkillSpec(
        name="home",
        description="Return the arm to its neutral home / rest pose.",
        params=[],
        preconditions=[],
        postconditions=["arm_idle", "arm_at_home"],
        embodiments=[Embodiment.SIM],
    )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def to_vla_request(self, call: SkillCall, ctx: dict[str, Any]) -> VlaTaskRequest:
        return VlaTaskRequest(
            skill_call_id=call.skill_call_id,
            embodiment=ctx.get("embodiment", Embodiment.SIM),
            task_name=TASK_NAME,
            instruction="return to the home position",
            checkpoint=ctx["checkpoint"],
            checkpoint_name=ctx.get("checkpoint_name"),
            gr00t_ver=ctx.get("gr00t_ver", "N1.7"),
            params={},
        )
