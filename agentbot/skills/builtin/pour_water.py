"""pour_water — pour water from one container into another.

Skeleton skill mapped to the PourWater sim dataset
(``OpenArm_O6_PourWater_Sim_Dataset_0519``). The exact gym id is confirmed during
the MVP phase; the placeholder below follows the project's task-naming convention.
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.skills.base import Skill

# TODO(MVP): confirm the registered IsaacLab gym id for the PourWater task.
TASK_NAME = "Isaac-Pour-Water-OpenArm-DexHand-v0"


class PourWaterSkill(Skill):
    _spec = SkillSpec(
        name="pour_water",
        description="Pour the water from the source container into the target container.",
        params=[],
        preconditions=["arm_idle"],
        postconditions=["water_poured"],
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
            instruction="pour the water into the cup",
            checkpoint=ctx["checkpoint"],
            checkpoint_name=ctx.get("checkpoint_name"),
            gr00t_ver=ctx.get("gr00t_ver", "N1.7"),
            params={},
        )
