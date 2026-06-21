"""pick — grasp and lift a named object (atomic primitive from the Skill Layer).

This is one of the "Pick, Place, …" atomic skills shown in architecture.png block 4.
Unlike ``sort_can`` (a fused pick+place), ``pick``/``place``/``home`` are composable
primitives the Brain can chain into multi-step plans (e.g. pick → place → home).

Skeleton: the gym id below follows the project's naming convention and is confirmed
during the MVP phase (a dedicated grasp task or a parameterized manipulation task).
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillParamSpec, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.skills.base import Skill

# TODO(MVP): confirm the registered IsaacLab gym id for the grasp primitive.
TASK_NAME = "Isaac-Pick-OpenArm-DexHand-v0"


class PickSkill(Skill):
    _spec = SkillSpec(
        name="pick",
        description="Pick up (grasp and lift) the named object.",
        params=[
            SkillParamSpec(
                name="object", type="string",
                description="The object to pick up, e.g. 'can', 'cup', 'block'.", required=True,
            ),
        ],
        preconditions=["arm_idle", "object_visible"],
        postconditions=["object_grasped"],
        embodiments=[Embodiment.SIM],
    )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def to_vla_request(self, call: SkillCall, ctx: dict[str, Any]) -> VlaTaskRequest:
        obj = call.args["object"]
        return VlaTaskRequest(
            skill_call_id=call.skill_call_id,
            embodiment=ctx.get("embodiment", Embodiment.SIM),
            task_name=TASK_NAME,
            instruction=f"pick up the {obj}",
            checkpoint=ctx["checkpoint"],
            checkpoint_name=ctx.get("checkpoint_name"),
            gr00t_ver=ctx.get("gr00t_ver", "N1.7"),
            params={"object": obj},
        )
