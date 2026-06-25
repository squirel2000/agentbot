"""place — put the currently held object onto a named target (atomic primitive).

Composes with ``pick``: a typical plan is [pick(object) → place(target) → home].
``preconditions=["object_grasped"]`` lets the Brain / orchestrator reason about
ordering (place requires a prior successful pick).

Skeleton: the gym id below follows the project's naming convention and is confirmed
during the MVP phase.
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillParamSpec, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.skills.base import Skill

# TODO(MVP): confirm the registered IsaacLab gym id for the place primitive.
TASK_NAME = "Isaac-Place-OpenArm-DexHand-v0"


class PlaceSkill(Skill):
    _spec = SkillSpec(
        name="place",
        description="Place the currently held object onto the named target location.",
        params=[
            SkillParamSpec(
                name="target", type="string",
                description="Where to place the held object, e.g. 'orange plate', 'tray', 'shelf'.",
                required=True,
            ),
        ],
        preconditions=["object_grasped"],
        postconditions=["object_released", "arm_idle"],
        embodiments=[Embodiment.SIM],
        sim_runnable=False,  # Isaac-Place-OpenArm-DexHand-v0 is not a registered gym id (skeleton)
    )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def to_vla_request(self, call: SkillCall, ctx: dict[str, Any]) -> VlaTaskRequest:
        target = call.args["target"]
        return VlaTaskRequest(
            skill_call_id=call.skill_call_id,
            embodiment=ctx.get("embodiment", Embodiment.SIM),
            task_name=TASK_NAME,
            instruction=f"place the held object on the {target}",
            checkpoint=ctx["checkpoint"],
            checkpoint_name=ctx.get("checkpoint_name"),
            gr00t_ver=ctx.get("gr00t_ver", "N1.7"),
            params={"target": target},
        )
