"""sort_can — place a can on a colored plate (the multitask Can-Sorting task).

Maps to IsaacLab gym id ``Isaac-Can-Sorting-OpenArm-DexHand-v0``. The instruction
and ``target_color`` mirror the multitask prompt logic in
``scripts/eval/gr00t_infer_agent.py`` (orange/green plate selection).
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.common import Embodiment
from agentbot.contracts.skills import SkillCall, SkillParamSpec, SkillSpec
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.skills.base import Skill

TASK_NAME = "Isaac-Can-Sorting-OpenArm-DexHand-v0"


class SortCanSkill(Skill):
    _spec = SkillSpec(
        name="sort_can",
        description="Pick up the can and place it on the plate of the given color.",
        params=[
            SkillParamSpec(
                name="target_color", type="enum", enum=["orange", "green"],
                description="Which colored plate to place the can on. Omit it for a bare "
                            "'sort can' — the Brain fills it from the can on the table.",
                required=False,
            ),
        ],
        preconditions=["arm_idle", "can_visible"],
        postconditions=["can_on_target_plate"],
        embodiments=[Embodiment.SIM],
    )

    @property
    def spec(self) -> SkillSpec:
        return self._spec

    def to_vla_request(self, call: SkillCall, ctx: dict[str, Any]) -> VlaTaskRequest:
        # Color precedence: explicit call arg -> scene color injected via ctx -> "orange".
        # sim_session re-syncs to the env's actual target anyway, so this only sets the prompt.
        color = call.args.get("target_color") or ctx.get("target_color") or "orange"
        return VlaTaskRequest(
            skill_call_id=call.skill_call_id,
            embodiment=ctx.get("embodiment", Embodiment.SIM),
            task_name=TASK_NAME,
            instruction=f"place the can on the {color} plate",
            checkpoint=ctx["checkpoint"],
            checkpoint_name=ctx.get("checkpoint_name"),
            gr00t_ver=ctx.get("gr00t_ver", "N1.7"),
            params={"target_color": color},
        )
