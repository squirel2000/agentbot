"""Hardware backend (blocks 6-7) — FUTURE.

When ``embodiment == hardware``, this bridges the VLA action chunk to the real
robot via the existing control node
``Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py`` (VLA -> ROS2
JointTrajectoryController -> OpenArm) in ``openarm_ros2``, and republishes ROS2
``/joint_states`` / ``/vla/diagnostics`` / safety into the AgentBot event bus
(optionally through an MQTT<->ROS2 bridge). Sim and hardware share the Brain,
Skill, and Monitor layers unchanged — only the backend differs.
"""
from __future__ import annotations

from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult
from agentbot.vla.isaac_runner import Publish


class HardwareBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
        raise NotImplementedError(
            "future: hardware backend (blocks 6-7) — bridge to "
            "Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py + openarm_ros2."
        )
