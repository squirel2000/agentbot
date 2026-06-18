"""IsaacLab rollout (block 5, System 1 closed-loop execution).

``run_rollout`` is the on-demand equivalent of one run in
``scripts/eval/gr00t_infer_agent.py``: build the IsaacLab env, connect the GR00T
policy client, and step the per-episode loop while streaming telemetry.

SCAFFOLD scope: ``run_rollout`` is defined with its real signature + a docstring
mapping each phase to the exact source to port, but the body raises
NotImplementedError (porting it needs ``env_isaaclab`` + a GPU = the MVP phase).
``FakeRunner`` provides a no-IsaacLab implementation that emits synthetic telemetry
+ a result so the queue/worker/event-bus path is fully testable today.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult, VlaTaskStatus, VlaTelemetry

Publish = Callable[[Event], None]


def _telemetry_event(req: VlaTaskRequest, tel: VlaTelemetry) -> Event:
    return Event(type=EventType.VLA_TELEMETRY, source="vla.isaac_runner",
                 task_id=req.task_id, payload=tel.model_dump())


class Runner(Protocol):
    def run_rollout(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult: ...


def run_rollout(req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
    """Run one IsaacLab episode for *req*; publish telemetry; return the result.

    PORT MAP (from scripts/eval/gr00t_infer_agent.py):
      1. AppLauncher + ``gym.make(req.task_name)``        -> env setup
      2. JointMapper(env)                                  -> scripts/eval/utils/joint_mapper.py
      3. obs -> GR00T state/video/annotation dict          -> obs assembly (multitask prompt = req.instruction)
      4. policy_client.get_action(obs)                     -> scripts/eval/utils/gr00t_client_adapter.py
      5. map action -> Isaac joints (+ filter.py)          -> action mapping
      6. env.step(action) x16, check task_done()           -> stepping + termination
      7. episode_data_saver.save(...) (if options.save_video)
    Publish a VlaTelemetry event per step; return VlaTaskResult with artifacts.
    """
    raise NotImplementedError(
        "MVP: port the per-episode loop from scripts/eval/gr00t_infer_agent.py "
        "(requires the env_isaaclab conda env + a GPU)."
    )


class FakeRunner:
    """No-IsaacLab rollout: deterministic synthetic episode for end-to-end plumbing tests."""

    def __init__(self, n_steps: int = 8, succeed: bool = True) -> None:
        self.n_steps = n_steps
        self.succeed = succeed

    def run_rollout(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
        t0 = time.time()
        for step in range(1, self.n_steps + 1):
            tel = VlaTelemetry(task_id=req.task_id, step=step, sim_time=step * 0.0417,
                               inference_latency_s=0.12)
            publish(_telemetry_event(req, tel))
        return VlaTaskResult(
            task_id=req.task_id,
            status=VlaTaskStatus.SUCCEEDED if self.succeed else VlaTaskStatus.FAILED,
            episodes=1, success=1 if self.succeed else 0,
            success_rate=1.0 if self.succeed else 0.0,
            steps=self.n_steps, duration_s=round(time.time() - t0, 4),
            artifacts={"note": "fake rollout (no IsaacLab)"},
        )
