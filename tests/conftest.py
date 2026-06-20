import asyncio
from dataclasses import dataclass
import pytest
from agentbot.monitor.event_bus import InProcEventBus
from agentbot.monitor.command_queue import InMemCommandQueue
from agentbot.monitor.results import ResultWaiter
from agentbot.records.store import Records
from agentbot.brain.orchestrator import Orchestrator
from agentbot.contracts.skills import SkillPlan, SkillCall
from agentbot.contracts.vla import VlaTaskResult, VlaTaskStatus
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.common import new_id


class StubAgent:
    """plan() returns a fixed ordered list of skill calls (names)."""
    def __init__(self, names): self._names = names
    async def plan(self, text, session_id):
        return SkillPlan(intent=text, calls=[SkillCall(name=n) for n in self._names])


class StubRegistry:
    def get(self, name): return object()   # non-None so the orchestrator dispatches


@dataclass
class FakeEnv:
    orchestrator: Orchestrator
    command_queue: InMemCommandQueue
    records: Records
    bus: InProcEventBus
    waiter_task: asyncio.Task


@pytest.fixture
async def fake_env():
    bus = InProcEventBus()
    cq = InMemCommandQueue()
    results = ResultWaiter(bus)
    waiter_task = asyncio.create_task(results.run())
    await asyncio.sleep(0)               # let ResultWaiter subscribe
    records = Records.in_memory()

    # success map: skill A fails its first attempt then succeeds; B succeeds first try.
    attempts: dict[str, int] = {}
    succeed_on_attempt = {"A": 1, "B": 0}   # 0-based attempt index that first succeeds

    async def fake_sim(task_id: str, skill: str):
        idx = attempts.get(skill, 0); attempts[skill] = idx + 1
        ok = idx >= succeed_on_attempt[skill]
        res = VlaTaskResult(task_id=task_id,
                            status=VlaTaskStatus.SUCCEEDED if ok else VlaTaskStatus.FAILED,
                            success=1 if ok else 0, success_rate=1.0 if ok else 0.0, steps=10)
        await bus.publish(Event(type=EventType.VLA_TELEMETRY, source="fakesim", task_id=task_id,
                                payload={"kind": "result", **res.model_dump()}))

    def dispatch(call, ctx):
        task_id = new_id("task")
        asyncio.ensure_future(fake_sim(task_id, call.name))   # publish a result shortly after
        return task_id

    orch = Orchestrator(StubAgent(["A", "B"]), StubRegistry(), cq, dispatch, results, records,
                        vla_ctx=lambda: {}, max_retries=2, skill_timeout_s=5.0)
    yield FakeEnv(orch, cq, records, bus, waiter_task)
    waiter_task.cancel()
