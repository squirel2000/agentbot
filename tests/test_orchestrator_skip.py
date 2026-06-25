"""Orchestrator skips skills with no runnable sim task (sim_runnable=False).

Regression for the 4090 bring-up: the Brain sometimes decomposes "sort the can…"
into [pick, sort_can], but `pick` maps to an unregistered gym id. The orchestrator
must SKIP such skeleton skills (not hang on the 30-min VLA timeout) and still run
the runnable ones.
"""
import asyncio

from agentbot.brain.orchestrator import Orchestrator
from agentbot.contracts.commands import Command, CommandStatus
from agentbot.contracts.common import new_id
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.skills import SkillCall, SkillPlan, SkillSpec
from agentbot.contracts.vla import VlaTaskResult, VlaTaskStatus
from agentbot.monitor.command_queue import InMemCommandQueue
from agentbot.monitor.event_bus import InProcEventBus
from agentbot.monitor.results import ResultWaiter
from agentbot.records.store import Records


class _Skill:
    def __init__(self, name, runnable):
        self.spec = SkillSpec(name=name, description="", sim_runnable=runnable)


class _Reg:
    def __init__(self, skills):
        self._skills = skills

    def get(self, name):
        return self._skills.get(name)


class _Agent:
    def __init__(self, names):
        self._names = names

    async def plan(self, text, session_id):
        # pick(can) then sort_can(orange) — pick is the skeleton skill to be skipped.
        args = {"pick": {"object": "can"}, "sort_can": {"target_color": "orange"}}
        return SkillPlan(calls=[SkillCall(name=n, args=args.get(n, {})) for n in self._names])


async def test_orchestrator_skips_non_sim_runnable_pick_and_runs_sort_can():
    bus = InProcEventBus()
    cq = InMemCommandQueue()
    results = ResultWaiter(bus)
    waiter = asyncio.create_task(results.run())
    await asyncio.sleep(0)
    records = Records.in_memory()

    reg = _Reg({"pick": _Skill("pick", False), "sort_can": _Skill("sort_can", True)})
    dispatched: list[str] = []

    async def fake_sim(task_id: str):
        res = VlaTaskResult(task_id=task_id, status=VlaTaskStatus.SUCCEEDED,
                            success=1, success_rate=1.0, steps=10)
        await bus.publish(Event(type=EventType.VLA_TELEMETRY, source="fakesim", task_id=task_id,
                                payload={"kind": "result", **res.model_dump()}))

    def dispatch(call, ctx):
        dispatched.append(call.name)
        tid = new_id("task")
        asyncio.ensure_future(fake_sim(tid))
        return tid

    orch = Orchestrator(_Agent(["pick", "sort_can"]), reg, cq, dispatch, results, records,
                        vla_ctx=lambda: {}, max_retries=0, skill_timeout_s=5.0)
    records.add_command(Command(command_id="cmd_skip", text="sort the can onto the orange plate"))
    cq.put(Command(command_id="cmd_skip", text="sort the can onto the orange plate"))
    await orch.run_one()
    waiter.cancel()

    # pick was skipped (never dispatched); only sort_can reached the VLA.
    assert dispatched == ["sort_can"]
    detail = records.command_detail("cmd_skip")
    assert detail["command"]["status"] == CommandStatus.DONE.value
    runs = detail["skill_runs"]
    assert any(r["skill"] == "pick" and r["status"] == "skipped" for r in runs)
    assert any(r["skill"] == "sort_can" and r["success"] == 1 for r in runs)
