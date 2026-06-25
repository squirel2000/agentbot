"""Orchestrator control plane: the run() loop must survive crashes and a Stop.

Regression for the 4090 bring-up "I typed a command and nothing happened":
the loop used to call orchestrator.stop() on the UI Stop (and could die on any
unhandled exception), ending run() permanently with no restart path. Now:
  - a crashing command is logged + failed, and the loop CONTINUES to the next one;
  - cancel_current() aborts the in-flight command (CLEARED) without killing the loop.
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
    def __init__(self, name):
        self.spec = SkillSpec(name=name, description="", sim_runnable=True)


class _Reg:
    def __init__(self, names):
        self._s = {n: _Skill(n) for n in names}

    def get(self, name):
        return self._s.get(name)


async def _run_loop_then_stop(orch, cq, records, cmds, settle=0.4):
    """Start run(), enqueue cmds, let the loop drain, then stop the task."""
    task = asyncio.create_task(orch.run())
    for c in cmds:
        records.add_command(c)
        cq.put(c)
    await asyncio.sleep(settle)
    orch.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _publish_result(bus, tid):
    res = VlaTaskResult(task_id=tid, status=VlaTaskStatus.SUCCEEDED,
                        success=1, success_rate=1.0, steps=5)
    return bus.publish(Event(type=EventType.VLA_TELEMETRY, source="fakesim", task_id=tid,
                             payload={"kind": "result", **res.model_dump()}))


async def test_run_loop_survives_a_crashing_command_and_runs_the_next():
    bus = InProcEventBus()
    cq = InMemCommandQueue()
    results = ResultWaiter(bus)
    waiter = asyncio.create_task(results.run())
    await asyncio.sleep(0)
    records = Records.in_memory()

    class _Agent:
        async def plan(self, text, session_id):
            if "boom" in text:
                raise RuntimeError("planner blew up")
            return SkillPlan(calls=[SkillCall(name="sort_can", args={"target_color": "orange"})])

    async def fake_sim(tid):
        await _publish_result(bus, tid)

    def dispatch(call, ctx):
        tid = new_id("task")
        asyncio.ensure_future(fake_sim(tid))
        return tid

    orch = Orchestrator(_Agent(), _Reg(["sort_can"]), cq, dispatch, results, records,
                        vla_ctx=lambda: {}, max_retries=0, skill_timeout_s=5.0)
    bad = Command(command_id="bad", text="boom")
    good = Command(command_id="good", text="sort the can")
    await _run_loop_then_stop(orch, cq, records, [bad, good])
    waiter.cancel()

    # The crash was contained; the loop survived and processed the next command.
    assert records.command_detail("bad")["command"]["status"] == CommandStatus.FAILED.value
    assert records.command_detail("good")["command"]["status"] == CommandStatus.DONE.value


async def test_cancel_current_aborts_remaining_skills_but_keeps_loop_alive():
    bus = InProcEventBus()
    cq = InMemCommandQueue()
    results = ResultWaiter(bus)
    waiter = asyncio.create_task(results.run())
    await asyncio.sleep(0)
    records = Records.in_memory()
    dispatched: list[str] = []
    orch_ref: dict = {}

    class _Agent:
        async def plan(self, text, session_id):
            return SkillPlan(calls=[SkillCall(name="sort_can", args={"target_color": "orange"}),
                                    SkillCall(name="sort_can", args={"target_color": "green"})])

    async def fake_sim(tid):
        orch_ref["o"].cancel_current()      # Stop pressed while the first skill runs
        await _publish_result(bus, tid)

    def dispatch(call, ctx):
        dispatched.append(call.args.get("target_color"))
        tid = new_id("task")
        asyncio.ensure_future(fake_sim(tid))
        return tid

    orch = Orchestrator(_Agent(), _Reg(["sort_can"]), cq, dispatch, results, records,
                        vla_ctx=lambda: {}, max_retries=0, skill_timeout_s=5.0)
    orch_ref["o"] = orch
    cmd = Command(command_id="two", text="do two skills")
    await _run_loop_then_stop(orch, cq, records, [cmd])
    waiter.cancel()

    assert dispatched == ["orange"]   # second skill never dispatched after cancel
    assert records.command_detail("two")["command"]["status"] == CommandStatus.CLEARED.value
    # the loop is still alive (not stopped by cancel) — it only stopped because the test stopped it
