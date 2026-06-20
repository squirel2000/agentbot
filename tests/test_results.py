"""Tests for ResultWaiter — awaits a VLA skill's terminal result by task_id."""
from __future__ import annotations

import asyncio

from agentbot.monitor.event_bus import InProcEventBus
from agentbot.monitor.results import ResultWaiter
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskResult, VlaTaskStatus


async def test_result_waiter_resolves_by_task_id():
    bus = InProcEventBus(); w = ResultWaiter(bus)
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0)
    fut = w.expect("task_1", timeout=1.0)
    res = VlaTaskResult(task_id="task_1", status=VlaTaskStatus.SUCCEEDED, success=1, success_rate=1.0)
    await bus.publish(Event(type=EventType.VLA_TELEMETRY, source="sim", task_id="task_1",
                            payload={"kind": "result", **res.model_dump()}))
    got = await fut
    assert got.success == 1 and got.status == VlaTaskStatus.SUCCEEDED
    task.cancel()
