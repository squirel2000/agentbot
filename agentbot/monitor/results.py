"""ResultWaiter — resolves asyncio Futures when a VLA terminal result arrives.

The Brain orchestrator uses this to block on a dispatched skill:

    task_id = await queue.enqueue(request)
    result  = await result_waiter.expect(task_id, timeout=120.0)

The sim/worker publishes Event(type=VLA_TELEMETRY, payload={"kind":"result", ...})
which is what ``run()`` watches for.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable

from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskResult
from agentbot.monitor.event_bus import EventBus

log = logging.getLogger(__name__)


class ResultWaiter:
    """Subscribe to the event bus and resolve per-task-id Futures on terminal results.

    Single responsibility: map incoming ``VLA_TELEMETRY`` events whose payload
    has ``kind == "result"`` onto the correct waiting Future, if any.
    Unregistered or duplicate result events are silently ignored.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._pending: dict[str, asyncio.Future[VlaTaskResult]] = {}

    async def run(self) -> None:
        """Drain VLA_TELEMETRY events and resolve registered futures.

        Runs indefinitely; cancel the task to stop it.
        """
        async for ev in self._bus.subscribe([EventType.VLA_TELEMETRY]):
            self._handle(ev)

    def _handle(self, ev: Event) -> None:
        """Process one event synchronously (separated for unit-testability)."""
        if ev.payload.get("kind") != "result":
            return
        task_id = ev.task_id
        if task_id is None or task_id not in self._pending:
            # No one is waiting — ignore duplicate / unexpected results.
            return
        fut = self._pending.pop(task_id)
        if fut.done():
            # Already cancelled/timed-out; nothing to do.
            return
        try:
            result = VlaTaskResult.model_validate(ev.payload)
        except Exception:  # noqa: BLE001
            log.warning("ResultWaiter: could not validate result payload for task_id=%s", task_id)
            return
        fut.set_result(result)

    def expect(self, task_id: str, timeout: float) -> Awaitable[VlaTaskResult]:
        """Return an awaitable that resolves to VlaTaskResult or raises asyncio.TimeoutError.

        Registers a Future under ``task_id`` and wraps it with ``asyncio.wait_for``.
        The registry entry is cleaned up on timeout so the ``run()`` loop stays healthy.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[VlaTaskResult] = loop.create_future()
        self._pending[task_id] = fut

        async def _guarded() -> VlaTaskResult:
            try:
                return await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                # Remove stale entry so run() doesn't try to resolve it later.
                self._pending.pop(task_id, None)
                raise

        return _guarded()
