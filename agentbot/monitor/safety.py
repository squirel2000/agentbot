"""Safety watchdog — the red "Immediate Safety" path in architecture.png.

Subscribes to SAFETY events; when one is flagged ``immediate``, it invokes the
registered emergency-stop callback synchronously (sim: abort the active task;
hardware future: ROS2 e-stop override). Kept tiny so it stays low-latency.
"""
from __future__ import annotations

from typing import Callable

from agentbot.contracts.events import Event, EventType, SafetyEvent
from agentbot.monitor.event_bus import EventBus


class SafetyWatchdog:
    def __init__(self, bus: EventBus, on_stop: Callable[[Event], None]) -> None:
        self._bus = bus
        self._on_stop = on_stop

    async def run(self) -> None:
        async for ev in self._bus.subscribe([EventType.SAFETY]):
            try:
                payload = SafetyEvent.model_validate(ev.payload)
            except Exception:  # noqa: BLE001 - a malformed safety event must not crash the watchdog
                continue
            if payload.immediate:
                self._on_stop(ev)
