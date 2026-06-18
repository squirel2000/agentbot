"""Ingestor — drains the event bus into the live state store (and episodic memory).

This is the durable bridge from the real-time Monitor (Redis/bus) to what the
Console reads: it keeps the latest snapshot of each state key and records terminal
VLA results per task_id (so ``GET /v1/vla/tasks/{id}`` can answer).
"""
from __future__ import annotations

from typing import Optional

from agentbot.contracts.events import Event, EventType
from agentbot.monitor.event_bus import EventBus
from agentbot.monitor.state_store import StateStore

_LIVE_STATE_TYPES = {EventType.ROBOT_STATE, EventType.SYSTEM_STATUS, EventType.ENVIRONMENT}


class Ingestor:
    def __init__(self, bus: EventBus, state_store: StateStore, episodic=None) -> None:
        self.bus = bus
        self.state = state_store
        self.episodic = episodic   # optional EpisodicMemory

    def handle(self, ev: Event) -> None:
        """Process one event (sync, unit-testable)."""
        if ev.type in _LIVE_STATE_TYPES:
            self.state.set_state(ev.type.value, ev.payload)
        if ev.type == EventType.VLA_TELEMETRY and ev.task_id:
            key = f"vla:{ev.task_id}"
            merged = {**self.state.get_state(key), **ev.payload}
            self.state.set_state(key, merged)

    async def run(self) -> None:
        async for ev in self.bus.subscribe():
            self.handle(ev)
