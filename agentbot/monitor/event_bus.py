"""Event bus (Monitor / block 3 backbone).

Transport-agnostic ``EventBus`` with two impls:
- ``InProcEventBus``  — asyncio fan-out for single-process dev/tests.
- ``RedisEventBus``   — Redis Streams (XADD/XREAD) for the real multi-process system.

The abstraction is deliberately small so a future ``MqttEventBus`` / ROS2 bridge
(for hardware, blocks 6-7) can drop in without touching producers/consumers.
"""
from __future__ import annotations

import abc
import asyncio
from typing import AsyncIterator, Iterable, Optional

from agentbot.contracts.events import Event, EventType


class EventBus(abc.ABC):
    @abc.abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abc.abstractmethod
    def subscribe(self, types: Optional[Iterable[EventType]] = None) -> AsyncIterator[Event]:
        """Return an async iterator of events, optionally filtered by type."""
        ...


class InProcEventBus(EventBus):
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event]] = []

    async def publish(self, event: Event) -> None:
        for q in list(self._subs):
            q.put_nowait(event)

    async def subscribe(self, types: Optional[Iterable[EventType]] = None) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue()
        self._subs.append(q)
        want = set(types) if types else None
        try:
            while True:
                ev = await q.get()
                if want is None or ev.type in want:
                    yield ev
        finally:
            self._subs.remove(q)


class RedisEventBus(EventBus):
    """Redis Streams. One stream per deployment; new subscribers see only new events ($)."""

    def __init__(self, url: str = "redis://localhost:6379/0", stream: str = "agentbot:events") -> None:
        import redis.asyncio as redis  # local import: optional dependency at runtime
        self._r = redis.from_url(url)
        self._stream = stream

    async def publish(self, event: Event) -> None:
        await self._r.xadd(self._stream, {"json": event.model_dump_json()})

    async def subscribe(self, types: Optional[Iterable[EventType]] = None) -> AsyncIterator[Event]:
        import redis as _redis
        want = set(types) if types else None
        last = "$"
        while True:
            # Finite block + tolerate timeouts/connection blips: redis-py's socket timeout
            # races with an indefinite (block=0) XREAD and raises TimeoutError, which would
            # otherwise kill this subscriber for good (and any future waiting on it).
            try:
                resp = await self._r.xread({self._stream: last}, block=5000, count=64)
            except (_redis.exceptions.TimeoutError, _redis.exceptions.ConnectionError):
                continue
            if not resp:
                continue
            for _stream, entries in resp:
                for eid, fields in entries:
                    last = eid
                    raw = fields[b"json"] if b"json" in fields else fields["json"]
                    ev = Event.model_validate_json(raw)
                    if want is None or ev.type in want:
                        yield ev
