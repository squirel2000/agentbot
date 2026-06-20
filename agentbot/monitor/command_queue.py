"""Command queue (Execution plane, UI -> Brain orchestrator).

The UI/console enqueues a ``Command``; the Brain orchestrator dequeues it.
``InMemCommandQueue`` for single-process dev/tests;
``RedisCommandQueue`` (List, LPUSH/BRPOP) for the real cross-process system.
"""
from __future__ import annotations

import abc
from collections import deque
from typing import Optional

from agentbot.contracts.commands import Command


class CommandQueue(abc.ABC):
    @abc.abstractmethod
    def put(self, cmd: Command) -> None: ...

    @abc.abstractmethod
    def next(self, timeout: float = 0.0) -> Optional[Command]:
        """Pop the next command, or ``None`` if none arrives within *timeout* seconds."""
        ...

    @abc.abstractmethod
    def clear(self) -> int:
        """Remove all pending commands. Returns the number of commands removed."""
        ...


class InMemCommandQueue(CommandQueue):
    def __init__(self) -> None:
        self._dq: deque[Command] = deque()

    def put(self, cmd: Command) -> None:
        self._dq.append(cmd)

    def next(self, timeout: float = 0.0) -> Optional[Command]:
        # FIFO: pop exactly the oldest item. `timeout` is a no-op for the in-mem queue
        # (it never blocks); the Redis impl honors it via BRPOP.
        return self._dq.popleft() if self._dq else None

    def clear(self) -> int:
        count = len(self._dq)
        self._dq.clear()
        return count


class RedisCommandQueue(CommandQueue):
    def __init__(self, url: str = "redis://localhost:6379/0", queue_key: str = "agentbot:commands") -> None:
        import redis
        self._r = redis.from_url(url)
        self._key = queue_key

    def put(self, cmd: Command) -> None:
        self._r.lpush(self._key, cmd.model_dump_json())

    def next(self, timeout: float = 0.0) -> Optional[Command]:
        # BRPOP blocks up to `timeout` seconds (0 = block forever); pair with LPUSH for FIFO.
        # Catch TimeoutError: redis-py's socket timeout can race with the BRPOP server-side
        # timeout and fire first, raising TimeoutError instead of returning None.
        import redis as _redis
        try:
            item = self._r.brpop([self._key], timeout=timeout)
        except _redis.exceptions.TimeoutError:
            return None
        if not item:
            return None
        _key, raw = item
        return Command.model_validate_json(raw)

    def clear(self) -> int:
        count = self._r.llen(self._key)
        self._r.delete(self._key)
        return count
