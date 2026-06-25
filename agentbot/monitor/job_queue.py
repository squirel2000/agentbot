"""VLA job queue (Execution plane, 4->5).

The Skill layer enqueues a ``VlaTaskRequest``; the separate ``env_isaaclab``
worker dequeues it. ``InMemJobQueue`` for single-process dev/tests;
``RedisJobQueue`` (List, LPUSH/BRPOP) for the real cross-process system.
"""
from __future__ import annotations

import abc
from collections import deque
from typing import Optional

from agentbot.contracts.vla import VlaTaskRequest


class JobQueue(abc.ABC):
    @abc.abstractmethod
    def put(self, req: VlaTaskRequest) -> None: ...

    @abc.abstractmethod
    def get(self, timeout: float = 0.0) -> Optional[VlaTaskRequest]:
        """Pop the next request, or ``None`` if none arrives within *timeout* seconds."""
        ...

    @abc.abstractmethod
    def clear(self) -> int:
        """Drop every queued request; return how many were dropped.

        Used by the dashboard ``Stop`` control to clear VLA tasks that were enqueued
        but not yet consumed by the worker (so a stuck/runaway plan can be cancelled).
        """
        ...


class InMemJobQueue(JobQueue):
    def __init__(self) -> None:
        self._dq: deque[VlaTaskRequest] = deque()

    def put(self, req: VlaTaskRequest) -> None:
        self._dq.append(req)

    def get(self, timeout: float = 0.0) -> Optional[VlaTaskRequest]:
        return self._dq.popleft() if self._dq else None

    def clear(self) -> int:
        n = len(self._dq)
        self._dq.clear()
        return n


class RedisJobQueue(JobQueue):
    def __init__(self, url: str = "redis://localhost:6379/0", queue_key: str = "agentbot:vla:tasks") -> None:
        import redis
        self._r = redis.from_url(url)
        self._key = queue_key

    def put(self, req: VlaTaskRequest) -> None:
        self._r.lpush(self._key, req.model_dump_json())

    def get(self, timeout: float = 0.0) -> Optional[VlaTaskRequest]:
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
        return VlaTaskRequest.model_validate_json(raw)

    def clear(self) -> int:
        # LLEN then DEL: report how many pending VLA tasks were dropped. (A task already
        # BRPOP'd by the worker is mid-episode and can't be cancelled here — that needs a
        # cooperative abort inside sim_session, a separate follow-up.)
        n = self._r.llen(self._key)
        self._r.delete(self._key)
        return int(n or 0)
