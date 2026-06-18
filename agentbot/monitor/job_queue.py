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


class InMemJobQueue(JobQueue):
    def __init__(self) -> None:
        self._dq: deque[VlaTaskRequest] = deque()

    def put(self, req: VlaTaskRequest) -> None:
        self._dq.append(req)

    def get(self, timeout: float = 0.0) -> Optional[VlaTaskRequest]:
        return self._dq.popleft() if self._dq else None


class RedisJobQueue(JobQueue):
    def __init__(self, url: str = "redis://localhost:6379/0", queue_key: str = "agentbot:vla:tasks") -> None:
        import redis
        self._r = redis.from_url(url)
        self._key = queue_key

    def put(self, req: VlaTaskRequest) -> None:
        self._r.lpush(self._key, req.model_dump_json())

    def get(self, timeout: float = 0.0) -> Optional[VlaTaskRequest]:
        # BRPOP blocks up to `timeout` seconds (0 = block forever); pair with LPUSH for FIFO.
        item = self._r.brpop([self._key], timeout=timeout)
        if not item:
            return None
        _key, raw = item
        return VlaTaskRequest.model_validate_json(raw)
