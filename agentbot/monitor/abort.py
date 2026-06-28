"""Abort flag (Execution plane) — a cross-process kill-switch for a running VLA episode.

The dashboard ■ Stop trips it; the persistent ``sim_session`` polls it once per action chunk
and ends the episode early (returning ABORTED). ``InMemAbortFlag`` for single-process dev/tests;
``RedisAbortFlag`` (one auto-expiring key) for the real cross-process system.
"""
from __future__ import annotations

import abc
from typing import Optional


class AbortFlag(abc.ABC):
    @abc.abstractmethod
    def trip(self, token: str = "all") -> None:
        """Request abort. ``token`` is a task_id, or ``"all"`` for any running task."""
        ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def tripped(self, task_id: str) -> bool:
        """True if an abort is pending for ``task_id`` (i.e. token is ``"all"`` or ``task_id``)."""
        ...


class InMemAbortFlag(AbortFlag):
    def __init__(self) -> None:
        self._tok: Optional[str] = None

    def trip(self, token: str = "all") -> None:
        self._tok = token

    def clear(self) -> None:
        self._tok = None

    def tripped(self, task_id: str) -> bool:
        return self._tok is not None and self._tok in ("all", task_id)


class RedisAbortFlag(AbortFlag):
    def __init__(self, url: str = "redis://localhost:6379/0", key: str = "agentbot:vla:abort") -> None:
        import redis
        self._r = redis.from_url(url)
        self._key = key

    def trip(self, token: str = "all") -> None:
        # ex=120: auto-expire so a missed clear() can't wedge future episodes permanently.
        self._r.set(self._key, token, ex=120)

    def clear(self) -> None:
        self._r.delete(self._key)

    def tripped(self, task_id: str) -> bool:
        v = self._r.get(self._key)
        if v is None:
            return False
        v = v.decode() if isinstance(v, bytes) else v
        return v in ("all", task_id)
