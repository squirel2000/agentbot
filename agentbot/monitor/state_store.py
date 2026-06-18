"""Live state snapshot (Monitor / block 3).

Holds the *current* value of each state key (robot_state, system_status,
environment, ...). Real-time and lightweight — history lives in the Brain's
memory (SQLite), not here. ``InMemStateStore`` for dev/tests; ``RedisStateStore``
(Hash) for the real system.
"""
from __future__ import annotations

import abc
import json
from typing import Any


class StateStore(abc.ABC):
    @abc.abstractmethod
    def set_state(self, key: str, value: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def get_state(self, key: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    def snapshot(self) -> dict[str, dict[str, Any]]: ...


class InMemStateStore(StateStore):
    def __init__(self) -> None:
        self._d: dict[str, dict[str, Any]] = {}

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self._d[key] = value

    def get_state(self, key: str) -> dict[str, Any]:
        return self._d.get(key, {})

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._d)


class RedisStateStore(StateStore):
    def __init__(self, url: str = "redis://localhost:6379/0", hash_key: str = "agentbot:state") -> None:
        import redis  # sync client: state ops are tiny point reads/writes
        self._r = redis.from_url(url)
        self._hash = hash_key

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self._r.hset(self._hash, key, json.dumps(value))

    def get_state(self, key: str) -> dict[str, Any]:
        raw = self._r.hget(self._hash, key)
        return json.loads(raw) if raw else {}

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k.decode() if isinstance(k, bytes) else k: json.loads(v)
                for k, v in self._r.hgetall(self._hash).items()}
