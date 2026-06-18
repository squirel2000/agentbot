"""Episodic memory — a log of past skill executions and their outcomes.

Lets the Brain answer "have I done this before / did it work?" and feeds the
Console's History tab. Populated by the Monitor ingestor (Task 7).
"""
from __future__ import annotations

import json
import time
from typing import Optional

from agentbot.brain.memory.store import MemoryStore
from agentbot.contracts.skills import SkillResult


class EpisodicMemory:
    def __init__(self, store: MemoryStore) -> None:
        self._c = store.conn

    def log(self, result: SkillResult, session_id: Optional[str] = None) -> None:
        self._c.execute(
            "INSERT INTO episodes(session_id, skill_call_id, skill, task_id, status, summary, metrics_json, ts) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (session_id, result.skill_call_id, result.name, result.task_id,
             result.status.value, result.summary, json.dumps(result.metrics), time.time()),
        )
        self._c.commit()

    def recent(self, n: int = 20) -> list[dict]:
        rows = self._c.execute(
            "SELECT skill, task_id, status, summary, ts FROM episodes ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def for_skill(self, skill: str, n: int = 20) -> list[dict]:
        rows = self._c.execute(
            "SELECT skill, task_id, status, summary, ts FROM episodes WHERE skill=? ORDER BY id DESC LIMIT ?",
            (skill, n),
        ).fetchall()
        return [dict(r) for r in rows]
