"""Short-term conversation memory — per-session turn history."""
from __future__ import annotations

import time

from agentbot.brain.memory.store import MemoryStore


class ConversationMemory:
    def __init__(self, store: MemoryStore) -> None:
        self._c = store.conn

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        self._c.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_ts) VALUES(?, ?)",
            (session_id, time.time()),
        )
        self._c.execute(
            "INSERT INTO turns(session_id, role, text, ts) VALUES(?, ?, ?, ?)",
            (session_id, role, text, time.time()),
        )
        self._c.commit()

    def recent(self, session_id: str, n: int = 10) -> list[dict]:
        rows = self._c.execute(
            "SELECT role, text, ts FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
