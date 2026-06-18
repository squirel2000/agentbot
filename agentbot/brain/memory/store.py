"""SQLite-backed durable memory for the Brain (block 2).

This is the COGNITIVE store (conversation + episodic history) — distinct from the
Monitor's real-time state (Redis). Schema: ``sessions``, ``turns``, ``episodes``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_ts REAL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,                 -- user | agent
    text TEXT,
    ts REAL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    skill_call_id TEXT,
    skill TEXT,
    task_id TEXT,
    status TEXT,
    summary TEXT,
    metrics_json TEXT,
    ts REAL
);
"""


class MemoryStore:
    def __init__(self, sqlite_path: str = "agentbot/var/agentbot.db") -> None:
        p = Path(sqlite_path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may touch it from worker threads.
        self.conn = sqlite3.connect(str(p), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    @classmethod
    def in_memory(cls) -> "MemoryStore":
        return cls(":memory:")
