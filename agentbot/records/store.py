"""SQLite-backed records store for command lifecycle and skill-run history.

This is the OPERATIONAL store (command + skill-run audit trail) — distinct from the
Brain's cognitive memory (conversation + episodic history in brain/memory/store.py).
Schema: ``commands``, ``skill_runs``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from agentbot.contracts.commands import Command, CommandStatus
from agentbot.contracts.common import now_ts
from agentbot.contracts.skills import SkillPlan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    command_id    TEXT PRIMARY KEY,
    session_id    TEXT,
    text          TEXT,
    status        TEXT,
    plan_json     TEXT,
    result_summary TEXT,
    created_ts    REAL,
    updated_ts    REAL
);
CREATE TABLE IF NOT EXISTS skill_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id  TEXT,
    skill       TEXT,
    task_id     TEXT,
    attempt     INTEGER,
    status      TEXT,
    success     INTEGER,
    steps       INTEGER,
    duration_s  REAL,
    ts          REAL
);
"""


class Records:
    def __init__(self, sqlite_path: str = "agentbot/var/records.db") -> None:
        p = Path(sqlite_path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may touch it from worker threads.
        self.conn = sqlite3.connect(str(p), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    @classmethod
    def in_memory(cls) -> "Records":
        return cls(":memory:")

    # ------------------------------------------------------------------
    # Command lifecycle
    # ------------------------------------------------------------------

    def add_command(self, cmd: Command) -> None:
        """Insert a new command row (store plan as JSON in plan_json)."""
        plan_json = cmd.plan.model_dump_json() if cmd.plan is not None else None
        self.conn.execute(
            """INSERT INTO commands
               (command_id, session_id, text, status, plan_json, result_summary, created_ts, updated_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cmd.command_id,
                cmd.session_id,
                cmd.text,
                cmd.status.value,
                plan_json,
                cmd.result_summary,
                cmd.created_ts,
                cmd.created_ts,
            ),
        )
        self.conn.commit()

    def set_status(
        self,
        command_id: str,
        status: CommandStatus,
        result_summary: str = "",
    ) -> None:
        """Update status, result_summary, and updated_ts for a command."""
        self.conn.execute(
            """UPDATE commands
               SET status = ?, result_summary = ?, updated_ts = ?
               WHERE command_id = ?""",
            (status.value, result_summary, now_ts(), command_id),
        )
        self.conn.commit()

    def set_plan(self, command_id: str, plan: SkillPlan) -> None:
        """Store the SkillPlan JSON for a command."""
        self.conn.execute(
            "UPDATE commands SET plan_json = ?, updated_ts = ? WHERE command_id = ?",
            (plan.model_dump_json(), now_ts(), command_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Skill run logging
    # ------------------------------------------------------------------

    def log_skill_run(
        self,
        command_id: str,
        skill: str,
        task_id: str,
        attempt: int,
        status: str,
        success: int,
        steps: int,
        duration_s: float,
    ) -> None:
        """Append a skill-run row for the given command."""
        self.conn.execute(
            """INSERT INTO skill_runs
               (command_id, skill, task_id, attempt, status, success, steps, duration_s, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (command_id, skill, task_id, attempt, status, success, steps, duration_s, now_ts()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_commands(self, n: int = 50) -> list[dict]:
        """Return the *n* most recent commands, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM commands ORDER BY created_ts DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def command_detail(self, command_id: str) -> dict:
        """Return ``{"command": {...}, "skill_runs": [...]}`` for a command.

        skill_runs are ordered by id ascending (insertion order).
        """
        cmd_row = self.conn.execute(
            "SELECT * FROM commands WHERE command_id = ?", (command_id,)
        ).fetchone()
        skill_rows = self.conn.execute(
            "SELECT * FROM skill_runs WHERE command_id = ? ORDER BY id ASC", (command_id,)
        ).fetchall()
        return {
            "command": dict(cmd_row) if cmd_row else {},
            "skill_runs": [dict(r) for r in skill_rows],
        }

    def stats(self) -> dict:
        """Return overall counts and per-skill aggregates.

        Shape::

            {
                "commands": int,
                "done": int,
                "failed": int,
                "per_skill": [
                    {
                        "skill": str,
                        "runs": int,
                        "successes": int,
                        "success_rate": float,
                        "avg_duration_s": float,
                    },
                    ...
                ],
            }
        """
        totals = self.conn.execute(
            """SELECT
                   COUNT(*) AS commands,
                   SUM(CASE WHEN status = 'done'   THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
               FROM commands"""
        ).fetchone()

        skill_rows = self.conn.execute(
            """SELECT
                   skill,
                   COUNT(*) AS runs,
                   SUM(success) AS successes,
                   CAST(SUM(success) AS REAL) / COUNT(*) AS success_rate,
                   AVG(duration_s) AS avg_duration_s
               FROM skill_runs
               GROUP BY skill
               ORDER BY skill"""
        ).fetchall()

        return {
            "commands": totals["commands"] or 0,
            "done": totals["done"] or 0,
            "failed": totals["failed"] or 0,
            "per_skill": [dict(r) for r in skill_rows],
        }
