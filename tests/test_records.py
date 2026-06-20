"""Tests for Records (SQLite-backed command lifecycle + skill-run store)."""
from agentbot.contracts.commands import Command
from agentbot.records import Records


def test_records_add_list_detail_stats():
    r = Records.in_memory()

    cmd = Command(text="pick the red cube")
    r.add_command(cmd)

    # Log two skill runs: one success, one failure.
    r.log_skill_run(cmd.command_id, "pick", "t1", 1, "success", 1, 10, 2.5)
    r.log_skill_run(cmd.command_id, "pick", "t2", 2, "failure", 0,  5, 1.0)

    # list_commands returns the command.
    listed = r.list_commands(1)
    assert len(listed) == 1
    assert listed[0]["command_id"] == cmd.command_id

    # command_detail returns two skill_runs.
    detail = r.command_detail(cmd.command_id)
    assert detail["command"]["command_id"] == cmd.command_id
    assert len(detail["skill_runs"]) == 2

    # stats: per_skill shows success_rate == 0.5 for "pick".
    s = r.stats()
    assert s["commands"] == 1
    pick_stat = next(ps for ps in s["per_skill"] if ps["skill"] == "pick")
    assert pick_stat["success_rate"] == 0.5
    assert pick_stat["runs"] == 2
    assert pick_stat["successes"] == 1
