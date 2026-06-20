"""Tests for Command contract and InMemCommandQueue."""
from agentbot.contracts.commands import Command, CommandStatus
from agentbot.monitor.command_queue import InMemCommandQueue


def test_command_defaults_and_roundtrip():
    c = Command(text="sort the can onto orange")
    assert c.command_id.startswith("cmd_") and c.status == CommandStatus.QUEUED
    assert Command.model_validate_json(c.model_dump_json()).text == c.text


def test_command_queue_fifo_and_clear():
    q = InMemCommandQueue()
    a = Command(text="first"); b = Command(text="second")
    q.put(a); q.put(b)
    assert q.next(timeout=0.1).command_id == a.command_id   # FIFO: oldest first
    assert q.next(timeout=0.1).command_id == b.command_id   # then the next oldest
    q.put(Command(text="third"))
    assert q.clear() == 1                                    # one pending (third) cleared
    assert q.next(timeout=0.01) is None
