"""AbortFlag — the ■ Stop control trips it so a running sim episode ends early.

token "all" aborts any task; a specific task_id aborts only that one. Tested with the
in-mem impl (the redis impl is the same logic over a single auto-expiring key)."""
from agentbot.monitor.abort import InMemAbortFlag


def test_trip_all_aborts_any_task_until_cleared():
    f = InMemAbortFlag()
    assert not f.tripped("task_1")
    f.trip("all")
    assert f.tripped("task_1") and f.tripped("task_2")
    f.clear()
    assert not f.tripped("task_1")


def test_trip_specific_task_only_aborts_that_task():
    f = InMemAbortFlag()
    f.trip("task_1")
    assert f.tripped("task_1")
    assert not f.tripped("task_2")


import types
from agentbot.contracts.commands import Command
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.monitor.command_queue import InMemCommandQueue
from agentbot.monitor.job_queue import InMemJobQueue
from agentbot.monitor.state_store import InMemStateStore


class _FakeDeps:
    def __init__(self):
        self.cancelled = False
        self.orchestrator = types.SimpleNamespace(
            cancel_current=lambda: setattr(self, "cancelled", True))
        self.command_queue = InMemCommandQueue()
        self.queue = InMemJobQueue()
        self.abort = InMemAbortFlag()
        self.state = InMemStateStore()


async def test_control_stop_trips_abort_and_drains_and_recovers():
    from agentbot.api.routes_commands import control_stop
    d = _FakeDeps()
    d.queue.put(VlaTaskRequest(task_name="Isaac-Can-Sorting", instruction="x", checkpoint="/c"))
    d.command_queue.put(Command(text="seed"))
    out = await control_stop(d)
    assert out["stopped"] and d.cancelled
    assert d.abort.tripped("any-task")                 # running episode will see the abort
    assert d.queue.get() is None                        # VLA queue drained
    assert d.command_queue.next(0.0) is None            # command queue also drained
    assert d.state.get_state("robot_state")["mode"] == "idle"
