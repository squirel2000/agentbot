"""JobQueue.clear() — the dashboard ``Stop`` control drains pending VLA tasks.

Regression for the 4090 bring-up: stopping a runaway plan must also drop tasks
already enqueued to ``agentbot:vla:tasks`` (not yet pulled by the worker), else the
worker keeps running stuck episodes after Stop. ``control_stop`` calls ``queue.clear()``.
"""
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.monitor.job_queue import InMemJobQueue


def _req(color: str) -> VlaTaskRequest:
    return VlaTaskRequest(
        task_name="Isaac-Can-Sorting",
        instruction=f"place the can on the {color} plate",
        checkpoint="/tmp/ckpt",
    )


def test_inmem_jobqueue_clear_drops_pending_and_returns_count():
    q = InMemJobQueue()
    q.put(_req("orange"))
    q.put(_req("green"))
    assert q.clear() == 2                 # both pending tasks dropped
    assert q.get(timeout=0.0) is None     # queue is now empty
    assert q.clear() == 0                 # idempotent on an empty queue
