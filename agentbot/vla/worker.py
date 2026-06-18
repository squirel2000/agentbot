"""VLA worker (block 5) — dequeue VlaTaskRequest, run the backend, publish events.

Runs in its OWN process. For real rollouts:
    conda activate env_isaaclab && python -m agentbot.vla.worker
For no-GPU plumbing tests:
    python -m agentbot.vla.worker --fake --selftest

It is intentionally synchronous (blocking ``BRPOP`` + a blocking subprocess
rollout, like ``scripts/eval/run_eval.py``). It publishes events with a *sync*
Redis client that is wire-compatible with the async ``RedisEventBus`` the API reads
(same stream, same ``{"json": ...}`` field), so the two processes interoperate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

from agentbot.contracts.events import Event, EventType, Severity
from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult, VlaTaskStatus
from agentbot.monitor.job_queue import InMemJobQueue, RedisJobQueue
from agentbot.settings import REPO_ROOT, AppConfig, load_config
from agentbot.vla.backends.sim import SimBackend
from agentbot.vla.isaac_runner import FakeRunner, backend_spec

Publish = Callable[[Event], None]


def _status_event(req: VlaTaskRequest, status: str, **payload) -> Event:
    sev = Severity.WARN if status in ("failed", "aborted") else Severity.INFO
    return Event(type=EventType.VLA_TELEMETRY, source="vla.worker", task_id=req.task_id,
                 severity=sev, payload={"status": status, **payload})


def _result_event(req: VlaTaskRequest, result: VlaTaskResult) -> Event:
    return Event(type=EventType.VLA_TELEMETRY, source="vla.worker", task_id=req.task_id,
                 payload={"kind": "result", **result.model_dump()})


def _load_backend_spec(cfg: AppConfig) -> dict:
    """Backend connection spec (host/port/version/embodiment/server_repo), reused from
    the eval configs (via isaac_runner) so worker and eval harness stay consistent."""
    return backend_spec(cfg.vla.gr00t_ver)


def make_backend(cfg: AppConfig, fake: bool) -> SimBackend:
    rollout_fn = FakeRunner().run_rollout if fake else None
    return SimBackend(repo_root=str(REPO_ROOT), backend_spec=_load_backend_spec(cfg),
                      rollout_fn=rollout_fn, manage_server=not fake)


def run_worker(config_path: Optional[str] = None, fake: bool = False) -> None:
    import redis  # sync client
    cfg = load_config(config_path)
    r = redis.from_url(cfg.redis.url)
    q = RedisJobQueue(cfg.redis.url, cfg.redis.tasks_queue)
    stream = cfg.redis.events_stream
    backend = make_backend(cfg, fake)

    def publish(ev: Event) -> None:
        r.xadd(stream, {"json": ev.model_dump_json()})

    print(f"[worker] polling {cfg.redis.tasks_queue} (fake={fake}); events -> {stream}")
    while True:
        req = q.get(timeout=5)
        if req is None:
            continue
        publish(_status_event(req, "running", task_name=req.task_name, checkpoint=req.checkpoint))
        try:
            result = backend.run(req, publish)
            publish(_result_event(req, result))
        except Exception as e:  # noqa: BLE001 - report failure, keep serving
            publish(_status_event(req, "failed", error=str(e)))


def _selftest(fake: bool = True) -> None:
    """In-proc, no Redis/IsaacLab: enqueue one request, run it, assert a result event."""
    events: list[Event] = []
    publish: Publish = events.append
    q = InMemJobQueue()
    q.put(VlaTaskRequest(task_name="Isaac-Can-Sorting-OpenArm-DexHand-v0",
                         instruction="place the can on the orange plate", checkpoint="dummy/ckpt"))
    backend = SimBackend(repo_root=str(REPO_ROOT), backend_spec={},
                         rollout_fn=FakeRunner().run_rollout, manage_server=False)
    req = q.get(timeout=0.1)
    assert req is not None
    publish(_status_event(req, "running"))
    result = backend.run(req, publish)
    publish(_result_event(req, result))
    n_result = sum(1 for e in events if e.payload.get("kind") == "result")
    assert n_result == 1 and result.status == VlaTaskStatus.SUCCEEDED, "selftest failed"
    print(f"[selftest] OK — {len(events)} events, result={result.status.value}, steps={result.steps}")


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="path to agentbot.yaml")
    ap.add_argument("--fake", action="store_true", help="use FakeRunner (no IsaacLab/GPU)")
    ap.add_argument("--selftest", action="store_true", help="run one in-proc fake task and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest(fake=True)
        return
    run_worker(args.config, fake=args.fake)


if __name__ == "__main__":
    cli()
