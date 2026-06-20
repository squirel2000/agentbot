"""FastAPI app — wires User Input (1) -> Brain (2) -> Skills (4) -> VLA (5) and the
Monitor (3) event stream, and serves the read-only Console.

Startup tasks: the Monitor ``Ingestor`` (bus -> live state), the ``SafetyWatchdog``,
and — in in-proc mode only — a small FakeRunner "VLA pump" that drains the in-mem
job queue so the whole chain is demoable in ONE process with no Redis/GPU. In
``redis`` mode the real ``env_isaaclab`` worker does that job instead.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agentbot.api import routes_chat, routes_commands, routes_events, routes_skills, routes_vla
from agentbot.api.deps import Deps, get_deps
from agentbot.vla.isaac_runner import FakeRunner
from agentbot.vla.worker import _result_event, _status_event

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


async def _fake_vla_pump(d: Deps) -> None:
    """In-proc demo executor: drain the in-mem queue, run FakeRunner, publish events."""
    runner = FakeRunner()
    while True:
        req = d.queue.get(timeout=0)   # in-mem queue is non-blocking
        if req is None:
            await asyncio.sleep(0.2)
            continue
        await d.bus.publish(_status_event(req, "running", task_name=req.task_name))
        events: list = []
        result = runner.run_rollout(req, events.append)   # FakeRunner is sync + fast
        for ev in events:
            await d.bus.publish(ev)
        await d.bus.publish(_result_event(req, result))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    d = get_deps()
    tasks = [
        asyncio.create_task(d.ingestor.run()),
        asyncio.create_task(d.watchdog.run()),
        asyncio.create_task(d.results.run()),
        asyncio.create_task(d.orchestrator.run()),
    ]
    if d.cfg.backbone != "redis":
        tasks.append(asyncio.create_task(_fake_vla_pump(d)))
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()


app = FastAPI(title="AgentBot API", version="0.1.0", lifespan=lifespan)
app.include_router(routes_chat.router)
app.include_router(routes_commands.router)
app.include_router(routes_skills.router)
app.include_router(routes_vla.router)
app.include_router(routes_events.router)
app.mount("/vendor", StaticFiles(directory=str(UI_DIR / "vendor")), name="vendor")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (UI_DIR / "console.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn
    cfg = get_deps().cfg
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    main()
