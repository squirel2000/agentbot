"""Commands / stats / control API (Phase 2a, Task 7).

POST   /v1/commands           — enqueue a natural-language command
GET    /v1/commands           — list recent commands
GET    /v1/commands/{id}      — detail (command + skill runs)
DELETE /v1/commands           — clear pending command queue

GET    /v1/stats              — aggregate stats (commands / done / failed / per_skill)

POST   /v1/control/stop       — graceful stop (orchestrator + clear queue)
POST   /v1/control/estop      — emergency stop (publish SAFETY/CRITICAL so watchdog trips)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agentbot.api.deps import Deps, get_deps
from agentbot.contracts.commands import Command
from agentbot.contracts.events import Event, EventType, SafetyEvent
from agentbot.contracts.common import Severity

router = APIRouter()


class CommandIn(BaseModel):
    text: str
    session_id: str = "console"


@router.post("/v1/commands")
async def post_command(body: CommandIn, d: Deps = Depends(get_deps)) -> dict:
    cmd = Command(text=body.text, session_id=body.session_id)
    d.records.add_command(cmd)
    d.command_queue.put(cmd)
    return {"command_id": cmd.command_id}


@router.get("/v1/commands")
async def list_commands(n: int = 50, d: Deps = Depends(get_deps)) -> dict:
    return {"commands": d.records.list_commands(n)}


@router.get("/v1/commands/{command_id}")
async def get_command(command_id: str, d: Deps = Depends(get_deps)) -> dict:
    detail = d.records.command_detail(command_id)
    if not detail["command"]:
        raise HTTPException(status_code=404, detail=f"command '{command_id}' not found")
    return detail


@router.delete("/v1/commands")
async def clear_commands(d: Deps = Depends(get_deps)) -> dict:
    n = d.command_queue.clear()
    return {"cleared": n}


@router.get("/v1/stats")
async def stats(d: Deps = Depends(get_deps)) -> dict:
    return d.records.stats()


@router.post("/v1/control/stop")
async def control_stop(d: Deps = Depends(get_deps)) -> dict:
    d.orchestrator.stop()
    d.command_queue.clear()
    return {"stopped": True}


@router.post("/v1/control/estop")
async def control_estop(d: Deps = Depends(get_deps)) -> dict:
    await d.bus.publish(Event(
        type=EventType.SAFETY,
        source="api",
        severity=Severity.CRITICAL,
        payload=SafetyEvent(kind="emergency", immediate=True).model_dump(),
    ))
    return {"estop": True}
