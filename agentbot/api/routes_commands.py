"""Commands / stats / control API (Phase 2a, Task 7).

POST   /v1/commands           — enqueue a natural-language command
GET    /v1/commands           — list recent commands
GET    /v1/commands/{id}      — detail (command + skill runs)
DELETE /v1/commands           — clear pending command queue

GET    /v1/stats              — aggregate stats (commands / done / failed / per_skill)

POST   /v1/control/stop       — recoverable stop (cancel current + drain queues + clear blocked)
POST   /v1/control/reset      — reset the sim scene (re-randomize target plate) + publish frame
POST   /v1/control/estop      — emergency stop (publish SAFETY/CRITICAL so watchdog trips)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agentbot.api.deps import Deps, get_deps
from agentbot.contracts.commands import Command
from agentbot.contracts.events import Event, EventType, SafetyEvent
from agentbot.contracts.common import Severity
from agentbot.contracts.vla import VlaTaskRequest

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
    # RECOVERABLE soft-stop. Do NOT call orchestrator.stop() — that ends the run() loop
    # permanently with no restart path (the "commands enqueue but nothing runs" bug). Instead:
    #   1) cancel the in-flight command (stop retrying/replanning; the running episode finishes),
    #   2) drain both queues — pending commands AND pending VLA tasks (`agentbot:vla:tasks`),
    #   3) clear the safety-blocked robot_state so the system is usable again after an E-STOP.
    d.orchestrator.cancel_current()
    cleared_commands = d.command_queue.clear()
    cleared_vla_tasks = d.queue.clear()
    d.state.set_state("robot_state", {"mode": "idle", "detail": "stopped"})
    return {"stopped": True, "cleared_commands": cleared_commands, "cleared_vla_tasks": cleared_vla_tasks}


@router.post("/v1/control/reset")
async def control_reset(d: Deps = Depends(get_deps)) -> dict:
    # Reset the IsaacLab scene in the persistent sim_session: cancel + drain first, then enqueue a
    # control task (params.control == "reset") on the same VLA queue. sim_session re-randomizes the
    # target plate, stabilizes, and publishes a fresh head frame + the new target color to state —
    # so the dashboard shows the live scene and a bare "sort can" resolves to the right plate.
    d.orchestrator.cancel_current()
    d.command_queue.clear()
    d.queue.clear()
    ctx = d.vla_ctx(None)
    req = VlaTaskRequest(
        task_name=d.cfg.vla.default_task,
        instruction="reset the scene",
        checkpoint=ctx["checkpoint"],
        params={"control": "reset"},
    )
    d.queue.put(req)
    d.state.set_state("robot_state", {"mode": "idle", "detail": "reset"})
    return {"reset": True, "task_id": req.task_id}


@router.post("/v1/control/estop")
async def control_estop(d: Deps = Depends(get_deps)) -> dict:
    await d.bus.publish(Event(
        type=EventType.SAFETY,
        source="api",
        severity=Severity.CRITICAL,
        payload=SafetyEvent(kind="emergency", immediate=True).model_dump(),
    ))
    return {"estop": True}
