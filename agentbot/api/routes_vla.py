"""Execution plane 4->5: submit VLA tasks directly + read status; list checkpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agentbot.api.deps import Deps, get_deps
from agentbot.contracts.vla import VlaTaskRequest

router = APIRouter()


class TaskIn(BaseModel):
    task_name: str
    instruction: str
    checkpoint: Optional[str] = None   # registry name OR literal path
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/v1/vla/tasks")
async def submit_task(body: TaskIn, d: Deps = Depends(get_deps)) -> dict:
    ctx = d.vla_ctx(body.checkpoint)
    req = VlaTaskRequest(
        task_name=body.task_name, instruction=body.instruction,
        checkpoint=ctx["checkpoint"], checkpoint_name=ctx["checkpoint_name"],
        gr00t_ver=ctx["gr00t_ver"], embodiment=ctx["embodiment"], params=body.params,
    )
    d.queue.put(req)
    d.state.set_state(f"vla:{req.task_id}", {"status": "queued", "task_name": req.task_name})
    return {"task_id": req.task_id}


@router.get("/v1/vla/tasks/{task_id}")
async def get_task(task_id: str, d: Deps = Depends(get_deps)) -> dict:
    st = d.state.get_state(f"vla:{task_id}")
    return st or {"status": "unknown"}


@router.get("/v1/vla/checkpoints")
async def checkpoints(d: Deps = Depends(get_deps)) -> dict:
    """The swappable checkpoint registry (so the Console can offer a dropdown)."""
    return {"default": d.cfg.vla.default_checkpoint, "checkpoints": d.cfg.vla.checkpoints}
