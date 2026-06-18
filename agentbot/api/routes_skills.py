"""Block 4 + the concrete 4->5 edge: list skills (as LLM tools) and invoke one."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentbot.api.deps import Deps, get_deps
from agentbot.contracts.skills import SkillCall

router = APIRouter()


@router.get("/v1/skills")
async def list_skills(d: Deps = Depends(get_deps)) -> dict:
    return {
        "skills": [s.model_dump() for s in d.registry.list_specs()],
        "tools": d.registry.as_tool_schemas(),
    }


class InvokeIn(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    checkpoint: Optional[str] = None   # registry name OR literal path; default = config default


@router.post("/v1/skills/{name}/invoke")
async def invoke(name: str, body: InvokeIn, d: Deps = Depends(get_deps)) -> dict:
    skill = d.registry.get(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"unknown skill '{name}'")
    call = SkillCall(name=name, args=body.args, constraints=body.constraints, safety_flags=body.safety_flags)
    ok, err = skill.validate(call)
    if not ok:
        raise HTTPException(status_code=422, detail=err)
    ctx = d.vla_ctx(body.checkpoint)
    req = skill.to_vla_request(call, ctx)
    d.queue.put(req)
    d.state.set_state(f"vla:{req.task_id}", {
        "status": "queued", "task_name": req.task_name,
        "checkpoint": req.checkpoint, "checkpoint_name": req.checkpoint_name,
    })
    return {"task_id": req.task_id, "checkpoint": req.checkpoint, "checkpoint_name": req.checkpoint_name}
