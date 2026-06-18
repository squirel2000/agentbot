"""Control plane 1->2: user message -> Brain -> AgentResponse."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agentbot.api.deps import Deps, get_deps

router = APIRouter()


class ChatIn(BaseModel):
    session_id: str
    text: str = ""
    modality: str = "text"
    image_path: Optional[str] = None


@router.post("/v1/chat")
async def chat(body: ChatIn, d: Deps = Depends(get_deps)) -> dict:
    msg = d.gateway.normalize(body.session_id, body.modality, body.text, body.image_path)
    resp = await d.agent.handle(msg)
    return resp.model_dump()


@router.websocket("/v1/chat/stream")
async def chat_stream(ws: WebSocket) -> None:
    await ws.accept()
    d = get_deps()
    try:
        while True:
            payload = await ws.receive_json()
            msg = d.gateway.normalize(payload.get("session_id", "ws"),
                                      payload.get("modality", "text"),
                                      payload.get("text", ""))
            await ws.send_json({"state": "thinking", "turn_id": msg.turn_id})
            resp = await d.agent.handle(msg)
            await ws.send_json(resp.model_dump())
    except WebSocketDisconnect:
        return
