"""Event plane (Monitor / block 3): live snapshot + the WS event stream."""
from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from agentbot.api.deps import Deps, get_deps

router = APIRouter()


@router.get("/v1/state")
async def state(d: Deps = Depends(get_deps)) -> dict:
    return d.state.snapshot()


@router.websocket("/v1/events")
async def events_ws(ws: WebSocket) -> None:
    await ws.accept()
    d = get_deps()
    try:
        async for ev in d.bus.subscribe():
            await ws.send_text(ev.model_dump_json())
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - client gone / bus closed; end the stream quietly
        return
