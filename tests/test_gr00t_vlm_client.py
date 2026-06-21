"""Tests for Gr00tVLMClient.complete() — real HTTP path + stub fallback."""
import json
from agentbot.brain.vlm_client import Gr00tVLMClient


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


async def test_complete_parses_tool_calls(monkeypatch):
    payload = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "sort_can", "arguments": json.dumps({"target_color": "orange"})}}]}}]}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _Resp(payload)

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    reply = await Gr00tVLMClient(base_url="http://x/v1").complete(
        [{"role": "user", "content": "sort onto orange"}],
        [{"name": "sort_can", "description": "d",
          "input_schema": {"type": "object", "properties": {}, "required": []}}])
    assert reply.calls[0].name == "sort_can" and reply.calls[0].args == {"target_color": "orange"}


async def test_complete_falls_back_to_stub_on_error(monkeypatch):
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): raise RuntimeError("down")

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    reply = await Gr00tVLMClient(base_url="http://x/v1").complete(
        [{"role": "user", "content": "sort_can onto orange"}],
        [{"name": "sort_can", "description": "d",
          "input_schema": {"type": "object", "properties": {
              "target_color": {"enum": ["orange", "green"]}}, "required": ["target_color"]}}])
    # endpoint error -> keyword stub still selects sort_can
    assert reply.calls and reply.calls[0].name == "sort_can"
