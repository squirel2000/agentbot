"""Pluggable VLM backend for the Brain (block 2).

Default = Qwen3-VL (OpenAI-compatible local server, per architecture.png); adapters
for Claude and OpenAI. All return a ``VLMReply`` = natural-language text + an
optional list of ``SkillCall`` selected via tool-calling.

Scaffold behavior: if a backend isn't reachable/configured, clients fall back to a
deterministic STUB reply so the API and tests run without a live model. Real
inference is wired in the MVP phase.
"""
from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentbot.contracts.skills import SkillCall


class VLMReply(BaseModel):
    text: str = ""
    calls: list[SkillCall] = Field(default_factory=list)


class VLMClient(abc.ABC):
    def __init__(self, model: str = "", base_url: str = "", api_key: str = "") -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @abc.abstractmethod
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        """Run one turn. ``tools`` are SkillSpec.to_tool_schema() dicts."""
        ...

    # Shared scaffold fallback: naive keyword match so the pipeline is demoable
    # before a real model is connected. Replaced by real tool-calls in MVP.
    @staticmethod
    def _stub_reply(messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        raw = messages[-1]["content"] if messages else ""
        # content may be a str (plain text) or a list of parts (multimodal OpenAI format).
        # Extract the plain-text portion for the keyword-matching stub.
        if isinstance(raw, list):
            text_parts = [p["text"] for p in raw if isinstance(p, dict) and p.get("type") == "text"]
            raw = " ".join(text_parts)
        last = raw.lower()
        for t in tools:
            tokens = [tok for tok in t["name"].split("_") if tok]
            if tokens and all(tok in last for tok in tokens):
                args: dict[str, Any] = {}
                props = t.get("input_schema", {}).get("properties", {})
                for pname, pschema in props.items():
                    if "enum" in pschema:
                        for opt in pschema["enum"]:
                            if str(opt).lower() in last:
                                args[pname] = opt
                                break
                return VLMReply(text=f"[stub] selected skill '{t['name']}'",
                                calls=[SkillCall(name=t["name"], args=args, rationale="stub keyword match")])
        return VLMReply(text="[stub] no skill matched; please rephrase.", calls=[])


class QwenVLClient(VLMClient):
    """OpenAI-compatible chat-completions endpoint serving Qwen3-VL."""

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        # MVP: POST to {base_url}/chat/completions with tools and parse tool_calls.
        return self._stub_reply(messages, tools)


class ClaudeClient(VLMClient):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        # MVP: anthropic Messages API with `tools=`; map tool_use blocks -> SkillCall.
        return self._stub_reply(messages, tools)


class OpenAIClient(VLMClient):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        # MVP: openai chat.completions with function tools.
        return self._stub_reply(messages, tools)


class Gr00tVLMClient(VLMClient):
    """The Brain VLM extracted + fine-tuned from GR00T N1.7 — the planned replacement
    for the open-source Qwen3-VL placeholder.

    It honors the same OpenAI-compatible tool-calling contract as ``QwenVLClient``; once
    the extracted VLM is served, point ``vlm.base_url`` at its endpoint and set
    ``vlm.backend: gr00t-vlm``. Stub (keyword fallback) until that model is available.
    """

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        import json
        import httpx
        from agentbot.contracts.skills import SkillCall

        payload = {
            "model": self.model or "gr00t-vlm",
            "messages": messages,
            "tools": [{"type": "function",
                       "function": {"name": t["name"], "description": t.get("description", ""),
                                    "parameters": t["input_schema"]}} for t in tools],
            "tool_choice": "auto",
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception:  # endpoint down/misconfigured -> safe stub so the loop survives
            return self._stub_reply(messages, tools)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(SkillCall(name=fn.get("name", ""), args=args, rationale=msg.get("content") or ""))
        return VLMReply(text=msg.get("content") or "", calls=calls)


def build_vlm(cfg) -> VLMClient:
    """Factory: pick the client from ``cfg.vlm.backend`` (qwen-vl | gr00t-vlm | claude | openai)."""
    import os
    backend = getattr(cfg.vlm, "backend", "qwen-vl")
    api_key = os.environ.get(cfg.vlm.api_key_env, "") if cfg.vlm.api_key_env else ""
    kwargs = dict(model=cfg.vlm.model, base_url=cfg.vlm.base_url, api_key=api_key)
    if backend == "gr00t-vlm":
        return Gr00tVLMClient(**kwargs)
    if backend == "claude":
        return ClaudeClient(**kwargs)
    if backend == "openai":
        return OpenAIClient(**kwargs)
    return QwenVLClient(**kwargs)
