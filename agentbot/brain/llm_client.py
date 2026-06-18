"""Pluggable LLM backend for the Brain (block 2).

Default = Qwen3-VL (OpenAI-compatible local server, per architecture.png); adapters
for Claude and OpenAI. All return an ``LLMReply`` = natural-language text + an
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


class LLMReply(BaseModel):
    text: str = ""
    calls: list[SkillCall] = Field(default_factory=list)


class LLMClient(abc.ABC):
    def __init__(self, model: str = "", base_url: str = "", api_key: str = "") -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @abc.abstractmethod
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        """Run one turn. ``tools`` are SkillSpec.to_tool_schema() dicts."""
        ...

    # Shared scaffold fallback: naive keyword match so the pipeline is demoable
    # before a real model is connected. Replaced by real tool-calls in MVP.
    @staticmethod
    def _stub_reply(messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        last = (messages[-1]["content"] if messages else "").lower()
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
                return LLMReply(text=f"[stub] selected skill '{t['name']}'",
                                calls=[SkillCall(name=t["name"], args=args, rationale="stub keyword match")])
        return LLMReply(text="[stub] no skill matched; please rephrase.", calls=[])


class QwenVLClient(LLMClient):
    """OpenAI-compatible chat-completions endpoint serving Qwen3-VL."""

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        # MVP: POST to {base_url}/chat/completions with tools and parse tool_calls.
        return self._stub_reply(messages, tools)


class ClaudeClient(LLMClient):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        # MVP: anthropic Messages API with `tools=`; map tool_use blocks -> SkillCall.
        return self._stub_reply(messages, tools)


class OpenAIClient(LLMClient):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        # MVP: openai chat.completions with function tools.
        return self._stub_reply(messages, tools)


class Gr00tVLMClient(LLMClient):
    """The Brain VLM extracted + fine-tuned from GR00T N1.7 — the planned replacement
    for the open-source Qwen3-VL placeholder.

    It honors the same OpenAI-compatible tool-calling contract as ``QwenVLClient``; once
    the extracted VLM is served, point ``llm.base_url`` at its endpoint and set
    ``llm.backend: gr00t-vlm``. Stub (keyword fallback) until that model is available.
    """

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> LLMReply:
        return self._stub_reply(messages, tools)


def build_llm(cfg) -> LLMClient:
    """Factory: pick the client from ``cfg.llm.backend`` (qwen-vl | gr00t-vlm | claude | openai)."""
    import os
    backend = getattr(cfg.llm, "backend", "qwen-vl")
    api_key = os.environ.get(cfg.llm.api_key_env, "") if cfg.llm.api_key_env else ""
    kwargs = dict(model=cfg.llm.model, base_url=cfg.llm.base_url, api_key=api_key)
    if backend == "gr00t-vlm":
        return Gr00tVLMClient(**kwargs)
    if backend == "claude":
        return ClaudeClient(**kwargs)
    if backend == "openai":
        return OpenAIClient(**kwargs)
    return QwenVLClient(**kwargs)
