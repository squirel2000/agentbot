"""BrainAgent (block 2) — reason -> plan -> select skills.

Orchestrates one user turn: load recent conversation, expose the skill registry as
LLM tools, call the LLM, assemble a structured ``SkillPlan`` from the returned skill
calls, persist memory, and emit agent-lifecycle events to the Monitor bus.

SCAFFOLD scope: produces the plan and persists memory. It does NOT yet dispatch the
plan to the VLA engine — that wiring (translate -> enqueue VlaTaskRequest) lands in
the API layer / MVP, marked below with ``# MVP:``.
"""
from __future__ import annotations

from typing import Optional

from agentbot.brain.gateway import Gateway
from agentbot.brain.vlm_client import VLMClient
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.messages import AgentResponse, AgentState, UserMessage
from agentbot.contracts.skills import SkillPlan
from agentbot.monitor.event_bus import EventBus
from agentbot.skills.registry import SkillRegistry


class BrainAgent:
    def __init__(self, llm: VLMClient, registry: SkillRegistry,
                 conversation: ConversationMemory, bus: Optional[EventBus] = None,
                 gateway: Optional[Gateway] = None) -> None:
        self.llm = llm
        self.registry = registry
        self.conversation = conversation
        self.bus = bus
        self.gateway = gateway or Gateway()

    async def _emit(self, session_id: str, payload: dict) -> None:
        if self.bus is not None:
            await self.bus.publish(Event(type=EventType.AGENT_LIFECYCLE, source="brain.agent",
                                         session_id=session_id, payload=payload))

    async def _complete(self, text: str, session_id: str) -> tuple[SkillPlan, str]:
        """Run the LLM decomposition for *text* against the current conversation history.

        Builds the message list from the N most-recent prior turns in *session_id*,
        then appends *text* as the current user message.  Returns ``(plan, reply_text)``
        so both ``plan()`` and ``handle()`` can use the result without duplicating logic
        or re-calling the LLM.
        """
        history = self.conversation.recent(session_id, n=10)
        messages = [{"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
                    for t in history]
        # Always append the current user message so the LLM sees it regardless of
        # whether the caller has already persisted it.
        messages.append({"role": "user", "content": text})
        tools = self.registry.as_tool_schemas()

        reply = await self.llm.complete(messages, tools)
        plan = SkillPlan(intent=text, calls=reply.calls)
        return plan, reply.text

    async def plan(self, text: str, session_id: str) -> SkillPlan:
        """Decompose *text* into an ordered ``SkillPlan`` without side-effects.

        Reads the existing conversation history for *session_id* but does NOT
        add any turns or emit events. Intended for the Phase-2 orchestrator to
        preview or execute the plan step by step.
        """
        plan, _ = await self._complete(text, session_id)
        return plan

    async def handle(self, msg: UserMessage) -> AgentResponse:
        await self._emit(msg.session_id, {"state": AgentState.THINKING.value, "turn_id": msg.turn_id})

        plan, reply_text = await self._complete(msg.text, msg.session_id)
        await self._emit(msg.session_id, {"state": AgentState.PLANNING.value,
                                          "turn_id": msg.turn_id, "n_calls": len(plan.calls)})

        # MVP: for each call -> registry.get(name).to_vla_request(call, ctx) -> job_queue.put(...)
        #      then await terminal VlaTaskResult/SkillResult via the event bus and continue the plan.

        self.conversation.add_turn(msg.session_id, "user", msg.text)
        self.conversation.add_turn(msg.session_id, "agent", reply_text)
        return self.gateway.format_response(
            session_id=msg.session_id, turn_id=msg.turn_id, text=reply_text,
            state=AgentState.DONE, plan=plan,
        )
