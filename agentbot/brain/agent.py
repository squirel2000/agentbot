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
from agentbot.brain.llm_client import LLMClient
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.messages import AgentResponse, AgentState, UserMessage
from agentbot.contracts.skills import SkillPlan
from agentbot.monitor.event_bus import EventBus
from agentbot.skills.registry import SkillRegistry


class BrainAgent:
    def __init__(self, llm: LLMClient, registry: SkillRegistry,
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

    async def handle(self, msg: UserMessage) -> AgentResponse:
        self.conversation.add_turn(msg.session_id, "user", msg.text)
        await self._emit(msg.session_id, {"state": AgentState.THINKING.value, "turn_id": msg.turn_id})

        history = self.conversation.recent(msg.session_id, n=10)
        messages = [{"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
                    for t in history]
        tools = self.registry.as_tool_schemas()

        reply = await self.llm.complete(messages, tools)
        plan = SkillPlan(intent=msg.text, calls=reply.calls)
        await self._emit(msg.session_id, {"state": AgentState.PLANNING.value,
                                          "turn_id": msg.turn_id, "n_calls": len(plan.calls)})

        # MVP: for each call -> registry.get(name).to_vla_request(call, ctx) -> job_queue.put(...)
        #      then await terminal VlaTaskResult/SkillResult via the event bus and continue the plan.

        self.conversation.add_turn(msg.session_id, "agent", reply.text)
        return self.gateway.format_response(
            session_id=msg.session_id, turn_id=msg.turn_id, text=reply.text,
            state=AgentState.DONE, plan=plan,
        )
