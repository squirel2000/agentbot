"""BrainAgent integration: a user message -> SkillPlan via the stub LLM + registry,
with conversation memory persisted and lifecycle events emitted."""
from agentbot.brain.agent import BrainAgent
from agentbot.brain.gateway import Gateway
from agentbot.brain.vlm_client import QwenVLClient
from agentbot.brain.memory.store import MemoryStore
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.monitor.event_bus import InProcEventBus
from agentbot.skills.registry import SkillRegistry
from agentbot.skills.builtin.sort_can import SortCanSkill


def _agent():
    reg = SkillRegistry()
    reg.register(SortCanSkill())
    conv = ConversationMemory(MemoryStore.in_memory())
    return BrainAgent(QwenVLClient(), reg, conv, bus=InProcEventBus())


async def test_brain_builds_plan_from_text():
    agent = _agent()
    msg = Gateway().normalize(session_id="s1", text="please sort the can onto the orange plate")
    resp = await agent.handle(msg)
    assert resp.plan is not None
    assert len(resp.plan.calls) == 1
    call = resp.plan.calls[0]
    assert call.name == "sort_can"
    assert call.args.get("target_color") == "orange"


async def test_brain_persists_conversation():
    agent = _agent()
    msg = Gateway().normalize(session_id="s2", text="sort the can onto green")
    await agent.handle(msg)
    turns = agent.conversation.recent("s2", n=10)
    assert [t["role"] for t in turns] == ["user", "agent"]


async def test_agent_plan_returns_skillplan():
    agent = _agent()   # QwenVLClient stub + sort_can registered
    plan = await agent.plan("sort the can onto the orange plate", session_id="s1")
    assert plan.calls and plan.calls[0].name == "sort_can"
    assert plan.calls[0].args.get("target_color") == "orange"
