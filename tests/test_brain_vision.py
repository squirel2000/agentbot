"""C1: build_user_content helper + BrainAgent frame_provider plumbing."""
from agentbot.brain.agent import BrainAgent, build_user_content
from agentbot.brain.gateway import Gateway
from agentbot.brain.vlm_client import QwenVLClient
from agentbot.brain.memory.store import MemoryStore
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.monitor.event_bus import InProcEventBus
from agentbot.skills.registry import SkillRegistry
from agentbot.skills.builtin.sort_can import SortCanSkill


# ---------------------------------------------------------------------------
# Unit tests for build_user_content
# ---------------------------------------------------------------------------

def test_text_only_when_no_image():
    assert build_user_content("sort it", None) == "sort it"


def test_text_only_when_empty_string_image():
    # empty string is falsy — same as None
    assert build_user_content("sort it", "") == "sort it"


def test_multimodal_when_image_present():
    c = build_user_content("sort it", "data:image/png;base64,aGk=")
    assert {"type": "text", "text": "sort it"} in c
    assert any(p["type"] == "image_url" for p in c)


def test_multimodal_image_url_value():
    img = "data:image/jpeg;base64,QQ=="
    c = build_user_content("look", img)
    img_part = next(p for p in c if p["type"] == "image_url")
    assert img_part["image_url"]["url"] == img


# ---------------------------------------------------------------------------
# Integration: frame_provider is called during plan()
# ---------------------------------------------------------------------------

def _agent(frame_provider=None):
    reg = SkillRegistry()
    reg.register(SortCanSkill())
    conv = ConversationMemory(MemoryStore.in_memory())
    return BrainAgent(QwenVLClient(), reg, conv, bus=InProcEventBus(),
                      frame_provider=frame_provider)


async def test_plan_with_frame_provider_does_not_raise():
    """frame_provider returning a data-uri must not break plan()."""
    called = []

    def provider():
        called.append(1)
        return "data:image/png;base64,aGk="

    agent = _agent(frame_provider=provider)
    plan = await agent.plan("sort the can onto the orange plate", session_id="s-vision")
    assert plan.calls and plan.calls[0].name == "sort_can"
    assert called, "frame_provider should have been called during plan()"


async def test_plan_with_none_frame_provider_still_works():
    """No frame_provider -> plain text path; no regression."""
    agent = _agent(frame_provider=None)
    plan = await agent.plan("sort the can onto the orange plate", session_id="s-no-vision")
    assert plan.calls and plan.calls[0].name == "sort_can"


async def test_handle_passes_image_path_to_complete():
    """handle() with image_path set must not raise and must return a plan."""
    agent = _agent()
    msg = Gateway().normalize(session_id="s-img", text="sort the can onto the orange plate")
    msg.image_path = "data:image/png;base64,aGk="
    resp = await agent.handle(msg)
    assert resp.plan is not None
    assert resp.plan.calls[0].name == "sort_can"
