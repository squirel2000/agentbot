"""Contracts are the protocol crossing every layer boundary. They must
round-trip through JSON (API + event bus) without loss."""
from agentbot.contracts.skills import SkillSpec, SkillParamSpec, SkillCall, SkillPlan
from agentbot.contracts.vla import VlaTaskRequest, VlaTaskStatus, VlaTaskResult
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.messages import UserMessage, AgentResponse


def test_skillspec_to_tool_schema():
    spec = SkillSpec(
        name="sort_can",
        description="sort a can onto a colored plate",
        params=[SkillParamSpec(name="target_color", type="enum", enum=["orange", "green"], description="plate color")],
    )
    t = spec.to_tool_schema()
    assert t["name"] == "sort_can"
    assert t["input_schema"]["properties"]["target_color"]["enum"] == ["orange", "green"]
    assert "target_color" in t["input_schema"]["required"]


def test_skillplan_roundtrip():
    plan = SkillPlan(
        intent="tidy table",
        goal="can on orange plate",
        calls=[SkillCall(name="sort_can", args={"target_color": "orange"}, safety_flags=["avoid_human"])],
    )
    back = SkillPlan.model_validate_json(plan.model_dump_json())
    assert back.calls[0].args["target_color"] == "orange"
    assert back.calls[0].safety_flags == ["avoid_human"]


def test_vla_request_defaults_and_event():
    r = VlaTaskRequest(
        task_name="Isaac-Can-Sorting-OpenArm-DexHand-v0",
        instruction="place the can on the orange plate",
        checkpoint="artifacts/checkpoints/gr00t/x",
    )
    assert r.task_id.startswith("task_") and r.embodiment.value == "sim"
    ev = Event(type=EventType.VLA_TELEMETRY, source="vla.worker", task_id=r.task_id, payload={"step": 1})
    assert Event.model_validate_json(ev.model_dump_json()).task_id == r.task_id


def test_vla_request_checkpoint_name_audit():
    r = VlaTaskRequest(
        task_name="t", instruction="i",
        checkpoint="artifacts/checkpoints/gr00t/N1_7_fft_0614_150k_lr5e5_no_tune_visual",
        checkpoint_name="n17_150k_lr5e5",
    )
    assert r.checkpoint_name == "n17_150k_lr5e5"


def test_vla_result_status_enum():
    res = VlaTaskResult(task_id="task_x", status=VlaTaskStatus.SUCCEEDED, episodes=1, success=1, success_rate=1.0)
    assert VlaTaskResult.model_validate_json(res.model_dump_json()).status == VlaTaskStatus.SUCCEEDED


def test_agentresponse_carries_plan():
    ar = AgentResponse(session_id="s1", turn_id="t1", text="on it", plan=SkillPlan(intent="x"))
    assert AgentResponse.model_validate_json(ar.model_dump_json()).plan.intent == "x"


def test_usermessage_defaults():
    m = UserMessage(session_id="s1", text="hello")
    assert m.modality.value == "text" and m.turn_id.startswith("turn_") and m.ts > 0
