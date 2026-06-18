"""Skills (block 4) validate a SkillCall against their spec and translate it into
a VlaTaskRequest. They never hardcode a checkpoint — it comes from ctx."""
from agentbot.skills.builtin.sort_can import SortCanSkill
from agentbot.contracts.skills import SkillCall
from agentbot.contracts.common import Embodiment

CTX = {
    "checkpoint": "artifacts/checkpoints/gr00t/N1_7_fft_0615_150k_lr1e4_absolute_no_tune_visual",
    "checkpoint_name": "n17_150k_lr1e4_absolute",
    "embodiment": Embodiment.SIM,
    "gr00t_ver": "N1.7",
}


def test_sort_can_validate_and_translate():
    sk = SortCanSkill()
    call = SkillCall(name="sort_can", args={"target_color": "orange"})
    ok, err = sk.validate(call)
    assert ok, err
    req = sk.to_vla_request(call, CTX)
    assert req.task_name == "Isaac-Can-Sorting-OpenArm-DexHand-v0"
    assert "orange" in req.instruction
    assert req.params["target_color"] == "orange"
    assert req.checkpoint == CTX["checkpoint"]
    assert req.checkpoint_name == "n17_150k_lr1e4_absolute"
    assert req.skill_call_id == call.skill_call_id


def test_sort_can_rejects_bad_enum():
    ok, err = SortCanSkill().validate(SkillCall(name="sort_can", args={"target_color": "purple"}))
    assert not ok and "target_color" in err


def test_sort_can_rejects_missing_required():
    ok, err = SortCanSkill().validate(SkillCall(name="sort_can", args={}))
    assert not ok and "target_color" in err
