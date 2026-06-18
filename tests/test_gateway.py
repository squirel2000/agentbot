"""Gateway (block 1->2): normalize raw input into a UserMessage; format replies."""
from agentbot.brain.gateway import Gateway
from agentbot.contracts.common import Modality
from agentbot.contracts.messages import AgentState


def test_gateway_normalizes_text():
    g = Gateway()
    msg = g.normalize(session_id="s1", modality="text", text="  sort the can onto orange  ")
    assert msg.session_id == "s1"
    assert msg.modality == Modality.TEXT
    assert msg.text == "sort the can onto orange"
    assert msg.turn_id.startswith("turn_")


def test_gateway_coerces_unknown_modality_to_text():
    g = Gateway()
    msg = g.normalize(session_id="s1", modality="weird", text="hi")
    assert msg.modality == Modality.TEXT


def test_gateway_format_response():
    g = Gateway()
    resp = g.format_response(session_id="s1", turn_id="turn_x", text="done", state=AgentState.DONE)
    assert resp.session_id == "s1" and resp.turn_id == "turn_x" and resp.text == "done"
