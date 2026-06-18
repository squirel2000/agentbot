"""Gateway (block 2 entry): session management + input normalization (block 1).

Turns raw voice/text/UI input into a canonical ``UserMessage`` and formats the
Brain's reply as an ``AgentResponse``. Voice ASR/TTS is a documented stub for now.
"""
from __future__ import annotations

from typing import Optional

from agentbot.contracts.common import Modality
from agentbot.contracts.messages import AgentResponse, AgentState, UserMessage
from agentbot.contracts.skills import SkillPlan


class Gateway:
    def normalize(self, session_id: str, modality: str = "text", text: str = "",
                  image_path: Optional[str] = None) -> UserMessage:
        try:
            mod = Modality(modality)
        except ValueError:
            mod = Modality.TEXT
        # Voice -> text would run ASR here (stub: text is assumed already transcribed).
        return UserMessage(session_id=session_id, modality=mod,
                           text=(text or "").strip(), image_path=image_path)

    def format_response(self, session_id: str, turn_id: str, text: str = "",
                        state: AgentState = AgentState.DONE,
                        plan: Optional[SkillPlan] = None) -> AgentResponse:
        return AgentResponse(session_id=session_id, turn_id=turn_id, text=text, state=state, plan=plan)
