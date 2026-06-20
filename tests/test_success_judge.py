"""Tests for SuccessJudge (Task 6 — Phase 2a).

In simulation the authoritative verdict is task_done() encoded in VlaTaskResult.
The VLM judge is a stub hook for hardware / partial-success detection (Phase 2c/3).
"""
from __future__ import annotations

from agentbot.monitor.success_judge import SuccessJudge
from agentbot.contracts.vla import VlaTaskResult, VlaTaskStatus


def test_judge_uses_task_done_in_sim():
    j = SuccessJudge()
    ok, source = j.judge(VlaTaskResult(task_id="t", status=VlaTaskStatus.SUCCEEDED, success=1, success_rate=1.0))
    assert ok is True and source == "task_done"
    bad, src2 = j.judge(VlaTaskResult(task_id="t", status=VlaTaskStatus.FAILED, success=0, success_rate=0.0))
    assert bad is False and src2 == "task_done"


def test_vlm_judge_hook_is_uncertain_stub():
    j = SuccessJudge()
    verdict, reason = j.vlm_judge(frame=None, goal="place the can on the orange plate")
    assert verdict == "uncertain"
