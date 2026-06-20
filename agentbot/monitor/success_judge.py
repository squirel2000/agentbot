"""SuccessJudge — centralised verdict on whether a VLA skill succeeded.

Two verdict paths:

sim (authoritative)
    In IsaacLab the simulator evaluates task_done() after every episode and
    encodes the result in run_manifest.json.  The worker reads that file and
    populates ``VlaTaskResult.success`` / ``VlaTaskResult.success_rate``.
    Because the source is the physics engine itself this is free, deterministic,
    and should always be preferred in simulation.  ``judge()`` implements this
    path and returns the string tag ``"task_done"`` so callers know which path
    fired.

hardware / ambiguous (VLM hook, Phase 2c / 3)
    On real hardware there is no task_done().  ``vlm_judge()`` is a stable hook
    that a future implementation will fill by POSTing the latest camera frame and
    the skill goal to a small served VLM (e.g. a distilled Qwen3-VL endpoint) and
    parsing a ``success | fail | uncertain`` verdict.  It is also useful in
    simulation for early "stuck / partial" detection before an episode formally
    ends.  For now the method is a documented stub that always returns
    ``"uncertain"`` so the orchestrator can fall back to task_done safely.
"""
from __future__ import annotations

from typing import Any

from agentbot.contracts.vla import VlaTaskResult


class SuccessJudge:
    """Centralised arbiter for VLA skill success.

    Single responsibility: given a completed (or in-progress) skill execution,
    return a verdict and the evidence source so the orchestrator can gate the
    next action.
    """

    # ------------------------------------------------------------------
    # Sim path — authoritative
    # ------------------------------------------------------------------

    def judge(self, result: VlaTaskResult) -> tuple[bool, str]:
        """Return ``(success, source)`` using the simulator's task_done verdict.

        Parameters
        ----------
        result:
            The ``VlaTaskResult`` produced by the IsaacLab worker.  The
            ``success`` field is populated from ``run_manifest.json`` which
            records the physics-engine task_done() evaluation — this is the
            authoritative source in simulation.

        Returns
        -------
        tuple[bool, str]
            ``(True, "task_done")`` when at least one episode succeeded,
            ``(False, "task_done")`` otherwise.  The ``"task_done"`` tag lets
            callers distinguish this path from the VLM path.
        """
        return (result.success > 0, "task_done")

    # ------------------------------------------------------------------
    # Hardware / ambiguous path — VLM hook (stub, Phase 2c / 3)
    # ------------------------------------------------------------------

    def vlm_judge(self, frame: Any, goal: str) -> tuple[str, str]:
        """Assess success from a camera frame + skill goal via a small VLM.

        This method is a **stable hook** for Phase 2c / 3.  A real
        implementation would:

        1. Encode ``frame`` (e.g. a numpy HWC image) as JPEG/base64.
        2. POST ``{image, goal}`` to a served VLM endpoint
           (e.g. a distilled Qwen3-VL inference server).
        3. Parse the model's response into one of
           ``"success" | "fail" | "uncertain"``.

        On real hardware there is no task_done(), so this is the *only*
        verdict path.  In simulation it can provide early partial/stuck
        detection before the episode formally ends.

        Parameters
        ----------
        frame:
            Latest camera observation (numpy array, PIL Image, or ``None``).
            ``None`` is accepted so callers can instantiate and test the hook
            without a real frame.
        goal:
            Natural-language description of the skill that was attempted,
            e.g. ``"place the can on the orange plate"``.

        Returns
        -------
        tuple[str, str]
            ``(verdict, reason)`` where verdict is one of
            ``"success" | "fail" | "uncertain"`` and reason is a human-readable
            explanation.  The stub always returns ``("uncertain", ...)`` so the
            orchestrator falls back to task_done safely until this is wired.
        """
        return ("uncertain", "vlm judge not wired (Phase 2c/3)")
