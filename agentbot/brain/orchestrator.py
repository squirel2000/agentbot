"""Brain Orchestrator — the persistent loop that drives command execution.

Lifecycle per command:
  1. Pop a Command from the CommandQueue.
  2. Ask the agent to plan() -> SkillPlan (ordered SkillCalls).
  3. Dispatch each SkillCall one at a time; await its result via ResultWaiter.
  4. Retry failed skills up to max_retries; if still failing, attempt replan.
  5. Record every step (set_status, set_plan, log_skill_run) in Records.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from agentbot.contracts.commands import Command, CommandStatus
from agentbot.contracts.skills import SkillCall
from agentbot.contracts.vla import VlaTaskResult

log = logging.getLogger(__name__)


class Orchestrator:
    """Drives command execution sequentially: plan -> dispatch -> await -> retry -> record.

    Args:
        agent:           Has ``async plan(text, session_id) -> SkillPlan``.
        registry:        Has ``get(name) -> SkillSpec | None``; None means unknown skill.
        command_queue:   ``CommandQueue`` with ``next(timeout) -> Command | None``.
        dispatch:        ``Callable[[SkillCall, dict], str]`` — sends a skill to the VLA
                         engine and returns a ``task_id``.
        results:         ``ResultWaiter`` — ``expect(task_id, timeout) -> VlaTaskResult``.
        records:         ``Records`` store for audit trail.
        vla_ctx:         ``Callable[[], dict]`` — produces per-dispatch context (e.g. scene).
        max_retries:     How many extra attempts after the first failure (total = max_retries+1).
        skill_timeout_s: Seconds to wait for a VLA result before treating it as timed-out.
    """

    def __init__(
        self,
        agent,
        registry,
        command_queue,
        dispatch: Callable[[SkillCall, dict], str],
        results,
        records,
        vla_ctx: Callable[[], dict],
        max_retries: int = 2,
        skill_timeout_s: float = 1800.0,
    ) -> None:
        self.agent = agent
        self.registry = registry
        self.cq = command_queue
        self.dispatch = dispatch
        self.results = results
        self.records = records
        self.vla_ctx = vla_ctx
        self.max_retries = max_retries
        self.skill_timeout_s = skill_timeout_s
        self._running = True

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Continuous loop — runs until ``stop()`` is called."""
        while self._running:
            cmd = await asyncio.to_thread(self.cq.next, 5)
            if cmd is not None:
                await self._run_command(cmd)

    async def run_one(self) -> None:
        """Pop and execute exactly one command (for tests and one-shot calls)."""
        cmd = await asyncio.to_thread(self.cq.next, 1)
        if cmd:
            await self._run_command(cmd)

    def stop(self) -> None:
        """Signal the ``run()`` loop to exit after the current command finishes."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_command(self, cmd: Command) -> None:
        """Execute a single command end-to-end and update its record."""
        self.records.set_status(cmd.command_id, CommandStatus.RUNNING)

        plan = await self.agent.plan(cmd.text, cmd.session_id)
        self.records.set_plan(cmd.command_id, plan)
        log.info("orchestrator: cmd=%s plan has %d skills", cmd.command_id, len(plan.calls))

        for call in plan.calls:
            succeeded = await self._run_skill_with_retry(cmd, call)
            if not succeeded:
                log.warning("orchestrator: skill %s failed after retries — attempting replan", call.name)
                replanned = await self._replan(cmd, call)
                if not replanned:
                    self.records.set_status(cmd.command_id, CommandStatus.FAILED,
                                            f"skill '{call.name}' failed; aborted")
                    return

        self.records.set_status(cmd.command_id, CommandStatus.DONE, "all skills complete")

    async def _run_skill_with_retry(self, cmd: Command, call: SkillCall) -> bool:
        """Dispatch ``call`` up to (max_retries + 1) times; return True on first success."""
        if self.registry.get(call.name) is None:
            log.error("orchestrator: unknown skill '%s'", call.name)
            self.records.log_skill_run(cmd.command_id, call.name, "", 0, "unknown_skill", 0, 0, 0.0)
            return False

        for attempt in range(self.max_retries + 1):
            task_id = self.dispatch(call, self.vla_ctx())
            log.debug("orchestrator: dispatched skill=%s attempt=%d task_id=%s", call.name, attempt, task_id)

            try:
                res: VlaTaskResult = await self.results.expect(task_id, self.skill_timeout_s)
            except asyncio.TimeoutError:
                log.warning("orchestrator: timeout waiting for task_id=%s skill=%s", task_id, call.name)
                self.records.log_skill_run(
                    cmd.command_id, call.name, task_id, attempt, "timeout", 0, 0, 0.0
                )
                continue

            self.records.log_skill_run(
                cmd.command_id, call.name, task_id, attempt,
                res.status.value, res.success, res.steps, res.duration_s,
            )

            if res.success:
                log.info("orchestrator: skill=%s attempt=%d succeeded", call.name, attempt)
                return True

            log.warning("orchestrator: skill=%s attempt=%d failed (status=%s)", call.name, attempt, res.status)

        return False

    async def _replan(self, cmd: Command, failed: SkillCall) -> bool:
        """Ask the agent to replan after a failed skill; execute the new plan.

        Returns True if the replan produced at least one skill and all succeeded,
        False otherwise.
        """
        replan_text = (
            f"{cmd.text} (the step '{failed.name}' failed; replan the remaining steps)"
        )
        new_plan = await self.agent.plan(replan_text, cmd.session_id)
        if not new_plan.calls:
            log.warning("orchestrator: replan returned empty plan for cmd=%s", cmd.command_id)
            return False

        self.records.set_plan(cmd.command_id, new_plan)
        for call in new_plan.calls:
            if not await self._run_skill_with_retry(cmd, call):
                return False
        return True
