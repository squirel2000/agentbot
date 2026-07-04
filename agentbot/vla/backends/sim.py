"""Simulation backend (block 5, today): GR00T policy server + IsaacLab rollout."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult
from agentbot.settings import AGENTBOT_DIR
from agentbot.vla.isaac_runner import Publish, run_rollout as real_run_rollout
from agentbot.vla.policy_server import PolicyServer

RolloutFn = Callable[[VlaTaskRequest, Publish], VlaTaskResult]


class SimBackend:
    def __init__(self, repo_root: str, backend_spec: dict,
                 conda_sh: str = "~/miniforge3/etc/profile.d/conda.sh",
                 rollout_fn: Optional[RolloutFn] = None, manage_server: bool = True) -> None:
        self.repo_root = repo_root
        self.backend_spec = backend_spec
        self.conda_sh = conda_sh
        self.rollout_fn: RolloutFn = rollout_fn or real_run_rollout
        self.manage_server = manage_server
        self._server: Optional[PolicyServer] = None

    def run(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
        if self.manage_server:
            self._ensure_server(req)
        return self.rollout_fn(req, publish)

    def _ensure_server(self, req: VlaTaskRequest) -> None:
        if self._server and self._server.health():
            return
        srv = PolicyServer(self.backend_spec, req.checkpoint, self.repo_root, self.conda_sh)
        log_path = AGENTBOT_DIR / "var" / "logs" / f"vla_server_{req.task_id}.log"
        srv.start(str(log_path))
        if not srv.wait_ready():
            raise RuntimeError("GR00T policy server failed to become ready")
        self._server = srv

    def shutdown(self) -> None:
        if self._server:
            self._server.stop()
