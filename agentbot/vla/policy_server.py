"""GR00T policy-server lifecycle (block 5, System 2 inference).

Lifts the server-launch + readiness-poll pattern from ``scripts/eval/run_eval.py``
(``server_cmd`` / ``wait_ready``). Import-light: subprocess + log polling only, no
IsaacLab/torch import, so it can run anywhere the repo is checked out.

Backend spec mirrors ``scripts/eval/configs/gr00t_n17_openarm_o6.json``:
    {host, port, version, embodiment_tag, server_repo, server_venv?, server_conda_env?}
"""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

READY_STRING = "Server is ready and listening"
READY_TIMEOUT_S = 900


class PolicyServer:
    def __init__(self, backend: dict, checkpoint: str, repo_root: str,
                 conda_sh: str = "~/miniforge3/etc/profile.d/conda.sh") -> None:
        self.backend = backend
        self.checkpoint = checkpoint
        self.repo_root = Path(repo_root).resolve()
        self.conda_sh = conda_sh
        self.proc: Optional[subprocess.Popen] = None

    @property
    def port(self) -> int:
        return int(self.backend.get("port", 5555))

    @property
    def host(self) -> str:
        return self.backend.get("host", "localhost")

    def _workdir_activate(self) -> tuple[Path, str]:
        workdir = self.repo_root / self.backend.get("server_repo", "Isaac-GR00T")
        if self.backend.get("server_venv"):          # N1.7 uses a uv/.venv, not conda
            return workdir, f"source {shlex.quote(str(workdir / self.backend['server_venv']))}/bin/activate"
        env = self.backend.get("server_conda_env", "env_gr00t")
        return workdir, f"source {shlex.quote(self.conda_sh)} && conda activate {env}"

    def build_cmd(self) -> str:
        """Full ``bash -c`` string: activate env, cd repo, launch the GR00T server."""
        workdir, activate = self._workdir_activate()
        ver = self.backend.get("version", "N1.5")
        emb = self.backend["embodiment_tag"]
        emb = emb.upper() if ver == "N1.6" else emb     # N1.6 tyro parses the enum NAME
        server = (f"python3 -u -m gr00t.eval.run_gr00t_server "
                  f"--model-path {shlex.quote(self.checkpoint)} "
                  f"--embodiment-tag {emb} --port {self.port}")
        return f"{activate} && cd {shlex.quote(str(workdir))} && {server}"

    def start(self, log_path: str) -> "PolicyServer":
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path
        self.proc = subprocess.Popen(["bash", "-c", self.build_cmd()],
                                     stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        return self

    def wait_ready(self, timeout: int = READY_TIMEOUT_S) -> bool:
        log = Path(self._log)
        waited = 0
        while waited < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                return False                          # server died before becoming ready
            if log.exists() and READY_STRING in log.read_text(errors="ignore"):
                return True
            time.sleep(4)
            waited += 4
        return False

    def health(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
