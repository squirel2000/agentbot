"""IsaacLab rollout (block 5, System 1 closed-loop execution).

MVP strategy (Phase 1): rather than re-implement IsaacLab's delicate import-time app
launch in-process, ``run_rollout`` drives the **proven** eval agent
``agents/evalbot/harness/gr00t_infer_agent.py`` for ONE episode and streams/parses its output.
This is exactly "block 5 reuses the eval harness" — battle-tested, low-risk. (A persistent
in-process sim is a Phase-2 optimization.)

It replicates ``run_eval.py``'s client invocation:
  cwd = <repo>/IsaacLab ; PYTHONPATH += <repo>/Isaac-GR00T_n1d7 (N1.7 ZMQ wire format) ;
  python = the worker's own interpreter (the worker runs in env_isaaclab).
The GR00T policy server is brought up separately by ``SimBackend`` (``policy_server.py``).

``run_rollout`` imports nothing heavy at module load, so this file imports cleanly in
any env (API/tests); IsaacLab is only touched inside the spawned agent subprocess.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult, VlaTaskStatus, VlaTelemetry
from agentbot.settings import REPO_ROOT, load_config

Publish = Callable[[Event], None]

# Parse the agent's stdout (see gr00t_infer_agent.py print statements).
_TEL = re.compile(r"Ep\s+\d+\s+\|\s+Step\s+(\d+)\s+\|\s+SimTime\s+([\d.]+)s:\s+Inference:\s+([\d.]+)s")
_FIN = re.compile(r"Episode\s+\d+\s+finished.*?Success:\s+(True|False).*?Terminated:\s+(True|False)"
                  r".*?Truncated:\s+(True|False)")

# gr00t_ver -> the eval backend-config JSON (host/port/embodiment/server_repo/client_pythonpath)
_BACKEND_CFG = {"N1.7": "gr00t_n17_openarm_o6", "N1.6": "gr00t_n16_openarm_o6", "N1.5": "gr00t_n15_openarm_o6"}


def eval_harness_dir(cfg=None):
    """The eval harness dir (gr00t_infer_agent.py, configs/, utils/) — location is
    config-driven (``vla.eval_harness``, workspace-root relative), never hardcoded."""
    cfg = cfg or load_config()
    return (REPO_ROOT / cfg.vla.eval_harness).resolve()


def backend_spec(gr00t_ver: str) -> dict:
    """Read the eval backend-config JSON so the worker and eval harness stay consistent."""
    name = _BACKEND_CFG.get(gr00t_ver, "gr00t_n17_openarm_o6")
    p = eval_harness_dir() / "configs" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {"host": "localhost", "port": 5555}


def _telemetry_event(req: VlaTaskRequest, tel: VlaTelemetry) -> Event:
    return Event(type=EventType.VLA_TELEMETRY, source="vla.isaac_runner",
                 task_id=req.task_id, payload=tel.model_dump())


class Runner(Protocol):
    def run_rollout(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult: ...


def run_rollout(req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
    """Run ONE IsaacLab episode for *req* by driving agents/evalbot/harness/gr00t_infer_agent.py.

    Assumes the GR00T policy server is already up on the backend's host:port (SimBackend
    ensures this) and that this process runs in env_isaaclab (IsaacLab importable).
    """
    cfg = load_config()
    spec = backend_spec(req.gr00t_ver)
    isaaclab = (REPO_ROOT / cfg.vla.isaaclab_repo).resolve()
    harness = eval_harness_dir(cfg)
    agent = harness / "gr00t_infer_agent.py"
    policy_config = harness / "configs" / f"{_BACKEND_CFG.get(req.gr00t_ver)}.json"
    host = spec.get("host", "localhost")
    port = str(spec.get("port", 5555))
    opts = req.options or {}
    save_dir = f"output/infer_record/agentbot_{req.task_id}"   # relative to IsaacLab cwd

    cmd = [sys.executable, "-u", str(agent),
           "--task", req.task_name,
           "--policy", "gr00t", "--gr00t_ver", req.gr00t_ver,
           "--policy_config", str(policy_config),
           "--host", host, "--port", port,
           "--max_eps_num", "1",
           "--openarm_hand_type", req.params.get("openarm_hand_type", "linkerhand_o6"),
           "--save_dir", save_dir,
           "--pov_list", "head"]
    if req.params.get("multitask", "Can-Sorting" in req.task_name):
        cmd.append("--multitask")
    if opts.get("filter", True):
        cmd.append("--filter")
    if opts.get("save_video", True):
        cmd.append("--save_video")
    if opts.get("headless", True):
        cmd.append("--headless")

    env = dict(os.environ)
    cpp = spec.get("client_pythonpath")
    if cpp:
        env["PYTHONPATH"] = str((REPO_ROOT / cpp).resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    publish(Event(type=EventType.VLA_TELEMETRY, source="vla.isaac_runner", task_id=req.task_id,
                  payload={"status": "running", "task_name": req.task_name, "instruction": req.instruction}))

    t0 = time.time()
    success = terminated = truncated = False
    steps = 0
    latencies: list[float] = []
    proc = subprocess.Popen(cmd, cwd=str(isaaclab), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        m = _TEL.search(line)
        if m:
            steps = int(m.group(1))
            lat = float(m.group(3))
            latencies.append(lat)
            publish(_telemetry_event(req, VlaTelemetry(
                task_id=req.task_id, step=steps, sim_time=float(m.group(2)), inference_latency_s=lat)))
        f = _FIN.search(line)
        if f:
            success = f.group(1) == "True"
            terminated = f.group(2) == "True"
            truncated = f.group(3) == "True"
    rc = proc.wait()

    # The agent's run_manifest.json is the AUTHORITATIVE result; stdout parse is the fallback.
    artifacts = {"save_dir": str(isaaclab / save_dir)}
    result_data: dict = {}
    manifest = _find_manifest(isaaclab / save_dir)
    if manifest:
        artifacts["manifest"] = str(manifest)
        try:
            result_data = json.loads(manifest.read_text()).get("result", {})
        except Exception:  # noqa: BLE001 - tolerate a partial/missing manifest
            result_data = {}

    n_success = int(result_data["success"]) if "success" in result_data else (1 if success else 0)
    success = n_success > 0
    success_rate = float(result_data.get("success_rate", 1.0 if success else 0.0))
    status = (VlaTaskStatus.SUCCEEDED if success
              else VlaTaskStatus.FAILED if rc == 0
              else VlaTaskStatus.ABORTED)
    return VlaTaskResult(
        task_id=req.task_id, status=status,
        episodes=int(result_data.get("episodes_total", 1)),
        success=n_success, success_rate=success_rate,
        steps=steps, duration_s=round(time.time() - t0, 2), artifacts=artifacts,
        error=None if rc == 0 else f"agent exited {rc}; check {save_dir}",
    )


def _find_manifest(save_dir: Path) -> Optional[Path]:
    """The agent writes run_manifest.json under save_dir/<timestamp>/; return the newest."""
    if not save_dir.exists():
        return None
    hits = sorted(save_dir.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


class FakeRunner:
    """No-IsaacLab rollout: deterministic synthetic episode for end-to-end plumbing tests."""

    def __init__(self, n_steps: int = 8, succeed: bool = True) -> None:
        self.n_steps = n_steps
        self.succeed = succeed

    def run_rollout(self, req: VlaTaskRequest, publish: Publish) -> VlaTaskResult:
        t0 = time.time()
        for step in range(1, self.n_steps + 1):
            publish(_telemetry_event(req, VlaTelemetry(
                task_id=req.task_id, step=step, sim_time=step * 0.0417, inference_latency_s=0.12)))
        return VlaTaskResult(
            task_id=req.task_id,
            status=VlaTaskStatus.SUCCEEDED if self.succeed else VlaTaskStatus.FAILED,
            episodes=1, success=1 if self.succeed else 0,
            success_rate=1.0 if self.succeed else 0.0,
            steps=self.n_steps, duration_s=round(time.time() - t0, 4),
            artifacts={"note": "fake rollout (no IsaacLab)"},
        )
