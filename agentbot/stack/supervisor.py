"""Sim-stack supervisor (block 3/5 glue): wake, watch, and tear down the stack.

Ports the proven ``scripts/stack/{start,stop}_stack.sh`` flow into agentbot so the
orchestrator owns process lifecycle:

    redis -> VLM Brain(:8000) -> GR00T policy(:5555) -> IsaacLab sim_session -> dashboard(:8780)

Every service is a plain ``bash -c`` command (activate env, cd repo, exec server) —
identical to what the shell scripts ran — launched detached with its log under
``<agentbot>/var/stack/``. Sibling-repo locations resolve through the workspace
map (:func:`agentbot.settings.workspace_path`), never hardcoded. Lifecycle changes
are emitted as best-effort SYSTEM_STATUS events on the redis bus so the Monitor /
dashboard can observe the stack.
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentbot.contracts.events import Event, EventType
from agentbot.settings import (
    AGENTBOT_DIR,
    REPO_ROOT,
    AppConfig,
    load_config,
    workspace_path,
)

# Redis keys cleared on up/down so a fresh stack never auto-execs stale tasks.
_CLEAR_KEYS = ("agentbot:vla:tasks", "agentbot:commands", "agentbot:state", "agentbot:events")
# Process patterns force-stopped on down (sim_session shuts down slowly on SIGTERM).
_KILL_PATTERNS = ("agentbot.vla.sim_session", "run_sim_session.sh")


@dataclass
class Service:
    name: str
    cmd: str                       # full `bash -c` string (env + cd + exec)
    port: Optional[int] = None     # liveness probe; None -> pid/pattern only


def port_open(port: int, host: str = "localhost", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_services(cfg: AppConfig, headless: Optional[bool] = None) -> list[Service]:
    """The stack as data. Mirrors start_stack.sh command-for-command."""
    st = cfg.stack
    brain = os.environ.get("BRAIN_MODEL") or str((REPO_ROOT / st.brain_model).resolve())
    ckpt = os.environ.get("GR00T_CKPT") or str((REPO_ROOT / st.gr00t_checkpoint).resolve())
    hd = st.headless if headless is None else headless
    conda_sh = os.path.expanduser(st.conda_sh)
    disp = os.environ.get("DISPLAY", ":0")
    xauth = os.environ.get("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

    vlm_repo = workspace_path("isaac_gr00t_vlm", "engines/vlm/Isaac-GR00T-VLM")
    n1d7 = workspace_path("isaac_gr00t_n1d7", "engines/vla/Isaac-GR00T_n1d7")
    isaaclab = workspace_path("isaaclab", "engines/sim/IsaacLab")
    simflag = " --headless" if hd else ""

    return [
        Service(
            "vlm",
            f"cd {shlex.quote(str(vlm_repo))} && VLM_MODEL_DIR={shlex.quote(brain)} "
            f"CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 exec bash examples/run_vlm_server.sh",
            st.vlm_port,
        ),
        Service(
            "gr00t",
            f"cd {shlex.quote(str(n1d7))} && exec .venv/bin/python -m gr00t.eval.run_gr00t_server "
            f"--model-path {shlex.quote(ckpt)} --embodiment-tag new_embodiment --port {st.gr00t_port}",
            st.gr00t_port,
        ),
        Service(
            "sim_session",
            f"source {shlex.quote(conda_sh)} && conda activate {cfg.vla.isaaclab_conda_env} && "
            f"export OMNI_KIT_ACCEPT_EULA=YES DISPLAY={shlex.quote(disp)} XAUTHORITY={shlex.quote(xauth)} && "
            f"cd {shlex.quote(str(isaaclab))} && exec python -m agentbot.vla.sim_session{simflag}",
            None,
        ),
        Service(
            "dashboard",
            f"cd {shlex.quote(str(AGENTBOT_DIR))} && exec uv run uvicorn agentbot.api.app:app "
            f"--host 0.0.0.0 --port {st.dashboard_port}",
            st.dashboard_port,
        ),
    ]


class StackSupervisor:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or load_config()
        self.log_dir = AGENTBOT_DIR / "var" / "stack"
        self.pid_file = self.log_dir / "stack.pids"

    # ---------------------------------------------------------------- events
    def _emit(self, payload: dict) -> None:
        """Best-effort SYSTEM_STATUS on the redis events stream (never raises)."""
        try:
            import redis

            ev = Event(type=EventType.SYSTEM_STATUS, source="stack.supervisor", payload=payload)
            r = redis.Redis.from_url(self.cfg.redis.url, socket_connect_timeout=1)
            r.xadd(self.cfg.redis.events_stream, {"json": ev.model_dump_json()})
        except Exception:
            pass

    # ----------------------------------------------------------------- redis
    def _redis_ping(self) -> bool:
        try:
            import redis

            return bool(redis.Redis.from_url(self.cfg.redis.url, socket_connect_timeout=1).ping())
        except Exception:
            return False

    def ensure_redis(self) -> None:
        if not self._redis_ping():
            subprocess.run(["redis-server", "--daemonize", "yes"], check=False)
            time.sleep(1)

    def clear_queues(self) -> None:
        try:
            import redis

            r = redis.Redis.from_url(self.cfg.redis.url, socket_connect_timeout=1)
            r.delete(*_CLEAR_KEYS)
        except Exception:
            pass

    # -------------------------------------------------------------------- up
    def up(self, headless: Optional[bool] = None, dry_run: bool = False) -> list[Service]:
        services = build_services(self.cfg, headless)
        if dry_run:
            for s in services:
                print(f"[stack:dry-run] {s.name}: bash -c {s.cmd!r}")
            return services

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text("")
        self.ensure_redis()
        self.clear_queues()
        print(f"[stack] redis: {'PONG' if self._redis_ping() else 'DOWN'} ; queues cleared")

        for s in services:
            log = self.log_dir / f"{s.name}.log"
            proc = subprocess.Popen(
                ["bash", "-c", s.cmd],
                stdout=open(log, "w"), stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.pid_file.open("a") as f:
                f.write(f"{s.name} {proc.pid}\n")
            print(f"[stack] {s.name} -> pid {proc.pid} ({log})")
            self._emit({"service": s.name, "status": "launched", "pid": proc.pid})

        print(f"[stack] all launched (pids in {self.pid_file}).")
        print(f"  logs:      tail -f {self.log_dir}/*.log")
        print(f"  UI:        http://localhost:{self.cfg.stack.dashboard_port}")
        print("  teardown:  python -m agentbot.stack down")
        return services

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        pids: dict[str, int] = {}
        if self.pid_file.is_file():
            for line in self.pid_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    pids[parts[0]] = int(parts[1])

        out: dict[str, dict] = {"redis": {"up": self._redis_ping()}}
        for s in build_services(self.cfg):
            pid = pids.get(s.name)
            alive = pid is not None and Path(f"/proc/{pid}").exists()
            entry: dict = {"pid": pid, "pid_alive": alive}
            if s.port is not None:
                entry["port"] = s.port
                entry["port_open"] = port_open(s.port)
                entry["up"] = entry["port_open"]
            else:
                entry["up"] = alive
            out[s.name] = entry
        return out

    # ------------------------------------------------------------------ down
    def down(self) -> None:
        print("[stack] shutting down…")
        # 1) pids from the pid file
        if self.pid_file.is_file():
            for line in self.pid_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    subprocess.run(["kill", parts[1]], check=False,
                                   stderr=subprocess.DEVNULL)
        # 2) by port (belt and braces)
        st = self.cfg.stack
        for port in (st.dashboard_port, st.vlm_port, st.gr00t_port):
            r = subprocess.run(["lsof", f"-ti:{port}"], capture_output=True, text=True)
            for pid in r.stdout.split():
                subprocess.run(["kill", pid], check=False, stderr=subprocess.DEVNULL)
        # 3) by pattern (sim_session tears down slowly; force leftovers)
        for pat in _KILL_PATTERNS:
            subprocess.run(["pkill", "-f", pat], check=False)
        time.sleep(3)
        subprocess.run(["pkill", "-9", "-f", _KILL_PATTERNS[0]], check=False)
        # 4) clear redis so next up never auto-execs stale tasks
        self.clear_queues()
        if self.pid_file.is_file():
            self.pid_file.write_text("")
        self._emit({"service": "stack", "status": "down"})
        print("[stack] done.")

    # ------------------------------------------------------------------ logs
    def logs(self) -> None:
        for s in build_services(self.cfg):
            log = self.log_dir / f"{s.name}.log"
            print(f"--- {log} ({'exists' if log.is_file() else 'missing'})")
