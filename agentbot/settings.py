"""Load + validate ``config/agentbot.yaml`` into a typed :class:`AppConfig`.

Pattern mirrors ``scripts/pipeline/core/pipeline_config.py``: an example YAML is
committed, the real ``agentbot.yaml`` (gitignored) overrides it. The checkpoint
path is **never** hardcoded in code — callers resolve it through
:meth:`VlaCfg.resolve_checkpoint`, so swapping models is a one-line config edit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

PKG_DIR = Path(__file__).resolve().parent
AGENTBOT_DIR = PKG_DIR.parent


def _find_workspace_root() -> Path:
    """The monorepo (workspace) root that holds IsaacLab/, artifacts/, scripts/ ...

    Resolution order: $AGENTBOT_WORKSPACE_ROOT override -> walk up from this
    package until a ``workspace.yaml`` root marker -> legacy fallback (agentbot
    checked out directly under the workspace root).
    """
    env = os.environ.get("AGENTBOT_WORKSPACE_ROOT")
    if env:
        return Path(env).resolve()
    for d in (AGENTBOT_DIR, *AGENTBOT_DIR.parents):
        if (d / "workspace.yaml").is_file():
            return d
    return AGENTBOT_DIR.parent


REPO_ROOT = _find_workspace_root()
_DEFAULT_CONFIG = AGENTBOT_DIR / "config" / "agentbot.yaml"
_EXAMPLE_CONFIG = AGENTBOT_DIR / "config" / "agentbot.example.yaml"


def workspace_path(key: str, default: str) -> Path:
    """Resolve a flat ``key: relpath`` line from the workspace-root workspace.yaml.

    The workspace map is how the monorepo tells agentbot where sibling repos live
    (they may move during restructures); *default* keeps things working when the
    marker file is absent (e.g. agentbot checked out standalone).
    """
    marker = REPO_ROOT / "workspace.yaml"
    rel = default
    if marker.is_file():
        for line in marker.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s.startswith("#") and s.startswith(f"{key}:"):
                rel = s.split(":", 1)[1].strip()
                break
    return (REPO_ROOT / rel).resolve()


class VlmCfg(BaseModel):
    backend: str = "qwen-vl"
    model: str = "Qwen3-VL-8B"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = ""


class RedisCfg(BaseModel):
    url: str = "redis://localhost:6379/0"
    events_stream: str = "agentbot:events"
    tasks_queue: str = "agentbot:vla:tasks"
    state_hash: str = "agentbot:state"
    abort_key: str = "agentbot:vla:abort"


class VectorCfg(BaseModel):
    backend: str = "chroma"
    path: str = "var/chroma"          # relative -> under the agentbot repo dir


class MemoryCfg(BaseModel):
    sqlite_path: str = "var/agentbot.db"   # relative -> under the agentbot repo dir
    vector: VectorCfg = Field(default_factory=VectorCfg)


class VlaCfg(BaseModel):
    embodiment: str = "sim"
    gr00t_ver: str = "N1.7"
    isaaclab_repo: str = "engines/sim/IsaacLab"
    isaaclab_conda_env: str = "env_isaaclab"
    # Eval harness location (relative to the workspace root) — gr00t_infer_agent.py,
    # backend-config JSONs, and the client utils/ package live there.
    eval_harness: str = "agents/evalbot/harness"
    default_task: str = "Isaac-Can-Sorting-OpenArm-DexHand-v0"
    options: dict = Field(default_factory=lambda: {"headless": True, "save_video": True, "filter": True})
    default_checkpoint: str = ""
    checkpoints: dict[str, str] = Field(default_factory=dict)

    def resolve_checkpoint(self, name: Optional[str] = None) -> str:
        """Resolve a checkpoint *name* (registry key) or literal *path* to a path.

        - ``None``  -> the ``default_checkpoint`` entry.
        - a registry key -> its mapped path.
        - anything that looks like a path (contains ``/`` or exists) -> returned as-is,
          so callers may pass either a name or a literal path interchangeably.
        """
        key = name or self.default_checkpoint
        if key in self.checkpoints:
            return self.checkpoints[key]
        if "/" in key or os.path.sep in key:   # already a path
            return key
        raise KeyError(
            f"Unknown checkpoint '{key}'. Known: {sorted(self.checkpoints)} "
            f"(or pass a literal path)."
        )


class ApiCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8780


class StackCfg(BaseModel):
    """Sim-stack supervisor knobs (``python -m agentbot.stack``). Paths are
    workspace-root relative; ``BRAIN_MODEL`` / ``GR00T_CKPT`` env vars override."""
    brain_model: str = ("artifacts/checkpoints/gr00t/lora_tuned_vlm_toolcall/"
                        "Cosmos-Reason2-2B-toolcall-merged")
    gr00t_checkpoint: str = ("artifacts/checkpoints/gr00t/"
                             "N1_7_fft_0614_150k_lr5e5_no_tune_visual/checkpoint-150000")
    headless: bool = False        # windowed by default (mirrors start_stack.sh)
    conda_sh: str = "~/miniforge3/etc/profile.d/conda.sh"
    vlm_port: int = 8000
    gr00t_port: int = 5555
    dashboard_port: int = 8780


class AppConfig(BaseModel):
    vlm: VlmCfg = Field(default_factory=VlmCfg)
    redis: RedisCfg = Field(default_factory=RedisCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    vla: VlaCfg = Field(default_factory=VlaCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    stack: StackCfg = Field(default_factory=StackCfg)
    backbone: str = "in-proc"   # in-proc | redis


def load_config(path: str | None = None) -> AppConfig:
    """Load config from *path*, else ``config/agentbot.yaml``, else the example."""
    if path:
        cfg_path = Path(path)
    elif _DEFAULT_CONFIG.exists():
        cfg_path = _DEFAULT_CONFIG
    else:
        cfg_path = _EXAMPLE_CONFIG
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    # Deprecated alias: old configs used `llm:` — silently promote to `vlm:`.
    if "llm" in data and "vlm" not in data:
        data["vlm"] = data.pop("llm")
    return AppConfig.model_validate(data)
