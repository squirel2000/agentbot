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

# agentbot/agentbot/settings.py -> repo root is three parents up.
PKG_DIR = Path(__file__).resolve().parent
AGENTBOT_DIR = PKG_DIR.parent
REPO_ROOT = AGENTBOT_DIR.parent
_DEFAULT_CONFIG = AGENTBOT_DIR / "config" / "agentbot.yaml"
_EXAMPLE_CONFIG = AGENTBOT_DIR / "config" / "agentbot.example.yaml"


class LlmCfg(BaseModel):
    backend: str = "qwen-vl"
    model: str = "Qwen3-VL-8B"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = ""


class RedisCfg(BaseModel):
    url: str = "redis://localhost:6379/0"
    events_stream: str = "agentbot:events"
    tasks_queue: str = "agentbot:vla:tasks"
    state_hash: str = "agentbot:state"


class VectorCfg(BaseModel):
    backend: str = "chroma"
    path: str = "agentbot/var/chroma"


class MemoryCfg(BaseModel):
    sqlite_path: str = "agentbot/var/agentbot.db"
    vector: VectorCfg = Field(default_factory=VectorCfg)


class VlaCfg(BaseModel):
    embodiment: str = "sim"
    gr00t_ver: str = "N1.7"
    isaaclab_repo: str = "IsaacLab"
    isaaclab_conda_env: str = "env_isaaclab"
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


class AppConfig(BaseModel):
    llm: LlmCfg = Field(default_factory=LlmCfg)
    redis: RedisCfg = Field(default_factory=RedisCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    vla: VlaCfg = Field(default_factory=VlaCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
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
    return AppConfig.model_validate(data)
