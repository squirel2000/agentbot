"""VLA Execution Engine (block 5) contracts — the 4->5 request and 5->4 result.

``checkpoint`` is the *resolved path* the worker loads; ``checkpoint_name`` is the
registry key it came from (audit/display only). Name->path resolution happens at
the Skill/API boundary via ``VlaCfg.resolve_checkpoint`` — this module holds no
default path.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import Embodiment, new_id, now_ts


class VlaTaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    skill_call_id: Optional[str] = None
    embodiment: Embodiment = Embodiment.SIM
    task_name: str                      # IsaacLab gym id, e.g. "Isaac-Can-Sorting-OpenArm-DexHand-v0"
    instruction: str                    # GR00T language prompt, e.g. "place the can on the orange plate"
    checkpoint: str                     # resolved path the worker loads
    checkpoint_name: Optional[str] = None   # registry key it came from (audit)
    gr00t_ver: str = "N1.7"
    params: dict[str, Any] = Field(default_factory=dict)       # e.g. {"target_color": "orange"}
    options: dict[str, Any] = Field(default_factory=lambda: {"headless": True, "save_video": True, "filter": True})
    ts: float = Field(default_factory=now_ts)


class VlaTaskStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class VlaTelemetry(BaseModel):
    task_id: str
    step: int = 0
    sim_time: float = 0.0
    inference_latency_s: Optional[float] = None
    success: Optional[bool] = None
    ts: float = Field(default_factory=now_ts)


class VlaTaskResult(BaseModel):
    task_id: str
    status: VlaTaskStatus
    episodes: int = 0
    success: int = 0
    success_rate: float = 0.0
    steps: int = 0
    duration_s: float = 0.0
    artifacts: dict[str, str] = Field(default_factory=dict)   # {"video":..., "parquet":..., "manifest":...}
    error: Optional[str] = None
    ts: float = Field(default_factory=now_ts)
