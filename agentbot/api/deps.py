"""Process-wide singletons wired from config (the composition root).

Selects in-proc vs Redis impls by ``cfg.backbone``. In-proc is the default so the
whole core (Gateway -> Brain -> Skills -> Monitor) runs in one process with no
Redis for local dev/tests; switch to ``redis`` to span the API and the separate
``env_isaaclab`` VLA worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from agentbot.brain.agent import BrainAgent
from agentbot.brain.gateway import Gateway
from agentbot.brain.vlm_client import build_vlm
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.brain.memory.episodic import EpisodicMemory
from agentbot.brain.memory.store import MemoryStore
from agentbot.brain.orchestrator import Orchestrator
from agentbot.contracts.common import Embodiment
from agentbot.contracts.events import Event
from agentbot.contracts.skills import SkillCall
from agentbot.monitor.command_queue import InMemCommandQueue, RedisCommandQueue
from agentbot.monitor.event_bus import EventBus, InProcEventBus, RedisEventBus
from agentbot.monitor.ingest import Ingestor
from agentbot.monitor.job_queue import InMemJobQueue, JobQueue, RedisJobQueue
from agentbot.monitor.results import ResultWaiter
from agentbot.monitor.safety import SafetyWatchdog
from agentbot.monitor.state_store import InMemStateStore, RedisStateStore, StateStore
from agentbot.records.store import Records
from agentbot.settings import AppConfig, REPO_ROOT, load_config
from agentbot.skills.builtin.home import HomeSkill
from agentbot.skills.builtin.pick import PickSkill
from agentbot.skills.builtin.place import PlaceSkill
from agentbot.skills.builtin.pour_water import PourWaterSkill
from agentbot.skills.builtin.sort_can import SortCanSkill
from agentbot.skills.registry import SkillRegistry


class Deps:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        if cfg.backbone == "redis":
            self.bus: EventBus = RedisEventBus(cfg.redis.url, cfg.redis.events_stream)
            self.state: StateStore = RedisStateStore(cfg.redis.url, cfg.redis.state_hash)
            self.queue: JobQueue = RedisJobQueue(cfg.redis.url, cfg.redis.tasks_queue)
        else:
            self.bus = InProcEventBus()
            self.state = InMemStateStore()
            self.queue = InMemJobQueue()

        if cfg.backbone == "redis":
            self.command_queue = RedisCommandQueue(cfg.redis.url)
        else:
            self.command_queue = InMemCommandQueue()

        self.registry = SkillRegistry()
        self.registry.register(SortCanSkill())
        self.registry.register(PourWaterSkill())
        # Composable atomic primitives (architecture.png block 4: "Pick, Place, …").
        self.registry.register(PickSkill())
        self.registry.register(PlaceSkill())
        self.registry.register(HomeSkill())

        # Resolve the sqlite path absolute (under REPO_ROOT if relative) so the DBs land in
        # the same place regardless of the process's cwd (API vs sim_session vs tests).
        db = Path(cfg.memory.sqlite_path)
        db = db if db.is_absolute() else REPO_ROOT / db
        store = MemoryStore(str(db))
        self.conversation = ConversationMemory(store)
        self.episodic = EpisodicMemory(store)
        self.records = Records(str(db.parent / "records.db"))
        self.results = ResultWaiter(self.bus)
        self.gateway = Gateway()
        def _latest_frame() -> Optional[str]:
            try:
                snap = self.state.snapshot()
            except Exception:
                return None
            return (snap.get("camera") or {}).get("frame")

        def _scene_color() -> Optional[str]:
            # The env's current target plate color, published by sim_session to state["environment"].
            try:
                return (self.state.get_state("environment") or {}).get("target_color")
            except Exception:
                return None

        self.agent = BrainAgent(build_vlm(cfg), self.registry, self.conversation,
                                bus=self.bus, gateway=self.gateway,
                                frame_provider=_latest_frame, scene_provider=_scene_color)
        self.ingestor = Ingestor(self.bus, self.state, self.episodic)
        self.watchdog = SafetyWatchdog(self.bus, on_stop=self._on_safety_stop)
        self.orchestrator = Orchestrator(
            self.agent, self.registry, self.command_queue, self._dispatch,
            self.results, self.records, vla_ctx=lambda: self.vla_ctx(None),
            max_retries=2,
        )

    def _dispatch(self, call: SkillCall, ctx: dict) -> str:
        skill = self.registry.get(call.name)
        if skill is None:
            raise ValueError(f"unknown skill '{call.name}'")
        req = skill.to_vla_request(call, ctx)
        self.queue.put(req)
        return req.task_id

    def _on_safety_stop(self, ev: Event) -> None:
        # sim: mark the robot blocked; hardware future: ROS2 e-stop override.
        self.state.set_state("robot_state", {"mode": "blocked", "detail": "safety stop"})

    def vla_ctx(self, checkpoint: Optional[str] = None) -> dict:
        """Build the skill->VLA context, resolving the checkpoint name/path here so
        skills never hardcode it. Returns checkpoint (path) + checkpoint_name (audit)."""
        cks = self.cfg.vla.checkpoints
        if checkpoint in cks:
            name, path = checkpoint, cks[checkpoint]
        elif checkpoint is None:
            name, path = self.cfg.vla.default_checkpoint, self.cfg.vla.resolve_checkpoint(None)
        else:
            name, path = None, checkpoint  # literal path passes through
        return {"checkpoint": path, "checkpoint_name": name,
                "embodiment": Embodiment(self.cfg.vla.embodiment), "gr00t_ver": self.cfg.vla.gr00t_ver}


_deps: Optional[Deps] = None


def get_deps() -> Deps:
    global _deps
    if _deps is None:
        _deps = Deps(load_config())
    return _deps


def set_deps(d: Deps) -> None:
    """Inject a Deps (used by tests to supply an isolated/in-proc instance)."""
    global _deps
    _deps = d
