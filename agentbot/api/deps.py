"""Process-wide singletons wired from config (the composition root).

Selects in-proc vs Redis impls by ``cfg.backbone``. In-proc is the default so the
whole core (Gateway -> Brain -> Skills -> Monitor) runs in one process with no
Redis for local dev/tests; switch to ``redis`` to span the API and the separate
``env_isaaclab`` VLA worker.
"""
from __future__ import annotations

from typing import Optional

from agentbot.brain.agent import BrainAgent
from agentbot.brain.gateway import Gateway
from agentbot.brain.llm_client import build_llm
from agentbot.brain.memory.conversation import ConversationMemory
from agentbot.brain.memory.episodic import EpisodicMemory
from agentbot.brain.memory.store import MemoryStore
from agentbot.contracts.common import Embodiment
from agentbot.contracts.events import Event
from agentbot.monitor.event_bus import EventBus, InProcEventBus, RedisEventBus
from agentbot.monitor.ingest import Ingestor
from agentbot.monitor.job_queue import InMemJobQueue, JobQueue, RedisJobQueue
from agentbot.monitor.safety import SafetyWatchdog
from agentbot.monitor.state_store import InMemStateStore, RedisStateStore, StateStore
from agentbot.settings import AppConfig, load_config
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

        self.registry = SkillRegistry()
        self.registry.register(SortCanSkill())
        self.registry.register(PourWaterSkill())

        store = MemoryStore(cfg.memory.sqlite_path)
        self.conversation = ConversationMemory(store)
        self.episodic = EpisodicMemory(store)
        self.gateway = Gateway()
        self.agent = BrainAgent(build_llm(cfg), self.registry, self.conversation,
                                bus=self.bus, gateway=self.gateway)
        self.ingestor = Ingestor(self.bus, self.state, self.episodic)
        self.watchdog = SafetyWatchdog(self.bus, on_stop=self._on_safety_stop)

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
