# AgentBot

**The orchestration layer that runs the whole agentic robot system.** A natural-language command flows
through an LLM/VLM **Brain** that reasons, plans, and decomposes it into atomic **skills**; each skill drives the
**GR00T VLA** to execute — today in the **IsaacLab** simulator, tomorrow on the real OpenArm. A **Monitor / event
bus** watches every layer and closes the feedback + immediate-safety loops. AgentBot also owns the **dashboard
(UI)**, the **command queue**, and the **records/History + stats**.

```
        ┌─────────────────────────── AgentBot controls all of this ───────────────────────────┐
 you ─► 1 User Input ─► 2 Brain (VLM) ─► 4 Skill Layer ─► 5 VLA Engine ─► (6 Robot ─► 7 Physical)
   (UI/dashboard)         reason·plan·       atomic           GR00T +            └ future hardware
                          decompose          skills           IsaacLab
                                  ▲                                │
                                  └──────── 3 Monitor / Event Bus ─┘   (live state · feedback · safety · History)
```

> 📊 **Full architecture, module breakdown & animated data-flow:** [`docs/agentbot_system.html`](docs/agentbot_system.html)
> 📋 **Status & roadmap:** [`TASKS.md`](TASKS.md) · **Phase-2 plan:** [`docs/plans/`](docs/plans/)

---

## Quick start

Prereq: [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Run everything from the `agentbot/` folder.

```bash
uv sync && uv run pytest -q          # install + run the unit suite (31 passed)
```

### A) No-GPU demo (one process, fake VLA)
Set `backbone: in-proc` in `config/agentbot.yaml`, then open <http://localhost:8780>:
```bash
uv run uvicorn agentbot.api.app:app --port 8780
```
Type a command in the dashboard → the Brain plans it → a fake VLA "runs" it → watch it reach `done`.

### B) Real flow on a GPU (4 processes)
Set `backbone: redis` in `config/agentbot.yaml`. Then, in 4 terminals:
```bash
redis-server                                                      # 1 · the cross-process bus
uv run uvicorn agentbot.api.app:app --port 8780                   # 2 · Core + dashboard + orchestrator
cd Isaac-GR00T_n1d7 && source .venv/bin/activate \
  && python -m gr00t.eval.run_gr00t_server --model-path <ckpt> --embodiment-tag new_embodiment --port 5555   # 3 · GR00T server
conda activate env_isaaclab && pip install -e agentbot \
  && cd IsaacLab && python -m agentbot.vla.sim_session --headless  # 4 · persistent IsaacLab (stays open)
```
Then send commands from the dashboard (or `POST /v1/commands`). The orchestrator dispatches skills one at a time
to the always-open IsaacLab session; the Monitor confirms each before the next.

No GPU? Smoke-test the worker path: `cd agents/agentbot && uv run python -m agentbot.vla.sim_session --selftest` (needs a GR00T server) or the in-proc demo (A).

---

## Configuration

**Everything configurable lives in one file: `config/agentbot.yaml`** (copy it from
`config/agentbot.example.yaml`; it's gitignored). **Every option is documented inline in that file** — there's
nothing to look up here. The big three: `backbone` (`in-proc` | `redis`), `vlm.backend` (the Brain model), and the
**swappable checkpoint registry** (`vla.checkpoints` + `default_checkpoint`; never hardcoded — change it there or
override per request with a `checkpoint` field).

---

## Code structure & reading guide

```
agentbot/agentbot/
  contracts/        ← THE PROTOCOL. Pydantic models crossing every boundary. Read this first.
      common·messages·skills·vla·events·commands
  brain/            ← block 2. The "thinking" layer.
      gateway·agent·orchestrator·vlm_client·memory/
  skills/           ← block 4. Semantic action tools (sort_can, pour_water) + registry.
  vla/              ← block 5. GR00T + IsaacLab execution.
      sim_session (persistent, Phase 2)·isaac_runner (one-shot, Phase 1)·policy_server·worker·backends/
  monitor/          ← block 3. event_bus·state_store·job_queue·command_queue·results·safety·success_judge·ingest
  records/          ← SQLite task-management store (commands, skill_runs, stats).
  api/              ← FastAPI: app·deps (composition root)·routes_{chat,commands,skills,vla,events}.
  ui/console.html   ← the hierarchical dashboard (vanilla, served at /).
```

**Reading order for a newcomer:** `contracts/` (the vocabulary) → `brain/orchestrator.py` (the command loop) →
`vla/sim_session.py` (how a skill becomes a real episode) → `api/deps.py` (how it's all wired) → `ui/console.html`.

---

## HTTP API (the dashboard uses these)

| Method | Path | Purpose |
|--------|------|---------|
| POST · GET · DELETE | `/v1/commands[/{id}]` | enqueue a command · list/history · drill-down · clear queue |
| GET | `/v1/stats` | per-skill success rate + counts (task management) |
| POST | `/v1/control/{stop,estop}` | stop the orchestrator / trip the safety watchdog |
| GET | `/v1/skills` · `/v1/vla/checkpoints` | registered skills · the checkpoint registry |
| GET | `/v1/state` · WS `/v1/events` | Monitor live snapshot · live event stream |
| POST | `/v1/chat` · `/v1/skills/{name}/invoke` | one-shot plan preview · dispatch a single skill |
