# AgentBot — the agentic stack (architecture.png)

A natural-language command → an LLM **Brain** that reasons/plans/selects **skills** → each skill drives the
**GR00T VLA** to execute in **IsaacLab** (today) or on real hardware (future). A **Monitor / event bus** watches
every layer and closes the feedback + immediate-safety loops.

```
1 User Input → 2 Brain/ClawBot → 4 Skill Layer → 5 VLA Engine → (6 Robot Control → 7 Physical)
                         3 Monitor / Event Bus observes all + feedback
```

Full design: [`docs/agentbot_architecture.html`](../docs/agentbot_architecture.html).
This package is the **Phase-0 scaffold**: contracts, services, event bus, API, console, and tests — runnable and
demoable end-to-end with a fake VLA (no GPU). Porting the real IsaacLab rollout is Phase 1.

## Layout → architecture blocks

| Path | Block |
|------|-------|
| `agentbot/contracts/` | the **protocol** (Pydantic; shared by all layers) |
| `agentbot/brain/` | **2** — gateway, pluggable LLM (Qwen3-VL default), memory |
| `agentbot/skills/` | **4** — semantic action tools + registry |
| `agentbot/vla/` | **5** — worker, IsaacLab runner, GR00T policy-server lifecycle (reuses `scripts/eval/*`) |
| `agentbot/monitor/` | **3** — event bus, live state, safety watchdog, ingest |
| `agentbot/api/` + `agentbot/ui/` | wires 1·2·4·5; read-only Console |

## Install

Prereq: [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync            # builds an isolated .venv from pyproject + uv.lock (does not touch conda envs)
uv run pytest -q   # 23 passed
```

`uv sync` also installs the `dev` group; add `--extra vector` for the optional semantic-memory store.

## Run

> All commands below run **from the `agentbot/` folder** (the uv project root), unless noted.

**Single-process demo** (in-proc bus + fake VLA, no Redis/GPU) — open <http://localhost:8780>:

```bash
cd agentbot
uv run uvicorn agentbot.api.app:app --port 8780
```

**Real, multi-process** — Core/Console and the GPU worker are **separate processes in separate terminals**.

```bash
# one-time: create your config and switch the backbone to redis
cp agentbot/config/agentbot.example.yaml agentbot/config/agentbot.yaml
#   then edit agentbot/config/agentbot.yaml  →  set the bottom line:  backbone: redis
```

```bash
# ── Terminal 1 · Redis (the cross-process bus) ───────────────────────────
redis-server

# ── Terminal 2 · Core + Console (uv venv) ────────────────────────────────
cd agentbot && uv run uvicorn agentbot.api.app:app --port 8780

# ── Terminal 3 · VLA worker, in env_isaaclab (it imports IsaacLab) ───────
#   install the agentbot package INTO env_isaaclab once — run this in the
#   agentbot/ folder (NOT in IsaacLab/), so `agentbot` is importable there:
conda activate env_isaaclab
cd agentbot && pip install -e .
python -m agentbot.vla.worker        # dequeues VLA tasks → runs the IsaacLab rollout
```

No GPU handy? Check the worker plumbing without IsaacLab:
`cd agentbot && uv run python -m agentbot.vla.worker --fake --selftest`

## Configuration

Copy `config/agentbot.example.yaml` → `config/agentbot.yaml` (gitignored) and edit. Key knobs: `llm.backend`
(`qwen-vl` | `claude` | `openai`), `backbone` (`in-proc` | `redis`), and the **swappable checkpoint registry**.

### Swap the checkpoint

The model path is **never hardcoded** — it lives in `vla.checkpoints` with a `default_checkpoint` pointer.
Three ways to swap:

```yaml
vla:
  default_checkpoint: n17_150k_lr1e4_absolute     # ← change this, or
  checkpoints:
    n17_150k_lr1e4_absolute: artifacts/checkpoints/gr00t/N1_7_fft_0615_150k_lr1e4_absolute_no_tune_visual
    n17_150k_lr5e5:          artifacts/checkpoints/gr00t/N1_7_fft_0614_150k_lr5e5_no_tune_visual  # ← add entries
```

```bash
# ← or override per request (registry name or a literal path):
curl -X POST :8780/v1/skills/sort_can/invoke \
  -d '{"args":{"target_color":"orange"},"checkpoint":"n17_150k_lr5e5"}'
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat` | user message → `AgentResponse` (+ `SkillPlan`) |
| WS | `/v1/chat/stream` | streaming dialogue |
| GET | `/v1/skills` | registered skills (as LLM tool schemas) |
| POST | `/v1/skills/{name}/invoke` | validate + dispatch a skill to the VLA engine |
| POST/GET | `/v1/vla/tasks[/{id}]` | submit / read a VLA task |
| GET | `/v1/vla/checkpoints` | the swappable checkpoint registry |
| GET | `/v1/state` · WS `/v1/events` | Monitor snapshot + live event stream |
