# Using the fine-tuned GR00T-VLM as AgentBot's Brain

Operator guide for running AgentBot with the **fine-tuned Cosmos-Reason2-2B** VLM as the Brain
(`vlm.backend: gr00t-vlm`). Full pipeline + 4090 build details: see
`IsaacLab-GR00T/docs/DELIVERY_4090.md` (super-repo). This doc is the **AgentBot-side** authoritative guide (config, sim, UI, monitoring).

---

## What the Brain does
A natural-language command (+ the camera frame + state) → the VLM returns **tool-calls**
(`sort_can`, `pick`, `place`, `home`, `pour_water`) → `Gr00tVLMClient` parses them into
`SkillCall`s → the Orchestrator dispatches each skill to the GR00T VLA in IsaacLab.

The VLM is served as an **OpenAI-compatible `/v1/chat/completions`** endpoint (separate process,
in `Isaac-GR00T-VLM`). AgentBot only needs `vlm.base_url` to reach it.

---

## Configure (`config/agentbot.yaml`, copy from `config/agentbot.example.yaml`)
```yaml
vlm:
  backend: gr00t-vlm                       # use the fine-tuned VLM (else: qwen-vl placeholder)
  model:   gr00t-vlm
  base_url: http://localhost:8000/v1       # where the VLM server listens
vla:
  default_checkpoint: <a key in vla.checkpoints>   # the GR00T action checkpoint
```
> If the VLM endpoint is unreachable, `Gr00tVLMClient` falls back to a keyword **stub** so the
> dashboard still works — check the URL if plans look wrong.

---

## A) No-GPU demo (one process, fake VLA) — verify the loop fast
```yaml
# config/agentbot.yaml
backbone: in-proc
```
```bash
cd agentbot && uv sync
uv run uvicorn agentbot.api.app:app --port 8780   # open http://localhost:8780
```
Type **"sort the can onto the orange plate"** in the dashboard Chat → see the structured plan →
a fake VLA "runs" it → command reaches `done`. (With `vlm.backend: gr00t-vlm` + a running VLM
server, the plan comes from the real VLM; otherwise the keyword stub.)

## B) Full GPU flow (4 processes) — real IsaacLab episode (run on the 4090)
Set `backbone: redis`. Then, in 4 terminals (see `DELIVERY_4090.md` §6 for the env build):
```bash
# 1  redis-server
# 2  VLM Brain server  (Isaac-GR00T-VLM, port 8000)   — DELIVERY_4090.md §5 step 4
# 3  GR00T policy server (port 5555)
cd Isaac-GR00T_n1d7 && python -m gr00t.eval.run_gr00t_server --model-path <ckpt> --embodiment-tag new_embodiment --port 5555
# 4  persistent IsaacLab session
conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES \
  && cd IsaacLab && python -m agentbot.vla.sim_session --headless
# Core + dashboard + orchestrator
cd agentbot && uv run uvicorn agentbot.api.app:app --port 8780
```
Dashboard → command → Brain(VLM) plans `sort_can(orange)` → Orchestrator dispatches → IsaacLab
runs one episode → Monitor confirms (`task_done`) → `done`.

---

## HTTP API (the dashboard uses these)
| Method | Path | Use |
|---|---|---|
| POST | `/v1/commands` `{"text": "..."}` | enqueue a command (full plan→execute) |
| POST | `/v1/chat` `{"text":"...","image_path":"<frame.png\|data-uri>"}` | one-shot plan preview (no exec) |
| POST | `/v1/skills/{name}/invoke` | dispatch a single skill |
| GET | `/v1/state` · WS `/v1/events` | live snapshot · event stream |
| GET | `/v1/skills` · `/v1/vla/checkpoints` · `/v1/stats` | registered skills · checkpoint registry · per-skill stats |
| POST | `/v1/control/stop` · `/v1/control/estop` | stop orchestrator · trip safety |

**Vision:** `/v1/chat` carries `image_path`; the orchestrator flow reads the live frame from the
Monitor `state["camera"]["frame"]` (publish it from `sim_session` — see `DELIVERY_4090.md` §2-C).

---

## Troubleshooting
- **Plans look like keyword guesses** → the VLM endpoint is down/misconfigured; `Gr00tVLMClient`
  fell back to the stub. Check `vlm.base_url` and that the VLM server's `/health` is `ok`.
- **`OMNI_KIT_ACCEPT_EULA`** must be `YES` for headless isaacsim/isaaclab.
- **`egl_probe` build failed** during `isaaclab.sh --install` → optional (robomimic EGL detection);
  isaaclab runs without it. To install: `sudo apt install libegl1-mesa-dev` + `conda install "cmake<4"`.
- **Wrong torch / CUDA** → ensure `torch 2.7.0+cu128` (not cu130) in `env_isaaclab` (see `DELIVERY_4090.md` §8 #1).
- **Switch the action model** → edit `vla.checkpoints` / `default_checkpoint` in `agentbot.yaml` (never hardcoded).
