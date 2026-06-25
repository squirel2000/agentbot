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

## C) 4090 windowed 啟動（建議：看得到 sim + 穩定大腦）

兩個重點：① **大腦用 `toolcall-merged`**（穩定吐單一 `sort_can`，不會過度拆出沒有 sim task 的 `pick`）；② **sim_session 用 windowed**（看得到畫面、且較不會 idle 自關）。orchestrator 也已會**自動跳過** skeleton skill（`pick`/`place`/`home`，`sim_runnable=False`），所以即使大腦偶爾吐 `pick` 也不會卡在 30 分鐘 timeout。

**先全部關閉**（安全的 `[x]` grep，不會誤殺自己的 shell）：
```bash
for p in 8780 8000 5555; do lsof -ti:$p | xargs -r kill; done
ps -eo pid,args | grep "[a]gentbot.vla.sim_session" | awk '{print $1}' | xargs -r kill
ps -eo pid,args | grep "[r]un_sim_session.sh"        | awk '{print $1}' | xargs -r kill
```

**依序啟動（5 個程序，各開一個終端機）**：
```bash
GR=/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t

# 0) redis
redis-cli ping || redis-server --daemonize yes

# 1) VLM Brain —— toolcall-merged，:8000
cd /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T-VLM
VLM_MODEL_DIR="$GR/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged" \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 bash examples/run_vlm_server.sh

# 2) GR00T policy server，:5555
cd /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T_n1d7
.venv/bin/python -m gr00t.eval.run_gr00t_server \
  --model-path "$GR/N1_7_fft_0614_150k_lr5e5_no_tune_visual/checkpoint-150000" \
  --embodiment-tag new_embodiment --port 5555

# 3) sim_session —— WINDOWED（不要加 --headless；視窗會開在 4090 螢幕）
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
cd /home/asus/Gits/IsaacLab-GR00T/IsaacLab
python -m agentbot.vla.sim_session          # 等 "[sim_session] ready; ... polling"

# 4) dashboard，:8780（0.0.0.0 → 遠端可開）
cd /home/asus/Gits/IsaacLab-GR00T/agentbot
uv run uvicorn agentbot.api.app:app --host 0.0.0.0 --port 8780
```
開 `http://<4090-ip>:8780`（或本機 `localhost:8780`），輸入 `sort the can onto the orange plate` → 規劃 `sort_can(orange)` → IsaacLab 視窗跑一集 → `done`。

> headless、無人值守時改用 `agentbot/run_sim_session.sh`（supervisor，會在 sim_session 因 idle 自關後自動重啟；指令在 redis 佇列緩衝，不會遺失）。但 headless 約 1 分鐘 idle 自關，**windowed 通常較穩、又看得到**。

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

## Dashboard controls & recovery
- **↺ Reset env** (`POST /v1/control/reset`) → asks `sim_session` to re-randomize the scene
  (target plate) and publish a fresh head frame + the new target color to the Monitor. Use it for
  a clean scene before a run. The top-bar **`🎯 target: <color> plate`** pill shows the live target.
- **✕ Clear queue** (`DELETE /v1/commands`) → drops pending *commands* only (lightest action).
- **■ Stop** (`POST /v1/control/stop`) → **recoverable**: cancels the in-flight command, drains BOTH
  the command queue and the VLA-task queue (`agentbot:vla:tasks`), and clears a SAFETY-blocked
  `robot_state`. It does **not** kill the orchestrator loop, so you can enqueue again immediately.
  (An already-running sim episode finishes — ~30 s — before the cancel takes effect; aborting
  mid-episode is a follow-up.)
- **⛔ E-STOP** trips the `SafetyWatchdog` (`robot_state → blocked`); press **■ Stop** or **↺ Reset env**
  to clear the blocked state and resume.
- **Type just `sort can`** → the Brain fills the plate color from the can on the table (the env's
  randomized target, published to `state['environment']`). An explicit "…green/orange plate" always wins.

## Troubleshooting
- **A command does nothing / no action after Enqueue** → the orchestrator loop was halted by an
  earlier ■ Stop on the *old* build (it permanently exited `run()`). Fixed: ■ Stop is now a
  recoverable soft-cancel and the loop survives crashes. If a stack is still on the old code,
  restart the dashboard process (`uvicorn agentbot.api.app:app`); the GR00T/VLM servers can stay up.
- **The robot repeats the action and "skill runs" is always `failed` even though it completes** →
  color mismatch, not a Monitor bug. The Can-Sorting env randomizes the target plate each reset and
  `task_done` judges success against *that* basket; `sim_session` now syncs the policy instruction
  to the env's actual target (`obs['scene_obs']['target_object_color']`, {0:orange,1:green}).
  **Limitation:** the *commanded* color is overridden by the env's random one — honoring the exact
  commanded color requires setting the env target on reset (a follow-up).
- **Plans look like keyword guesses** → the VLM endpoint is down/misconfigured; `Gr00tVLMClient`
  fell back to the stub. Check `vlm.base_url` and that the VLM server's `/health` is `ok`.
- **`OMNI_KIT_ACCEPT_EULA`** must be `YES` for headless isaacsim/isaaclab.
- **`egl_probe` build failed** during `isaaclab.sh --install` → optional (robomimic EGL detection);
  isaaclab runs without it. To install: `sudo apt install libegl1-mesa-dev` + `conda install "cmake<4"`.
- **Wrong torch / CUDA** → ensure `torch 2.7.0+cu128` (not cu130) in `env_isaaclab` (see `DELIVERY_4090.md` §8 #1).
- **Switch the action model** → edit `vla.checkpoints` / `default_checkpoint` in `agentbot.yaml` (never hardcoded).
