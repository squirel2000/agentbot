# AgentBot 4090 Sim-Loop Bring-up Implementation Plan（繁體中文）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development（建議）或 superpowers:executing-plans，逐 task 執行。步驟用 `- [ ]` 追蹤。
>
> **COMMITS：** 維護者擁有所有 commit。每個 task 測試通過後**只 stage、停下來等維護者 commit**；不要自行 `git commit` / `git push`。

**目標（Goal）：** 在單台 **RTX 4090** 上把 `agentbot` 完整跑起來——以**產物 A**（`Cosmos-Reason2-2B-lora-merged-6phase`，或更可靠的 `toolcall-merged`）當 Brain，VLA 跑在 `env_isaaclab` 的 IsaacLab 模擬環境，從**儀表板 UI** 下一句自然語言指令，讓 `sort_can` 真的在 sim 跑一集並到達 `done`。ROS2／實機為**後續階段**（先在 sim 驗證成功再切換）。

**架構（Architecture）：** agentbot 已大致建好（Phase 0 骨架 + Phase 1 程式已實作）。本計畫＝**4090 bring-up + 一個程式缺口補完 + 端到端 sim 驗證**，非從零開發。四程序：① Redis（跨程序匯流排）② VLM Brain server（:8000，Isaac-GR00T-VLM）③ GR00T policy server（:5555，Isaac-GR00T_n1d7／uv）④ 常駐 IsaacLab `sim_session`（env_isaaclab／conda）＋ core/dashboard（:8780）。資料流：UI 指令 → Brain(VLM) 規劃 `tool_calls` → `Gr00tVLMClient` → `SkillCall` → Orchestrator 派工到 Redis job queue → `sim_session` BRPOP 執行一集 → Monitor 確認 → `done`。

**技術棧：** FastAPI/uvicorn（:8780）、Redis、httpx、IsaacLab/IsaacSim 5.1.0（conda `env_isaaclab`、py3.11、torch 2.7.0+cu128）、GR00T N1.7（`Isaac-GR00T_n1d7/.venv`、uv）、PIL（縮圖 JPEG）。

## 全域限制（Global Constraints）

- **不要 `git commit`/`push`**；只 stage。
- **Brain 模型**：依使用者指示用**產物 A**＝`artifacts/checkpoints/gr00t/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase`；DELIVERY 建議改用 `lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged`（tool-call 格式更可靠）——兩者皆可，Task 3 以產物 A 為主、toolcall 為 fallback。
- **GR00T 動作 checkpoint**：`agentbot.yaml` 的 `vla.default_checkpoint`（目前指向 `N1_7_fft_0614_150k_lr5e5_no_tune_visual/checkpoint-150000`）。
- **任務 gym id（已知）**：`Isaac-Can-Sorting-OpenArm-DexHand-v0`；顏色 `target_color ∈ {orange, green}`。
- **EULA**：任何 headless isaacsim/isaaclab 都要 `export OMNI_KIT_ACCEPT_EULA=YES`。
- **torch**：`env_isaaclab` 必須是 `torch 2.7.0+cu128`（勿讓 cu130 混入，見 DELIVERY §8 #1）。
- **VRAM 風險**：VLM(~5G)+GR00T policy(~12G)+IsaacSim(~5-7G) 共駐 24G **很緊**；Task 3 須實測並備好降載方案。
- 受限模型 `nvidia/Cosmos-Reason2-2B` 須在 HF cache（`HF_HUB_OFFLINE=1`）。

## 背景：目前狀態（已驗證自 DELIVERY_4090.md §10 + 程式碼）

- ✅ 已完成：VLM serve + tool-calling、`Gr00tVLMClient`（已 e2e 驗證 → 真實 `SkillCall`）、Brain 端視覺 plumbing（`brain/agent.py` 的 `frame_provider`/`build_user_content`）、`monitor/ingest.py` 處理 `CAMERA` 事件 → `state["camera"]`、`isaac_runner.run_rollout`/`sim_session`（Phase 1 程式）、UI `ui/console.html`、46 測試綠燈、env_isaaclab。
- ▶ 未完成（本計畫處理）：① `sim_session` **未發布** `CAMERA` 事件（讀相機只餵 policy，未給 Brain 看）② 4 程序迴路**尚未在 4090 實跑驗證** ③ 確認一集到達 `done`。

---

### Task 1：sim_session 發布 CAMERA 事件（Brain 看得到 sim 相機）

**Files:**
- Modify: `agentbot/agentbot/vla/sim_session.py`（加 `_frame_to_datauri` 與 run_skill 內的 CAMERA publish）
- Read first: `agentbot/agentbot/contracts/events.py`（確認 `EventType.CAMERA` 與 `Event` 建構子簽名）
- Test: `agentbot/tests/test_sim_camera_datauri.py`（CPU-only，測 encode helper）

**Interfaces:**
- Produces: `_frame_to_datauri(img: np.ndarray, max_side: int = 320, quality: int = 70) -> str`，回傳 `"data:image/jpeg;base64,…"`。

- [ ] **Step 1：先讀 events.py 確認介面**

Run: `grep -nE "class Event|EventType|CAMERA|def __init__|payload" agentbot/agentbot/contracts/events.py`
Expected: 看到 `EventType.CAMERA` 與 `Event(type=…, payload=…, …)` 的建構方式（mirror `worker._status_event` 的用法）。

- [ ] **Step 2：寫 failing test（encode helper）**

```python
# agentbot/tests/test_sim_camera_datauri.py
import numpy as np

def test_frame_to_datauri_shrinks_and_encodes():
    # import the pure helper without importing IsaacLab (it lives in sim_session, which
    # imports isaaclab at module top — so the helper must be importable in isolation).
    from agentbot.vla.frame_encode import frame_to_datauri
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    uri = frame_to_datauri(img, max_side=320)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 100
```

- [ ] **Step 3：跑測試確認 fail**

Run: `cd agentbot && uv run pytest tests/test_sim_camera_datauri.py -v`
Expected: FAIL（`ModuleNotFoundError: agentbot.vla.frame_encode`）。

- [ ] **Step 4：建 `agentbot/agentbot/vla/frame_encode.py`（純函式、不 import isaaclab）**

```python
"""Pure helper: numpy RGB frame -> downsized JPEG data-uri (no IsaacLab import)."""
import base64
import io

import numpy as np


def frame_to_datauri(img: np.ndarray, max_side: int = 320, quality: int = 70) -> str:
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
```

- [ ] **Step 5：跑測試確認 pass**

Run: `cd agentbot && uv run pytest tests/test_sim_camera_datauri.py -v`
Expected: PASS。

- [ ] **Step 6：在 sim_session.run_skill 發布 CAMERA 事件**

在 `sim_session.py` 頂部 import 區加：
```python
from agentbot.contracts.events import Event, EventType
from agentbot.vla.frame_encode import frame_to_datauri
```
在 run_skill 迴圈中、`cam_imgs` 算出後（約現行 line 142 之後），每 15 步發布一次 head 相機縮圖（用既有的 `publish`）：
```python
            if "head" in cam_imgs and steps % 15 == 0:
                publish(Event(type=EventType.CAMERA,
                              payload={"frame": frame_to_datauri(cam_imgs["head"])}))
```
（若 events.py 的 `Event` 需要其他必填欄位，比照 `worker._status_event` 的建構方式補上。）

- [ ] **Step 7：全套 CPU 測試不回歸**

Run: `cd agentbot && uv run pytest -q -k "not gpu"`
Expected: 既有測試仍綠（46+1 passed），輸出乾淨。

- [ ] **Step 8：Stage**

```bash
cd agentbot && git add agentbot/agentbot/vla/frame_encode.py agentbot/agentbot/vla/sim_session.py agentbot/tests/test_sim_camera_datauri.py
# 建議訊息：feat(vla): publish CAMERA event from sim_session so the Brain sees the live frame
```

---

### Task 2：env_isaaclab 就緒檢查（4 個 runtime 事實 + 安裝 agentbot）

**Files:** 無程式改動；只驗證環境。

- [ ] **Step 1：確認 conda env 與 torch cu128**

```bash
conda activate env_isaaclab
python -c "import torch;print('cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
```
Expected: `cuda 12.8 avail True`（若是 cu130 → 依 DELIVERY §8 #1 重釘 `torch==2.7.0+cu128`）。

- [ ] **Step 2：把 agentbot 裝進 env_isaaclab（worker 要 import）**

```bash
conda activate env_isaaclab && pip install -e /home/asus/Gits/IsaacLab-GR00T/agentbot
python -c "import agentbot, redis; print('agentbot+redis ok')"
```
Expected: `agentbot+redis ok`。

- [ ] **Step 3：確認 N1.7 client_pythonpath**

```bash
python -c "import json;print(json.load(open('/home/asus/Gits/IsaacLab-GR00T/scripts/eval/configs/gr00t_n17_openarm_o6.json'))['client_pythonpath'])"
```
Expected: `Isaac-GR00T_n1d7`（`sim_session._make_client` 會把它加到 sys.path）。

- [ ] **Step 4：確認顏色觀測 key 仍有效（多工 sort_can）**

記錄 TASKS.md 的待確認事實：`sort_can` gym id = `Isaac-Can-Sorting-OpenArm-DexHand-v0`、顏色經 `params.target_color` 傳入（0=orange/1=green）。此步在 Task 4 的 selftest 一併驗證（不需另跑）。

---

### Task 3：啟動四程序迴路（執行）

> **VRAM 提醒**：三個模型共駐 24G 很緊。先用 `nvidia-smi` 盯著；若 OOM，降載順序：① Brain 模型用較小（或 toolcall-merged 同樣 4G）② 確認沒有殘留行程佔 GPU（`nvidia-smi`）③ 必要時把 Brain 移到 CPU 推論或另一張卡。

- [ ] **Step 1：Redis**

```bash
redis-server --save '' --appendonly no &   # 或系統服務
redis-cli ping   # PONG
```

- [ ] **Step 2：VLM Brain server（產物 A，:8000）**

```bash
cd /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T-VLM
GR=/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t
VLM_MODEL_DIR="$GR/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged-6phase" \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 bash examples/run_vlm_server.sh   # :8000（背景）
# 就緒檢查：
curl -s localhost:8000/health   # {"status":"ok","model_loaded":true}
```

- [ ] **Step 3：GR00T policy server（:5555）**

```bash
cd /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T_n1d7
.venv/bin/python -m gr00t.eval.run_gr00t_server \
  --model-path /home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/gr00t/N1_7_fft_0614_150k_lr5e5_no_tune_visual/checkpoint-150000 \
  --embodiment-tag new_embodiment --port 5555   # 等待 "Server is ready and listening"
```

- [ ] **Step 4：設定 agentbot.yaml（已大致設好，確認）**

確認 `agentbot/config/agentbot.yaml`：`backbone: redis`、`vlm.backend: gr00t-vlm`、`vlm.base_url: http://localhost:8000/v1`、`vla.default_checkpoint` 指向上面同一個 GR00T ckpt。

- [ ] **Step 5：常駐 IsaacLab sim_session（env_isaaclab）**

```bash
conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
cd /home/asus/Gits/IsaacLab-GR00T/IsaacLab
python -m agentbot.vla.sim_session --headless   # 等 "[sim_session] ready; ... polling agentbot:vla:tasks"
```

- [ ] **Step 6：core + dashboard（:8780）**

```bash
cd /home/asus/Gits/IsaacLab-GR00T/agentbot
uv run uvicorn agentbot.api.app:app --port 8780   # 開 http://localhost:8780
```

- [ ] **Step 7：四程序就緒檢查**

```bash
redis-cli ping                       # PONG
curl -s localhost:8000/health        # model_loaded:true
curl -s localhost:8780/v1/state | head -c 300   # core 有回應
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader   # 確認沒 OOM
```
Expected: 四者皆健康、VRAM 未爆。

---

### Task 4：端到端 sim 驗證（UI 指令 → done）

- [ ] **Step 1（先離線驗證 sim_session 本身）：selftest（不經 Redis/Brain）**

```bash
conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
cd /home/asus/Gits/IsaacLab-GR00T/IsaacLab
python -m agentbot.vla.sim_session --selftest   # 跑兩集 can-sorting（orange/green）
```
Expected: 兩集都印出 `-> SUCCEEDED/FAILED ... steps ...`、app 跨兩集不關。確認 sort_can 在此環境能動（顏色 key 正確）。

- [ ] **Step 2：經 API 下單一指令（full plan→execute）**

```bash
curl -s -X POST localhost:8780/v1/commands -H 'Content-Type: application/json' \
  -d '{"text":"sort the can onto the orange plate"}'
# 觀察事件流：
curl -s localhost:8780/v1/state | python3 -m json.tool | grep -iE "status|skill|done|success"
```
Expected: Brain 規劃出 `sort_can(target_color=orange)` → orchestrator 派工 → sim_session 跑一集 → 指令狀態到 `done`、`success` 為真。

- [ ] **Step 3：從 UI 驗證**

開 `http://localhost:8780`，Chat 輸入「sort the can onto the orange plate」→ 看到結構化 plan → VLA 在 IsaacLab 跑 → Console 即時遙測 → 指令 `done`。

- [ ] **Step 4：確認 Brain 真的有看到相機（非 stub）**

`curl -s localhost:8780/v1/state | grep -i camera` 應看到 `state["camera"]["frame"]` 有 data-uri（Task 1 生效）；且 Brain 的 plan 非關鍵字 stub（plan 來自真實 VLM）。

- [ ] **Step 5：記錄結果**

把成功率／步數／一張 dashboard 截圖記到 `agentbot/docs/USING_VLM_BRAIN.md` 或本計畫附註；更新 `TASKS.md` 把 Phase 1 勾為完成。

---

### Task 5（後續階段，本計畫不執行）：切換 ROS2／實機

sim 全綠後，Phase 3：`agentbot/vla/backends/hardware.py` → ROS2 → OpenArm（架構圖 6·7）。同一個 `CAMERA` 事件接縫由 ROS2→event-bus 橋接器發布，Brain 不用改。**先確保 sim 端 Task 1–4 全部成功再進此階段。**

---

## 自我檢視（Self-review）

- **覆蓋**：使用者目標＝agentbot 完整實現（VLM=產物 A）→ Task 3-4；VLA launch 在 env_isaaclab → Task 3 Step 5；UI 下指令 → Task 4 Step 3；sim 測試 → Task 4；ROS2 後續 → Task 5。程式缺口（相機）→ Task 1。
- **風險**：VRAM 共駐（Task 3 提醒 + 降載方案）；IsaacSim bring-up 的已知坑（DELIVERY §8：torch cu130、port 殺法、EGL、EULA）。
- **介面一致**：`frame_to_datauri`、`EventType.CAMERA`、`state["camera"]["frame"]` 在 Task 1↔4 一致；GR00T ckpt 在 Task 3 Step 3/4 一致。
