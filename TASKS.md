# AgentBot · 任務追蹤（TASKS）

> 這份檔案記錄整個 AgentBot 的**規劃、目前進度、剩餘任務**。每完成一項就更新勾選 `[ ]→[x]`。
> 完整設計見 [`docs/agentbot_architecture.html`](../docs/agentbot_architecture.html)；總體計畫見 `~/.claude/plans/architecture-png-robot-cozy-anchor.md`。
> **最後更新：2026-06-18**

---

## 名詞解釋（先看這個）

- **Phase（階段）**：里程碑。Phase 0＝骨架、Phase 1＝MVP、Phase 2＝加廣、Phase 3＝實機。
- **scaffold（骨架）**：所有結構／介面／測試都在，但**關鍵執行用「假的」替身**（`FakeRunner`）頂著，所以不需要 GPU 也能跑通整條流程示範。
- **MVP（Minimum Viable Product，最小可行產品）**：能**端到端真正跑通的最小一條「垂直線」**。在這裡＝
  「一句話指令 → Brain → 一個技能（如 sort_can）→ VLA **真的**在 IsaacLab 跑一集 → 成功/失敗結果回傳」。
  「MVP slice（切片）」就是這條穿過全部分層、但範圍最小、真的會動的線。它和骨架的差別只有一個：把 `FakeRunner` 換成**真的** `run_rollout`。

---

## 目前狀態總覽

| Phase | 內容 | 狀態 |
|---|---|---|
| **0** | 骨架 + 架構報告（contracts／各層服務／event bus／API／Console／測試） | ✅ **完成**（23 測試綠燈） |
| **0.5** | 動畫流程圖重繪、改用 uv、TASKS 檔、README | ✅ **完成** |
| **0.6** | submodule（squirel2000/agentbot）+ README 多終端機澄清 | ✅ **完成** |
| **1** | **MVP** — 真 IsaacLab rollout（Option B：驅動 gr00t_infer_agent 跑單集）+ 真 LLM | 🔄 **進行中**（已實作，4090 驗證中） |
| **2** | 加廣 — 多技能、多步規劃、記憶 RAG、**階層式 Dashboard（含控制）** | ◻ 未開始 |
| **3** | 實機 — `backends/hardware` → ROS2 → OpenArm（架構圖 6·7） | ◻ 未開始 |

---

## 你的下一步（重點看這段）

1. **先把 Phase 0 跑一遍、熟悉它**（不需要 GPU）：
   ```bash
   cd agentbot && uv sync && uv run pytest -q          # 23 passed
   uv run uvicorn agentbot.api.app:app --port 8780     # 開 http://localhost:8780，在 Chat 輸入「sort the can onto orange」
   ```
   你會看到：指令 → 結構化 plan → 派工到（假的）VLA → Console 即時顯示遙測。這就是整條 1·2·4·5+3。

2. **要不要提供 Phase 1 的資料給我？** —— 大部分**不用**，因為 `gr00t_infer_agent.py`、`joint_mapper.py`、eval 設定都在 repo 裡，**我可以自己讀**來移植。真正只有「在 GPU 上跑得起來」這件事我看不到，所以分工是：
   - **我做**：讀 repo，把單集 rollout 移植進 `agentbot/vla/isaac_runner.py:run_rollout`（目前是 `NotImplementedError`）。
   - **你做**：在有 GPU 的機器（env_isaaclab）實際跑一次，把 **log／錯誤**貼回來，我再修。
   - **請你順手確認的 4 個 runtime 事實**（只有實機/實跑才知道）：
     - [ ] `pour_water` 在 IsaacLab 的實際 gym id（`sort_can` 已知為 `Isaac-Can-Sorting-OpenArm-DexHand-v0`）
     - [ ] 多工（顏色）觀測的實際 key（目前 eval 用 `obs["scene_obs"]["target_object_color"]`，0=orange / 1=green，確認仍正確）
     - [ ] N1.7 client 連線是否需要 `client_pythonpath`（見 `scripts/eval/configs/gr00t_n17_openarm_o6.json`）
     - [ ] 是否已在 `env_isaaclab` 裡 `pip install -e agentbot`（worker 要 import agentbot）

3. 給我綠燈後，我就照下面 **Phase 1** 的清單開工。

---

## Phase 0 · 骨架 + 報告 ✅ 完成

- [x] `agentbot/` 新頂層套件（與 Isaac-GR00T / IsaacLab / scripts 並列）
- [x] **contracts**（協定）：`UserMessage · SkillPlan · SkillCall · VlaTaskRequest · VlaTaskResult · Event`（Pydantic）
- [x] **monitor**（block 3）：`event_bus`（in-proc + Redis）· `state_store` · `job_queue` · `safety` · `ingest`
- [x] **skills**（block 4）：`base · registry · builtin/{sort_can, pour_water}`
- [x] **brain**（block 2）：`gateway · agent · llm_client`（可插拔，預設 Qwen3-VL）· `memory/{store,conversation,episodic,semantic}`
- [x] **vla**（block 5）：`worker · isaac_runner（FakeRunner 可跑、真 run_rollout 待 Phase 1）· policy_server · backends/{sim,hardware}`
- [x] **api**：`app · deps · routes_{chat,skills,vla,events}`（串起 1·2·4·5）
- [x] **ui**：`console.html`（Monitor / Chat / History / Memory 四分頁，唯讀）
- [x] 23 個單元測試綠燈
- [x] 可替換 checkpoint 名冊（config，不寫死；預設 `n17_150k_lr1e4_absolute`，未來 `n17_150k_lr5e5` 已登錄）
- [x] 架構報告 `docs/agentbot_architecture.html`

## Phase 0.5 · 本輪追加 ✅ 完成

- [x] **重繪「執行期拓樸」流程圖**：雙軌正交佈線（任務軌／事件軌互不交叉）、**流動虛線動畫**（沿箭頭方向）、文字不出框；獨立檔 `docs/agentbot_runtime_topology.svg`
- [x] **重新產生「系統地圖」**（取代低解析 `architecture.png`）：高解析、動畫、**資料流已校正**、整合圖例與註解；獨立檔 `docs/agentbot_system_map.svg`
- [x] 報告內嵌上述兩張動畫圖（`<img src=*.svg>` 會動畫），並提供「開啟獨立大圖」連結
- [x] **改用 uv** 架設環境：`pyproject` 改 PEP 735 dependency-groups、`uv sync` 產生 `uv.lock`、`uv run pytest` 綠燈
- [x] README 改寫為 uv 用法
- [x] 本 TASKS 追蹤檔

## Phase 1 · MVP 🔄 進行中（核心已完成並在 4090 驗證）

**已完成並驗證 ✅**
- [x] 在 `env_isaaclab` 裡 `pip install -e agentbot --no-deps`（+redis），worker 可 import
- [x] `agentbot/vla/isaac_runner.py:run_rollout`（**Option B**：以子程序驅動 proven 的 `gr00t_infer_agent.py` 跑單集，
      複製 run_eval 的啟動方式：cwd=IsaacLab、`client_pythonpath` 上 PYTHONPATH、worker 自身的 env_isaaclab python）；
      串流解析遙測，並以 `run_manifest.json` 為**權威結果**來源
- [x] `policy_server.py` 強化（`exec`+new session+`killpg`，乾淨關閉不留孤兒）
- [x] **4090 實跑驗證**：sort_can 單集，GR00T N1.7 server 開機+IsaacLab 一集，93s 完成；
      `success_rate=1.0`、遙測延遲 ~0.06–0.14s；server 乾淨關閉、GPU 釋放。
      （過程抓到並修正一個 bug：原本用 stdout 判斷成敗會誤報，改讀 manifest 後正確回報 SUCCEEDED。）
- [x] `brain/llm_client.py`：新增 `gr00t-vlm` 後端 seam（你的抽離 VLM 之後改 config 即可換）

**剩餘 Phase 1 收尾（較低風險）**
- [ ] `brain/llm_client.py`：把 stub 換成**真的** Qwen3-VL tool-calling（需先有 Qwen OpenAI 相容 server 才能測）
- [ ] `brain/agent.py`：plan 產生後**自動派工**到 VLA（目前是 Console 按「▶ 執行」或 `POST /v1/skills/{name}/invoke` 顯式派工）
- [ ] 多集成功率：跑 N 集得統計（單集無法代表成功率；可挑表現較好的 checkpoint，如 `n17_150k_lr1e4_absolute`）

**驗證指令（可重現）**：`env_isaaclab` 下 `python /tmp/verify_phase1.py`，或正式三終端機流程（見 README）。

## Phase 2 · 加廣（長任務編排 + Dashboard + RAG）📋 已詳細規劃

詳細計畫：**[`docs/plans/2026-06-19-phase2-orchestration-core.md`](docs/plans/2026-06-19-phase2-orchestration-core.md)**。
已確認決策：執行順序＝**先編排核心**；Dashboard＝**沿用 vanilla**；失敗＝**重試 N 次→大腦重規劃/中止**。

- [ ] **2a · 編排核心（先做）**：命令佇列（submit 後清空輸入、FIFO、可清除）→ 大腦把命令拆成有序 atomic skills →
      **常駐 IsaacLab session**（`vla/sim_session.py`，App 全程開著）逐一執行 → Monitor 確認（`task_done` + 小 VLM judge hook，回答你的 #3）→
      成功推進 / 失敗重試 N 次→重規劃或中止 → 全程紀錄 + 統計（任務管理）。`llm`→`vlm` 改名也在此（你的 #1）。
- [ ] **2b · 階層式 Dashboard + 控制**（vanilla 擴充 Console）：可展開 sidebar/navigator、下鑽（命令→計畫→技能→單集）、控制鈕（停止/清除/E-stop/選 checkpoint/手動觸發）。
- [ ] **2c · 記憶 RAG**：把過去 episode（skill/參數/結果/失敗原因）+ lessons 入向量庫，規劃時檢索回注，讓大腦參考歷史。
- [ ] 配套：多新增幾個 atomic skill（pick/place/move…）讓多步計畫更有料；真實安全事件來源。

## Phase 3 · 實機（架構圖 6·7）◻

- [ ] `vla/backends/hardware.py` → 橋接 `Isaac-GR00T/scripts/sim2real/gr00t_control_robot.py` + `openarm_ros2`（ROS2）
- [ ] 把 ROS2 `/joint_states`、`/vla/diagnostics`、安全訊號 republish 進同一條 event bus（可加 MQTT↔ROS2 橋）
- [ ] sim / real 一致性驗證；`embodiment=hardware` 切換

---

## 變更紀錄

- 2026-06-17 Phase 0 完成（骨架 + 報告，23 測試）。
- 2026-06-18 Phase 0.5 完成（動畫圖重繪、系統地圖重生、改用 uv、TASKS 檔、README）。
