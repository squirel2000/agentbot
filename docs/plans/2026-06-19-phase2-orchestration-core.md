# AgentBot Phase 2 — 編排核心（Orchestration Core）實作計畫

> **給執行的 agent：** 必用子技能：superpowers:subagent-driven-development（建議）或 superpowers:executing-plans，逐任務執行。步驟用 `- [ ]` 勾選框追蹤。
>
> **Commit 由使用者擁有。** 不要執行 `git commit`。每個任務結束於一個「可審查的 checkpoint」（未 commit），由使用者 review 後自行 commit（在 `agentbot` submodule 內 commit，再更新父 repo 的 gitlink）。見記憶 `commits-are-user-owned`。

**目標：** 把 AgentBot 從「單一技能、隨點隨跑」升級成**長任務編排器**：UI 命令進佇列 → 大腦把命令**拆解成有序的 atomic skills** → 一個**常駐（全程開著）的 IsaacLab session** 逐一執行 —— Monitor 確認每個 skill 完成、大腦才推進（失敗則重試/重規劃）—— 全程紀錄，供任務管理檢視。

**架構：** Redis 上的**兩層佇列**。**命令佇列**（`agentbot:commands`，FIFO）放自然語言目標；UI 輸入框在送出後清空、使用者可清除佇列。**大腦編排器**（常駐 async loop）取出最舊命令，用 LLM 拆解成有序 `SkillCall` 的 `SkillPlan`，**一次一個**派到**任務佇列**（`agentbot:vla:tasks`）。一個**常駐 `sim_session`**（env_isaaclab，只啟動一次）消費任務，在活著的 IsaacLab env 內把每個 skill 跑到 `task_done()`/逾時，發布遙測 + 終局結果。編排器等每個結果（Monitor 把關），失敗的 skill 重試最多 N 次（反應式策略重跑 ≈ 從當前狀態繼續），仍失敗則重規劃或中止。紀錄落地 SQLite 供統計/下鑽。閒置 → 等待。

**技術棧：** 沿用既有 AgentBot 棧（Pydantic contracts、Redis backbone、FastAPI、vanilla Console）+ 一個常駐 IsaacLab session（把 `scripts/eval/gr00t_infer_agent.py` 的 `while simulation_app.is_running()` 外層迴圈，重構成命令驅動的迴圈）+ SQLite 紀錄。

---

## 背景（Context）

**為什麼。** Phase 1 已證明「一個 skill → 一集真實 IsaacLab」（在 4090 驗證）。Phase 2 讓它變成真正的 agent：**常駐模擬**（IsaacLab 全程開著，直到關 UI / 按停止）、**多步計畫**（一個命令變多個 atomic skill 依序執行、Monitor 把關）、**非同步命令佇列**（送出即清空、FIFO、可清除）、**紀錄/統計**（任務管理系統）、**階層式 Dashboard + 控制**、**記憶 RAG**。

**本回合已確認的決策：**
- **執行順序：** 先做編排核心（本計畫＝ **Phase 2a**）；**階層式 Dashboard（2b）**與 **RAG（2c）**為後續里程碑（本文 outline，屆時各出獨立計畫）。
- **Dashboard 前端：** vanilla JS，擴充既有 `ui/console.html`（不引入建置鏈）。
- **失敗處理：** 同一 skill 從當前狀態重試最多 **N** 次（反應式策略重跑＝從停下處繼續）；仍失敗則大腦重規劃或中止整個長任務。

**沿用、不重寫。** 每步 obs→action→step 迴圈、`JointMapper`、`filter`、`episode_data_saver`、`task_done()`、GR00T policy client 全來自 `scripts/eval/*`（已驗證的棧），與 Phase 1 一樣。常駐 session 只重構 `gr00t_infer_agent.py` 的外層迴圈。

**Phase 1 既有（可用基礎）。** contracts（`UserMessage·SkillPlan·SkillCall·VlaTaskRequest·VlaTaskResult·Event`）；Redis/in-proc 的 `event_bus·state_store·job_queue·safety·ingest`；skills（`sort_can`,`pour_water`）；brain（`gateway·agent·llm_client` 可插拔、`memory/`）；vla（`worker·isaac_runner` Option-B 子程序·`policy_server`）；FastAPI API + 唯讀 Console。Phase-1 的每集 `isaac_runner.run_rollout` 保留供一次性執行；Phase 2 新增編排器使用的**常駐**路徑。

---

## Phase 2 路線圖（本文詳述 2a；2b/2c 於文末 outline）

| 里程碑 | 範圍 | 本計畫 |
|---|---|---|
| **2a · 編排核心** | 命令佇列 · 大腦拆解 · 常駐模擬 · Monitor 把關的循序技能 · 重試/重規劃 · 紀錄/統計 · 最小 Console 串接 | **以下完整拆解** |
| **2b · 階層式 Dashboard + 控制** | 可展開 sidebar/navigator · 下鑽（命令→計畫→技能→單集）· 控制（暫停/中止/清除/E-stop/checkpoint/手動觸發） | outline |
| **2c · 記憶 RAG** | 規劃時向量檢索過去 episode/lessons | outline + 設計 |

---

## Phase 2a 新的執行期拓樸

```
 UI ──POST /v1/commands──►  Redis: agentbot:commands (FIFO, 自然語言目標)   [送出即清空輸入框]
                                    │  next()
                          ┌─────────▼──────────────┐
                          │ Brain Orchestrator      │  LLM 拆解 → SkillPlan(calls[])
                          │ (常駐 async loop)        │  逐一: dispatch → 等 result
                          └─────────┬──────────────┘  重試 N → 重規劃/中止；全程紀錄
                                    │ LPUSH VlaTaskRequest（一次一個 skill）
                          Redis: agentbot:vla:tasks ──────────► ┌──────────────────────────┐
                                    ▲ 終局結果 (XADD)            │ 常駐 sim_session          │
                          Redis: agentbot:events ◄──────────────┤ (env_isaaclab, App 開著)  │
                          (編排器依 task_id 等結果)                │ 跑 skill→task_done→reset │
                                                                 └─────────────┬────────────┘
                                                                   ZMQ :5555   │
                                                              GR00T policy server (自有 venv)
```

三個常駐程序：**Core**（API+Console+Orchestrator+Monitor+紀錄，uv venv）·**sim_session**（env_isaaclab，持有開著的 IsaacLab App）·**GR00T server**（自有 venv）。Redis 是接縫。狀態機：`IDLE → RUNNING(命令) → IDLE`；停止 → 清空回 IDLE/等待。

---

## 檔案結構（File Structure）

```
agentbot/agentbot/
  contracts/commands.py            # 新: Command, CommandStatus
  monitor/command_queue.py         # 新: CommandQueue (Redis list + in-mem), FIFO + clear
  monitor/results.py               # 新: ResultWaiter — 依 task_id 等終局 VlaTaskResult（經 bus）
  monitor/success_judge.py         # 新: SuccessJudge — task_done(模擬,權威) + VLM-judge hook
  brain/agent.py                   # 改: 加 plan(text, session_id) -> SkillPlan（多 call）
  brain/orchestrator.py            # 新: Orchestrator — 命令迴圈、循序技能、重試/重規劃
  vla/sim_session.py               # 新: 常駐 IsaacLab session（env_isaaclab entrypoint）
  records/__init__.py records/store.py   # 新: SQLite 紀錄 (commands, skill_runs) + 統計查詢
  api/routes_commands.py           # 新: /v1/commands (POST/GET/DELETE), /v1/stats, /v1/control
  api/deps.py                      # 改: 接 CommandQueue, Orchestrator, ResultWaiter, records
  api/app.py                       # 改: 啟動時拉起 Orchestrator（backbone=redis 時）
  ui/console.html                  # 改(最小): 命令框(送出清空) + 佇列/歷史面板
  brain/llm_client.py + settings.py + config  # 改: llm.* -> vlm.* / LLMClient -> VLMClient
tests/
  test_commands.py test_orchestrator.py test_results.py test_records.py   # 新
```

---

## 任務（Phase 2a）

> 在 `agentbot/` 下跑測試：`uv run pytest -q`。具邏輯的單元用 TDD（commands、results waiter、orchestrator、records）。`sim_session` 用 GPU 驗證（在 4090 實跑，如 Phase 1）。**不要 `git commit`** —— 留 checkpoint 給使用者。

### Task 1: Command contracts + 佇列 + 紀錄 schema（TDD）

**檔案：** 新增 `contracts/commands.py`, `monitor/command_queue.py`, `records/store.py`；測試 `tests/test_commands.py`, `tests/test_records.py`。

- [ ] **Step 1: 失敗測試** `tests/test_commands.py`：
```python
from agentbot.contracts.commands import Command, CommandStatus
from agentbot.monitor.command_queue import InMemCommandQueue

def test_command_defaults_and_roundtrip():
    c = Command(text="sort the can onto orange")
    assert c.command_id.startswith("cmd_") and c.status == CommandStatus.QUEUED
    assert Command.model_validate_json(c.model_dump_json()).text == c.text

def test_command_queue_fifo_and_clear():
    q = InMemCommandQueue()
    a = Command(text="first"); b = Command(text="second")
    q.put(a); q.put(b)
    assert q.next(timeout=0.1).command_id == a.command_id   # FIFO（最舊先）
    q.put(Command(text="third"))
    assert q.clear() == 1                                    # 清掉一個待處理(third)
    assert q.next(timeout=0.01) is None
```
- [ ] **Step 2: 跑 → FAIL。**
- [ ] **Step 3: 實作 `contracts/commands.py`**
```python
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from .common import new_id, now_ts
from .skills import SkillPlan

class CommandStatus(str, Enum):
    QUEUED = "queued"; RUNNING = "running"; DONE = "done"; FAILED = "failed"; CLEARED = "cleared"

class Command(BaseModel):
    command_id: str = Field(default_factory=lambda: new_id("cmd"))
    session_id: str = "console"
    text: str
    status: CommandStatus = CommandStatus.QUEUED
    plan: Optional[SkillPlan] = None
    result_summary: str = ""
    created_ts: float = Field(default_factory=now_ts)
```
- [ ] **Step 4: 實作 `monitor/command_queue.py`** — `CommandQueue` ABC（`put`, `next(timeout)->Command|None`, `clear()->int`）；`InMemCommandQueue`（deque）+ `RedisCommandQueue`（LPUSH/BRPOP on `agentbot:commands`，`clear()` = `DEL` 回傳先前長度）。沿用 `job_queue.py` 模式（含其已加的 `redis.exceptions.TimeoutError` 防護）。
- [ ] **Step 5: 實作 `records/store.py`** — SQLite（沿用 `brain/memory/store.py` 連線模式）。表：
```sql
CREATE TABLE IF NOT EXISTS commands(
  command_id TEXT PRIMARY KEY, session_id TEXT, text TEXT, status TEXT,
  plan_json TEXT, result_summary TEXT, created_ts REAL, updated_ts REAL);
CREATE TABLE IF NOT EXISTS skill_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, command_id TEXT, skill TEXT, task_id TEXT,
  attempt INTEGER, status TEXT, success INTEGER, steps INTEGER, duration_s REAL, ts REAL);
```
`Records` 方法：`add_command`, `set_status`, `set_plan`, `log_skill_run`, `list_commands(n)`, `command_detail(id)`（command + 其 skill_runs）, `stats()`（每 skill 成功率 + 平均時長；整體計數）。
- [ ] **Step 6: `tests/test_records.py`** — 加一個 command、記 2 筆 skill_runs（1 成功 1 失敗），驗證 `stats()` 成功率與 `command_detail` 的 join；跑 → PASS。
- [ ] **Step 7: Checkpoint，留給使用者 review。**

### Task 2: ResultWaiter — 依 task_id 等某 skill 的終局結果（TDD）

**檔案：** 新增 `monitor/results.py`；測試 `tests/test_results.py`。

- [ ] **Step 1: 失敗測試** `tests/test_results.py`：
```python
import asyncio, pytest
from agentbot.monitor.event_bus import InProcEventBus
from agentbot.monitor.results import ResultWaiter
from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskResult, VlaTaskStatus

async def test_result_waiter_resolves_by_task_id():
    bus = InProcEventBus(); w = ResultWaiter(bus)
    task = asyncio.create_task(w.run())
    await asyncio.sleep(0)
    fut = w.expect("task_1", timeout=1.0)
    res = VlaTaskResult(task_id="task_1", status=VlaTaskStatus.SUCCEEDED, success=1, success_rate=1.0)
    await bus.publish(Event(type=EventType.VLA_TELEMETRY, source="sim", task_id="task_1",
                            payload={"kind": "result", **res.model_dump()}))
    got = await fut
    assert got.success == 1 and got.status == VlaTaskStatus.SUCCEEDED
    task.cancel()
```
- [ ] **Step 2: 跑 → FAIL。**
- [ ] **Step 3: 實作 `monitor/results.py`** — `ResultWaiter(bus)`：`run()` 訂閱 `VLA_TELEMETRY`；遇到 `payload["kind"]=="result"` 的事件就解析對應的 `asyncio.Future`。`expect(task_id, timeout) -> Future[VlaTaskResult]` 註冊 future（用 `asyncio.wait_for` 包逾時）。這是編排器等每個 skill 的機制（終局結果事件就是 `sim_session`/Phase-1 worker 已用 `_result_event` 發的）。
- [ ] **Step 4: 跑 → PASS。Checkpoint。**

### Task 3: 大腦拆解 — `BrainAgent.plan()`（TDD）

**檔案：** 改 `brain/agent.py`；測試擴充 `tests/test_brain.py`。

- [ ] **Step 1: 失敗測試**（stub LLM 回 ≥1 call；多步目標回多 call）：
```python
async def test_agent_plan_returns_skillplan():
    agent = _agent()   # 既有 helper：QwenVLClient stub + 註冊 sort_can
    plan = await agent.plan("sort the can onto the orange plate", session_id="s1")
    assert plan.calls and plan.calls[0].name == "sort_can"
    assert plan.calls[0].args.get("target_color") == "orange"
```
- [ ] **Step 2: 跑 → FAIL**（`plan` 未定義）。
- [ ] **Step 3: 實作** — 把規劃從 `handle()` 抽出成 `async def plan(self, text, session_id) -> SkillPlan`（組 messages + tools → `llm.complete` → 組 `SkillPlan(intent=text, calls=reply.calls)`）；讓 `handle()` 改呼叫 `plan()`。真實 Qwen3-VL（接好後）對多步目標會回多個 `SkillCall`；stub 回關鍵字命中的 skill。在 prompt 契約裡載明：模型應輸出**有序**的 atomic skill 清單。
- [ ] **Step 4: 跑 → PASS。Checkpoint。**

### Task 4: Orchestrator — 命令迴圈、循序技能、重試/重規劃（TDD）

**檔案：** 新增 `brain/orchestrator.py`；測試 `tests/test_orchestrator.py`。

- [ ] **Step 1: 失敗測試**（FakeSim 對每個派出的 task 發一個結果；驗證循序 + 重試 + 紀錄）：
```python
import asyncio
from agentbot.brain.orchestrator import Orchestrator
from agentbot.contracts.commands import Command, CommandStatus
# FakeSim：當 VlaTaskRequest 被「派出」時發一個結果（skill X 第一次失敗、之後成功）
async def test_orchestrator_runs_plan_sequentially_with_retry(fake_env):
    orch = fake_env.orchestrator                 # 接 stub agent(回 [A,B] 計畫)、fake sim、records
    fake_env.command_queue.put(Command(text="do A then B"))
    await orch.run_one()                          # 只處理一個命令（測試 hook）
    cmd = fake_env.records.list_commands(1)[0]
    assert cmd["status"] == CommandStatus.DONE.value
    runs = fake_env.records.command_detail(cmd["command_id"])["skill_runs"]
    assert [r["skill"] for r in runs] == ["A", "A", "B"]   # A 重試一次後成功，再 B
    assert runs[0]["success"] == 0 and runs[1]["success"] == 1
```
（提供 `conftest.py` 的 `fake_env` fixture：in-proc bus、`InMemCommandQueue`、`ResultWaiter`、一個 `plan()` 回 [A,B] 的 stub agent、一個把 task 入列的 fake dispatch + 一個依腳本成功表發結果的 fake sim、in-memory `Records`。）
- [ ] **Step 2: 跑 → FAIL。**
- [ ] **Step 3: 實作 `brain/orchestrator.py`**
```python
from __future__ import annotations
import asyncio
from typing import Callable
from agentbot.contracts.commands import Command, CommandStatus
from agentbot.contracts.skills import SkillCall
from agentbot.contracts.vla import VlaTaskResult

class Orchestrator:
    def __init__(self, agent, registry, command_queue, dispatch, results, records,
                 vla_ctx, max_retries: int = 2, skill_timeout_s: float = 1800.0):
        self.agent = agent; self.registry = registry; self.cq = command_queue
        self.dispatch = dispatch          # Callable[[SkillCall, dict], str] -> task_id（入列 VlaTaskRequest）
        self.results = results            # ResultWaiter
        self.records = records; self.vla_ctx = vla_ctx
        self.max_retries = max_retries; self.skill_timeout_s = skill_timeout_s
        self._running = True

    async def run(self) -> None:          # 常駐迴圈（app startup 拉起）
        while self._running:
            cmd = await self.cq.next_async(timeout=5)
            if cmd is not None:
                await self._run_command(cmd)

    async def run_one(self) -> None:      # 測試 hook：只處理一個待命令
        cmd = await self.cq.next_async(timeout=1)
        if cmd: await self._run_command(cmd)

    async def _run_command(self, cmd: Command) -> None:
        self.records.set_status(cmd.command_id, CommandStatus.RUNNING)
        plan = await self.agent.plan(cmd.text, cmd.session_id)
        self.records.set_plan(cmd.command_id, plan)
        for call in plan.calls:
            if not await self._run_skill_with_retry(cmd, call):
                if not await self._replan(cmd, call):
                    self.records.set_status(cmd.command_id, CommandStatus.FAILED, "skill failed; aborted")
                    return
        self.records.set_status(cmd.command_id, CommandStatus.DONE, "all skills complete")

    async def _run_skill_with_retry(self, cmd: Command, call: SkillCall) -> bool:
        skill = self.registry.get(call.name)
        if skill is None:
            self.records.log_skill_run(cmd.command_id, call.name, "", 0, "unknown_skill", 0, 0, 0.0)
            return False
        for attempt in range(self.max_retries + 1):
            task_id = self.dispatch(call, self.vla_ctx())     # 入列 VlaTaskRequest，拿 task_id
            try:
                res: VlaTaskResult = await self.results.expect(task_id, self.skill_timeout_s)
            except asyncio.TimeoutError:
                self.records.log_skill_run(cmd.command_id, call.name, task_id, attempt, "timeout", 0, 0, 0.0)
                continue
            self.records.log_skill_run(cmd.command_id, call.name, task_id, attempt,
                                       res.status.value, res.success, res.steps, res.duration_s)
            if res.success:
                return True
        return False

    async def _replan(self, cmd: Command, failed: SkillCall) -> bool:
        """請大腦從失敗處重規劃（MVP：一次重規劃）。新計畫全跑完回 True；否則 False（中止）。"""
        new_plan = await self.agent.plan(
            f"{cmd.text} (the step '{failed.name}' failed; replan the remaining steps)", cmd.session_id)
        for call in new_plan.calls:
            if not await self._run_skill_with_retry(cmd, call):
                return False
        return bool(new_plan.calls)

    def stop(self) -> None:
        self._running = False
```
（`command_queue.next_async` = 對阻塞 `next` 的 async 包裝；`InMemCommandQueue` 很簡單，Redis 版用 `asyncio.to_thread` 在執行緒跑 `next`。）
- [ ] **Step 4: 跑 → PASS。Checkpoint。**

### Task 5: 常駐 sim session（env_isaaclab；GPU 驗證）

**檔案：** 新增 `vla/sim_session.py`。在 `env_isaaclab` 執行；**不**被 API/測試 import。

- [ ] **Step 1:** 結構照 `scripts/eval/gr00t_infer_agent.py` —— **AppLauncher + 參數解析放最上面，在 import gym/isaaclab 之前**（此順序是強制的）。然後是命令驅動迴圈：
  1. 啟動 app（從 config 取 headless）；設 carb（`use_joint_space`, `robot_type`）。
  2. 為 `cfg.vla.default_task` 建 env；`JointMapper`；`build_policy_client`（連 GR00T server host:port —— server 另外啟動，如今）。
  3. 連 Redis：`RedisJobQueue(agentbot:vla:tasks)` + 一個 sync 事件發布器（`r.xadd`，與 `RedisEventBus` wire 相容，完全照 `vla/worker.py`）。
  4. 迴圈：
     - `req = queue.get(block)`；若 `req` 是 shutdown sentinel → break。
     - 若 `req.task_name != current_task`：`env.close()` + 為新 task 重建 env（**保持 app + policy server 活著**）。
     - 從 `req.instruction` 設 `task_description`（多工 Can-Sorting 從 `req.params` 設 `target_color`，沿用 agent 的 `TARGET_COLOR_ID_TO_NAME`）。
     - 跑每 chunk 的 obs→`policy_client.get_action`→map+filter→`env.step`×16，直到 `task_done()`/terminated/truncated/`max_steps`；每 chunk 發 `VlaTelemetry` + 一個終局**結果事件**（`{"kind":"result", ...VlaTaskResult}`）；有開就 `episode_data_saver`。
     - `env.reset()` + 穩定化；回去等。
  5. `scripts/eval/utils/{joint_mapper,filter,episode_data_saver}` 與 obs/action 程式**逐字沿用** `gr00t_infer_agent.py`（搬其內層迴圈，約該檔 282–345 行）。
- [ ] **Step 2:** CLI：`python -m agentbot.vla.sim_session`（在 env_isaaclab、GR00T server 起來後跑）。加 `--max-skills N` 給有界測試。
- [ ] **Step 3: GPU 驗證**（4090）：起 GR00T server + `sim_session`；LPUSH 兩個 `VlaTaskRequest`（如 sort_can orange、再 green）；確認 **app 跨兩個 skill 維持開著**、兩個都發結果事件。（無單元測試 —— 如 Phase 1 的實跑驗證。）
- [ ] **Step 4: Checkpoint。**

### Task 6: monitor/success_judge.py — task_done + VLM-judge hook

**檔案：** 新增 `monitor/success_judge.py`。

- [ ] **Step 1:** `SuccessJudge`：模擬中**權威**訊號是 `task_done()`（已在 sim 發布的結果裡）—— `judge(result) -> (success: bool, source: "task_done")`。加一個 `VlmJudge` hook（stub）：`vlm_judge(frame, goal) -> (success|fail|uncertain, reason)`，用小/已服務的 VLM（回答你的 #3）—— 給實機（無 `task_done()`）或提早抓部分/卡住。2a 階段 `SuccessJudge` 回 sim 的 `task_done` 結果並露出 VLM hook（stub 回 "uncertain"）；2c/Phase 3 接真。
- [ ] **Step 2: smoke** — 單元測試：`SuccessJudge().judge(VlaTaskResult(success=1,...)) == (True, "task_done")`。Checkpoint。

### Task 7: API — commands、stats、control；啟動 orchestrator

**檔案：** 新增 `api/routes_commands.py`；改 `api/deps.py`, `api/app.py`。

- [ ] **Step 1: `routes_commands.py`**
  - `POST /v1/commands {text, session_id?}` → `records.add_command` + `command_queue.put` → 回 `{command_id}`（UI 收到 200 後清空輸入框）。
  - `GET /v1/commands?n=50` → `records.list_commands`（待處理+執行中+歷史，新到舊）。
  - `GET /v1/commands/{id}` → `records.command_detail`（command + plan + skill_runs）供下鑽。
  - `DELETE /v1/commands` → `command_queue.clear()` + 把這些標 `CLEARED` → `{cleared: n}`。
  - `GET /v1/stats` → `records.stats()`（每 skill 成功率、平均時長、總數）。
  - `POST /v1/control/stop` → `orchestrator.stop()` + `command_queue.clear()` → IDLE；`POST /v1/control/estop` → 發一個 `safety` 事件（`immediate=True`）讓 watchdog 跳。
- [ ] **Step 2: `deps.py`** — 加 `command_queue`、`records`、`ResultWaiter`、`Orchestrator` 單例；`dispatch(call, ctx)` closure = `skill.to_vla_request(call, ctx)` → `job_queue.put` → 回 `task_id`。（`backbone=redis` 時用 Redis 版。）
- [ ] **Step 3: `app.py`** — startup 時（`backbone=redis`）：`create_task(ResultWaiter.run())` + `create_task(Orchestrator.run())`。（in-proc dev 仍用 Phase-1 fake pump 給無 GPU demo。）
- [ ] **Step 4: smoke** — `POST /v1/commands` 回 id；`GET /v1/commands` 看得到；`DELETE /v1/commands` 清除；`GET /v1/stats` 形狀正確。Checkpoint。

### Task 8: Console — 非同步命令框 + 佇列/歷史（最小；完整 UI 是 2b）

**檔案：** 改 `ui/console.html`。

- [ ] **Step 1:** Chat 分頁：送出 → `POST /v1/commands` → **立刻清空輸入框**（非同步；可繼續打下一個命令）。加一個 **Queue/History** 面板輪詢 `GET /v1/commands`（狀態 chip：queued/running/done/failed），含 **Clear queue** 鈕（`DELETE /v1/commands`）與 **Stop** 鈕（`POST /v1/control/stop`）。點某命令 → 抓 `/v1/commands/{id}` → 顯示 plan + skill_runs（下鑽）。保持 vanilla（這是 2b dashboard 擴充的種子）。
- [ ] **Step 2: smoke** — 開 `/`，快速送兩個命令（每次輸入框清空），看它們入列、狀態推進、清除佇列。Checkpoint。

### Task 9: 改名 `llm` → `vlm`（cosmetic；大腦本質是 VLM）

**檔案：** 改 `brain/llm_client.py`→`brain/vlm_client.py`, `settings.py`（`LlmCfg`→`VlmCfg`, `llm:`→`vlm:`）, `config/agentbot.example.yaml`, `api/deps.py`, imports。

- [ ] **Step 1:** 改 `LLMClient`→`VLMClient`, `build_llm`→`build_vlm`, 檔名 `llm_client.py`→`vlm_client.py`；config 鍵 `llm:`→`vlm:`（`load_config` 保留讀 `llm:` 為過渡別名一個版本）。更新 `docs/agentbot_architecture.html` 用字。跑全測 → 綠。Checkpoint。

### Task 10: 在 4090 端到端驗證

- [ ] **Step 1:** `redis-server`；起 GR00T server；`conda activate env_isaaclab && python -m agentbot.vla.sim_session`；`uv run uvicorn agentbot.api.app:app`（backbone=redis）。
- [ ] **Step 2:** `POST /v1/commands {"text":"sort the can onto orange then onto green"}`（兩 skill 命令）。確認：命令 QUEUED→RUNNING；大腦計畫＝2 calls；sim 跑 skill 1、Monitor 結果、再 skill 2 —— **app 全程開著**；紀錄有 2 筆 skill_runs；命令 DONE；系統回 IDLE/等待。`GET /v1/stats` 反映。
- [ ] **Step 3:** 負向路徑：強制一個失敗 skill（如 max_steps 很小）→ 確認重試最多 N 次 → 重規劃/中止 → 命令 FAILED、有紀錄。Checkpoint。

---

## 2b · 階層式 Dashboard + 控制（outline — 屆時出獨立計畫）

vanilla，擴充 `console.html`。**可展開左側 sidebar / navigator**：System → Processes（Core/sim_session/GR00T）→ Blocks（1·2·4·5）→ Commands → Plan → Skills → Episodes（可摺疊樹）。**主面板**依選取切換（即時 Monitor、命令明細含技能時間軸、用內附 Chart.js 的統計圖、記憶瀏覽）。**控制列**（非唯讀）：Stop、Clear queue、E-stop、checkpoint 下拉（`/v1/vla/checkpoints`）、暫停/續跑、手動觸發 skill。靠 2a 的端點 + WS `/v1/events`。不引入建置鏈。

## 2c · 記憶 RAG（outline + 設計建議）

episode 列（`skill_runs`）+ Chroma 向量庫（已是 optional dep）。**索引：** 每個完成的 skill/command，嵌入一段短文 —— `"{skill}({args}) on {task} → {status} in {steps} steps; {failure_reason}"` —— 加上提煉的 **lessons**（`failure → fix` 筆記）。**規劃時檢索：** `agent.plan()` 前，`SemanticMemory.query(command_text + skill_names)` 回 top-k 過去嘗試；注入為 context（"相關歷史：sort_can(orange) 成功 8/10；被遮擋時失敗 → 先靠近"）。也檢索 skill spec/docs。效果：大腦依過去成效規劃與重試。藏在 `memory.vector.backend` 後（已 gated；無 Chroma 時 no-op）。

---

## 驗證（Verification）

1. **單元測試（無 GPU/Redis）：** `uv run pytest -q` — commands+佇列+clear、records+stats、ResultWaiter 依 task_id 解析、agent.plan 多 call、**orchestrator 循序 + 重試 + 重規劃 + 紀錄**（FakeSim）、success_judge。全綠。
2. **API（in-proc）：** commands 入列/列出/清除、`/v1/stats`、`/v1/control/stop`；Console 送出清空輸入、佇列推進。
3. **常駐 sim（4090）：** `sim_session` 跨 ≥2 個 skill 維持開著；結果串回。
4. **E2E（4090, redis）：** 兩 skill 命令拆解 → Monitor 把關循序 → 紀錄 → DONE → IDLE；負向路徑重試後重規劃/中止。

## 自我檢查（Self-Review）

- **規格覆蓋：** 常駐開著的 IsaacLab → Task 5；命令紀錄 + 派到大腦 → Tasks 1,7,8；大腦拆解 → 有序 atomic skills → Task 3；循序派發 + Monitor 確認 + 推進 → Task 4（+ Task 2 ResultWaiter, Task 6 judge）；重試/重規劃（重做/從失敗處續）→ Task 4 `_run_skill_with_retry`+`_replan`；長任務完成 → 下一個、否則 IDLE/等待 → Task 4 `run` 迴圈 + Task 7 狀態；紀錄/統計（任務管理）→ Tasks 1,7；多步 + 非同步送出即清空 + 可清除佇列 → Tasks 7,8；階層 Dashboard + 控制 → 2b；RAG → 2c；大腦-VLM 命名（Q1）→ Task 9；Monitor 小 VLM judge（Q3）→ Task 6。✔
- **型別一致：** `Command{command_id,text,status,plan}`；`Orchestrator.dispatch(call, ctx)->task_id` + `ResultWaiter.expect(task_id,timeout)->VlaTaskResult` 在 Tasks 2,4,7 一致；`records.log_skill_run(command_id, skill, task_id, attempt, status, success, steps, duration)` 在 Task 1(定義)、Task 4(呼叫)、Task 7(讀) 一致。沿用既有 `SkillPlan/SkillCall/VlaTaskRequest/VlaTaskResult/Event`。✔
- **無 placeholder：** 可測單元都有完整程式碼；`sim_session` 搬 `gr00t_infer_agent.py` 已驗證的內層迴圈（引用行號）；VLM-judge 是給 2c/Phase 3 的明確 hook。

---

## 執行交接（Execution Handoff）

計畫存於 `agentbot/docs/plans/2026-06-19-phase2-orchestration-core.md`。使用者已選 **subagent 驅動**執行。Tasks 1–4、6–9 不需 GPU 即可單元測試；Tasks 5 與 10 需 4090（如 Phase 1，可在本機跑）。
