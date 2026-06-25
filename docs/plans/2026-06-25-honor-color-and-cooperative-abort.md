# Honor Commanded Color + Cooperative Episode Abort Implementation Plan（繁體中文）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development（建議）或 superpowers:executing-plans，逐 task 執行。步驟用 `- [ ]` 追蹤。
>
> **COMMITS：** 維護者擁有所有 commit。每個 task 測試通過後**只 stage、停下來等維護者 commit**；不要自行 `git commit` / `git push`。

**目標（Goal）：** 補完上一輪 sim-loop bring-up 留下的兩個限制——(A) 讓「指令指定的顏色」真正驅動環境目標（reset 時設定 env target，而非被隨機目標覆寫）；(B) 讓 ■ Stop 能**即時中止正在跑的那一集**（cooperative abort），而非等它跑完約 30s。

**架構（Architecture）：** 兩個獨立小特性，可分開實作/審查。**Part A** 在 IsaacLab 的 `reset_random_choose_object` 加一個 carb 覆寫開關（`/pickplace_env/force_target_object`，未設定時維持原本 `random.choice`），由 `sim_session` 在 `env.reset()` 前依指令顏色設定它。**Part B** 新增一個 redis 旗標 `AbortFlag`（跨程序），`sim_session` 的 chunk 迴圈每個 chunk 輪詢一次；■ Stop 觸發旗標，那一集就提早結束並回報 `ABORTED`。

**技術棧：** IsaacLab/IsaacSim（conda `env_isaaclab`）、carb settings、Redis、FastAPI、pydantic、pytest（agentbot `.venv`，`uv run --no-sync python -m pytest`）。

## 全域限制（Global Constraints）

- **不要 `git commit`/`push`**；每個 task 結束只 `git add`（指定檔案，**不要** `git add -A`，避免帶入 `.superpowers/`、`uv.lock` 變動）。
- **跨 repo（submodule）**：Part A 同時改 `IsaacLab/`（submodule）與 `agentbot/`（submodule）；stage 時要在**對應的 submodule** 內 `git -C <path> add`。
- **顏色↔can↔id 對應（verbatim，勿改）**：instruction `orange` ↔ `red_can` ↔ `target_object_color = 0`；instruction `green` ↔ `blue_can` ↔ `target_object_color = 1`。（來源：`can_sorting/mdp/observations.py:104`、`terminations.py:61`。）
- **向後相容**：`/pickplace_env/force_target_object` 未設定（或非 `red_can`/`blue_can`）時，`reset_random_choose_object` 必須維持原本 `random.choice` 行為（eval harness `gr00t_infer_agent.py` 不設此值，不可被影響）。
- **basket 維持隨機**：force 只能作用在「can 清單」（含 `red_can`/`blue_can` 的 `asset_cfg_list`），不可影響 `respawn_basket`（清單是 `red_basket`/`blue_basket`）。
- **abort 在 in-proc 模式為 no-op**：`backbone != redis` 時用 `InMemAbortFlag`；`_fake_vla_pump`/`FakeRunner` 不檢查它（即時 episode，無需中止）。
- **abort 旗標 token**：`"all"` 代表中止任何 task；否則只中止符合 `task_id` 者。redis 旗標設 `ex=120`（120s 自動過期，避免漏清造成永久卡住）。
- **測試指令**：`cd agentbot && uv run --no-sync python -m pytest <path> -v`。改 `sim_session.py` 後（需 IsaacLab，無法在 agentbot venv import）只能 `python -m py_compile` + stack 實機驗證。
- **Brain 模型 / 動作 checkpoint / EULA / torch 版本**：沿用 `agentbot.yaml` 與 `scripts/stack/start_stack.sh`（toolcall-merged Brain、N1.7 150k、`OMNI_KIT_ACCEPT_EULA=YES`、`torch 2.7.0+cu128`）。

---

## 檔案結構（File Structure）

**Part A — honor commanded color**
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/playground_openarm/dexhand_bimanual/task_scenes/can_sorting/mdp/events.py` — `reset_random_choose_object` 加 force 覆寫。
- `agentbot/agentbot/vla/sim_session.py` — 新增 `force_can_for_color` / `set_force_target` / `clear_force_target`；`run_skill` 在 reset 前依指令顏色 set；startup 與 reset-control 改 clear（維持隨機）。
- `agentbot/tests/test_force_target.py` —（新）`force_can_for_color` 純函式對應測試。

**Part B — cooperative abort**
- `agentbot/agentbot/monitor/abort.py` —（新）`AbortFlag` ABC + `InMemAbortFlag` + `RedisAbortFlag`。
- `agentbot/agentbot/settings.py` — `RedisCfg` 加 `abort_key`。
- `agentbot/agentbot/api/deps.py` — 建 `self.abort`（依 backbone 選 impl）。
- `agentbot/agentbot/api/routes_commands.py` — `control_stop` 觸發 `d.abort.trip()`。
- `agentbot/agentbot/vla/sim_session.py` — 建 `RedisAbortFlag`，傳入 `run_skill`；chunk 迴圈輪詢；回報 `ABORTED`。
- `agentbot/agentbot/ui/console.html` — ■ Stop tooltip 補「abort running episode」。
- `agentbot/tests/test_abort.py` —（新）`InMemAbortFlag` 行為 + `control_stop` 觸發 abort/清佇列。

---

## Part A — Honor commanded color（reset 時設定 env target）

### Task A1: IsaacLab `reset_random_choose_object` 支援 force 覆寫

**Files:**
- Modify: `IsaacLab/.../can_sorting/mdp/events.py:60-62`（`reset_random_choose_object` 內選 target 的兩行）

**Interfaces:**
- Consumes: carb 設定 `/pickplace_env/force_target_object`（string；`red_can`/`blue_can`/空字串）。
- Produces: 行為改變——當 force 命中 `asset_cfg_list` 內某資產時，以它為 target（並照原邏輯 `set_string("/pickplace_env/target_object", asset_name)`）；否則 `random.choice`。

> 此 task 牽涉 IsaacLab+carb+torch，無法在 agentbot venv 單元測試；以「程式碼審查 + Part A 末端實機驗證（Task A3）」為驗收。請嚴格保持未設定時的隨機行為。

- [ ] **Step 1: 閱讀現況**

讀 `IsaacLab/.../can_sorting/mdp/events.py:34-99`。確認第 60-62 行目前為：
```python
    # Randomly choose one asset as the target
    choosed_asset_cfg = random.choice(asset_cfg_list)
    target_asset_name = choosed_asset_cfg.name
```
並確認檔案頂端已有 `import random`（第 19 行）與 `carb_settings_iface`（第 32 行）。

- [ ] **Step 2: 套用 force 覆寫**

把第 60-62 行替換為：
```python
    # Choose the target asset. Honor an operator-pinned target if carb
    # "/pickplace_env/force_target_object" names an asset IN THIS list (only the can list
    # contains red_can/blue_can, so basket selection stays random); else fall back to random.
    # Unset/empty/foreign value => random (keeps eval + default behavior unchanged).
    forced = carb_settings_iface.get("/pickplace_env/force_target_object")
    names = [cfg.name for cfg in asset_cfg_list]
    if forced in names:
        target_asset_name = forced
        choosed_asset_cfg = asset_cfg_list[names.index(forced)]
    else:
        choosed_asset_cfg = random.choice(asset_cfg_list)
        target_asset_name = choosed_asset_cfg.name
```

- [ ] **Step 3: 語法檢查**

Run: `python -m py_compile IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/playground_openarm/dexhand_bimanual/task_scenes/can_sorting/mdp/events.py`
Expected: 無輸出（成功）。

- [ ] **Step 4: Stage（IsaacLab submodule）**

```bash
git -C IsaacLab add source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/playground_openarm/dexhand_bimanual/task_scenes/can_sorting/mdp/events.py
```
（不要 commit；交給維護者。）

---

### Task A2: `sim_session` 依指令顏色設定 force target

**Files:**
- Modify: `agentbot/agentbot/vla/sim_session.py`（新增 3 個 helper；`run_skill`、`_serve` startup、reset-control 三處）
- Test: `agentbot/tests/test_force_target.py`（新）

**Interfaces:**
- Consumes: 模組層既有 `_carb = carb.settings.get_settings()`（sim_session.py:72）；`VlaTaskRequest.params["target_color"]`。
- Produces: `force_can_for_color(color: str | None) -> str | None`（`"orange"->"red_can"`, `"green"->"blue_can"`, 其他/None->`None`）；`set_force_target(color)`（set carb 字串，未知顏色設空字串）；`clear_force_target()`（set 空字串）。

- [ ] **Step 1: 寫失敗測試**

建立 `agentbot/tests/test_force_target.py`：
```python
"""force_can_for_color maps an instruction color to the can asset task_done checks.

orange -> red_can (target_object_color 0); green -> blue_can (1). Unknown -> None.
sim_session sets carb /pickplace_env/force_target_object so the env's reset picks that can,
making the commanded color the actual target (terminations.py checks the matching basket).
"""
import importlib.util
from pathlib import Path

# sim_session.py imports isaaclab at module top (env_isaaclab only); load just the helper by
# exec'ing the function's source is brittle, so import lazily and skip if isaaclab is absent.
import pytest

SS = "agentbot.vla.sim_session"


def _load():
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab not importable in this venv; helper verified on the stack")
    import agentbot.vla.sim_session as m
    return m


def test_force_can_for_color_maps_orange_and_green():
    m = _load()
    assert m.force_can_for_color("orange") == "red_can"
    assert m.force_can_for_color("green") == "blue_can"
    assert m.force_can_for_color("GREEN") == "blue_can"   # case-insensitive
    assert m.force_can_for_color("purple") is None
    assert m.force_can_for_color(None) is None
```

> 註：sim_session.py 在 import 時就 `from isaaclab.app import AppLauncher`，agentbot 的 `.venv` 沒有 isaaclab，故測試以 `find_spec("isaaclab")` 判斷並 `skip`（在 env_isaaclab 內跑才會真正執行）。這仍是有效的回歸測試，且 Task A3 會實機驗證對應。

- [ ] **Step 2: 跑測試確認 skip（非 fail）**

Run: `cd agentbot && uv run --no-sync python -m pytest tests/test_force_target.py -v`
Expected: `1 skipped`（因為 agentbot venv 無 isaaclab）。

- [ ] **Step 3: 新增 helper（sim_session.py）**

在 `sim_session.py` 的 `_scene_target_color` 定義**之前**（即 `run_stabilization` 之後）插入：
```python
FORCE_TARGET_SETTING = "/pickplace_env/force_target_object"
COLOR_TO_CAN = {"orange": "red_can", "green": "blue_can"}


def force_can_for_color(color):
    """Instruction color -> the can asset task_done checks ('orange'->red_can, 'green'->blue_can),
    or None for unknown. Mirrors target_object_color {0: red_can/orange, 1: blue_can/green}."""
    return COLOR_TO_CAN.get((color or "").lower())


def set_force_target(color) -> None:
    """Pin the env's next reset target to *color*'s can (so the commanded color becomes the
    actual target). Unknown color clears the pin (-> random)."""
    _carb.set_string(FORCE_TARGET_SETTING, force_can_for_color(color) or "")


def clear_force_target() -> None:
    """Release the pin so the env re-randomizes its target on reset."""
    _carb.set_string(FORCE_TARGET_SETTING, "")
```

- [ ] **Step 4: `run_skill` 在 reset 前 set force**

在 `run_skill` 內，把
```python
    with torch.inference_mode():
        obs, _ = env.reset()
```
改為：
```python
    with torch.inference_mode():
        # Honor the commanded color: pin the env's target to this color's can BEFORE reset, so
        # the randomized target == the command (task_done then checks the right basket). The
        # color-sync below becomes a no-op when they already match; it stays as a safety net.
        set_force_target(req.params.get("target_color"))
        obs, _ = env.reset()
```

- [ ] **Step 5: startup 與 reset-control 改成 clear（維持隨機）**

在 `_serve` 的 startup reset（`# Reset once at startup ...` 區塊）把
```python
    with torch.inference_mode():
        obs, _ = env.reset()
        obs = run_stabilization(env, idle)
```
改為：
```python
    with torch.inference_mode():
        clear_force_target()           # startup scene is freshly random
        obs, _ = env.reset()
        obs = run_stabilization(env, idle)
```
並在 reset-control 區塊（`if req.params.get("control") == "reset":`）把其中的
```python
            with torch.inference_mode():
                obs, _ = env.reset()
                obs = run_stabilization(env, idle)
```
改為：
```python
            with torch.inference_mode():
                clear_force_target()   # ↺ Reset env => fresh random target
                obs, _ = env.reset()
                obs = run_stabilization(env, idle)
```

- [ ] **Step 6: 語法檢查 + 測試**

Run: `cd agentbot && python -m py_compile agentbot/vla/sim_session.py && uv run --no-sync python -m pytest tests/test_force_target.py -q`
Expected: py_compile 無輸出；pytest `1 skipped`。

- [ ] **Step 7: Stage（agentbot submodule）**

```bash
git -C agentbot add agentbot/vla/sim_session.py tests/test_force_target.py
```

---

### Task A3: Part A 端到端實機驗證

**Files:** 無（僅驗證）。前置：Task A1+A2 完成、stack 以**新程式**重啟。

- [ ] **Step 1: 重啟受影響程序（載入新 events.py + sim_session.py）**

events.py 在 IsaacLab 內、由 `sim_session` import 鏈載入，故需重啟 sim_session（與 dashboard）。可用：
```bash
bash scripts/stack/stop_stack.sh && bash scripts/stack/start_stack.sh   # 全部重啟（最穩）
```
或只重啟 sim_session + dashboard（保留 vlm:8000 / gr00t:5555），等 `output/stack/sim_session.log` 出現 `ready; task=`。

- [ ] **Step 2: 先 Reset 成「已知顏色」，再下相反顏色的明確指令**

```bash
# 取得目前隨機場景顏色
curl -s -XPOST http://localhost:8780/v1/control/reset >/dev/null; sleep 3
redis-cli HGET agentbot:state environment            # 例如 {"target_color":"green",...}
# 下「相反」顏色的明確指令（若上面是 green，就下 orange）
curl -s -XPOST http://localhost:8780/v1/commands -H 'content-type: application/json' \
  -d '{"text":"sort the can onto the orange plate","session_id":"verifyA"}'
```

- [ ] **Step 3: 確認環境目標被改成指令顏色且成功**

```bash
redis-cli HGET agentbot:state environment            # 應變成 {"target_color":"orange",...}
sleep 30                                             # 等 episode 跑完（一集約 30s）
# 取最後一筆指令明細，確認 sort_can succeeded
CID=$(curl -s 'http://localhost:8780/v1/commands?n=1' | python3 -c "import sys,json;print(json.load(sys.stdin)['commands'][0]['command_id'])")
curl -s http://localhost:8780/v1/commands/$CID | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['command']['status'], d['skill_runs'])"
```
Expected：`environment.target_color` == 指令顏色（orange）；指令 `done`、skill_runs 內 `sort_can ... succeeded`。`sim_session.log` 不應出現 `syncing instruction`（因為已一致）。

- [ ] **Step 4: 確認 bare「sort can」仍正常（force 不破壞自動配色）**

```bash
curl -s -XPOST http://localhost:8780/v1/control/reset >/dev/null; sleep 3
COLOR=$(redis-cli HGET agentbot:state environment | python3 -c "import sys,json;print(json.load(sys.stdin)['target_color'])")
echo "scene=$COLOR"
curl -s -XPOST http://localhost:8780/v1/commands -H 'content-type: application/json' -d '{"text":"sort can","session_id":"verifyA2"}'
sleep 35
CID=$(curl -s 'http://localhost:8780/v1/commands?n=1' | python3 -c "import sys,json;print(json.load(sys.stdin)['commands'][0]['command_id'])")
curl -s http://localhost:8780/v1/commands/$CID | python3 -c "import sys,json;d=json.load(sys.stdin);import json as j;p=j.loads(d['command']['plan_json']);print('plan_color=',[c['args'].get('target_color') for c in p['calls']],'status=',d['command']['status'])"
```
Expected：`plan_color` == `scene`（Reset 後的隨機色）；`status=done`。

---

## Part B — Cooperative in-episode abort

### Task B1: `AbortFlag` 抽象 + redis 設定鍵

**Files:**
- Create: `agentbot/agentbot/monitor/abort.py`
- Modify: `agentbot/agentbot/settings.py:32-36`（`RedisCfg` 加 `abort_key`）
- Test: `agentbot/tests/test_abort.py`（新）

**Interfaces:**
- Produces: `AbortFlag`（ABC）：`trip(token: str = "all") -> None`、`clear() -> None`、`tripped(task_id: str) -> bool`。`InMemAbortFlag`、`RedisAbortFlag(url, key)`。`RedisCfg.abort_key: str = "agentbot:vla:abort"`。

- [ ] **Step 1: 寫失敗測試**

建立 `agentbot/tests/test_abort.py`：
```python
"""AbortFlag — the ■ Stop control trips it so a running sim episode ends early.

token "all" aborts any task; a specific task_id aborts only that one. Tested with the
in-mem impl (the redis impl is the same logic over a single auto-expiring key)."""
from agentbot.monitor.abort import InMemAbortFlag


def test_trip_all_aborts_any_task_until_cleared():
    f = InMemAbortFlag()
    assert not f.tripped("task_1")
    f.trip("all")
    assert f.tripped("task_1") and f.tripped("task_2")
    f.clear()
    assert not f.tripped("task_1")


def test_trip_specific_task_only_aborts_that_task():
    f = InMemAbortFlag()
    f.trip("task_1")
    assert f.tripped("task_1")
    assert not f.tripped("task_2")
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `cd agentbot && uv run --no-sync python -m pytest tests/test_abort.py -v`
Expected: FAIL（`ModuleNotFoundError: agentbot.monitor.abort`）。

- [ ] **Step 3: 實作 `abort.py`**

建立 `agentbot/agentbot/monitor/abort.py`：
```python
"""Abort flag (Execution plane) — a cross-process kill-switch for a running VLA episode.

The dashboard ■ Stop trips it; the persistent ``sim_session`` polls it once per action chunk
and ends the episode early (returning ABORTED). ``InMemAbortFlag`` for single-process dev/tests;
``RedisAbortFlag`` (one auto-expiring key) for the real cross-process system.
"""
from __future__ import annotations

import abc
from typing import Optional


class AbortFlag(abc.ABC):
    @abc.abstractmethod
    def trip(self, token: str = "all") -> None:
        """Request abort. ``token`` is a task_id, or ``"all"`` for any running task."""
        ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def tripped(self, task_id: str) -> bool:
        """True if an abort is pending for ``task_id`` (i.e. token is ``"all"`` or ``task_id``)."""
        ...


class InMemAbortFlag(AbortFlag):
    def __init__(self) -> None:
        self._tok: Optional[str] = None

    def trip(self, token: str = "all") -> None:
        self._tok = token

    def clear(self) -> None:
        self._tok = None

    def tripped(self, task_id: str) -> bool:
        return self._tok is not None and self._tok in ("all", task_id)


class RedisAbortFlag(AbortFlag):
    def __init__(self, url: str = "redis://localhost:6379/0", key: str = "agentbot:vla:abort") -> None:
        import redis
        self._r = redis.from_url(url)
        self._key = key

    def trip(self, token: str = "all") -> None:
        # ex=120: auto-expire so a missed clear() can't wedge future episodes permanently.
        self._r.set(self._key, token, ex=120)

    def clear(self) -> None:
        self._r.delete(self._key)

    def tripped(self, task_id: str) -> bool:
        v = self._r.get(self._key)
        if v is None:
            return False
        v = v.decode() if isinstance(v, bytes) else v
        return v in ("all", task_id)
```

- [ ] **Step 4: `RedisCfg` 加 `abort_key`**

`settings.py` 的 `RedisCfg`（第 32-36 行）改為：
```python
class RedisCfg(BaseModel):
    url: str = "redis://localhost:6379/0"
    events_stream: str = "agentbot:events"
    tasks_queue: str = "agentbot:vla:tasks"
    state_hash: str = "agentbot:state"
    abort_key: str = "agentbot:vla:abort"
```

- [ ] **Step 5: 跑測試確認 pass**

Run: `cd agentbot && uv run --no-sync python -m pytest tests/test_abort.py -v`
Expected: `2 passed`。

- [ ] **Step 6: Stage**

```bash
git -C agentbot add agentbot/monitor/abort.py agentbot/settings.py tests/test_abort.py
```

---

### Task B2: deps 接上 `AbortFlag`；■ Stop 觸發 abort

**Files:**
- Modify: `agentbot/agentbot/api/deps.py`（import + 建 `self.abort`）
- Modify: `agentbot/agentbot/api/routes_commands.py`（`control_stop` 加 `d.abort.trip()`）
- Modify: `agentbot/agentbot/ui/console.html`（■ Stop tooltip）
- Test: 追加到 `agentbot/tests/test_abort.py`

**Interfaces:**
- Consumes: `Deps.abort: AbortFlag`、`Deps.cfg.redis.abort_key`。
- Produces: `control_stop` 觸發 `cancel_current()` + 清兩佇列 + 清 blocked state +（新）`abort.trip()`。

- [ ] **Step 1: 寫失敗測試（control_stop 觸發 abort + 清佇列）**

在 `tests/test_abort.py` 末端追加：
```python
import types
import pytest
from agentbot.contracts.vla import VlaTaskRequest
from agentbot.monitor.command_queue import InMemCommandQueue
from agentbot.monitor.job_queue import InMemJobQueue
from agentbot.monitor.state_store import InMemStateStore


class _FakeDeps:
    def __init__(self):
        self.cancelled = False
        self.orchestrator = types.SimpleNamespace(
            cancel_current=lambda: setattr(self, "cancelled", True))
        self.command_queue = InMemCommandQueue()
        self.queue = InMemJobQueue()
        self.abort = InMemAbortFlag()
        self.state = InMemStateStore()


async def test_control_stop_trips_abort_and_drains_and_recovers():
    from agentbot.api.routes_commands import control_stop
    d = _FakeDeps()
    d.queue.put(VlaTaskRequest(task_name="Isaac-Can-Sorting", instruction="x", checkpoint="/c"))
    out = await control_stop(d)
    assert out["stopped"] and d.cancelled
    assert d.abort.tripped("any-task")                 # running episode will see the abort
    assert d.queue.get() is None                        # VLA queue drained
    assert d.state.get_state("robot_state")["mode"] == "idle"
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `cd agentbot && uv run --no-sync python -m pytest tests/test_abort.py::test_control_stop_trips_abort_and_drains_and_recovers -v`
Expected: FAIL（`AttributeError`：`control_stop` 尚未呼叫 `d.abort.trip()`，`d.abort.tripped(...)` 為 False）。

- [ ] **Step 3: deps 建 `self.abort`**

`deps.py` import 區加：
```python
from agentbot.monitor.abort import AbortFlag, InMemAbortFlag, RedisAbortFlag
```
在 `__init__` 內 `self.queue = ...` 那段之後（redis/in-proc 兩分支都要）補：
```python
        if cfg.backbone == "redis":
            self.abort: AbortFlag = RedisAbortFlag(cfg.redis.url, cfg.redis.abort_key)
        else:
            self.abort = InMemAbortFlag()
```
（放在現有 `if cfg.backbone == "redis": ... else: ...`（建 bus/state/queue 與 command_queue）之後另起一段即可。）

- [ ] **Step 4: `control_stop` 觸發 abort**

`routes_commands.py` 的 `control_stop`，在 `d.orchestrator.cancel_current()` 之後加一行：
```python
    d.orchestrator.cancel_current()
    d.abort.trip()                       # also abort the episode already running in sim_session
    cleared_commands = d.command_queue.clear()
```

- [ ] **Step 5: UI tooltip**

`console.html` 的 ■ Stop 按鈕 title 末尾補「+ abort running episode」：
```html
      <button class="btn bad" onclick="stopOrch()" title="Stop &amp; recover: cancel current, abort the running episode, drain command + VLA queues, clear blocked state (loop stays alive)">■ Stop</button>
```

- [ ] **Step 6: 跑測試確認 pass（含全 abort 測試）**

Run: `cd agentbot && uv run --no-sync python -m pytest tests/test_abort.py -v`
Expected: `3 passed`。

- [ ] **Step 7: Stage**

```bash
git -C agentbot add agentbot/api/deps.py agentbot/api/routes_commands.py agentbot/ui/console.html tests/test_abort.py
```

---

### Task B3: `sim_session` 在 chunk 迴圈輪詢 abort → 回報 ABORTED

**Files:**
- Modify: `agentbot/agentbot/vla/sim_session.py`（`run_skill` 簽章 + 迴圈 + result；`_serve` 建 flag 並傳入、收尾 clear）

**Interfaces:**
- Consumes: `RedisAbortFlag(cfg.redis.url, cfg.redis.abort_key)`、`VlaTaskStatus.ABORTED`（已存在於 `contracts/vla.py`）。
- Produces: `run_skill(env, mapper, idle, task_done, client, req, publish, abort=None)`；被中止時回 `VlaTaskResult(status=ABORTED, success=0, ...)`。

> 此 task 需 IsaacLab，無單元測試；以 py_compile + Task B4 實機驗證為準。

- [ ] **Step 1: `run_skill` 簽章加 `abort=None`**

把
```python
def run_skill(env, mapper, idle, task_done, client, req: VlaTaskRequest, publish) -> VlaTaskResult:
```
改為：
```python
def run_skill(env, mapper, idle, task_done, client, req: VlaTaskRequest, publish, abort=None) -> VlaTaskResult:
```

- [ ] **Step 2: 進迴圈前清除舊 abort、初始化旗標**

在 `run_skill` 內 `success = terminated = truncated = False` 那行附近，加入 `aborted = False`：
```python
    success = terminated = truncated = False
    aborted = False
    steps = 0
```
並在 `with torch.inference_mode():` 之後、`set_force_target(...)`/`env.reset()` 之前，清掉任何殘留 abort（這是新派發的 skill，先前的 abort 已過時）：
```python
    with torch.inference_mode():
        if abort is not None:
            abort.clear()               # stale abort from a previous skill must not kill this one
```
（接著才是 `set_force_target(...)`（若 Part A 已套用）與 `obs, _ = env.reset()`。若 Part A 尚未做，就接 `obs, _ = env.reset()`。）

- [ ] **Step 3: 每個 chunk 後檢查 abort**

把內層
```python
            publish(_telemetry_event(req, VlaTelemetry(task_id=req.task_id, step=steps,
                                                       inference_latency_s=lat, success=success)))
            if terminated or truncated or success:
                break
```
改為：
```python
            publish(_telemetry_event(req, VlaTelemetry(task_id=req.task_id, step=steps,
                                                       inference_latency_s=lat, success=success)))
            if abort is not None and abort.tripped(req.task_id):
                aborted = True
                print(f"[sim_session] abort requested; ending episode at step {steps}", flush=True)
                break
            if terminated or truncated or success:
                break
```

- [ ] **Step 4: result 反映 ABORTED**

把
```python
    result = VlaTaskResult(
        task_id=req.task_id,
        status=VlaTaskStatus.SUCCEEDED if success else VlaTaskStatus.FAILED,
        episodes=1, success=1 if success else 0, success_rate=1.0 if success else 0.0,
        steps=steps, duration_s=round(time.time() - t0, 2),
    )
```
改為：
```python
    if aborted:
        status = VlaTaskStatus.ABORTED
    elif success:
        status = VlaTaskStatus.SUCCEEDED
    else:
        status = VlaTaskStatus.FAILED
    result = VlaTaskResult(
        task_id=req.task_id,
        status=status,
        episodes=1, success=1 if success else 0, success_rate=1.0 if success else 0.0,
        steps=steps, duration_s=round(time.time() - t0, 2),
    )
```

- [ ] **Step 5: `_serve` 建立 flag、傳入、收尾清除**

在 `_serve` 內 `client = _make_client(cfg)` 之後加：
```python
    from agentbot.monitor.abort import RedisAbortFlag
    abort = RedisAbortFlag(cfg.redis.url, cfg.redis.abort_key)
    abort.clear()                         # startup: clear any stale abort
```
把呼叫 `run_skill(env, mapper, idle, task_done, client, req, publish)` 改為帶 `abort`：
```python
        run_skill(env, mapper, idle, task_done, client, req, publish, abort=abort)
        abort.clear()                     # done/aborted -> release so the next skill runs
        last_scene_pub = time.time()
```

- [ ] **Step 6: selftest 不傳 abort（保持可跑）**

確認 `_selftest` 內呼叫仍是 `run_skill(env, mapper, idle, task_done, client, req, publish)`（`abort` 預設 None）——不需修改。

- [ ] **Step 7: 語法檢查**

Run: `cd agentbot && python -m py_compile agentbot/vla/sim_session.py`
Expected: 無輸出。

- [ ] **Step 8: Stage**

```bash
git -C agentbot add agentbot/vla/sim_session.py
```

---

### Task B4: Part B 端到端實機驗證

**Files:** 無（僅驗證）。前置：B1–B3 完成、重啟 dashboard + sim_session。

- [ ] **Step 1: 開始一集長任務，中途按 Stop**

```bash
curl -s -XPOST http://localhost:8780/v1/control/reset >/dev/null; sleep 3
CID=$(curl -s -XPOST http://localhost:8780/v1/commands -H 'content-type: application/json' \
  -d '{"text":"sort can","session_id":"verifyB"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['command_id'])")
sleep 6                                   # 讓 episode 跑起來（remember: 一集約 30s）
T0=$(date +%s)
curl -s -XPOST http://localhost:8780/v1/control/stop                      # 觸發 abort
```

- [ ] **Step 2: 確認那一集在數秒內結束、指令為 cleared**

```bash
for i in $(seq 1 8); do
  S=$(curl -s http://localhost:8780/v1/commands/$CID | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['command']['status'], [(r['skill'],r['status']) for r in d['skill_runs']])")
  echo "$(($(date +%s)-T0))s $S"; echo "$S" | grep -qE "cleared|failed|done" && break; sleep 2
done
grep -n "abort requested" output/stack/sim_session.log | tail -2
```
Expected：`sim_session.log` 出現 `abort requested; ending episode at step N`；自按下 Stop 起 **數秒內**（約 ≤ 一個 chunk，~1–2s 模擬 + 推論）episode 結束；指令 `cleared`（被 cancel）。對照修復前需等整集約 30s。

- [ ] **Step 3: 確認 abort 不會誤殺下一集**

```bash
curl -s -XPOST http://localhost:8780/v1/control/reset >/dev/null; sleep 3
CID2=$(curl -s -XPOST http://localhost:8780/v1/commands -H 'content-type: application/json' \
  -d '{"text":"sort can","session_id":"verifyB2"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['command_id'])")
sleep 35
curl -s http://localhost:8780/v1/commands/$CID2 | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['command']['status'], [(r['skill'],r['status'],r.get('success')) for r in d['skill_runs']])"
```
Expected：新指令 `done`、`sort_can ... succeeded`（abort 旗標已被清除，未殘留）。

---

## 完成後（Done）

- [ ] 跑全套 agentbot 測試：`cd agentbot && uv run --no-sync python -m pytest -q`，Expected：全 pass（新增 `test_abort.py` 3 筆、`test_force_target.py` 1 skip）。
- [ ] 確認只 stage、未 commit；列出 staged 檔案交給維護者 review。
- [ ] 更新 `agentbot/docs/USING_VLM_BRAIN.md`：把「① 指令顏色現在會真正設定 env target（Reset 才回隨機）」「② ■ Stop 會即時中止當前 episode」兩點補進 controls/troubleshooting（移除舊的「limitation」字眼）。

---

## 自我檢查（Self-Review）

- **Spec coverage：** ①「reset 時設定 env target」= Task A1（events.py force）+A2（sim_session set）+A3（驗證）；②「episode 中途 cooperative abort」= B1（AbortFlag）+B2（Stop 觸發）+B3（sim_session 輪詢）+B4（驗證）。皆涵蓋。
- **向後相容：** force 未設定 → `random.choice`（eval 不受影響）；`run_skill(abort=None)` 預設不檢查（selftest/worker 路徑不受影響）；in-proc 用 `InMemAbortFlag`。
- **型別/命名一致：** `force_can_for_color`/`set_force_target`/`clear_force_target`、`AbortFlag.trip/clear/tripped`、`RedisCfg.abort_key`、`VlaTaskStatus.ABORTED`、carb 鍵 `/pickplace_env/force_target_object` 在各 task 一致；顏色↔can 對應與 `observations.py`/`terminations.py` 一致。
- **No placeholders：** 每個改碼步驟都附完整程式碼與確切指令/預期輸出。
