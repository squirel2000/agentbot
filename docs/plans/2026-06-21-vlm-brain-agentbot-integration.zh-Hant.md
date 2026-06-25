# VLM Brain ⇄ AgentBot Integration Implementation Plan（繁體中文）

> **給自動化代理人員：** 必要子技能：請使用 superpowers:subagent-driven-development（推薦）或 superpowers:executing-plans 逐任務實作本計畫。步驟採用核取方塊（`- [ ]`）語法。
>
> **提交規範：** 所有提交由維護者負責。每個任務測試通過後，**暫存**變更並**停下讓維護者提交**。建議訊息已提供；請勿自行執行 `git commit`/`push`。
>
> **此文件取代** `docs/plans/2026-06-21-vlm-brain-deployment.md`（獨立 brain 草稿）。AgentBot 已實作了 Brain/SkillRegistry/SkillCall/executor/FastAPI，因此該草稿中的 `src/vlm_lora/brain/`（schema、skills、prompt、core、serve、client、mock_executor、bridge）**予以廢除**。本計畫改為將我們的 VLM 作為 AgentBot 明確為了使用它而建置的端點提供服務。

**目標：** 讓微調後的 **Cosmos-Reason2-2B** 成為 AgentBot 真正的 Brain——在 `Isaac-GR00T-VLM` 中將其作為 **OpenAI 相容的工具呼叫端點**提供服務，把 AgentBot 的 `Gr00tVLMClient` 接上這個端點，**將攝影機影像接入 Brain**，並微調 VLM 以發出 AgentBot 真實技能（pick/place/home/sort_can/pour_water）的有效**工具呼叫**。

**架構：** 兩個 repo。**(1) `Isaac-GR00T-VLM`** 獲得 `src/vlm_lora/serve/` — 一個 FastAPI `/v1/chat/completions` 伺服器，載入合併後的 VLM，接受 OpenAI `messages`+`tools`（含多模態 `image_url`），建構工具呼叫提示，生成，並將模型的 `<tool_call>{…}</tool_call>` 輸出解析成 OpenAI `tool_calls`。**(2) `agentbot`** 獲得一個真實的 `Gr00tVLMClient.complete()`（httpx → 解析 `tool_calls` → `SkillCall`）、config 切換（`vlm.backend: gr00t-vlm`），以及**視覺管道**，使目前的攝影機幀能傳達至 VLM。**工具呼叫 LoRA**（資料：在 AgentBot 技能 schema 上的 `instruction+image → <tool_call>`）使分解更可靠。AgentBot 現有的 orchestrator/sim_session 在 GR00T+IsaacLab 上執行所產生的技能。

**技術棧：** FastAPI + uvicorn + pydantic v2、`transformers`（Qwen3-VL）、`peft`、`httpx`、現有的 `vlm_lora` 套件；AgentBot 以 `uv` 管理，有自己的合約。僅 CPU 的測試對模型/mock httpx 進行 stub。

---

## 背景說明——AgentBot 已為我們提供的與待接縫的部分

`agentbot/`（未追蹤的同層 repo，`squirel2000/agentbot`）實作了完整的 architecture.png 技術棧（Phase 0/0.5/1/2a 已完成，4090 已驗證）。VLM 接縫是**明確的，等待接入**：

- `agentbot/agentbot/brain/vlm_client.py` → `Gr00tVLMClient`：*「從 GR00T N1.7 提取 + 微調的 VLM——Qwen3-VL 佔位符的計劃替代品；遵守相同的 OpenAI 相容工具呼叫合約；將 `vlm.base_url` 指向其端點並設定 `vlm.backend: gr00t-vlm`。」* 目前為關鍵字 **stub**。
- 合約（`agentbot/agentbot/contracts/skills.py`，**已驗證**）：`VLMClient.complete(messages, tools) -> VLMReply{text, calls:[SkillCall]}`；`SkillCall{skill_call_id, name, args, constraints, safety_flags, rationale}`；`SkillSpec.to_tool_schema() -> {name, description, input_schema:{type,properties,required}}`。
- 技能（**已驗證**）：`sort_can{target_color: enum[orange,green]}`、`pick{object: string}`、`place{target: string}`、`home{}`、`pour_water{}`；各 `to_vla_request()` → `VlaTaskRequest`。已在 `SkillRegistry` 中註冊；透過 `registry.as_tool_schemas()` 公開。
- 視覺接縫（**已驗證**）：`UserMessage.image_path: Optional[str]`（"path or data-uri for VLM input"）流向 `routes_chat → Gateway.normalize → UserMessage`，**但 `BrainAgent._complete` 建構純文字訊息並忽略它**——這就是本計畫要填補的差距。
- Config（**已驗證**，`agentbot/config/agentbot.example.yaml`）：`vlm:{backend, model, base_url, api_key_env}`。（`vla.checkpoints` 是**動作**模型/GR00T registry——與透過 `vlm.base_url` 到達的 brain VLM 分開。）

**工具呼叫連線格式（決策）：** 我們的伺服器和微調都使用 Qwen/Hermes 慣例——模型發出 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`（每個選擇的技能一個）；伺服器將這些解析成 OpenAI `tool_calls`。在服務端和訓練端都掌控格式，確保兩者一致，不論 Cosmos-Reason2-2B 的基底 chat template 支援什麼。

---

## 目標檔案結構

```
Isaac-GR00T-VLM/
  src/vlm_lora/
    serve/
      __init__.py
      toolcall.py       # PURE: build tool-calling prompt (+multimodal) ; parse <tool_call> -> calls
      model.py          # ToolCallVLM: load merged VLM once (bf16) ; generate(messages,tools)->text
      openai_app.py     # FastAPI: POST /v1/chat/completions (tools) ; GET /v1/models ; GET /health
    gen_toolcall_data.py  # Phase D: OpenArm frames + skill schemas -> (instruction+image -> <tool_call>) JSONL
    eval_toolcall.py      # Phase D: valid-call% / skill-name acc / arg acc
  examples/
    run_vlm_server.sh   # uvicorn launcher (env: VLM_MODEL_DIR, PORT)
  configs/
    agentbot_skills.sample.json  # frozen copy of registry.as_tool_schemas() for offline gen/tests
  tests/
    test_serve_toolcall.py   # pure prompt-build + parse
    test_serve_app.py        # FastAPI TestClient + stub model
    test_gen_toolcall_data.py
    test_eval_toolcall.py

agentbot/agentbot/
  brain/vlm_client.py   # MODIFY: implement Gr00tVLMClient.complete() (httpx -> tool_calls -> SkillCall)
  brain/agent.py        # MODIFY: plumb image (msg.image_path / frame_provider) into messages
  api/deps.py           # MODIFY: pass a frame_provider into BrainAgent (orchestrator-flow image)
agentbot/tests/
  test_gr00t_vlm_client.py   # NEW: mock endpoint -> parse tool_calls
  test_brain_vision.py       # NEW: image_path -> multimodal message content
agentbot/config/agentbot.example.yaml  # MODIFY: document gr00t-vlm + a vision note
```

現有的 `vlm_lora` 微調模組重用不變（`train_vlm_lora`、`vqa_dataset`、`merge_lora`、`infer_vlm`、`hf_utils`、`gen_vqa_from_lerobot`）。

---

## Phase A — VLM OpenAI 相容工具呼叫伺服器（`Isaac-GR00T-VLM`）

### Task A1: 依賴項 + 套件
**檔案：** 修改 `pyproject.toml`；建立 `src/vlm_lora/serve/__init__.py`

- [ ] **步驟 1：** 附加至 `[project].dependencies`：`"fastapi"`、`"uvicorn"`、`"pydantic>=2"`、`"httpx"`。
- [ ] **步驟 2：** 建立 `src/vlm_lora/serve/__init__.py`：

```python
"""OpenAI-compatible tool-calling server for the fine-tuned VLM (AgentBot's Brain backend)."""
```

- [ ] **步驟 3：** `uv sync --extra dev && uv run python -c "import fastapi, httpx, pydantic; print('ok')"` → `ok`。
- [ ] **步驟 4：暫存** — 訊息：`chore(serve): add fastapi/httpx deps + serve package`。

### Task A2: `toolcall.py` — 提示建構 + 解析（TDD，純粹）
**檔案：** 建立 `src/vlm_lora/serve/toolcall.py`；測試 `tests/test_serve_toolcall.py`

- [ ] **步驟 1：撰寫失敗的測試**

```python
# tests/test_serve_toolcall.py
from vlm_lora.serve.toolcall import build_tool_system, split_text_and_images, parse_tool_calls

TOOLS = [{"name": "sort_can", "description": "place can on a colored plate",
          "input_schema": {"type": "object",
                           "properties": {"target_color": {"enum": ["orange", "green"]}},
                           "required": ["target_color"]}}]

def test_system_lists_tools_and_format():
    sys = build_tool_system(TOOLS)
    assert "sort_can" in sys and "target_color" in sys and "<tool_call>" in sys

def test_parse_single_tool_call():
    txt = 'ok\n<tool_call>{"name": "sort_can", "arguments": {"target_color": "orange"}}</tool_call>'
    calls = parse_tool_calls(txt)
    assert calls == [{"name": "sort_can", "arguments": {"target_color": "orange"}}]

def test_parse_multiple_and_ignores_prose():
    txt = ('<tool_call>{"name":"pick","arguments":{"object":"can"}}</tool_call> then '
           '<tool_call>{"name":"place","arguments":{"target":"orange plate"}}</tool_call>')
    assert [c["name"] for c in parse_tool_calls(txt)] == ["pick", "place"]

def test_parse_none_when_absent():
    assert parse_tool_calls("I cannot do that.") == []

def test_split_text_and_images_handles_multimodal():
    msg = {"role": "user", "content": [
        {"type": "text", "text": "sort it"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}]}
    text, imgs = split_text_and_images(msg)
    assert text == "sort it" and imgs == ["data:image/png;base64,aGk="]
```

- [ ] **步驟 2：執行 → 失敗** — `uv run python -m pytest tests/test_serve_toolcall.py -v`
- [ ] **步驟 3：實作**

```python
# src/vlm_lora/serve/toolcall.py
"""Pure helpers for OpenAI tool-calling over a Qwen3-VL chat model: render the available
tools into a system prompt, pull text/images out of OpenAI multimodal message content, and
parse the model's <tool_call>{...}</tool_call> output back into call dicts."""
from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def build_tool_system(tools: list[dict]) -> str:
    """A system message describing the callable skills + the required output format."""
    lines = ["You are the planning brain of a robot. Decompose the user's instruction into an "
             "ordered sequence of skill-calls, choosing ONLY from these skills:"]
    for t in tools:
        props = t.get("input_schema", {}).get("properties", {})
        req = set(t.get("input_schema", {}).get("required", []))
        args = []
        for name, sch in props.items():
            kind = sch.get("enum") or sch.get("type", "string")
            args.append(f"{name}{'*' if name in req else ''}: {kind}")
        lines.append(f"- {t['name']}({', '.join(args) or ''}) — {t.get('description', '')}")
    lines.append(
        "\nFor EACH step emit exactly one line:\n"
        '<tool_call>{"name": "<skill>", "arguments": {<args>}}</tool_call>\n'
        "Use only the listed skill names and fill every required (*) arg. "
        "Emit nothing else if no skill applies."
    )
    return "\n".join(lines)


def split_text_and_images(message: dict) -> tuple[str, list[str]]:
    """From one chat message, return (joined_text, [image_url_or_datauri, ...]).
    Accepts OpenAI content that is a plain string or a list of typed parts."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            images.append(part.get("image_url", {}).get("url", ""))
    return " ".join(t for t in texts if t).strip(), [u for u in images if u]


def parse_tool_calls(text: str) -> list[dict]:
    """Extract every <tool_call>{json}</tool_call> block as {name, arguments}."""
    out = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            out.append({"name": obj["name"], "arguments": obj.get("arguments", {}) or {}})
    return out
```

- [ ] **步驟 4：執行 → 通過**；lint `uv run ruff check src tests`。
- [ ] **步驟 5：暫存** — 訊息：`feat(serve): tool-calling prompt + parser (pure)`。

### Task A3: `model.py` — ToolCallVLM 載入/生成（GPU；未進行單元測試）
**檔案：** 建立 `src/vlm_lora/serve/model.py`

- [ ] **步驟 1：實作**（重用 `infer_vlm`/`hf_utils` 模式）

```python
# src/vlm_lora/serve/model.py
"""Load the merged VLM once and generate tool-calling text for a chat request.
Decodes OpenAI multimodal image_url (data-uri or path) into PIL images for the processor."""
from __future__ import annotations

import base64
import io

from vlm_lora.serve.toolcall import build_tool_system, parse_tool_calls, split_text_and_images


def _load_image(url: str):
    from PIL import Image

    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return Image.open(url).convert("RGB")


class ToolCallVLM:
    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from vlm_lora.hf_utils import resolve_model_path

        md = resolve_model_path(model_dir)
        self.processor = AutoProcessor.from_pretrained(md, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            md, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )

    def generate(self, messages: list[dict], tools: list[dict], max_new_tokens: int = 512) -> str:
        # 1) prepend a tool-describing system message
        chat = [{"role": "system", "content": build_tool_system(tools)}]
        images = []
        for m in messages:
            text, imgs = split_text_and_images(m)
            images += imgs
            if imgs:  # keep an <image> placeholder so the processor aligns image tokens
                chat.append({"role": m["role"],
                             "content": [{"type": "image"}, {"type": "text", "text": text}]})
            else:
                chat.append({"role": m["role"], "content": text})
        pil = [_load_image(u) for u in images] or None
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[prompt], images=pil, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

    def complete(self, messages: list[dict], tools: list[dict], max_new_tokens: int = 512) -> dict:
        """Return (raw_text, parsed_calls)."""
        raw = self.generate(messages, tools, max_new_tokens)
        return {"text": raw, "calls": parse_tool_calls(raw)}
```

- [ ] **步驟 2：** import 檢查 `uv run python -c "import vlm_lora.serve.model"`；lint；**暫存** — 訊息：`feat(serve): ToolCallVLM loader + multimodal generate`。

### Task A4: `openai_app.py` — FastAPI 伺服器（TDD，stub 模型）
**檔案：** 建立 `src/vlm_lora/serve/openai_app.py`；測試 `tests/test_serve_app.py`

- [ ] **步驟 1：撰寫失敗的測試**

```python
# tests/test_serve_app.py
from fastapi.testclient import TestClient
from vlm_lora.serve import openai_app as A

class _Stub:
    def complete(self, messages, tools, max_new_tokens=512):
        return {"text": "ok", "calls": [{"name": "sort_can", "arguments": {"target_color": "orange"}}]}

def _client():
    A.STATE["model"] = _Stub()
    return TestClient(A.app)

def test_health():
    assert _client().get("/health").json()["status"] == "ok"

def test_chat_completions_returns_tool_calls():
    body = {"model": "gr00t-vlm", "messages": [{"role": "user", "content": "sort the can onto orange"}],
            "tools": [{"type": "function", "function": {"name": "sort_can", "description": "x",
                       "parameters": {"type": "object", "properties": {}, "required": []}}}]}
    r = _client().post("/v1/chat/completions", json=body).json()
    tc = r["choices"][0]["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "sort_can"
    assert '"target_color": "orange"' in tc[0]["function"]["arguments"]
```

- [ ] **步驟 2：執行 → 失敗**
- [ ] **步驟 3：實作**（接受 OpenAI `tools=[{type:function, function:{name,description,parameters}}]`，映射至我們的 `{name,description,input_schema}`，回傳 OpenAI `tool_calls`）

```python
# src/vlm_lora/serve/openai_app.py
"""Minimal OpenAI-compatible /v1/chat/completions that backs AgentBot's Gr00tVLMClient.
Launch:  VLM_MODEL_DIR=<merged_vlm> uv run uvicorn vlm_lora.serve.openai_app:app --port 8000"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="GR00T VLM (OpenAI-compatible)")
STATE: dict = {"model": None}


class ChatMessage(BaseModel):
    role: str
    content: Any = ""           # str OR list of typed parts (multimodal)


class ChatCompletionRequest(BaseModel):
    model: str = "gr00t-vlm"
    messages: list[ChatMessage]
    tools: list[dict] = Field(default_factory=list)
    tool_choice: Any = "auto"
    max_tokens: int = 512
    temperature: float = 0.0


@app.on_event("startup")
def _load():
    if STATE["model"] is not None:
        return
    from vlm_lora.serve.model import ToolCallVLM

    STATE["model"] = ToolCallVLM(os.environ["VLM_MODEL_DIR"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "gr00t-vlm", "object": "model"}]}


def _to_internal_tools(tools: list[dict]) -> list[dict]:
    """OpenAI tools [{type:function, function:{name,description,parameters}}] -> internal
    [{name, description, input_schema}]. Also accept the internal shape directly."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", fn.get("input_schema", {}))})
    return out


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [m.model_dump() for m in req.messages]
    result = STATE["model"].complete(messages, _to_internal_tools(req.tools), req.max_tokens)
    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
        for i, c in enumerate(result["calls"])
    ]
    message = {"role": "assistant", "content": result["text"] if not tool_calls else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-gr00tvlm", "object": "chat.completion", "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }
```

- [ ] **步驟 4：執行 → 通過**；lint。
- [ ] **步驟 5：啟動腳本** `examples/run_vlm_server.sh`：

```bash
#!/usr/bin/env bash
# Serve the fine-tuned VLM as AgentBot's Brain. Env: VLM_MODEL_DIR (required), PORT, CUDA_VISIBLE_DEVICES.
set -euo pipefail
: "${VLM_MODEL_DIR:?set VLM_MODEL_DIR to the merged VLM dir}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec uv run uvicorn vlm_lora.serve.openai_app:app --host 0.0.0.0 --port "${PORT:-8000}"
```

- [ ] **步驟 6：** `bash -n examples/run_vlm_server.sh`；**暫存** — 訊息：`feat(serve): OpenAI-compatible /v1/chat/completions + launcher`。

### Task A5: GPU 煙霧測試（H100/4090，手動關卡）
- [ ] **步驟 1：** `VLM_MODEL_DIR=<…/lora_tuned_vlm/Cosmos-Reason2-2B-lora-merged> bash examples/run_vlm_server.sh`
- [ ] **步驟 2：** `curl /v1/chat/completions` 搭配真實 OpenArm 幀（作為 `image_url` data-uri）+ tools=[sort_can]。**預期：** HTTP 200；若 base（VQA 微調）模型未發出 `<tool_call>`，這是預期行為 → 促使 Phase D。記錄原始輸出。

---

## Phase B — AgentBot 消費端點

### Task B1: 實作 `Gr00tVLMClient.complete()`
**檔案：** 修改 `agentbot/agentbot/brain/vlm_client.py:87`（stub 本體）

- [ ] **步驟 1：撰寫失敗的測試** `agentbot/tests/test_gr00t_vlm_client.py`

```python
import json
import pytest
from agentbot.brain.vlm_client import Gr00tVLMClient

class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

@pytest.mark.asyncio
async def test_complete_parses_tool_calls(monkeypatch):
    payload = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "sort_can", "arguments": json.dumps({"target_color": "orange"})}}]}}]}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _Resp(payload)

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    reply = await Gr00tVLMClient(base_url="http://x/v1").complete(
        [{"role": "user", "content": "sort onto orange"}],
        [{"name": "sort_can", "description": "d", "input_schema": {"type": "object", "properties": {}, "required": []}}])
    assert reply.calls[0].name == "sort_can" and reply.calls[0].args == {"target_color": "orange"}
```

（注意：AgentBot 的測試套件已使用 asyncio；若 `pytest.mark.asyncio` 未配置，改用 `asyncio.run` 包裝。）

- [ ] **步驟 2：執行 → 失敗**（`cd agentbot && uv run python -m pytest tests/test_gr00t_vlm_client.py -v`）
- [ ] **步驟 3：實作** — 替換 `Gr00tVLMClient.complete` stub 本體：

```python
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict]) -> VLMReply:
        import json
        import httpx
        from agentbot.contracts.skills import SkillCall

        payload = {
            "model": self.model or "gr00t-vlm",
            "messages": messages,
            "tools": [{"type": "function",
                       "function": {"name": t["name"], "description": t.get("description", ""),
                                    "parameters": t["input_schema"]}} for t in tools],
            "tool_choice": "auto",
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # endpoint down/misconfigured -> safe stub so the loop survives
            return self._stub_reply(messages, tools)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(SkillCall(name=fn.get("name", ""), args=args, rationale=msg.get("content") or ""))
        return VLMReply(text=msg.get("content") or "", calls=calls)
```

- [ ] **步驟 4：執行 → 通過**；`cd agentbot && uv run ruff check agentbot tests`（符合他們的 lint 規範）。
- [ ] **步驟 5：暫存** — 訊息：`feat(brain): wire Gr00tVLMClient to the served VLM endpoint`。

### Task B2: gr00t-vlm backend 的 config + 文件
**檔案：** 修改 `agentbot/config/agentbot.example.yaml:14-21`

- [ ] **步驟 1：** 在 `vlm:` 下新增一個有文件說明的範例區塊（保持注解狀態，不改變預設值）：

```yaml
  # To use the fine-tuned GR00T VLM as the Brain, run the server (Isaac-GR00T-VLM:
  #   VLM_MODEL_DIR=<merged_vlm> uv run uvicorn vlm_lora.serve.openai_app:app --port 8000)
  # then set:
  #   backend: gr00t-vlm
  #   model:   gr00t-vlm
  #   base_url: http://<vlm-host>:8000/v1
```

- [ ] **步驟 2：暫存** — 訊息：`docs(config): how to point the Brain at the gr00t-vlm endpoint`。

---

## Phase C — 視覺管道（攝影機 → Brain）

> 兩個流程：**聊天**流程已攜帶 `UserMessage.image_path`；**orchestrator** 流程（`agent.plan(text, session_id)`）沒有影像，因此 Brain 從 `frame_provider` → Monitor 的 `state["camera"]["frame"]` 提取最新幀。**Brain 透過一個事件匯流排接縫與幀來源解耦**（Task C0）：生產者發布 `CAMERA` 事件；`ingest` 寫入 `state["camera"]`。在**模擬**中，`sim_session` 是生產者；在**硬體**上（Phase 3），ROS2→事件匯流排 bridge 以*相同*事件重新發布攝影機主題——Brain 從不改變。這與 AgentBot Phase-3「將 ROS2 重新發布至同一事件匯流排」的設計一致。

### Task C0: 攝影機幀 → Monitor `state["camera"]`（單一、與實施無關的接縫）
**檔案：** 修改 `agentbot/agentbot/contracts/events.py`（新增 `CAMERA` EventType）、`agentbot/agentbot/monitor/ingest.py`（儲存 `state["camera"]`）、`agentbot/agentbot/vla/sim_session.py`（發布縮小後的幀）；測試 `agentbot/tests/test_camera_ingest.py`

設計：Brain 僅讀取 `state["camera"]["frame"]`。生產者發布 `Event(type=CAMERA, payload={frame:<jpeg data-uri>, ts})`；`ingest` 儲存它。Sim = `sim_session` 生產（將 IsaacLab 攝影機 obs 縮小至 ~384px JPEG，以約 1–2 Hz 節流的閒置泵發布，並在每次 episode 開始時發布一次）。硬體（Phase 3）= ROS2→事件匯流排 bridge 以相同事件重新發布攝影機主題。幀縮小保持 redis state 較小；規劃不需要完整解析度。

- [ ] **步驟 1：失敗的測試** `agentbot/tests/test_camera_ingest.py`：

```python
from agentbot.contracts.events import Event, EventType
from agentbot.monitor.ingest import Ingestor
from agentbot.monitor.state_store import InMemoryStateStore  # use the concrete in-mem store

def test_camera_event_lands_in_state():
    st = InMemoryStateStore()
    ing = Ingestor(bus=None, state_store=st)
    ing.handle(Event(type=EventType.CAMERA, source="vla.sim_session",
                     payload={"frame": "data:image/jpeg;base64,QQ==", "ts": 1.0}))
    assert st.get_state("camera")["frame"].startswith("data:image/jpeg")
```

（實作時確認具體的 in-mem StateStore 類別名稱；調整 import。）

- [ ] **步驟 2：執行 → 失敗**
- [ ] **步驟 3：在 `EventType` 中新增 `CAMERA`**（`agentbot/agentbot/contracts/events.py`）：`CAMERA = "camera"`。
- [ ] **步驟 4：ingest 儲存它** — 在 `Ingestor.handle` 中新增：`if ev.type == EventType.CAMERA: self.state.set_state("camera", ev.payload)`。
- [ ] **步驟 5：執行 → 通過。**
- [ ] **步驟 6（GPU/sim，關卡）：sim_session 發布幀。** 先閱讀 `agentbot/agentbot/vla/sim_session.py` + `scripts/eval/gr00t_infer_agent.py` 以找到確切的攝影機 obs 存取器；然後新增一個輔助函式，將最新 obs 幀縮小至 ~384px JPEG data-uri 並發布 `Event(type=CAMERA, ...)` — 在閒置泵上（透過單調時間戳節流，約 1–2 Hz）以及 episode 開始時。**停下/詢問** 若閒置泵在不步進模擬的情況下沒有幀可用。
- [ ] **步驟 7：暫存** — 訊息：`feat(monitor): CAMERA event + ingest; sim_session publishes downsized frames`。

### Task C1: `BrainAgent` 建構多模態訊息（TDD）
**檔案：** 修改 `agentbot/agentbot/brain/agent.py`；測試 `agentbot/tests/test_brain_vision.py`

- [ ] **步驟 1：失敗的測試**

```python
# agentbot/tests/test_brain_vision.py
from agentbot.brain.agent import build_user_content

def test_text_only_when_no_image():
    assert build_user_content("sort it", None) == "sort it"

def test_multimodal_when_image_present():
    c = build_user_content("sort it", "data:image/png;base64,aGk=")
    assert {"type": "text", "text": "sort it"} in c
    assert any(p["type"] == "image_url" for p in c)
```

- [ ] **步驟 2：執行 → 失敗**
- [ ] **步驟 3：實作** — 新增模組級輔助函式並使用它；穿入可選影像：

```python
# agentbot/agentbot/brain/agent.py  (add near top)
def build_user_content(text: str, image: str | None):
    """OpenAI message content: plain text, or [text, image_url] when an image is available.
    `image` is a path or data-uri (UserMessage.image_path) or a live frame from frame_provider."""
    if not image:
        return text
    url = image if image.startswith("data:") else image  # server accepts data-uri or path
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]
```

然後更新 `BrainAgent.__init__`，接受 `frame_provider: Optional[Callable[[], str | None]] = None`（預設 None），並修改 `_complete` 以接受可選影像並將其用於最終使用者輪次：

```python
    async def _complete(self, text: str, session_id: str, image: str | None = None):
        history = self.conversation.recent(session_id, n=10)
        messages = [{"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
                    for t in history]
        messages.append({"role": "user", "content": build_user_content(text, image)})
        tools = self.registry.as_tool_schemas()
        reply = await self.llm.complete(messages, tools)
        return SkillPlan(intent=text, calls=reply.calls), reply.text

    async def plan(self, text: str, session_id: str) -> SkillPlan:
        image = self.frame_provider() if self.frame_provider else None
        plan, _ = await self._complete(text, session_id, image)
        return plan
```

以及在 `handle()` 中：`plan, reply_text = await self._complete(msg.text, msg.session_id, msg.image_path)`。

- [ ] **步驟 4：執行 → 通過**；確保現有的 brain 測試仍通過（`uv run python -m pytest tests/test_brain.py -v`）。
- [ ] **步驟 5：暫存** — 訊息：`feat(brain): plumb camera image into VLM messages (chat + frame_provider)`。

### Task C2: 在 `deps.py` 中從 Monitor 接入 `frame_provider`
**檔案：** 修改 `agentbot/agentbot/api/deps.py`

- [ ] **步驟 1：** 建構 `BrainAgent` 時，傳入一個從 Monitor state 讀取最新幀的 provider（若無則回退到 None）：

```python
        def _latest_frame():
            snap = self.state.snapshot() if hasattr(self, "state") else {}
            return (snap.get("camera") or {}).get("frame")  # data-uri/path, or None
        self.agent = BrainAgent(build_vlm(cfg), self.registry, self.conversation,
                                bus=self.bus, gateway=self.gateway, frame_provider=_latest_frame)
```

- [ ] **步驟 2：** 執行套件 `cd agentbot && uv run python -m pytest -q`（預期全部通過；provider 在幀未發布時回傳 None）。
- [ ] **步驟 3：暫存** — 訊息：`feat(api): provide latest Monitor frame to the Brain`。

---

## Phase D — 工具呼叫微調（可靠性）

### Task D1: 凍結 AgentBot 技能 schema 供離線使用
**檔案：** 建立 `Isaac-GR00T-VLM/configs/agentbot_skills.sample.json`

- [ ] **步驟 1：** 從 AgentBot 生成一次並提交副本（以便 gen/eval/tests 不需要 import agentbot）：

```bash
cd agentbot && uv run python -c "import json; from agentbot.api.deps import Deps; from agentbot.settings import load_config; \
print(json.dumps(Deps(load_config()).registry.as_tool_schemas(), indent=2))" \
> ../Isaac-GR00T-VLM/configs/agentbot_skills.sample.json
```

- [ ] **步驟 2：** 健全性檢查，確認列出了 sort_can/pick/place/home/pour_water；**暫存** — 訊息：`chore(serve): freeze AgentBot skill schemas for offline gen/tests`。

### Task D2: `gen_toolcall_data.py`（TDD）
**檔案：** 建立 `src/vlm_lora/gen_toolcall_data.py`；測試 `tests/test_gen_toolcall_data.py`

- [ ] **步驟 1：失敗的測試**

```python
# tests/test_gen_toolcall_data.py
from vlm_lora.gen_toolcall_data import template_calls, to_training_row

def test_template_calls_sort_can():
    calls = template_calls("place the can on the orange plate")
    assert calls == [{"name": "sort_can", "arguments": {"target_color": "orange"}}]

def test_to_training_row_emits_tool_call_tags():
    row = to_training_row("img.png", "put can on green plate",
                          [{"name": "sort_can", "arguments": {"target_color": "green"}}])
    assert row["images"] == ["img.png"]
    a = row["messages"][1]["content"]
    assert a.startswith("<tool_call>") and '"sort_can"' in a and a.rstrip().endswith("</tool_call>")
```

- [ ] **步驟 2：執行 → 失敗**
- [ ] **步驟 3：實作**（範本下限將 OpenArm 任務映射至 sort_can；教師 VLM 可選，用於措辭多樣性；重用 `gen_vqa_from_lerobot._extract_frame/_sample_frames`）

```python
# src/vlm_lora/gen_toolcall_data.py
"""Generate tool-calling training data (instruction+image -> <tool_call> sequence) over
AgentBot's real skill schemas, from OpenArm LeRobot frames. Output JSONL matches vqa_dataset's
`messages` schema so train_vlm_lora trains on it unchanged."""
from __future__ import annotations

import json
import os
import re

import tyro

_COLORS = ["orange", "green"]


def template_calls(task: str) -> list[dict]:
    """OpenArm 0403 is the can-sorting task -> a single sort_can call with the plate color."""
    for c in _COLORS:
        if re.search(rf"\b{c}\b", task.lower()):
            return [{"name": "sort_can", "arguments": {"target_color": c}}]
    return [{"name": "sort_can", "arguments": {"target_color": "orange"}}]


def _render_calls(calls: list[dict]) -> str:
    return "\n".join(f'<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>' for c in calls)


def to_training_row(image_rel: str, instruction: str, calls: list[dict]) -> dict:
    return {"images": [image_rel], "type": "ToolCall", "messages": [
        {"role": "user", "content": f"<image>\n{instruction}"},
        {"role": "assistant", "content": _render_calls(calls)}]}


def generate(dataset_path: str, out_dir: str, num_episodes: int = 100, frames_per_episode: int = 2,
             val_ratio: float = 0.1, seed: int = 42) -> None:
    import random

    from vlm_lora.gen_vqa_from_lerobot import _extract_frame, _sample_frames

    random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    eps = [json.loads(x) for x in open(os.path.join(dataset_path, "meta/episodes.jsonl"),
                                       encoding="utf-8") if x.strip()]
    random.shuffle(eps)
    eps = eps[:num_episodes]
    vpat = os.path.join(dataset_path, "videos/chunk-000/observation.images.camera/episode_{:06d}.mp4")
    rows = []
    for ep in eps:
        ei, task, length = ep["episode_index"], ep["task"], ep["length"]
        if not os.path.exists(vpat.format(ei)):
            continue
        for fidx, _phase in _sample_frames(length, frames_per_episode):
            img = _extract_frame(vpat.format(ei), fidx)
            if img is None:
                continue
            rel = f"images/ep{ei:06d}_f{fidx:06d}.png"
            img.save(os.path.join(out_dir, rel))
            rows.append(to_training_row(rel, task, template_calls(task)))
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    for name, sub in {"data.val.jsonl": rows[:n_val], "data.train.jsonl": rows[n_val:],
                      "data.jsonl": rows}.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_toolcall] {len(rows)} rows ({len(rows) - n_val} train / {n_val} val) -> {out_dir}")


if __name__ == "__main__":
    tyro.cli(generate)
```

- [ ] **步驟 4：執行 → 通過**；lint；**暫存** — 訊息：`feat(finetune): generate tool-calling data over AgentBot skills`。

### Task D3: 訓練 + 合併工具呼叫 LoRA（H100 GPU）
- [ ] **步驟 1：** `... -m vlm_lora.gen_toolcall_data --dataset-path <OpenArm> --out-dir artifacts/toolcall --num-episodes 100`
- [ ] **步驟 2：** `train_vlm_lora --dataset-path artifacts/toolcall/data.train.jsonl --image-root artifacts/toolcall --output-dir artifacts/cosmos_r2_toolcall_lora --max-steps 1500`（重用，不變）。
- [ ] **步驟 3：** `merge_lora --adapter-dir artifacts/cosmos_r2_toolcall_lora --out-dir <…>/lora_tuned_vlm_toolcall/Cosmos-Reason2-2B-toolcall-merged`。
- [ ] **步驟 4：** 將伺服器指向該檢查點（`VLM_MODEL_DIR=<…toolcall-merged>`）。**停下/詢問** 若 OOM。

### Task D4: `eval_toolcall.py`（TDD）+ 前後對比
**檔案：** 建立 `src/vlm_lora/eval_toolcall.py`；測試 `tests/test_eval_toolcall.py`

- [ ] **步驟 1：失敗的測試**

```python
# tests/test_eval_toolcall.py
from vlm_lora.eval_toolcall import score

def test_exact_match():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"orange"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["valid"] == 1 and s["name_ok"] == 1 and s["args_ok"] == 1

def test_wrong_arg():
    s = score('<tool_call>{"name":"sort_can","arguments":{"target_color":"green"}}</tool_call>',
              [{"name": "sort_can", "arguments": {"target_color": "orange"}}], allowed={"sort_can"})
    assert s["name_ok"] == 1 and s["args_ok"] == 0
```

- [ ] **步驟 2：執行 → 失敗；實作** `score(raw, gold_calls, allowed)`（重用 `serve.toolcall.parse_tool_calls`；指標：解析有效、所有名稱在 vocab 內、名稱序列匹配、args 完全相符）+ 一個 `evaluate(model_dir, val_jsonl, image_root, skills_json, out_json)` 函式，鏡像 `eval_vlm_vqa.py` 在 val 集上執行 `ToolCallVLM`。
- [ ] **步驟 3：執行 → 通過；暫存** — 訊息：`feat(finetune): tool-calling accuracy eval`。
- [ ] **步驟 4（GPU）：** 在 `artifacts/toolcall/data.val.jsonl` 上評估 base 合併 VLM vs toolcall-merged → 前後對比表（valid% / name-acc / args-acc）。

---

## Phase E — 端到端整合

### Task E1: 對真實 VLM 的內部行程儀表板示範（無 IsaacLab）
- [ ] **步驟 1：** 服務 toolcall-merged VLM（Task D3）；在 `agentbot/config/agentbot.yaml` 中設定 `backend: in-proc`、`vlm.backend: gr00t-vlm`、`vlm.base_url: http://<host>:8000/v1`。
- [ ] **步驟 2：** `cd agentbot && uv run uvicorn agentbot.api.app:app --port 8780`；在儀表板中輸入 "sort the can onto orange"；**預期：** 計畫顯示 VLM 發出的真實 `sort_can{target_color: orange}`（非關鍵字 stub），fake-VLA 到達 `done`。

### Task E2: 在 4090 上的真實流程（手動關卡，可選）
- [ ] **步驟 1：** 使用 `vlm.backend: gr00t-vlm` 的 4 行程流程（README §B）；發送指令；**預期：** VLM 規劃 → orchestrator 派發 `sort_can` → sim_session 執行 IsaacLab episode → `done`。記錄 success_rate。**停下/詢問** 若 VLM 發出不在 registry 中的技能（→ 收緊 Phase D 資料 / few-shot）。

---

## Phase F — 文件

### Task F1: 廢除舊計畫 + 更新 Isaac-GR00T-VLM 報告
**檔案：** 修改 `Isaac-GR00T-VLM/docs/plans/2026-06-21-vlm-brain-deployment.md`（新增廢除橫幅）、`docs/project_report.html`、`docs/architecture_dataflow.html`、`README.md`

- [ ] **步驟 1：** 在舊計畫頂部新增：`> ⚠️ SUPERSEDED by 2026-06-21-vlm-brain-agentbot-integration.md (integrates with the existing agentbot stack).`
- [ ] **步驟 2：** `README.md` — 新增 "Serve as AgentBot's Brain" 章節：執行 `run_vlm_server.sh`，在 agentbot 中設定 `vlm.backend: gr00t-vlm`，連結此計畫。
- [ ] **步驟 3：** `project_report.html` §8 — "VLM as AgentBot's Brain"：OpenAI 工具呼叫合約、`<tool_call>` 格式、VQA→工具呼叫差距 + LoRA 答案，以及（若 D4 已執行）前後對比表。圖表/公式保留英文，prose 使用繁中。
- [ ] **步驟 4：** `architecture_dataflow.html` — 新增流程：`instruction+image → [VLM /v1/chat/completions] → tool_calls → [AgentBot Brain→Skill→VLA]`。**暫存** — 訊息：`docs: VLM-as-AgentBot-Brain (report + dataflow + supersede)`。

### Task F2: 在 AgentBot 的 TASKS 中注記
**檔案：** 修改 `agentbot/TASKS.md`

- [ ] **步驟 1：** 在 Phase 1 剩餘項目「把 stub 換成真的 Qwen3-VL tool-calling」下，新增一個子行，說明 `gr00t-vlm` backend 現已實作 + 從 Isaac-GR00T-VLM `serve/` 服務。**暫存** — 訊息：`docs(tasks): gr00t-vlm Brain backend wired`。

---

## 驗證（端到端）

1. **Isaac-GR00T-VLM CPU 測試通過：** `uv run python -m pytest tests/test_serve_toolcall.py tests/test_serve_app.py tests/test_gen_toolcall_data.py tests/test_eval_toolcall.py -v`。
2. **AgentBot 測試通過：** `cd agentbot && uv run python -m pytest -q`（現有 31 個 + `test_gr00t_vlm_client` + `test_brain_vision` 全部通過）。
3. **伺服器：** `run_vlm_server.sh` → `/health` ok；`/v1/chat/completions` 搭配 tools=[sort_can] + 影像回傳 `tool_calls` 陣列。
4. **客戶端：** `Gr00tVLMClient.complete()` 將端點的 `tool_calls` 解析成 `SkillCall(name=..., args=...)`。
5. **視覺：** 設定 `image_path`（聊天）或 frame provider（orchestrator）後，對伺服器的請求攜帶 `image_url` 部分。
6. **可靠性（Phase D）：** toolcall-merged VLM 在 valid%/name-acc/args-acc 上勝過 base。
7. **迴路：** 內部行程儀表板顯示真實 VLM 產生的 `sort_can{orange}` 計畫；（可選）4090 執行 episode 至 `done`。
8. **無迴歸 / 無 n1d7 編輯：** AgentBot 現有套件保持通過；`Isaac-GR00T_n1d7` 未被觸及。

---

## 已解決的衝突（對比廢除的草稿）

| 廢除的草稿 | 現實（AgentBot） | 解決方案 |
|---|---|---|
| 自行實作 `SkillRegistry`、`SkillCall{skill,args}`、`/plan` 自由 JSON | `contracts/skills.py` `SkillCall{name,args,…}`，工具呼叫 | 廢除我們的；服務 OpenAI 工具呼叫；SkillCall 存在於 AgentBot |
| FastAPI brain 伺服器、mock executor、GR00T bridge | orchestrator + sim_session + 真實 VLA 已存在 | 廢除；我們只服務模型 |
| 微調 → 通用計畫 JSON | 需要 AgentBot 技能 tool_calls | 微調目標 = 在真實技能上的 `<tool_call>` |
| 範例技能 manifest | `registry.as_tool_schemas()` | 凍結一份副本（`agentbot_skills.sample.json`）供離線 gen/tests |

## 差距與開放問題

1. **攝影機幀來源（已解決 → Task C0）。** 決策：一個事件匯流排接縫——生產者發布 `CAMERA` 事件，`ingest` 寫入 `state["camera"]["frame"]`，Brain 透過 `frame_provider` 讀取。Sim = `sim_session` 生產（現在）；硬體 = ROS2→事件匯流排 bridge 生產相同事件（Phase 3），因此 Brain 與實施無關。唯一需要在實作時驗證的細節是確切的 IsaacLab 攝影機 obs 存取器 + 閒置泵不步進模擬是否能渲染幀（Task C0 步驟 6）。
2. **Cosmos-Reason2-2B 的 chat template 是否原生支援 `tools`？** 我們不依賴它——我們注入工具系統提示並解析 `<tool_call>`。若其 template *確實*支援 Hermes 工具 token，Phase D 可切換至原生格式；在 A5 期間驗證。
3. **兩個 repo 的提交。** 工作跨越 `Isaac-GR00T-VLM` 和 `agentbot`（其自己的 repo/submodule）。依照維護者規則，我只暫存；維護者提交各自的 repo。請確認兩個 repo 都在編輯範圍內。
4. **`place`/`home`/`pick`/`pour_water` gym id 在 AgentBot 中是骨架**（只有 `sort_can` 是確認的 IsaacLab 任務）。對於多步驟計畫的實際執行，這些任務必須存在——但 Brain/工具呼叫側是獨立的，現在可以建置/測試。

---

## 自我審查

- **規格覆蓋率：** "研究 agentbot 架構" → Context + 已驗證的 schema。"整合舊計畫" → Conflicts 表 + supersede（F1）。"完整計畫" → Phases A–F。"衝突" → Conflicts 表。"差距 + 詢問" → Gaps & open questions + 3 個 AskUserQuestion 決策已回答（shim / vision-on / tool-calling fine-tune）。每個答案都已反映：shim = Phase A；vision = Phase C；tool-calling fine-tune = Phase D。
- **佔位符掃描：** 每個程式碼步驟都有真實程式碼；指令有預期輸出；唯一的 TODO 類項目是 AgentBot 自身預先存在的骨架 gym id（已指出，非此處引入）。
- **型別一致性：** 伺服器發出 OpenAI `tool_calls[{function:{name,arguments(JSON str)}}]`；`Gr00tVLMClient.complete` 將其解析至 `SkillCall(name, args)`；`to_internal_tools` ↔ `as_tool_schemas()` `{name,description,input_schema}`；`<tool_call>{"name","arguments"}` 由 `gen_toolcall_data` 生成，由 `serve.toolcall.parse_tool_calls` 解析，由 `eval_toolcall.score` 評分——全程統一格式。

---

## 執行交接

兩種選項（我**只暫存；您提交**——且編輯跨越 `Isaac-GR00T-VLM` 和 `agentbot`）：
1. **子代理驅動（推薦）** — 每個任務一個全新子代理 + 每 phase 間的兩階段審查（superpowers:subagent-driven-development）。
2. **內聯** — 此 session，每個 phase 設置檢查點（superpowers:executing-plans）。

GPU 任務（A5、D3、D4、E2）透過 `pegasus.py` 在 H100/4090 上執行。您想選擇哪種方式——還是先調整計畫？
