# 系統設計文件：LiteLLM Proxy + APISIX AI Gateway → AWS Bedrock

> 版本：v1.1 | 日期：2026-03-27

---

## 目錄

1. [架構概覽](#1-架構概覽)
2. [APISIX Route 與 ai-proxy-multi 設計](#2-apisix-route-與-ai-proxy-multi-設計)
3. [Python Server 設計](#3-python-server-設計)
4. [Upstream Mapping DB 設計（SQLite）](#4-upstream-mapping-db-設計sqlite)
5. [Request 流程：從 APISIX 到 Bedrock](#5-request-流程從-apisix-到-bedrock)
6. [Streaming 與 Non-Streaming 處理](#6-streaming-與-non-streaming-處理)
7. [目錄結構](#7-目錄結構)

---

## 1. 架構概覽

```
Client
  │  POST /v1/chat/completions
  ▼
APISIX (3.15)
  │  ai-proxy-multi plugin
  │  • 多個 instance（每個對應不同 region 的 LiteLLM Python server）
  │  • roundrobin 負載均衡 + fallback on 429/5xx
  │  • 透過 auth.header 注入 X-LiteLLM-Instance: {instance_name}
  │
  ▼
Python LiteLLM Proxy Server（FastAPI + asyncio）
  │  POST /v1/chat/completions
  │  1. 讀取 X-LiteLLM-Instance header
  │  2. 查詢 SQLite → { bedrock_base_url, api_key, model_id }
  │  3. 呼叫 litellm.acompletion(
  │       model="bedrock/{model_id}",
  │       api_base=bedrock_base_url,   ← 覆寫 VPC endpoint
  │       api_key=api_key              ← Bearer token，跳過 SigV4
  │     )
  │
  ▼
AWS Bedrock（VPC Endpoint）
  POST https://{bedrock_base_url}/model/{model_id}/converse[-stream]
  Authorization: Bearer {api_key}
```

---

## 2. APISIX Route 與 ai-proxy-multi 設計

### 2.1 設計決策

`ai-proxy-multi` 本身是設計來「直接呼叫 LLM」的，但我們用 `provider: "openai-compatible"` + `override.endpoint` 指向 Python server，讓 Python server 負責實際呼叫 Bedrock。

**為什麼不讓 APISIX 直接打 Bedrock？**
- Bedrock 用 SigV4 簽名或 Bearer token，APISIX ai-proxy-multi 沒有內建 AWS SigV4 支援。
- VPC endpoint URL 與 model ID 的 mapping 邏輯需要在 Python server 處理。

**如何讓 Python server 知道選了哪個 instance？**
- 每個 APISIX instance 在 `auth.header` 裡放不同的 `X-LiteLLM-Instance` header 值。
- Python server 讀取此 header，查 DB 取得對應的 `bedrock_base_url`、`api_key`、`model_id`。

### 2.2 APISIX Route 設定（Admin API）

```bash
curl http://127.0.0.1:9180/apisix/admin/routes \
  -X PUT \
  -H "X-API-KEY: ${ADMIN_KEY}" \
  -d '{
    "id": "bedrock-chat-completion",
    "uri": "/v1/chat/completions",
    "methods": ["POST"],
    "plugins": {
      "ai-proxy-multi": {
        "fallback_strategy": ["rate_limiting", "http_5xx"],
        "balancer": {
          "algorithm": "roundrobin"
        },
        "logging": {
          "summaries": true
        },
        "instances": [
          {
            "name": "bedrock-us-east-1",
            "provider": "openai-compatible",
            "weight": 5,
            "priority": 1,
            "auth": {
              "header": {
                "X-LiteLLM-Instance": "bedrock-us-east-1"
              }
            },
            "options": {
              "model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
            },
            "override": {
              "endpoint": "http://litellm-proxy:8000/v1/chat/completions"
            }
          },
          {
            "name": "bedrock-ap-northeast-1",
            "provider": "openai-compatible",
            "weight": 5,
            "priority": 0,
            "auth": {
              "header": {
                "X-LiteLLM-Instance": "bedrock-ap-northeast-1"
              }
            },
            "options": {
              "model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
            },
            "override": {
              "endpoint": "http://litellm-proxy:8000/v1/chat/completions"
            }
          }
        ]
      }
    }
  }'
```

**重點說明：**

| 欄位 | 值 | 說明 |
|------|----|------|
| `provider` | `"openai-compatible"` | 讓 APISIX 把請求轉發給任意 OpenAI-compatible server |
| `override.endpoint` | `http://litellm-proxy:8000/v1/chat/completions` | 兩個 instance 指向同一個 Python server |
| `auth.header.X-LiteLLM-Instance` | 各自的 instance name | Python server 靠此 header 查 DB |
| `priority` | 1 / 0 | 1 優先；0 的 instance 作為 fallback |
| `fallback_strategy` | `["rate_limiting", "http_5xx"]` | Bedrock 429 或 5xx 時自動切換 instance |
| `options.model` | `"bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"` | 覆寫 client 傳來的 model（APISIX 注入） |

> **注意**：`options.model` 欄位會由 APISIX 覆寫 request body 中的 `model`，
> 所以 client 不管傳什麼 model 名稱，到 Python server 都是固定值。
> Python server 最終會忽略此值，改用從 DB 查到的 `model_id`。

---

## 3. Python Server 設計

### 3.1 技術選型

| 元件 | 選擇 | 原因 |
|------|------|------|
| Framework | FastAPI | 原生 async，OpenAPI 自動文件 |
| LLM client | `litellm.acompletion()` | async，支援 streaming，Bedrock 整合完善 |
| DB client | `aiosqlite` | async SQLite，不阻塞 event loop |
| Streaming | SSE via `StreamingResponse` | 與 OpenAI streaming 格式相容 |

### 3.2 核心邏輯

```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_litellm_instance: Optional[str] = Header(None, alias="X-LiteLLM-Instance"),
):
    if not x_litellm_instance:
        raise HTTPException(status_code=400, detail="Missing X-LiteLLM-Instance header")

    # 1. 查 DB
    cfg = await get_instance_config(x_litellm_instance)
    # cfg = { bedrock_base_url, api_key, model_id, aws_region_name }

    # 2. 呼叫 litellm（忽略 client 傳來的 model，改用 DB 查到的 model_id）
    litellm_kwargs = dict(
        model=f"bedrock/{cfg['model_id']}",
        messages=body.messages,
        stream=body.stream or False,
        api_base=cfg["bedrock_base_url"],   # 覆寫 VPC endpoint base URL
        api_key=cfg["api_key"],             # Bearer token，litellm 跳過 SigV4
        # 傳入 region_name 備用（litellm 用於 log，endpoint URL 已由 api_base 覆寫）
        aws_region_name=cfg["aws_region_name"],
    )
    # 轉發 client 的其他 optional params
    for field in ["temperature", "max_tokens", "top_p", "stop",
                  "tools", "tool_choice", "stream_options"]:
        val = getattr(body, field, None)
        if val is not None:
            litellm_kwargs[field] = val

    response = await litellm.acompletion(**litellm_kwargs)

    # 3. 回應
    if body.stream:
        return StreamingResponse(
            _stream_generator(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(response.model_dump())
```

### 3.3 litellm 的 `api_base` 與最終 URL 構成

**原始碼確認**（`converse_handler.py:352-363`）：

```python
# litellm 取得 base URL
endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
    api_base=api_base,   # ← 直接使用傳入值，不修改
    ...
)
# 再自行附加路徑
if stream:
    endpoint_url = f"{endpoint_url}/model/{modelId}/converse-stream"
else:
    endpoint_url = f"{endpoint_url}/model/{modelId}/converse"
```

**結論：**
- DB 中 `bedrock_base_url` 存 **base URL，不含 `/model/...`**。
- 例：`https://bedrock-runtime.us-east-1.amazonaws.com`（公有 endpoint）
  或 `https://vpce-xxx.bedrock-runtime.us-east-1.vpce.amazonaws.com`（VPC endpoint）
- litellm 自動構建完整 URL：`{bedrock_base_url}/model/{model_id}/converse[-stream]`

### 3.4 AWS Bedrock API Key 認證（Bearer Token）

**原始碼確認**（`base_aws_llm.py:1222-1244`）：

```python
def get_request_headers(self, ..., api_key: Optional[str] = None):
    if api_key is not None:
        aws_bearer_token = api_key
    else:
        aws_bearer_token = get_secret_str("AWS_BEARER_TOKEN_BEDROCK")

    if aws_bearer_token:
        # 直接用 Bearer token，完全跳過 SigV4 簽名
        headers["Authorization"] = f"Bearer {aws_bearer_token}"
    else:
        # fallback：SigV4 簽名（需要 access_key_id + secret）
        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
        ...
```

**結論：**
- 傳入 `api_key="your-bedrock-api-key"` 時，litellm 直接用 `Authorization: Bearer {api_key}`，**不需要 IAM access key / secret key**。
- SigV4 簽名邏輯完全繞過。

---

## 4. Upstream Mapping DB 設計（SQLite）

### 4.1 Schema

```sql
CREATE TABLE IF NOT EXISTS bedrock_instances (
    instance_name    TEXT PRIMARY KEY,
    aws_region_name  TEXT NOT NULL,
    -- base URL only，不含 /model/... 路徑
    -- 例：https://bedrock-runtime.us-east-1.amazonaws.com
    -- 例：https://vpce-xxx.bedrock-runtime.us-east-1.vpce.amazonaws.com
    bedrock_base_url TEXT NOT NULL,
    -- Bedrock API Key（Bearer token）
    api_key          TEXT NOT NULL,
    -- 完整 Bedrock model ID，不含 "bedrock/" prefix
    -- litellm 呼叫時會組成 "bedrock/{model_id}"
    model_id         TEXT NOT NULL DEFAULT 'anthropic.claude-sonnet-4-5-20250929-v1:0',
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 範例：us-east-1（主要 region）
INSERT INTO bedrock_instances
    (instance_name, aws_region_name, bedrock_base_url, api_key, model_id)
VALUES (
    'bedrock-us-east-1',
    'us-east-1',
    'https://bedrock-runtime.us-east-1.amazonaws.com',
    'bedrock-api-key-us-east-1-placeholder',
    'anthropic.claude-sonnet-4-5-20250929-v1:0'
);

-- 範例：ap-northeast-1（fallback region）
INSERT INTO bedrock_instances
    (instance_name, aws_region_name, bedrock_base_url, api_key, model_id)
VALUES (
    'bedrock-ap-northeast-1',
    'ap-northeast-1',
    'https://bedrock-runtime.ap-northeast-1.amazonaws.com',
    'bedrock-api-key-ap-northeast-1-placeholder',
    'anthropic.claude-sonnet-4-5-20250929-v1:0'
);
```

### 4.2 DB 存取層

```python
# db.py
import aiosqlite
from typing import Optional

DB_PATH = "instances.db"

# 啟動時快取，避免每 request 都 I/O
_cache: dict[str, dict] = {}

async def init_cache():
    """Server 啟動時呼叫，載入所有 active instances"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bedrock_instances WHERE is_active = 1"
        ) as cursor:
            rows = await cursor.fetchall()
    _cache.clear()
    for row in rows:
        _cache[row["instance_name"]] = dict(row)

async def get_instance_config(instance_name: str) -> dict:
    """從快取取得 instance config，查不到時拋 ValueError"""
    if instance_name in _cache:
        return _cache[instance_name]
    # fallback：直接查 DB（config 更新時快取尚未刷新）
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bedrock_instances WHERE instance_name = ? AND is_active = 1",
            (instance_name,)
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Instance not found or inactive: {instance_name}")
    return dict(row)
```

### 4.3 設計考量

- **快取**：Server 啟動時 `await init_cache()` 把 active instances 全部載入記憶體，正常路徑不需要 DB I/O。
- **憑證安全**：`api_key` 目前明文存 DB（MVP）。生產環境應從 AWS Secrets Manager 或環境變數讀取，DB 只存 secret 的 key name。
- **擴展性**：若未來需要動態更新 config（不重啟 server），可以加一個 `POST /admin/reload` endpoint 呼叫 `await init_cache()` 刷新快取。

---

## 5. Request 流程：從 APISIX 到 Bedrock

```
1. Client
   POST /v1/chat/completions
   Body: { "model": "claude-4", "messages": [...], "stream": false }

2. APISIX ai-proxy-multi
   • roundrobin + priority → 選擇 "bedrock-us-east-1"
   • 覆寫 model → "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
   • 注入 header → X-LiteLLM-Instance: bedrock-us-east-1
   • 轉發 POST → http://litellm-proxy:8000/v1/chat/completions

3. Python Server
   • 讀取 X-LiteLLM-Instance: "bedrock-us-east-1"
   • 查快取 → {
       bedrock_base_url: "https://bedrock-runtime.us-east-1.amazonaws.com",
       api_key: "brk-...",
       model_id: "anthropic.claude-sonnet-4-5-20250929-v1:0",
       aws_region_name: "us-east-1"
     }
   • 呼叫 litellm.acompletion(
       model="bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
       api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
       api_key="brk-..."
     )

4. litellm（內部）
   • get_llm_provider() → provider="bedrock"
   • get_runtime_endpoint(api_base=...) → 原樣返回 base URL
   • api_key 不為 None → 設定 Authorization: Bearer brk-...（跳過 SigV4）
   • 最終 URL：https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-5-20250929-v1:0/converse

5. AWS Bedrock
   • 驗證 Bearer token，處理請求，回傳 response

6. litellm → Python Server
   • 轉換為 OpenAI-compatible ModelResponse

7. Python Server → APISIX → Client
   • non-streaming：JSON response
   • streaming：SSE text/event-stream
```

---

## 6. Streaming 與 Non-Streaming 處理

### 6.1 Non-Streaming

```python
response = await litellm.acompletion(..., stream=False)
# response: litellm.ModelResponse（OpenAI-compatible）
return JSONResponse(response.model_dump())
```

### 6.2 Streaming

litellm `acompletion(stream=True)` 回傳 `AsyncGenerator`，每個 chunk 是 OpenAI streaming delta 格式。

```python
async def _stream_generator(litellm_stream):
    async for chunk in litellm_stream:
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"

# 回傳
return StreamingResponse(
    _stream_generator(response),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # 避免 nginx/APISIX 緩衝
    },
)
```

**litellm 內部 streaming 路徑：**
- `acompletion(stream=True)` → `async_streaming()` → endpoint 改為 `/converse-stream`
- 回傳 `CustomStreamWrapper`（async iterator），逐 chunk yield OpenAI-compatible delta

### 6.3 Error Handling

| 情境 | Python Server | APISIX 行為 |
|------|---------------|-------------|
| `X-LiteLLM-Instance` 缺失或查不到 | HTTP 400 | — |
| Bedrock 回 429 | `litellm.RateLimitError` → HTTP 429 | 觸發 fallback，切換另一 instance |
| Bedrock 回 5xx | `litellm.APIError` → HTTP 502 | 觸發 fallback |
| 全部 instance 失敗 | — | 回傳最後一次錯誤給 client |
| Streaming 中途斷線 | `async for` loop 結束，generator 自然關閉 | — |

---

## 7. 目錄結構

```
litellm_proxy/
├── main.py              # FastAPI app + route handlers
├── db.py                # SQLite 存取層（aiosqlite + 記憶體快取）
├── models.py            # Pydantic request/response 模型
├── init_db.py           # 建立 schema 與插入初始測試資料
├── instances.db         # SQLite DB（git-ignored）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml   # APISIX + etcd + Python server
└── apisix/
    ├── config.yaml      # APISIX 全域設定
    └── routes/
        └── bedrock-chat.sh  # Admin API 建立 route 的腳本
```
