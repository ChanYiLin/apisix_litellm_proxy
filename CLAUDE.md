# CLAUDE.md — AI 工作接手指引

本文件提供後續 AI 工作者接手此專案所需的完整背景與行為規範。

---

## 專案目的

建立一個 **LiteLLM Proxy + APISIX AI Gateway** 的整合系統，將 OpenAI-compatible 的 `/v1/chat/completions` 請求路由到 **AWS Bedrock Claude 模型**，支援：
- 多 region 備援（透過 APISIX ai-proxy-multi）
- 自動 failover（429 / 5xx）
- 串流（SSE）與非串流回應
- Bedrock API Key（Bearer token）認證，不使用 SigV4

詳細設計請閱讀 [design.md](design.md)。

---

## 關鍵技術決策（已確認，勿更動）

### 1. APISIX → Python Server 的 instance 識別

APISIX 的 `ai-proxy-multi` plugin 透過 `auth.header` 注入 `X-LiteLLM-Instance: {instance_name}` header。Python server 讀取此 header 後查詢 SQLite，取得對應的 `bedrock_base_url`、`api_key`、`model_id`。

**不使用 path-based routing**（例如 `/v1/chat/completions/bedrock-us-east-1`），因為這樣 client 端的 path 就需要知道 instance 名稱。

### 2. Bedrock 認證：Bearer token（API Key），非 SigV4

呼叫 litellm 時只傳 `api_key`，**不傳** `aws_access_key_id` / `aws_secret_access_key`。

```python
await litellm.acompletion(
    model=f"bedrock/{cfg['model_id']}",
    api_base=cfg["bedrock_base_url"],
    api_key=cfg["api_key"],           # ← Bearer token
    aws_region_name=cfg["aws_region_name"],
    ...
)
```

litellm 源碼確認（`base_aws_llm.py:get_request_headers()`）：`api_key` 不為 None 時直接設 `Authorization: Bearer {api_key}`，SigV4 邏輯完全繞過。

### 3. `api_base` 只傳 base URL

DB 的 `bedrock_base_url` 存 **不含 `/model/...` 的 base URL**（例如 `https://bedrock-runtime.us-east-1.amazonaws.com`）。litellm 會自動附加 `/model/{model_id}/converse[-stream]`。

源碼：`converse_handler.py:352-363`。

### 4. model_id 存在 DB 中

每個 instance 有獨立的 `model_id`（不從 client 的 request body 取）。Python server 組成 `bedrock/{model_id}` 傳給 litellm，**忽略** APISIX 覆寫後的 `model` 欄位。

---

## 檔案說明

| 檔案 | 職責 |
|------|------|
| `main.py` | FastAPI app。啟動時呼叫 `init_db()`。處理 `/v1/chat/completions`、`/health`、`/admin/reload` |
| `db.py` | SQLite 存取 + 記憶體快取（`_cache` dict）。`init_db()` 建表並載入快取；`reload_cache()` 重新載入；`get_instance_config()` 查快取再 fallback DB |
| `models.py` | `ChatMessage`、`ChatCompletionRequest` Pydantic models，`extra="allow"` 讓未知欄位通過 |
| `init_db.py` | 獨立腳本，建 schema + INSERT OR IGNORE seed data。Docker CMD 在啟動 uvicorn 前先執行此腳本 |
| `apisix/config.yaml` | APISIX 全域設定，啟用 `ai-proxy-multi` 等 plugins |
| `apisix/routes/setup-route.sh` | 用 APISIX Admin API 建立 `/v1/chat/completions` route，可透過環境變數 `APISIX_ADMIN_KEY`、`APISIX_ADMIN_URL`、`LITELLM_PROXY_URL` 覆寫預設值 |

---

## DB Schema

```sql
CREATE TABLE IF NOT EXISTS bedrock_instances (
    instance_name    TEXT PRIMARY KEY,
    aws_region_name  TEXT NOT NULL,
    bedrock_base_url TEXT NOT NULL,  -- base URL, 不含 /model/...
    api_key          TEXT NOT NULL,  -- Bedrock Bearer token
    model_id         TEXT NOT NULL DEFAULT 'anthropic.claude-sonnet-4-5-20250929-v1:0',
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
)
```

---

## 新增 Region 的完整流程

1. 在 SQLite 新增一筆 `bedrock_instances`（直接 INSERT 或修改 `init_db.py`）
2. 在 `setup-route.sh` 的 `instances` 陣列加入新 instance，設定：
   - `name`: 與 `instance_name` 相同
   - `auth.header.X-LiteLLM-Instance`: 與 `instance_name` 相同
   - `priority` / `weight`: 依備援策略調整
3. 重新執行 `./apisix/routes/setup-route.sh`
4. 呼叫 `POST http://localhost:8000/admin/reload` 刷新快取

---

## 已知限制與未來改進方向

- **憑證安全**：`api_key` 目前明文存 SQLite（MVP 用）。生產環境應從 AWS Secrets Manager 讀取，DB 只存 secret name。
- **快取一致性**：`_cache` 為模組級 dict，若多 worker 部署（`uvicorn --workers N`）需改用外部快取（Redis）。目前單 worker 無此問題。
- **APISIX auth.header 的自定義 header**：設計上假設 `ai-proxy-multi` 的 `auth.header` 可以夾帶任意 header（包含 `X-LiteLLM-Instance`）。若實測發現只有 `Authorization` 有效，備案是改用 path-based routing：每個 instance 的 `override.endpoint` 改為 `http://litellm-proxy:8000/v1/chat/completions/{instance_name}`，server 端改成 path parameter。
- **Streaming 中的 error handling**：streaming 中途若 Bedrock 出錯，目前 generator 會靜默結束。可考慮在 `_stream_generator` 加 try/except，yield 一個 error event。

---

## 開發注意事項

- Python 版本：3.11+（`dict[str, dict]` 型別語法需要 3.9+）
- 所有 DB 操作都是 async（`aiosqlite`），不可在 async context 中用 sync 方式呼叫
- `litellm.drop_params = True` 已在 module level 設定，不支援的參數會被靜默丟棄
- APISIX 的 `ai-proxy-multi` 使用 `options.model` 覆寫 client 傳來的 model，但 Python server 會忽略它，改用 DB 查到的 `model_id`
- 目前 target model 固定為 `anthropic.claude-sonnet-4-5-20250929-v1:0`（Claude Sonnet 4.5）
