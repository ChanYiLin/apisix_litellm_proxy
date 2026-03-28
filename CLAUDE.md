# CLAUDE.md — AI 工作接手指引

本文件提供後續 AI 工作者接手此專案所需的完整背景與行為規範。

---

## 專案目的

建立一個 **LiteLLM Proxy + APISIX AI Gateway** 的整合系統，將 OpenAI-compatible 的 `/v1/chat/completions` 請求依 `model` 欄位路由到對應的 LLM 雲端服務，支援：
- **AWS Bedrock**：Claude 系列模型，Bearer token 認證
- **Google AI Studio**：Gemini 系列模型，API Key 認證
- **Google Vertex AI**：Gemini 系列模型，Service Account OAuth2 認證
- 多 region 備援（透過 APISIX ai-proxy-multi）
- 自動 failover（429 / 5xx）
- 串流（SSE）與非串流回應

詳細設計請閱讀 [design.md](design.md)。

---

## 關鍵技術決策（已確認，勿更動）

### 1. APISIX → Python Server 的 instance 識別

APISIX 的 `ai-proxy-multi` plugin 透過 `auth.header` 注入 `X-LiteLLM-Instance: {instance_name}` header。Python server 讀取此 header 後查詢 MariaDB `llm_instances` 表，取得對應 provider 與 credentials。

**不使用 path-based routing**（例如 `/v1/chat/completions/gemini-flash`），因為這樣 client 端的 path 就需要知道 instance 名稱。

### 2. APISIX 依 `model` 欄位路由

每個 model group 對應一個 APISIX route，使用 `vars: [["post_arg_model", "==", "<display_model>"]]` 匹配 request body 的 `model` 欄位。APISIX 不需要解析 JSON body 的 Lua 腳本，直接用 `post_arg_model` built-in 變數。

### 3. `display_model` vs `model_id` 分離

- `display_model`：client 送的 human-readable model 名稱（e.g. `claude-sonnet-4-5`、`gemini-2.0-flash`），APISIX `vars` 用此匹配 route。
- `model_id`：Provider-specific 的真實 ID（e.g. `anthropic.claude-sonnet-4-5-20250929-v1:0`），Python server 查 DB 後傳給 litellm，**忽略** client body 中的 `model` 欄位。

### 4. Bedrock 認證：Bearer token（API Key），非 SigV4

呼叫 litellm 時只傳 `api_key`，**不傳** `aws_access_key_id` / `aws_secret_access_key`。

```python
litellm.acompletion(
    model=f"bedrock/{cfg['model_id']}",
    api_base=cfg["bedrock_base_url"],
    api_key=cfg["bedrock_api_key"],     # ← Bearer token
    aws_region_name=cfg["aws_region_name"],
)
```

litellm 源碼確認（`base_aws_llm.py:get_request_headers()`）：`api_key` 不為 None 時直接設 `Authorization: Bearer {api_key}`，SigV4 邏輯完全繞過。

### 5. Gemini AI Studio：API Key，走 `gemini/` prefix

```python
litellm.acompletion(
    model=f"gemini/{cfg['model_id']}",
    api_key=cfg["gemini_api_key"],      # ← AIza... API Key
)
```

litellm 用 `gemini/` prefix 時走 Google AI Studio endpoint（`generativelanguage.googleapis.com`），API Key 放 URL query param，不需 OAuth。

### 6. Vertex AI：Service Account JSON，走 `vertex_ai/` prefix

```python
litellm.acompletion(
    model=f"vertex_ai/{cfg['model_id']}",
    vertex_project=cfg["vertex_project"],
    vertex_location=cfg["vertex_location"],
    vertex_credentials=cfg["vertex_credentials"],  # ← SA JSON string
)
```

`vertex_ai/` prefix 強制走 OAuth2，**不支援 API Key**。`vertex_credentials` 傳 Service Account JSON 字串（非檔案路徑），litellm 內部用 google-auth 取 Bearer token。`bedrock_base_url` 同理，只傳 base URL，litellm 自動附加路徑。

### 7. `api_base` 只傳 base URL（Bedrock）

DB 的 `bedrock_base_url` 存 **不含 `/model/...` 的 base URL**（例如 `https://bedrock-runtime.us-east-1.amazonaws.com`）。litellm 會自動附加 `/model/{model_id}/converse[-stream]`。源碼：`converse_handler.py:352-363`。

---

## 檔案說明

| 檔案 | 職責 |
|------|------|
| `main.py` | FastAPI app。啟動時呼叫 `init_db()`。`_build_litellm_kwargs()` 依 provider 組出 litellm 參數。處理 `/v1/chat/completions`、`/health`、`/admin/reload` |
| `db.py` | MariaDB 存取 + 記憶體快取（`_cache` dict）。操作 `llm_instances` 表。`init_db()` 建表並載入快取；`reload_cache()` 重新載入；`get_instance_config()` 查快取再 fallback DB |
| `models.py` | `ChatMessage`、`ChatCompletionRequest` Pydantic models，`extra="allow"` 讓未知欄位通過 |
| `init_db.py` | 獨立腳本，建 schema + INSERT IGNORE seed data（含三個 provider 的範例）。Docker CMD 在啟動 uvicorn 前先執行此腳本 |
| `apisix/config.yaml` | APISIX 全域設定，啟用 `ai-proxy-multi` 等 plugins |
| `apisix/routes/setup-route.sh` | 用 APISIX Admin API 建立 per-model routes（含 `vars` 條件）。可透過環境變數 `APISIX_ADMIN_KEY`、`APISIX_ADMIN_URL`、`LITELLM_PROXY_URL` 覆寫預設值 |

---

## DB Schema（MariaDB `llm_instances`）

```sql
CREATE TABLE IF NOT EXISTS llm_instances (
    instance_name    VARCHAR(255) PRIMARY KEY,
    provider         ENUM('bedrock', 'gemini', 'vertex_ai') NOT NULL,
    model_id         VARCHAR(255) NOT NULL,   -- litellm 實際使用的 provider-specific ID
    display_model    VARCHAR(255) NOT NULL,   -- client 送的 model 名稱，APISIX post_arg_model 匹配用
    is_active        TINYINT(1) NOT NULL DEFAULT 1,

    -- Bedrock（provider='bedrock' 時必填）
    aws_region_name  VARCHAR(100),
    bedrock_base_url TEXT,          -- base URL，不含 /model/...
    bedrock_api_key  TEXT,          -- Bedrock Bearer token

    -- Gemini AI Studio（provider='gemini' 時必填）
    gemini_api_key   TEXT,          -- AIza... API Key
    gemini_api_base  TEXT,          -- 可選，自訂 proxy endpoint

    -- Vertex AI（provider='vertex_ai' 時必填）
    vertex_project     VARCHAR(255),
    vertex_location    VARCHAR(100),  -- e.g. us-central1
    vertex_credentials TEXT,          -- Service Account JSON 字串，直接存
    vertex_api_base    TEXT,          -- 可選，PSC endpoint

    created_at       DATETIME NOT NULL DEFAULT NOW(),
    updated_at       DATETIME NOT NULL DEFAULT NOW()
)
```

> **舊表 `bedrock_instances`** 已廢棄，在 `db.py` 和 `init_db.py` 中以 comment 保留，標注 DEPRECATED。

---

## 新增 Provider / Model 的完整流程

1. 在 MariaDB 新增一筆 `llm_instances`（直接 INSERT 或修改 `init_db.py` 的 `_SEED_ROWS`），指定：
   - `provider`：`bedrock` / `gemini` / `vertex_ai`
   - `display_model`：client 將送的 model 名稱（決定 APISIX route 的 `vars` 條件）
   - 對應 provider 的 credentials 欄位
2. 在 `setup-route.sh` 新增一個 route block，`vars` 條件填 `display_model` 的值，`instances` 填對應的 instance（含 `X-LiteLLM-Instance` header 注入）
3. 重新執行 `./apisix/routes/setup-route.sh`
4. 呼叫 `POST http://localhost:8000/admin/reload` 刷新快取

---

## 已知限制與未來改進方向

- **憑證安全**：所有 credentials 目前明文存 MariaDB（MVP 用）。生產環境應從 AWS Secrets Manager / GCP Secret Manager 讀取，DB 只存 secret name。MariaDB 密碼透過 docker-compose env 傳入，生產環境應改用 Docker secrets 或外部 secret manager。
- **快取一致性**：`_cache` 為模組級 dict，若多 worker 部署（`uvicorn --workers N`）需改用外部快取（Redis）。目前單 worker 無此問題。
- **Streaming 中的 error handling**：streaming 中途若 provider 出錯，目前 generator 會靜默結束。可考慮在 `_stream_generator` 加 try/except，yield 一個 error event。
- **Vertex AI credentials rotation**：SA JSON 目前靜態存 DB，litellm 每次請求都重新做 OAuth token exchange。若需優化可在 proxy 層快取 access token（有效期 1 小時）。

---

## 開發注意事項

- Python 版本：3.11+（`match/case` 語法需要 3.10+，`dict[str, dict]` 型別語法需要 3.9+）
- 所有 DB 操作都是 async（`aiomysql`），不可在 async context 中用 sync 方式呼叫
- MariaDB 連線參數透過環境變數設定：`DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`
- SQL 佔位符使用 `%s`（MySQL 格式），**非** SQLite 的 `?`
- `litellm.drop_params = True` 已在 module level 設定，不支援的參數會被靜默丟棄
- Python server 永遠忽略 client body 中的 `model` 欄位，改用 DB 查到的 `model_id`
- Vertex AI `vertex_credentials` 傳字串（JSON content），不是檔案路徑；litellm 源碼（`vertex_llm_base.py:load_auth()`）會先用 `os.path.exists()` 判斷，不存在才當 JSON string 解析
