# LiteLLM Multi-Provider Proxy

透過 APISIX AI Gateway + LiteLLM，將 OpenAI-compatible 的 `/v1/chat/completions` 請求路由到多種 LLM 雲端服務（AWS Bedrock、Google AI Studio、Google Vertex AI），支援多 region 備援、自動 failover、串流回應。

---

## 架構

```
Client
  │  POST /v1/chat/completions  {"model": "gemini-2.0-flash", ...}
  ▼
APISIX (port 9080)
  │  per-model route matching（vars: post_arg_model）
  │  ai-proxy-multi plugin
  │  • roundrobin 負載均衡（多 region / 多 instance）
  │  • 自動 failover（429 / 5xx）
  │  • 注入 X-LiteLLM-Instance header
  ▼
Python LiteLLM Proxy（port 8000）
  │  讀取 X-LiteLLM-Instance → 查 MariaDB（llm_instances 表）
  │  依 provider 組出對應 litellm 參數
  │  呼叫 litellm.acompletion()
  ▼
LLM Provider
  ├── AWS Bedrock（bedrock-runtime endpoint，Bearer token 認證）
  ├── Google AI Studio（generativelanguage.googleapis.com，API Key）
  └── Google Vertex AI（aiplatform.googleapis.com，Service Account OAuth2）
```

### 元件

| 元件 | 角色 |
|------|------|
| **APISIX 3.x** | API Gateway，依 `model` 欄位路由至對應 upstream，負責 failover |
| **ai-proxy-multi plugin** | 多 instance 負載均衡，支援 priority + fallback strategy |
| **Python FastAPI Server** | 查詢 instance 設定，依 provider 組出 litellm 參數 |
| **LiteLLM** | 統一 LLM 呼叫介面，處理各 provider 的 API 格式轉換與認證 |
| **MariaDB 11** | 儲存各 instance 的 provider、endpoint、credentials、model ID |

---

## 支援的 Provider

| Provider | `model` 值範例 | 認證方式 | 備援策略 |
|----------|--------------|---------|---------|
| AWS Bedrock | `claude-sonnet-4-5` | Bearer token（API Key） | 多 region（US / AP） |
| Google AI Studio | `gemini-2.0-flash`、`gemini-2.5-pro` | API Key | 單一全球端點 |
| Google Vertex AI | `vertex/gemini-2.0-flash` | Service Account JSON（OAuth2） | 多 region（US / Asia） |

---

## 目錄結構

```
litellm_proxy/
├── main.py                  # FastAPI server（route handlers + _build_litellm_kwargs）
├── db.py                    # MariaDB 存取層 + 記憶體快取
├── models.py                # Pydantic request models
├── init_db.py               # DB 初始化腳本（建 schema + seed data）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml       # APISIX + etcd + litellm-proxy + mariadb
├── .gitignore
├── design.md                # 完整系統設計文件
└── apisix/
    ├── config.yaml          # APISIX 全域設定
    └── routes/
        └── setup-route.sh   # 建立 APISIX per-model routes 的腳本
```

---

## 快速開始

### 前置需求

- Docker & Docker Compose
- 至少一種 provider 的憑證（Bedrock API Key、Gemini API Key 或 GCP Service Account JSON）

### 1. 設定憑證

編輯 `init_db.py`，將 seed data 中的 `REPLACE_ME_*` 替換為真實憑證：

**Bedrock：**
```python
{
    "instance_name":   "bedrock-us-east-1",
    "bedrock_api_key": "your-bedrock-bearer-token",
    "bedrock_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
    ...
}
```

**Gemini AI Studio：**
```python
{
    "instance_name":  "gemini-flash-global",
    "gemini_api_key": "AIza...",
    ...
}
```

**Vertex AI：**
```python
{
    "instance_name":      "vertex-flash-us",
    "vertex_project":     "my-gcp-project",
    "vertex_location":    "us-central1",
    "vertex_credentials": '{"type":"service_account","project_id":"...","private_key":"..."}',
    ...
}
```

### 2. 啟動服務

```bash
docker-compose up -d
```

服務啟動順序：etcd + mariadb → litellm-proxy → apisix

### 3. 建立 APISIX Routes

等待 APISIX 啟動完成後（約 15–30 秒），執行：

```bash
./apisix/routes/setup-route.sh
```

成功會印出：
```
[OK] bedrock-claude-sonnet-4-5 route created
[OK] gemini-2-0-flash route created
[OK] gemini-2-5-pro route created
[OK] vertex-gemini-2-0-flash route created
[OK] health route created
```

### 4. 測試

**Bedrock Claude（non-streaming）：**
```bash
curl -X POST http://localhost:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

**Gemini AI Studio（streaming）：**
```bash
curl -X POST http://localhost:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

**Vertex AI：**
```bash
curl -X POST http://localhost:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex/gemini-2.0-flash",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

**健康檢查：**
```bash
curl http://localhost:9080/health
```

---

## API

### `POST /v1/chat/completions`

OpenAI-compatible Chat Completions endpoint。

**Request Body（OpenAI 格式）：**

| 欄位 | 類型 | 說明 |
|------|------|------|
| `model` | string | **必填**。決定路由目標（見支援的 Provider 表）|
| `messages` | array | 必填。對話歷史 |
| `stream` | boolean | 選填。`true` 啟用 SSE streaming |
| `temperature` | float | 選填 |
| `max_tokens` | int | 選填 |
| `top_p` | float | 選填 |
| `stop` | string/array | 選填 |
| `tools` | array | 選填。Function calling |
| `tool_choice` | string/object | 選填 |

**Response：**
- Non-streaming：OpenAI-compatible JSON（`ChatCompletion` 格式）
- Streaming：`text/event-stream`，每行 `data: {...}\n\n`，結尾 `data: [DONE]\n\n`

### `GET /health`

回傳 `{"status": "ok"}`。

### `POST /admin/reload`

重新從 MariaDB 載入 instance 設定到記憶體快取。更新 DB 後不需重啟 server：

```bash
curl -X POST http://localhost:8000/admin/reload
```

---

## Instance Mapping（MariaDB `llm_instances`）

資料存在 MariaDB `litellm` 資料庫（Docker volume `mariadb-data`）。

**Schema 共用欄位：**

| 欄位 | 類型 | 說明 |
|------|------|------|
| `instance_name` | `VARCHAR(255) PK` | 唯一識別名稱，對應 APISIX 注入的 `X-LiteLLM-Instance` header 值 |
| `provider` | `ENUM` | `bedrock` / `gemini` / `vertex_ai` |
| `model_id` | `VARCHAR(255)` | Provider-specific model ID，litellm 實際使用 |
| `display_model` | `VARCHAR(255)` | Client 送的 model 名稱，APISIX `post_arg_model` 匹配用 |
| `is_active` | `TINYINT(1)` | `1` = 啟用，`0` = 停用 |

**Provider-specific 欄位：**

| Provider | 欄位 | 說明 |
|---------|------|------|
| `bedrock` | `aws_region_name`, `bedrock_base_url`, `bedrock_api_key` | Bedrock Bearer token 認證 |
| `gemini` | `gemini_api_key`, `gemini_api_base`（可選） | Google AI Studio API Key |
| `vertex_ai` | `vertex_project`, `vertex_location`, `vertex_credentials`, `vertex_api_base`（可選） | GCP SA JSON + region |

---

## APISIX 流量設計

### Per-model Route 匹配

每個 route 使用 `vars: [["post_arg_model", "==", "<model>"]]` 匹配 request body 的 `model` 欄位，路由到對應的 `ai-proxy-multi` 設定：

| Route ID | 匹配 `model` 值 | Instances | Failover |
|---|---|---|---|
| `bedrock-claude-sonnet-4-5` | `claude-sonnet-4-5` | bedrock-us-east-1（priority 1）, bedrock-ap-northeast-1（priority 0） | 是 |
| `gemini-2-0-flash` | `gemini-2.0-flash` | gemini-flash-global | 否 |
| `gemini-2-5-pro` | `gemini-2.5-pro` | gemini-pro-global | 否 |
| `vertex-gemini-2-0-flash` | `vertex/gemini-2.0-flash` | vertex-flash-us（priority 1）, vertex-flash-asia（priority 0） | 是 |

### Failover 條件

`fallback_strategy: ["rate_limiting", "http_5xx"]`

- Provider 回 `429 Too Many Requests` → 自動切換到下一個 instance
- Provider 回 `5xx` → 自動切換到下一個 instance

---

## 新增 Provider / Model

1. `INSERT INTO llm_instances`（指定 `provider`、`display_model`、對應 credentials 欄位）
2. 在 `setup-route.sh` 新增一個 route block，`vars` 條件填 `display_model` 值
3. 重新執行 `./apisix/routes/setup-route.sh`
4. `POST http://localhost:8000/admin/reload`

---

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `DB_HOST` | `localhost` | MariaDB 主機名稱 |
| `DB_PORT` | `3306` | MariaDB 連接埠 |
| `DB_USER` | `litellm` | MariaDB 使用者名稱 |
| `DB_PASSWORD` | _(空)_ | MariaDB 密碼 |
| `DB_NAME` | `litellm` | MariaDB 資料庫名稱 |
| `APISIX_ADMIN_KEY` | `edd1c9f034335f136f87ad84b625c8f1` | APISIX Admin API key |
| `APISIX_ADMIN_URL` | `http://localhost:9180` | APISIX Admin API URL |
| `LITELLM_PROXY_URL` | `http://litellm-proxy:8000` | Python server URL（供 APISIX 設定使用） |

---

## 技術細節

### litellm provider 對照

| Provider | litellm model prefix | 認證參數 | 端點 |
|---------|---------------------|---------|------|
| `bedrock` | `bedrock/{model_id}` | `api_key`（Bearer token） | Bedrock runtime endpoint |
| `gemini` | `gemini/{model_id}` | `api_key`（API Key，放 URL query param） | `generativelanguage.googleapis.com` |
| `vertex_ai` | `vertex_ai/{model_id}` | `vertex_credentials`（SA JSON → OAuth2） | `{location}-aiplatform.googleapis.com` |

### Gemini vs Vertex AI 認證差異

- **Gemini AI Studio**：API Key 直接可用，不需 OAuth。litellm 用 `gemini/` prefix，key 放 URL query param。
- **Vertex AI**：強制 OAuth2，**不接受** API Key。需傳 Service Account JSON（`vertex_credentials`），litellm 內部用 google-auth 取 Bearer token。

### Bedrock `api_base` 只傳 base URL

`bedrock_base_url` 存 **不含 `/model/...` 的 base URL**。litellm 自動附加路徑：
```
{api_base}/model/{model_id}/converse          # non-streaming
{api_base}/model/{model_id}/converse-stream   # streaming
```
