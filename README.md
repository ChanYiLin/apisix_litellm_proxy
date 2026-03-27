# LiteLLM Bedrock Proxy

透過 APISIX AI Gateway + LiteLLM，將 OpenAI-compatible 的 `/v1/chat/completions` 請求路由到 AWS Bedrock Claude 模型，支援多 region 備援、自動 failover、串流回應。

---

## 架構

```
Client
  │  POST /v1/chat/completions
  ▼
APISIX (port 9080)
  │  ai-proxy-multi plugin
  │  • roundrobin 負載均衡（多 region）
  │  • 自動 failover（429 / 5xx）
  │  • 注入 X-LiteLLM-Instance header
  ▼
Python LiteLLM Proxy（port 8000）
  │  讀取 X-LiteLLM-Instance → 查 SQLite
  │  呼叫 litellm.acompletion()
  ▼
AWS Bedrock（VPC Endpoint）
  各 region 的 bedrock-runtime endpoint
```

### 元件

| 元件 | 角色 |
|------|------|
| **APISIX 3.x** | API Gateway，負責流量分配與 failover |
| **ai-proxy-multi plugin** | 多 upstream 負載均衡，支援 priority + fallback strategy |
| **Python FastAPI Server** | 查詢 instance mapping，呼叫 litellm |
| **LiteLLM** | 統一 LLM 呼叫介面，處理 Bedrock API 格式轉換與 Bearer token 認證 |
| **SQLite** | 儲存各 instance 的 Bedrock endpoint URL、API Key、model ID |

---

## 目錄結構

```
litellm_proxy/
├── main.py                  # FastAPI server（route handlers）
├── db.py                    # SQLite 存取層 + 記憶體快取
├── models.py                # Pydantic request models
├── init_db.py               # DB 初始化腳本（建 schema + seed data）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml       # APISIX + etcd + litellm-proxy
├── .gitignore
├── design.md                # 完整系統設計文件
└── apisix/
    ├── config.yaml          # APISIX 全域設定
    └── routes/
        └── setup-route.sh   # 建立 APISIX route 的腳本
```

---

## 快速開始

### 前置需求

- Docker & Docker Compose
- AWS Bedrock API Key（Bearer token）

### 1. 設定 Bedrock API Key

編輯 `init_db.py`，將 seed data 中的 `REPLACE_ME_*` 換成真實的 Bedrock API Key：

```python
# init_db.py
_SEED_ROWS = [
    {
        "instance_name": "bedrock-us-east-1",
        "bedrock_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "api_key": "your-actual-bedrock-api-key-us-east-1",
        ...
    },
    ...
]
```

若使用 VPC Endpoint，將 `bedrock_base_url` 替換為 VPC endpoint 的 base URL，例如：
```
https://vpce-xxxxxxxxxxxx-xxxxxxxx.bedrock-runtime.us-east-1.vpce.amazonaws.com
```

### 2. 啟動服務

```bash
docker-compose up -d
```

服務啟動順序：etcd → litellm-proxy → apisix

### 3. 建立 APISIX Route

等待 APISIX 啟動完成後（約 15–30 秒），執行：

```bash
./apisix/routes/setup-route.sh
```

成功會印出：
```
[OK] bedrock-chat-completion route created
[OK] health route created
```

### 4. 測試

**Non-streaming：**
```bash
curl -X POST http://localhost:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

**Streaming：**
```bash
curl -X POST http://localhost:9080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
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
| `messages` | array | 必填。對話歷史 |
| `model` | string | 選填。由 APISIX 覆寫，可忽略 |
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

回傳 `{"status": "ok"}`，供 APISIX 與 Docker healthcheck 使用。

### `POST /admin/reload`

重新從 SQLite 載入 instance 設定到記憶體快取。更新 DB 後不需重啟 server，呼叫此 endpoint 即可生效。

```bash
curl -X POST http://localhost:8000/admin/reload
```

---

## Instance Mapping（SQLite）

資料存在 `instances.db`（Docker volume `litellm-data`）。

**Schema：**

| 欄位 | 說明 |
|------|------|
| `instance_name` | 唯一識別名稱，對應 APISIX config 中的 `X-LiteLLM-Instance` header 值 |
| `aws_region_name` | AWS region，例如 `us-east-1` |
| `bedrock_base_url` | Bedrock runtime base URL（不含 `/model/...`） |
| `api_key` | Bedrock API Key（Bearer token） |
| `model_id` | Bedrock model ID，例如 `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `is_active` | `1` = 啟用，`0` = 停用 |

**新增 instance：**
```sql
INSERT INTO bedrock_instances (instance_name, aws_region_name, bedrock_base_url, api_key, model_id)
VALUES ('bedrock-eu-west-1', 'eu-west-1', 'https://bedrock-runtime.eu-west-1.amazonaws.com', 'your-key', 'anthropic.claude-sonnet-4-5-20250929-v1:0');
```

新增後需同步更新 APISIX route（在 `setup-route.sh` 加入對應 instance）並執行，以及呼叫 `/admin/reload` 刷新快取。

---

## APISIX 流量設計

### 負載均衡

`ai-proxy-multi` 使用 `roundrobin` 算法，搭配 `priority` 控制主備關係：

| Instance | weight | priority | 角色 |
|----------|--------|----------|------|
| `bedrock-us-east-1` | 5 | 1 | 主要（優先接流量） |
| `bedrock-ap-northeast-1` | 5 | 0 | 備援（主要失敗時啟用） |

### Failover 條件

`fallback_strategy: ["rate_limiting", "http_5xx"]`

- Bedrock 回 `429 Too Many Requests` → 自動切換到下一個 instance
- Bedrock 回 `5xx` → 自動切換到下一個 instance

### 新增 region

1. 在 SQLite 新增一筆 `bedrock_instances`
2. 在 `setup-route.sh` 的 `instances` 陣列加入新 instance（設定對應的 `X-LiteLLM-Instance` header 值與 priority/weight）
3. 重新執行 `./apisix/routes/setup-route.sh`
4. 呼叫 `POST /admin/reload`

---

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `DB_PATH` | `instances.db` | SQLite 檔案路徑 |
| `APISIX_ADMIN_KEY` | `edd1c9f034335f136f87ad84b625c8f1` | APISIX Admin API key |
| `APISIX_ADMIN_URL` | `http://localhost:9180` | APISIX Admin API URL |
| `LITELLM_PROXY_URL` | `http://litellm-proxy:8000` | Python server URL（供 APISIX 設定使用） |

---

## 技術細節

### litellm 的 Bedrock 認證

本專案使用 Bedrock API Key（Bearer token）認證，**不需要** IAM access key / secret key。

當 `api_key` 傳入時，litellm 直接設定 `Authorization: Bearer {api_key}` header，**完全跳過 SigV4 簽名流程**（源碼：`litellm/llms/bedrock/base_aws_llm.py:get_request_headers()`）。

### VPC Endpoint URL 構建

`api_base` 只需傳入 base URL，litellm 自動附加路徑：

```
{api_base}/model/{model_id}/converse          # non-streaming
{api_base}/model/{model_id}/converse-stream   # streaming
```

源碼：`litellm/llms/bedrock/chat/converse_handler.py:352-363`
