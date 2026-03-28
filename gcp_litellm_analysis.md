# LiteLLM 源碼解析：completion() → GCP Gemini API 完整執行路徑

> **範圍**：以 `provider: vertex_ai`、`model: gemini-*` 為主要分析對象。
> **目的**：說明從 `litellm.completion()` 呼叫到真正送出 HTTP 請求至 GCP Vertex AI Gemini API 的每個環節。

---

## 目錄

1. [架構概覽](#1-架構概覽)
2. [進入點：`completion()` 函數](#2-進入點completion-函數)
3. [Provider 路由偵測：`get_llm_provider()`](#3-provider-路由偵測get_llm_provider)
4. [參數處理與 optional_params 對映](#4-參數處理與-optional_params-對映)
5. [環境變數彙整](#5-環境變數彙整)
6. [認證授權流程](#6-認證授權流程)
7. [能否直接用 API Key 跳過 OAuth？](#7-能否直接用-api-key-跳過-oauth)
8. [URL 建構](#8-url-建構)
9. [Request Body 轉換：OpenAI → Gemini 格式](#9-request-body-轉換openai--gemini-格式)
10. [HTTP 請求發送](#10-http-請求發送)
11. [Response 轉換：Gemini → OpenAI 格式](#11-response-轉換gemini--openai-格式)
12. [Streaming 處理](#12-streaming-處理)
13. [附錄：精簡版概念 Code](#13-附錄精簡版概念-code)

---

## 1. 架構概覽

```
litellm.completion(model="vertex_ai/gemini-2.0-flash", messages=[...])
│
├── [validation] validate_and_fix_openai_messages()
├── [routing]    get_llm_provider()           → custom_llm_provider = "vertex_ai"
├── [params]     get_optional_params()        → map_openai_params() via VertexGeminiConfig
│
└── [dispatch] main.py 大型 if/elif 路由
    └── custom_llm_provider == "vertex_ai"
        └── get_vertex_ai_model_route(model)  → VertexAIModelRoute.GEMINI
            └── vertex_chat_completion.completion()   ← VertexLLM instance
                │
                ├── [auth]    _ensure_access_token()  → Bearer token (OAuth2)
                ├── [url]     _get_token_and_url()    → Vertex AI endpoint URL
                ├── [headers] validate_environment()  → {"Authorization": "Bearer ...", "Content-Type": "application/json"}
                ├── [body]    sync_transform_request_body()
                │               └── _transform_request_body()
                │                   └── _gemini_convert_messages_with_history()
                │
                ├── [http]    client.post(url, headers, json=data)
                │
                └── [resp]    VertexGeminiConfig().transform_response()
                              └── _transform_google_generate_content_to_openai_model_response()
```

**關鍵模組檔案：**

| 角色 | 檔案 |
|------|------|
| 進入點 | `litellm/main.py` |
| Provider 路由 | `litellm/utils.py` → `get_llm_provider()` |
| Model 路由（Vertex 內部） | `litellm/llms/vertex_ai/common_utils.py` → `get_vertex_ai_model_route()` |
| Gemini Handler | `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py` → `VertexLLM` |
| 認證基底 | `litellm/llms/vertex_ai/vertex_llm_base.py` → `VertexBase` |
| Request 轉換 | `litellm/llms/vertex_ai/gemini/transformation.py` |
| Schema 工具 | `litellm/llms/vertex_ai/common_utils.py` |

---

## 2. 進入點：`completion()` 函數

**位置：** [`litellm/main.py:1050`](litellm/main.py#L1050)

```python
def completion(  # type: ignore # noqa: PLR0915
    model: str,
    messages: List = [],
    # -- OpenAI 標準參數 --
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    seed: Optional[int] = None,
    reasoning_effort: Optional[Literal["none","minimal","low","medium","high","xhigh","default"]] = None,
    # -- LiteLLM 特有 --
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,  # 可覆蓋 Vertex AI endpoint
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
```

### 2.1 進入後的初步處理

```python
# main.py:1148-1159
### VALIDATE Request ###
messages = validate_and_fix_openai_messages(messages=messages)    # 格式修正
tools     = validate_and_fix_openai_tools(tools=tools)            # tools 驗證
tool_choice = validate_chat_completion_tool_choice(tool_choice)   # tool_choice 驗證
stop      = validate_openai_optional_params(stop=stop)
thinking  = validate_and_fix_thinking_param(thinking=thinking)    # camelCase 正規化
```

### 2.2 Provider 偵測與 Model 別名解析

```python
# main.py:1356-1374
if litellm.model_alias_map and model in litellm.model_alias_map:
    model = litellm.model_alias_map[model]   # 支援 alias 機制

model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
    model=model,
    custom_llm_provider=custom_llm_provider,
    api_base=api_base,
    api_key=api_key,
)
```

---

## 3. Provider 路由偵測：`get_llm_provider()`

**位置：** `litellm/utils.py`

此函數從 model string 推斷 provider。GCP Gemini 支援兩種寫法：

| 呼叫方式 | 解析結果 |
|---------|---------|
| `model="vertex_ai/gemini-2.0-flash"` | `custom_llm_provider="vertex_ai"`, `model="gemini-2.0-flash"` |
| `model="gemini/gemini-2.0-flash"` | `custom_llm_provider="gemini"` (Google AI Studio) |
| `model="gemini-2.0-flash", custom_llm_provider="vertex_ai"` | 直接使用 |

### 3.1 Vertex AI 內部 Model 路由

**位置：** [`litellm/llms/vertex_ai/common_utils.py:46`](litellm/llms/vertex_ai/common_utils.py#L46)

```python
class VertexAIModelRoute(str, Enum):
    PARTNER_MODELS  = "partner_models"   # llama, mistral, claude (Anthropic on Vertex)
    GEMINI          = "gemini"           # gemini-* 模型 ← 我們的目標
    GEMMA           = "gemma"            # gemma/* 模型
    BGE             = "bge"             # 嵌入模型
    MODEL_GARDEN    = "model_garden"    # openai/* 相容模型
    NON_GEMINI      = "non_gemini"      # 舊式 chat-bison, text-bison 等
    OPENAI_COMPATIBLE = "openai"
    AGENT_ENGINE    = "agent_engine"    # Reasoning Engines

def get_vertex_ai_model_route(model: str, litellm_params: Optional[dict] = None) -> VertexAIModelRoute:
    # 優先檢查 litellm_params["base_model"] 是否含 "gemini"
    if litellm_params and litellm_params.get("base_model") is not None:
        if "gemini" in litellm_params["base_model"]:
            return VertexAIModelRoute.GEMINI

    # 數字型 endpoint ID（Fine-tuned 模型）且有自訂 api_base（PSC）
    if model.isdigit() and litellm_params and litellm_params.get("api_base"):
        return VertexAIModelRoute.GEMINI

    # partner models 偵測（llama, mistral, claude, etc.）
    if VertexAIPartnerModels.is_vertex_partner_model(model=model):
        return VertexAIModelRoute.PARTNER_MODELS

    # gemma/ 前綴
    if "gemma/" in model:
        return VertexAIModelRoute.GEMMA

    # 🎯 "gemini" in model → 路由到 Gemini handler
    if "gemini" in model:
        return VertexAIModelRoute.GEMINI

    return VertexAIModelRoute.NON_GEMINI
```

---

## 4. 參數處理與 optional_params 對映

**位置：** [`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:998`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L998)

透過 `VertexGeminiConfig.map_openai_params()` 將 OpenAI 參數對映至 Gemini 參數。

```python
def map_openai_params(
    self,
    non_default_params: Dict,  # 使用者傳入的非預設值參數
    optional_params: Dict,     # 輸出：Gemini 對應的參數字典
    model: str,
    drop_params: bool,
) -> Dict:
    for param, value in non_default_params.items():

        # temperature → temperature (直接轉，但 Gemini 3 有警告)
        if param == "temperature":
            optional_params["temperature"] = value

        # top_p → top_p（直接轉）
        elif param == "top_p":
            optional_params["top_p"] = value

        # n → candidate_count
        elif param == "n":
            optional_params["candidate_count"] = value

        # stop → stop_sequences（string 轉 list）
        elif param == "stop":
            optional_params["stop_sequences"] = [value] if isinstance(value, str) else value

        # max_tokens / max_completion_tokens → max_output_tokens
        elif param in ("max_tokens", "max_completion_tokens"):
            optional_params["max_output_tokens"] = value

        # response_format → response_mime_type + response_schema / response_json_schema
        elif param == "response_format" and isinstance(value, dict):
            self.apply_response_schema_transformation(value, optional_params, model)

        # frequency_penalty / presence_penalty（僅 Gemini 2.x 以下支援）
        elif param == "frequency_penalty":
            if self._supports_penalty_parameters(model):
                optional_params["frequency_penalty"] = value
        elif param == "presence_penalty":
            if self._supports_penalty_parameters(model):
                optional_params["presence_penalty"] = value

        # logprobs → responseLogprobs
        elif param == "logprobs":
            optional_params["responseLogprobs"] = value

        # top_logprobs → logprobs（注意：Gemini 的 logprobs 對應 OpenAI 的 top_logprobs）
        elif param == "top_logprobs":
            optional_params["logprobs"] = value

        # tools / functions → Gemini tools 格式
        elif param in ("tools", "functions") and isinstance(value, list) and value:
            mapped_tools = self._map_function(value=value, optional_params=optional_params)
            optional_params = self._add_tools_to_optional_params(optional_params, mapped_tools)

        # tool_choice → ToolConfig (functionCallingConfig)
        elif param == "tool_choice":
            _tool_choice_value = self.map_tool_choice_values(model=model, tool_choice=value)
            if _tool_choice_value is not None:
                optional_params["tool_choice"] = _tool_choice_value

        # seed → seed（直接轉）
        elif param == "seed":
            optional_params["seed"] = value

        # reasoning_effort → thinkingConfig（Gemini 2.x 用 thinkingBudget，Gemini 3+ 用 thinkingLevel）
        elif param == "reasoning_effort":
            if VertexGeminiConfig._is_gemini_3_or_newer(model):
                optional_params["thinkingConfig"] = self._map_reasoning_effort_to_thinking_level(value, model)
            else:
                optional_params["thinkingConfig"] = self._map_reasoning_effort_to_thinking_budget(value, model)

        # audio → speechConfig（TTS 模型用）
        elif param == "audio" and isinstance(value, dict):
            optional_params["speechConfig"] = self._map_audio_params(value)
```

### 4.1 參數對映完整對照表

| OpenAI 參數 | Gemini 參數 | 備註 |
|------------|------------|------|
| `temperature` | `temperature` | Gemini 3 不建議 < 1.0 |
| `top_p` | `top_p` | 直接轉 |
| `n` | `candidate_count` | 候選回應數量 |
| `stop` | `stop_sequences` | 字串轉 list |
| `max_tokens` | `max_output_tokens` | |
| `max_completion_tokens` | `max_output_tokens` | 同上 |
| `response_format` | `response_mime_type` + `response_schema` 或 `response_json_schema` | 詳見 §4.2 |
| `frequency_penalty` | `frequency_penalty` | 僅 Gemini 1.x/2.x |
| `presence_penalty` | `presence_penalty` | 僅 Gemini 1.x/2.x |
| `logprobs` | `responseLogprobs` | |
| `top_logprobs` | `logprobs` | 注意名稱互換 |
| `tools` | `tools` (FunctionDeclarations) | 詳見 §8.3 |
| `tool_choice` | `toolConfig.functionCallingConfig` | |
| `seed` | `seed` | |
| `reasoning_effort` | `thinkingConfig` | 詳見 §4.3 |
| `stream` | `stream` | 控制 endpoint 選擇 |

### 4.2 `response_format` 轉換

**位置：** [`vertex_and_google_ai_studio_gemini.py:689`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L689)

```python
def apply_response_schema_transformation(self, value: dict, optional_params: dict, model: str):
    # Gemini 2.0+ 支援 responseJsonSchema（標準 JSON Schema）
    # Gemini 1.5 只支援 responseSchema（OpenAPI 格式，大寫 type）
    use_json_schema = supports_response_json_schema(model)  # "gemini-2" 以上回傳 True

    if value.get("type") == "json_object":
        optional_params["response_mime_type"] = "application/json"
    elif value.get("type") == "json_schema":
        schema = value["json_schema"]["schema"]
        optional_params["response_mime_type"] = "application/json"
        if use_json_schema:
            # Gemini 2.0+：保留標準 JSON Schema，不做型別轉換
            optional_params["response_json_schema"] = _build_json_schema(deepcopy(schema))
        else:
            # Gemini 1.5：轉換成 OpenAPI 格式（string→STRING, object→OBJECT）
            optional_params["response_schema"] = self._map_response_schema(value=schema)
```

**轉換前後範例：**

```json
// 輸入（OpenAI response_format）
{
  "type": "json_schema",
  "json_schema": {
    "name": "Result",
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "score": {"type": "number"}
      },
      "required": ["name", "score"],
      "additionalProperties": false
    }
  }
}

// 輸出（Gemini 1.5 responseSchema，OpenAPI 格式）
{
  "response_mime_type": "application/json",
  "response_schema": {
    "type": "OBJECT",         // 大寫
    "properties": {
      "name": {"type": "STRING"},
      "score": {"type": "NUMBER"}
    },
    "required": ["name", "score"],
    "propertyOrdering": ["name", "score"]  // 新增順序欄位
    // additionalProperties 被移除
  }
}

// 輸出（Gemini 2.0+ responseJsonSchema，標準 JSON Schema）
{
  "response_mime_type": "application/json",
  "response_json_schema": {
    "type": "object",         // 小寫，保持不變
    "properties": { ... },
    "required": [...],
    "additionalProperties": false  // 保留
  }
}
```

### 4.3 `reasoning_effort` 轉換

```python
# Gemini 2.5 系列：thinkingBudget（整數 tokens）
reasoning_effort="high"  →  {"thinkingBudget": 16000, "includeThoughts": True}
reasoning_effort="medium" → {"thinkingBudget": 8000,  "includeThoughts": True}
reasoning_effort="low"   →  {"thinkingBudget": 1024,  "includeThoughts": True}
reasoning_effort="none"  →  {"thinkingBudget": 0,     "includeThoughts": False}

# Gemini 3+ 系列：thinkingLevel（字串）
reasoning_effort="high"  →  {"thinkingLevel": "high",   "includeThoughts": True}
reasoning_effort="medium" → {"thinkingLevel": "medium", "includeThoughts": True}
reasoning_effort="low"   →  {"thinkingLevel": "low",    "includeThoughts": True}
```

### 4.4 `tool_choice` 對映

```python
def map_tool_choice_values(self, model: str, tool_choice: Union[str, dict]) -> Optional[ToolConfig]:
    if tool_choice == "none":
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
    elif tool_choice == "required":
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
    elif tool_choice == "auto":
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
    elif isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "my_func"}}
        name = tool_choice.get("function", {}).get("name", "")
        return ToolConfig(
            functionCallingConfig=FunctionCallingConfig(
                mode="ANY", allowed_function_names=[name]
            )
        )
```

---

## 5. 環境變數彙整

### 5.1 Vertex AI（主要路徑）

| 環境變數 | 用途 | 取值優先順序 |
|---------|------|------------|
| `VERTEXAI_PROJECT` | GCP Project ID | `optional_params["vertex_project"]` → `optional_params["vertex_ai_project"]` → `litellm.vertex_project` → `get_secret("VERTEXAI_PROJECT")` |
| `VERTEXAI_LOCATION` | GCP Region（如 `us-central1`） | `optional_params["vertex_location"]` → `optional_params["vertex_ai_location"]` → `litellm.vertex_location` → `get_secret("VERTEXAI_LOCATION")` |
| `VERTEXAI_CREDENTIALS` | Service Account JSON（路徑或 JSON 字串） | `optional_params["vertex_credentials"]` → `optional_params["vertex_ai_credentials"]` → `get_secret("VERTEXAI_CREDENTIALS")` |
| `VERTEXAI_API_BASE` | 自訂 Vertex AI endpoint（如 PSC endpoint） | `api_base` → `litellm.api_base` → `get_secret("VERTEXAI_API_BASE")` |
| `GOOGLE_APPLICATION_CREDENTIALS` | ADC（Application Default Credentials）標準路徑 | google-auth library 自動讀取 |

**位置：** [`main.py:3464`](litellm/main.py#L3464)

```python
# main.py:3464-3483
elif custom_llm_provider == "vertex_ai":
    vertex_ai_project = (
        optional_params.pop("vertex_project", None)
        or optional_params.pop("vertex_ai_project", None)
        or litellm.vertex_project
        or get_secret("VERTEXAI_PROJECT")
    )
    vertex_ai_location = (
        optional_params.pop("vertex_location", None)
        or optional_params.pop("vertex_ai_location", None)
        or litellm.vertex_location
        or get_secret("VERTEXAI_LOCATION")
    )
    vertex_credentials = (
        optional_params.pop("vertex_credentials", None)
        or optional_params.pop("vertex_ai_credentials", None)
        or get_secret("VERTEXAI_CREDENTIALS")
    )
    api_base = api_base or litellm.api_base or get_secret("VERTEXAI_API_BASE")
```

### 5.2 Google AI Studio（`provider=gemini` 時）

| 環境變數 | 用途 |
|---------|------|
| `GEMINI_API_KEY` | Google AI Studio API Key |
| `GOOGLE_API_KEY` | 同上（備用） |

---

## 6. 認證授權流程

**位置：** [`litellm/llms/vertex_ai/vertex_llm_base.py:79`](litellm/llms/vertex_ai/vertex_llm_base.py#L79)

`VertexBase.load_auth()` 依 credentials 類型進行不同處理：

```python
def load_auth(self, credentials: Optional[VERTEX_CREDENTIALS_TYPES], project_id: Optional[str]) -> Tuple[Any, str]:
    if credentials is not None:
        # credentials 可能是：字串（路徑 or JSON 文字）、dict
        if isinstance(credentials, str):
            if os.path.exists(credentials):
                json_obj = json.load(open(credentials))    # 讀取檔案
            else:
                json_obj = json.loads(credentials)         # 解析 JSON 字串
        elif isinstance(credentials, dict):
            json_obj = credentials

        # 依 credentials type 選擇對應的 google-auth 方式
        if json_obj.get("type") == "external_account":
            # Workload Identity Federation（WIF）
            if "aws" in json_obj["credential_source"].get("environment_id", ""):
                creds = VertexAIAwsWifAuth.credentials_from_explicit_aws(...)  # AWS WIF
            else:
                creds = identity_pool.Credentials.from_info(json_obj)           # 其他 WIF

        elif json_obj.get("type") == "authorized_user":
            # gcloud auth application-default login 產生的 credentials
            creds = google.oauth2.credentials.Credentials.from_authorized_user_info(json_obj, scopes=[...])

        else:
            # Service Account（最常見）
            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

    else:
        # 沒有 credentials → 使用 ADC（Application Default Credentials）
        # 依序嘗試：GOOGLE_APPLICATION_CREDENTIALS → gcloud ADC → GCE metadata server
        creds, creds_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if project_id is None:
            project_id = creds_project_id

    self.refresh_auth(creds)  # 觸發 token refresh（取得 access_token）
    return creds, project_id
```

### 6.1 認證類型與使用場景

| 類型 | `json_obj["type"]` | 使用場景 |
|-----|-------------------|---------|
| Service Account | `"service_account"` | 生產環境，服務帳號 JSON key |
| Authorized User | `"authorized_user"` | 本機開發，`gcloud auth application-default login` |
| External Account (WIF-AWS) | `"external_account"` + `aws` in `environment_id` | AWS 跨雲 WIF |
| External Account (WIF) | `"external_account"` | 其他 WIF（如 GitHub Actions） |
| ADC（無 credentials 傳入） | N/A | Cloud Run / GKE 自動取得，或 GOOGLE_APPLICATION_CREDENTIALS |

### 6.2 Token 在請求中的使用

取得 Bearer token 後，放入 HTTP Authorization header：

```python
# validate_environment() in VertexGeminiConfig
def validate_environment(self, api_key: Optional[Union[str, Dict]], headers: Optional[Dict], ...) -> Dict:
    default_headers = {
        "Content-Type": "application/json",
    }
    if api_key is not None:
        default_headers["Authorization"] = f"Bearer {api_key}"  # Bearer OAuth2 token
    if headers is not None:
        default_headers.update(headers)
    return default_headers
```

Google AI Studio（`provider=gemini`）則使用 API Key 作為 URL query parameter，不用 Bearer token：
```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}
```

---

## 7. 能否直接用 API Key 跳過 OAuth？

**簡答：`vertex_ai` provider 不行。`gemini` provider 可以。**

### 7.1 `vertex_ai` provider — 硬性要求 OAuth2，無 API Key 支援

**位置：** [`main.py:3511`](litellm/main.py#L3511)、[`main.py:3524`](litellm/main.py#L3524)

```python
# main.py:3511-3532
elif model_route == VertexAIModelRoute.GEMINI:
    model_response = vertex_chat_completion.completion(
        model=model,
        messages=messages,
        vertex_location=vertex_ai_location,
        vertex_project=vertex_ai_project,
        vertex_credentials=vertex_credentials,
        gemini_api_key=None,          # ← 硬編碼 None，不會用 api_key 參數
        custom_llm_provider=custom_llm_provider,  # "vertex_ai"
        ...
    )
```

在 `VertexLLM.completion()` 裡，`_ensure_access_token()` 的判斷：

```python
# vertex_llm_base.py:329-346
def _ensure_access_token(self, credentials, project_id, custom_llm_provider):
    if custom_llm_provider == "gemini":
        return "", ""        # ← gemini provider：直接跳過，不取 token
    else:
        return self.get_access_token(  # ← vertex_ai / vertex_ai_beta：一定走 OAuth
            credentials=credentials,
            project_id=project_id,
        )
```

`custom_llm_provider="vertex_ai"` 永遠走 `get_access_token()`，即使 `credentials=None`，也會嘗試 ADC。如果 ADC 也沒設定則直接拋錯，**沒有降級至 API Key 的路徑**。

### 7.2 `gemini` provider — 支援 API Key，但走 Google AI Studio 端點

**位置：** [`main.py:3414`](litellm/main.py#L3414)、[`main.py:3433`](litellm/main.py#L3433)

```python
# main.py:3414  ← vertex_ai_beta 和 gemini 共用同一個 elif 分支
elif custom_llm_provider == "vertex_ai_beta" or custom_llm_provider == "gemini":
    gemini_api_key = (
        api_key                          # 呼叫時直接傳入
        or get_api_key_from_env()        # GEMINI_API_KEY 或 GOOGLE_API_KEY
        or get_secret("PALM_API_KEY")    # 舊 Palm API key 也支援
        or litellm.api_key
    )
    # gemini_api_key 會被傳入 vertex_chat_completion.completion()
```

`get_api_key_from_env()` 定義（[`litellm/llms/gemini/common_utils.py:47`](litellm/llms/gemini/common_utils.py#L47)）：

```python
@staticmethod
def get_api_key(api_key=None):
    return (
        api_key
        or get_secret_str("GOOGLE_API_KEY")
        or get_secret_str("GEMINI_API_KEY")
    )
```

當 `custom_llm_provider == "gemini"` 時，URL 建構走 `_get_gemini_url()`，API Key 直接嵌入 URL query param：

```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}
```

Auth header 對這個 provider 不使用（`_ensure_access_token` 直接回傳 `"", ""`）。

### 7.3 `vertex_ai_beta` provider — 有 `gemini_api_key` 但仍走 OAuth

`vertex_ai_beta` 雖然和 `gemini` 共用同一個 elif 分支（也設定了 `gemini_api_key`），但因為 `_ensure_access_token()` 只對 `"gemini"` 特判跳過，`vertex_ai_beta` 仍然走 OAuth 取 token。`gemini_api_key` 在這個路徑下**只在使用自訂 `api_base`（Cloudflare AI Gateway 等代理）時才生效**：

```python
# vertex_llm_base.py:389-402  _check_custom_proxy()
if api_base:
    if custom_llm_provider == "gemini":
        # gemini + custom api_base → 用 x-goog-api-key header
        auth_header = {"x-goog-api-key": gemini_api_key}
    else:
        # vertex_ai / vertex_ai_beta + custom api_base → 只替換 URL，不改 auth
        url = "{}:{}".format(api_base, endpoint)
```

### 7.4 三種 Provider 認證方式對照

| Provider | 認證方式 | 端點 | API Key 可用？ |
|---------|---------|------|--------------|
| `vertex_ai` | OAuth2（service account / ADC） | `{location}-aiplatform.googleapis.com` | ❌ 不支援 |
| `vertex_ai_beta` | OAuth2（service account / ADC） | `{location}-aiplatform.googleapis.com` | ❌ 不支援（gemini_api_key 僅供 custom proxy 用） |
| `gemini` | API Key（URL query param）| `generativelanguage.googleapis.com` | ✅ 支援 |

### 7.5 只有 API Key 時的正確用法

如果手上只有 `GEMINI_API_KEY`（Google AI Studio key），應使用 `gemini/` prefix：

```python
import litellm

# ✅ 正確：使用 gemini provider，走 Google AI Studio
response = litellm.completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "你好"}],
    api_key="AIza...",         # 或設 GEMINI_API_KEY / GOOGLE_API_KEY 環境變數
)

# ❌ 不支援：vertex_ai provider 無法接受 API Key
response = litellm.completion(
    model="vertex_ai/gemini-2.0-flash",
    messages=[{"role": "user", "content": "你好"}],
    api_key="AIza...",         # 這個 api_key 在 vertex_ai route 下會被忽略
    # 仍然需要 VERTEXAI_PROJECT + VERTEXAI_CREDENTIALS 或 ADC
)
```

> **注意：** `gemini/` provider 使用的是 Google AI Studio endpoint（`generativelanguage.googleapis.com`），與 Vertex AI（`aiplatform.googleapis.com`）是不同的服務。兩者支援相同的 Gemini 模型，但在 quota、SLA、企業功能（VPC-SC、CMEK 等）上有所差異。

---

## 8. URL 建構

**位置：** [`litellm/llms/vertex_ai/common_utils.py:278`](litellm/llms/vertex_ai/common_utils.py#L278)

### 8.1 Vertex AI Gemini URL

```python
def _get_vertex_url(mode, model, stream, vertex_project, vertex_location, vertex_api_version) -> Tuple[str, str]:
    base_url = get_vertex_base_url(vertex_location)
    # get_vertex_base_url():
    #   if vertex_location == "global" → "https://aiplatform.googleapis.com"
    #   else → f"https://{vertex_location}-aiplatform.googleapis.com"

    if mode == "chat":
        endpoint = "generateContent"
        if stream is True:
            endpoint = "streamGenerateContent"

        if model.isdigit():
            # Fine-tuned 模型（endpoint ID）
            url = f"{base_url}/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
        else:
            # 標準 Gemini 模型
            url = f"{base_url}/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"

        if stream is True:
            url += "?alt=sse"  # Server-Sent Events 格式

    return url, endpoint
```

**實際 URL 範例：**

```
# 非 streaming（us-central1）
https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent

# Streaming（us-central1）
https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse

# Fine-tuned 模型
https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/4965075652664360960:generateContent

# Global endpoint
https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-2.0-flash:generateContent
```

### 8.2 API 版本選擇

```python
# vertex_api_version 取決於是否使用 v1beta1 特性
should_use_v1beta1_features = self.is_using_v1beta1_features(optional_params=optional_params)
# 若使用 cachedContent、response_json_schema 等 beta 特性 → "v1beta1"
# 否則 → "v1"
```

### 8.3 Google AI Studio URL（`provider=gemini`）

```python
def _get_gemini_url(mode, model, stream, gemini_api_key) -> Tuple[str, str]:
    _gemini_model_name = f"models/{model}"
    # Gemini 3+ 使用 v1alpha；其餘使用 v1beta
    api_version = "v1alpha" if VertexGeminiConfig._is_gemini_3_or_newer(model) else "v1beta"

    if mode == "chat":
        if stream:
            url = f"https://generativelanguage.googleapis.com/{api_version}/{_gemini_model_name}:streamGenerateContent?key={gemini_api_key}&alt=sse"
        else:
            url = f"https://generativelanguage.googleapis.com/{api_version}/{_gemini_model_name}:generateContent?key={gemini_api_key}"
```

---

## 8. Request Body 轉換：OpenAI → Gemini 格式

### 9.1 整體流程

**位置：** [`litellm/llms/vertex_ai/gemini/transformation.py:776`](litellm/llms/vertex_ai/gemini/transformation.py#L776)

```python
def sync_transform_request_body(...) -> RequestBody:
    # 1. 檢查 context caching（若有 cached_content 則建立快取）
    messages, optional_params, cached_content = context_caching_endpoints.check_and_create_cache(...)

    # 2. 執行核心轉換
    return _transform_request_body(
        messages=messages,
        model=model,
        optional_params=optional_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
    )
```

**核心轉換函數 `_transform_request_body()`：**

```python
def _transform_request_body(messages, model, optional_params, custom_llm_provider, litellm_params, cached_content) -> RequestBody:

    # 步驟 1：分離 system message
    supports_system_message = get_supports_system_message(model=model, custom_llm_provider=...)
    system_instructions, messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )
    # system message → {"role": "system", "content": "..."} 提取到 system_instruction 欄位

    # 步驟 2：處理 response_schema 相容性
    if "response_schema" in optional_params:
        supports_response_schema = get_supports_response_schema(model=model, ...)
        if not supports_response_schema:
            # 不支援的模型：把 schema 轉成 user message 提示
            user_response_schema_message = response_schema_prompt(model, schema)
            messages.append({"role": "user", "content": user_response_schema_message})
            optional_params.pop("response_schema")

    # 步驟 3：轉換 messages → contents
    content = litellm.VertexGeminiConfig()._transform_messages(messages=messages, model=model)
    # → _gemini_convert_messages_with_history()

    # 步驟 4：從 optional_params 提取 Gemini 特有頂層欄位
    tools: Optional[Tools]                           = optional_params.pop("tools", None)
    tool_choice: Optional[ToolConfig]               = optional_params.pop("tool_choice", None)
    safety_settings: Optional[List[SafetySettings]] = optional_params.pop("safety_settings", None)
    labels: Optional[dict]                           = optional_params.pop("labels", None)

    # 步驟 5：其餘 optional_params 整理為 GenerationConfig
    config_fields = GenerationConfig.__annotations__.keys()
    filtered_params = {
        k: v for k, v in optional_params.items()
        if _get_equivalent_key(k, set(config_fields))  # 只保留 Gemini GenerationConfig 支援的欄位
    }
    generation_config = GenerationConfig(**filtered_params)

    # 步驟 6：組裝 RequestBody
    data = RequestBody(contents=content)
    if system_instructions is not None:
        data["system_instruction"] = system_instructions
    if tools is not None:
        data["tools"] = tools
    if tool_choice is not None:
        data["toolConfig"] = tool_choice
    if safety_settings is not None:
        data["safetySettings"] = safety_settings
    if generation_config:
        data["generationConfig"] = generation_config
    if cached_content is not None:
        data["cachedContent"] = cached_content
    if labels and custom_llm_provider != "gemini":  # Vertex AI only
        data["labels"] = labels

    # 步驟 7：合併 extra_body（使用者自訂 Gemini 特有欄位）
    _pop_and_merge_extra_body(data, optional_params)

    return data
```

### 9.2 Messages 轉換：`_gemini_convert_messages_with_history()`

**位置：** [`litellm/llms/vertex_ai/gemini/transformation.py:291`](litellm/llms/vertex_ai/gemini/transformation.py#L291)

**核心規則：**
- Gemini 要求 `role` 必須交替出現（`user` / `model`），連續相同 role 的 message 會被合併
- OpenAI `role: "system"` 在支援 system message 的模型中被提取到 `system_instruction`；不支援的則合併進 user message
- OpenAI `role: "tool"` / `role: "function"` 轉換為 Gemini 的 function response part

```python
def _gemini_convert_messages_with_history(messages, model) -> List[ContentType]:
    user_message_types = {"user", "system"}
    contents: List[ContentType] = []

    while msg_i < len(messages):
        # -- 合併連續的 user/system messages --
        user_content: List[PartType] = []
        while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
            _message_content = messages[msg_i].get("content")
            if isinstance(_message_content, list):
                for element in _message_content:
                    if element["type"] == "text":
                        _parts.append(PartType(text=element["text"]))
                    elif element["type"] == "image_url":
                        _part = _process_gemini_media(image_url=..., format=..., model=model)
                        _parts.append(_part)
                    elif element["type"] == "input_audio":
                        # 轉換音訊格式 → inlineData
                        ...
                    elif element["type"] == "file":
                        # 處理 file_id 或 file_data → fileData or inlineData
                        ...
            elif isinstance(_message_content, str):
                user_content.append(PartType(text=_message_content))
            msg_i += 1

        if user_content:
            # ⚠️ Gemini 要求至少有一個 text part
            if not _check_text_in_content(user_content):
                user_content.append(PartType(text=" "))  # 補空白避免 API 錯誤
            contents.append(ContentType(role="user", parts=user_content))

        # -- 合併連續的 assistant messages --
        assistant_content = []
        while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
            if reasoning_content is not None:
                assistant_content.append(PartType(thought=True, text=reasoning_content))  # 思考內容
            if _message_content is not None:
                assistant_content.append(PartType(text=_message_content))
            if tool_calls:
                gemini_tool_call_parts = convert_to_gemini_tool_call_invoke(assistant_msg)
                assistant_content.extend(gemini_tool_call_parts)
            msg_i += 1

        if assistant_content:
            contents.append(ContentType(role="model", parts=assistant_content))

        # -- tool response messages --
        if msg_i < len(messages) and messages[msg_i]["role"] in ["tool", "function"]:
            _part = convert_to_gemini_tool_call_result(messages[msg_i], last_assistant_msg)
            tool_call_responses.append(_part)
            msg_i += 1
```

**轉換前後範例：**

```json
// 輸入（OpenAI messages）
[
  {"role": "system", "content": "你是一位助理"},
  {"role": "user",   "content": "你好"},
  {"role": "assistant", "content": "你好！有什麼可以幫你？"},
  {"role": "user",   "content": "今天幾號？"}
]

// 輸出（Gemini RequestBody）
{
  "system_instruction": {
    "role": "user",
    "parts": [{"text": "你是一位助理"}]
  },
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "你好"}]
    },
    {
      "role": "model",
      "parts": [{"text": "你好！有什麼可以幫你？"}]
    },
    {
      "role": "user",
      "parts": [{"text": "今天幾號？"}]
    }
  ]
}
```

### 9.3 Tools 轉換

```json
// 輸入（OpenAI tools）
[{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather info",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
}]

// 輸出（Gemini tools + toolConfig）
{
  "tools": [{
    "functionDeclarations": [{
      "name": "get_weather",
      "description": "Get weather info",
      "parameters": {
        "type": "OBJECT",                    // 大寫（Gemini 1.x/2.x）
        "properties": {
          "location": {"type": "STRING"},
          "unit": {
            "type": "STRING",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }]
  }],
  "toolConfig": {
    "functionCallingConfig": {
      "mode": "AUTO"   // tool_choice="auto"
    }
  }
}
```

### 9.4 完整 GenerationConfig 欄位對照

```json
// GenerationConfig 最終範例
{
  "generationConfig": {
    "temperature": 0.7,
    "topP": 0.95,
    "maxOutputTokens": 1024,
    "stopSequences": ["END"],
    "candidateCount": 1,
    "responseMimeType": "application/json",
    "responseSchema": { ... },          // Gemini 1.5
    // 或
    "responseJsonSchema": { ... },      // Gemini 2.0+
    "responseLogprobs": true,
    "logprobs": 5,
    "seed": 42,
    "frequencyPenalty": 0.5,           // 僅 Gemini 2.x 以下
    "presencePenalty": 0.3,            // 僅 Gemini 2.x 以下
    "thinkingConfig": {                // reasoning 模型
      "thinkingBudget": 8000,
      "includeThoughts": true
    }
  }
}
```

---

## 9. HTTP 請求發送

**位置：** [`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2981`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L2981)

```python
# 同步非串流完成路徑（VertexLLM.completion() 末段）

## SYNC COMPLETION CALL ##
if client is None or isinstance(client, AsyncHTTPHandler):
    _params = {}
    if timeout is not None:
        timeout = httpx.Timeout(timeout)
        _params["timeout"] = timeout
    client = _get_httpx_client(params=_params)   # 建立 httpx 同步 client

try:
    response = client.post(
        url=url,                    # Vertex AI generateContent endpoint
        headers=headers,            # {"Authorization": "Bearer <token>", "Content-Type": "application/json"}
        json=data,                  # Gemini RequestBody dict
        logging_obj=logging_obj,
    )
    response.raise_for_status()     # 非 2xx → 拋出 HTTPStatusError

except httpx.HTTPStatusError as err:
    raise VertexAIError(
        status_code=err.response.status_code,
        message=err.response.text,
        headers=err.response.headers,
    )
except httpx.TimeoutException:
    raise VertexAIError(status_code=408, message="Timeout error occurred.")
```

### 9.1 完整請求範例

```http
POST https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent
Authorization: Bearer ya29.c.b0AXv0zTu...
Content-Type: application/json

{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "你好，今天天氣怎麼樣？"}]
    }
  ],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 1024
  }
}
```

---

## 10. Response 轉換：Gemini → OpenAI 格式

**位置：** [`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2275`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L2275)

### 11.1 transform_response() 流程

```python
def transform_response(self, model, raw_response, model_response, ...) -> ModelResponse:
    # 1. 解析 JSON 為 GenerateContentResponseBody
    completion_response = GenerateContentResponseBody(**raw_response.json())

    # 2. 轉換為 OpenAI 格式
    return self._transform_google_generate_content_to_openai_model_response(
        completion_response=completion_response,
        model_response=model_response,
        model=model,
        ...
    )

def _transform_google_generate_content_to_openai_model_response(...) -> ModelResponse:
    model_response.model = model

    # 3. 檢查 prompt 被 block（promptFeedback.blockReason）
    if "promptFeedback" in completion_response and "blockReason" in completion_response["promptFeedback"]:
        return self._handle_blocked_response(...)  # finish_reason = "content_filter"

    # 4. 處理 candidates
    _candidates = completion_response.get("candidates")
    if _candidates:
        # 檢查 finishReason 是否為內容違規（SAFETY, RECITATION, BLOCKLIST, ...）
        if _candidates[0].get("finishReason") in content_policy_violations:
            return self._handle_content_policy_violation(...)

        # 5. 轉換 candidates → choices
        VertexGeminiConfig._process_candidates(_candidates, model_response, ...)

    # 6. 計算 usage
    usage = VertexGeminiConfig._calculate_usage(completion_response=completion_response)
    setattr(model_response, "usage", usage)

    return model_response
```

### 11.2 候選結果處理：`_process_candidates()`

```python
@staticmethod
def _process_candidates(_candidates, model_response, standard_optional_params, ...):
    for idx, candidate in enumerate(_candidates):
        parts = candidate.get("content", {}).get("parts", [])

        # 提取文字內容和 reasoning 內容
        content_str, reasoning_content_str = self.get_assistant_content_message(parts)
        # get_assistant_content_message():
        #   - part["thought"] == True → reasoning_content
        #   - 其餘 text → content
        #   - inlineData audio/image → 分別處理

        # 提取 function calls
        tools: Optional[List[ChatCompletionToolCallChunk]] = None
        if "functionCall" in parts:
            tools = [
                ChatCompletionToolCallChunk(
                    id=f"call_{uuid4()}",
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=part["functionCall"]["name"],
                        arguments=json.dumps(part["functionCall"]["args"]),
                    ),
                )
                for part in parts if "functionCall" in part
            ]

        # 建立 ChatCompletionMessage
        chat_completion_message = ChatCompletionResponseMessage(
            role="assistant",
            content=content_str,
            reasoning_content=reasoning_content_str,
            tool_calls=tools,
        )

        # finish_reason 對映
        finish_reason = self._check_finish_reason(chat_completion_message, candidate.get("finishReason"))
        # STOP → "stop", MAX_TOKENS → "length", SAFETY → "content_filter", ...

        choice = litellm.Choices(
            finish_reason=finish_reason,
            index=candidate.get("index", idx),
            message=chat_completion_message,
            logprobs=chat_completion_logprobs,
        )
        model_response.choices.append(choice)
```

### 11.3 Usage 計算

```python
@staticmethod
def _calculate_usage(completion_response) -> Usage:
    usage_metadata = completion_response["usageMetadata"]
    # 基本 token 計數
    prompt_tokens     = usage_metadata.get("promptTokenCount", 0)
    completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
    total_tokens      = usage_metadata.get("totalTokenCount", 0)
    cached_tokens     = usage_metadata.get("cachedContentTokenCount")  # Optional

    # 各模態明細
    # promptTokensDetails → prompt_audio_tokens, prompt_image_tokens, prompt_text_tokens
    # candidatesTokensDetails → response_audio_tokens, response_image_tokens, etc.
    # thinkingTokenCount → reasoning_tokens

    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
            audio_tokens=prompt_audio_tokens,
            ...
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=reasoning_tokens,
            ...
        ),
    )
```

### 11.4 轉換前後格式範例

```json
// Gemini API 原始 Response
{
  "responseId": "abc123",
  "candidates": [{
    "content": {
      "role": "model",
      "parts": [{"text": "台北今天晴天，氣溫約 28°C。"}]
    },
    "finishReason": "STOP",
    "index": 0,
    "safetyRatings": [...]
  }],
  "usageMetadata": {
    "promptTokenCount": 15,
    "candidatesTokenCount": 12,
    "totalTokenCount": 27
  }
}

// 轉換後（OpenAI 格式 ModelResponse）
{
  "id": "abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "gemini-2.0-flash",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "台北今天晴天，氣溫約 28°C。"
    },
    "finish_reason": "stop"    // STOP → "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 12,
    "total_tokens": 27
  }
}
```

### 11.5 Finish Reason 對映

| Gemini `finishReason` | OpenAI `finish_reason` |
|----------------------|----------------------|
| `STOP` | `stop` |
| `MAX_TOKENS` | `length` |
| `SAFETY` | `content_filter` |
| `RECITATION` | `content_filter` |
| `BLOCKLIST` | `content_filter` |
| `PROHIBITED_CONTENT` | `content_filter` |
| `SPII` | `content_filter` |
| `MALFORMED_FUNCTION_CALL` | `tool_calls` |
| `OTHER` | `stop` |
| `FINISH_REASON_UNSPECIFIED` | `stop` |

---

## 11. Streaming 處理

### 12.1 Streaming URL

串流模式下，endpoint 從 `generateContent` 改為 `streamGenerateContent`，並加上 `?alt=sse`：

```
POST https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse
```

### 12.2 串流 Response 建立

**位置：** [`vertex_and_google_ai_studio_gemini.py:2944`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L2944)

```python
if stream is True:
    request_data_str = json.dumps(data)
    streaming_response = CustomStreamWrapper(
        completion_stream=None,    # 尚未建立，lazy init
        make_call=partial(         # 延遲執行的 HTTP 呼叫
            make_sync_call,
            gemini_client=client,
            api_base=url,
            data=request_data_str,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            headers=headers,
        ),
        model=model,
        custom_llm_provider="vertex_ai_beta",
        logging_obj=logging_obj,
    )
    return streaming_response  # 回傳 iterator，實際 HTTP 在第一次 iteration 時才執行
```

### 12.3 SSE 串流解析

**位置：** [`vertex_and_google_ai_studio_gemini.py:3012`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L3012) — `ModelResponseIterator`

```python
class ModelResponseIterator:
    def chunk_parser(self, chunk: dict) -> Optional[ModelResponseStream]:
        processed_chunk = GenerateContentResponseBody(**chunk)
        response_id = processed_chunk.get("responseId")
        model_response = ModelResponseStream(choices=[], id=response_id)

        # 檢查 prompt blocking
        blocked_response = VertexGeminiConfig._check_prompt_level_content_filter(processed_chunk, response_id)
        if blocked_response is not None:
            return blocked_response

        _candidates = processed_chunk.get("candidates")
        if _candidates:
            # 每個 SSE chunk 包含部分 candidates，逐一轉換
            VertexGeminiConfig._process_candidates(_candidates, model_response, ...)

        # 最後一個 chunk 包含完整的 usageMetadata
        if "usageMetadata" in processed_chunk:
            usage = VertexGeminiConfig._calculate_usage(processed_chunk)
            model_response.usage = usage

        return model_response
```

**Gemini SSE 格式（raw）：**

```
data: {"candidates":[{"content":{"parts":[{"text":"台北"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":2}}

data: {"candidates":[{"content":{"parts":[{"text":"今天"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":4}}

data: {"candidates":[{"content":{"parts":[{"text":"晴天。"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":6,"totalTokenCount":21}}
```

**轉換後（OpenAI delta 格式）：**

```json
// chunk 1
{"id":"resp_123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"台北"},"finish_reason":null}]}

// chunk 2
{"id":"resp_123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"今天"},"finish_reason":null}]}

// chunk 3（最後）
{"id":"resp_123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"晴天。"},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":6,"total_tokens":21}}
```

---

## 12. 附錄：精簡版概念 Code

以下為去除非 GCP Gemini 處理邏輯後的概念性重寫，涵蓋完整執行路徑：

```python
"""
概念 Code：LiteLLM completion() → GCP Vertex AI Gemini API
（移除非 Gemini 邏輯，保留核心路徑）
"""

import json
import os
from typing import Dict, List, Optional, Union

import httpx
import google.auth
import google.auth.transport.requests
import google.oauth2.service_account


# ─────────────────────────────────────────────
# 1. 進入點
# ─────────────────────────────────────────────

def completion(
    model: str,                  # e.g. "vertex_ai/gemini-2.0-flash"
    messages: List[Dict],
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    response_format: Optional[Dict] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    # Vertex AI 認證
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    vertex_credentials: Optional[str] = None,   # service account JSON path or string
) -> Dict:

    # Step 1: 解析 provider 和 model name
    if "/" in model:
        provider, model_name = model.split("/", 1)  # "vertex_ai", "gemini-2.0-flash"
    else:
        model_name = model

    # Step 2: 讀取認證設定（優先用傳入值，否則讀環境變數）
    project  = vertex_project  or os.environ.get("VERTEXAI_PROJECT")
    location = vertex_location or os.environ.get("VERTEXAI_LOCATION", "us-central1")
    creds_str = vertex_credentials or os.environ.get("VERTEXAI_CREDENTIALS")

    # Step 3: 取得 Bearer Token
    access_token, resolved_project = get_access_token(creds_str, project)

    # Step 4: 建構 endpoint URL
    url = build_vertex_url(model_name, resolved_project, location, stream)

    # Step 5: 轉換 OpenAI 格式的 optional params → Gemini 格式
    gemini_params = map_openai_to_gemini_params(
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        model=model_name,
    )

    # Step 6: 轉換 messages → Gemini contents 格式
    request_body = build_request_body(messages, gemini_params, model_name)

    # Step 7: 送出 HTTP 請求
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    if stream:
        return handle_stream_response(url, headers, request_body, model_name)
    else:
        return handle_sync_response(url, headers, request_body, model_name)


# ─────────────────────────────────────────────
# 2. 認證：取得 Bearer Token
# ─────────────────────────────────────────────

def get_access_token(creds_str: Optional[str], project: Optional[str]):
    """
    支援三種認證方式：
    1. Service Account JSON
    2. gcloud authorized_user（gcloud auth application-default login）
    3. ADC（Application Default Credentials）
    """
    if creds_str:
        # 嘗試解析為 JSON（可能是路徑或 JSON 文字）
        if os.path.exists(creds_str):
            json_obj = json.load(open(creds_str))
        else:
            json_obj = json.loads(creds_str)

        cred_type = json_obj.get("type")

        if cred_type == "service_account":
            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            resolved_project = project or json_obj.get("project_id")

        elif cred_type == "authorized_user":
            creds = google.oauth2.credentials.Credentials.from_authorized_user_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            resolved_project = project or json_obj.get("quota_project_id")

        else:
            raise ValueError(f"Unsupported credential type: {cred_type}")
    else:
        # ADC：依序嘗試 GOOGLE_APPLICATION_CREDENTIALS → gcloud → GCE metadata
        creds, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        resolved_project = project or adc_project

    # Refresh token（若已過期）
    request = google.auth.transport.requests.Request()
    creds.refresh(request)

    return creds.token, resolved_project


# ─────────────────────────────────────────────
# 3. URL 建構
# ─────────────────────────────────────────────

def build_vertex_url(model: str, project: str, location: str, stream: bool) -> str:
    base_url = f"https://{location}-aiplatform.googleapis.com"
    api_version = "v1"
    endpoint = "streamGenerateContent" if stream else "generateContent"
    url = f"{base_url}/{api_version}/projects/{project}/locations/{location}/publishers/google/models/{model}:{endpoint}"
    if stream:
        url += "?alt=sse"
    return url


# ─────────────────────────────────────────────
# 4. 參數對映：OpenAI → Gemini
# ─────────────────────────────────────────────

def map_openai_to_gemini_params(
    temperature, max_tokens, response_format, tools, tool_choice, model
) -> Dict:
    gemini_params = {}

    if temperature is not None:
        gemini_params["temperature"] = temperature

    if max_tokens is not None:
        gemini_params["max_output_tokens"] = max_tokens   # 名稱轉換

    if response_format:
        if response_format.get("type") == "json_object":
            gemini_params["response_mime_type"] = "application/json"
        elif response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            gemini_params["response_mime_type"] = "application/json"
            # Gemini 2.0+ 使用標準 JSON Schema；1.5 需要轉換成大寫 type
            if "gemini-2" in model or "gemini-3" in model:
                gemini_params["response_json_schema"] = schema  # 直接使用
            else:
                gemini_params["response_schema"] = convert_schema_to_openapi_format(schema)

    if tools:
        function_declarations = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": convert_schema_to_openapi_format(t["function"]["parameters"]),
            }
            for t in tools if t.get("type") == "function"
        ]
        gemini_params["tools"] = [{"functionDeclarations": function_declarations}]

    if tool_choice and tools:
        mode_map = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
        gemini_params["tool_choice"] = {
            "functionCallingConfig": {"mode": mode_map.get(tool_choice, "AUTO")}
        }

    return gemini_params


def convert_schema_to_openapi_format(schema: Dict) -> Dict:
    """
    遞迴將 JSON Schema 轉換為 Gemini OpenAPI 格式：
    - type: string → STRING, object → OBJECT, array → ARRAY, etc.
    - anyOf with null → nullable: true（移除 null type）
    """
    type_map = {
        "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
        "boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY"
    }
    result = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result["type"] = type_map.get(v, v.upper())
        elif k == "properties":
            result["properties"] = {pk: convert_schema_to_openapi_format(pv) for pk, pv in v.items()}
        elif k == "items":
            result["items"] = convert_schema_to_openapi_format(v)
        elif k == "anyOf":
            # anyOf with null → nullable
            non_null = [x for x in v if x.get("type") != "null"]
            if len(v) != len(non_null):  # 有 null type
                if len(non_null) == 1:
                    result.update(convert_schema_to_openapi_format(non_null[0]))
                    result["nullable"] = True
                else:
                    result["anyOf"] = [convert_schema_to_openapi_format(x) for x in non_null]
                    result["nullable"] = True
        elif k not in ("$schema", "$defs", "additionalProperties", "strict"):
            result[k] = v   # 直接保留其他欄位（description, required, enum 等）
    return result


# ─────────────────────────────────────────────
# 5. Request Body 建構
# ─────────────────────────────────────────────

def build_request_body(messages: List[Dict], gemini_params: Dict, model: str) -> Dict:
    # 分離 system message
    system_instruction = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = {"role": "user", "parts": [{"text": msg["content"]}]}
        else:
            filtered_messages.append(msg)

    # 轉換 messages → Gemini contents
    contents = convert_messages_to_gemini_format(filtered_messages)

    # 組裝 GenerationConfig
    config_keys = {"temperature", "top_p", "max_output_tokens", "stop_sequences",
                   "candidate_count", "response_mime_type", "response_schema",
                   "response_json_schema", "seed", "frequency_penalty", "presence_penalty"}
    generation_config = {k: v for k, v in gemini_params.items() if k in config_keys}

    body = {"contents": contents}
    if system_instruction:
        body["system_instruction"] = system_instruction
    if "tools" in gemini_params:
        body["tools"] = gemini_params["tools"]
    if "tool_choice" in gemini_params:
        body["toolConfig"] = gemini_params["tool_choice"]
    if generation_config:
        body["generationConfig"] = generation_config

    return body


def convert_messages_to_gemini_format(messages: List[Dict]) -> List[Dict]:
    """
    OpenAI messages → Gemini contents
    - 合併連續相同 role 的 messages
    - user/system → role: "user"
    - assistant → role: "model"
    - tool → role: "user"（function response）
    """
    contents = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] in ("user", "system"):
            # 合併連續的 user/system
            parts = []
            while i < len(messages) and messages[i]["role"] in ("user", "system"):
                content = messages[i]["content"]
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if part["type"] == "text":
                            parts.append({"text": part["text"]})
                        elif part["type"] == "image_url":
                            # 處理圖片 URL → inlineData or fileData
                            parts.append(process_image_part(part["image_url"]["url"]))
                i += 1
            contents.append({"role": "user", "parts": parts})

        elif msg["role"] == "assistant":
            # 合併連續的 assistant messages
            parts = []
            while i < len(messages) and messages[i]["role"] == "assistant":
                content = messages[i].get("content")
                if content:
                    parts.append({"text": content})
                # 處理 tool_calls
                for tc in messages[i].get("tool_calls", []):
                    parts.append({
                        "functionCall": {
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]),
                        }
                    })
                i += 1
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif msg["role"] == "tool":
            # tool response → function response part
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": msg.get("name", ""),
                        "response": {"result": msg["content"]},
                    }
                }]
            })
            i += 1
        else:
            i += 1

    return contents


def process_image_part(image_url: str) -> Dict:
    if image_url.startswith("gs://"):
        # Google Cloud Storage URI
        return {"fileData": {"mimeType": "image/jpeg", "fileUri": image_url}}
    elif image_url.startswith("data:"):
        # base64 內嵌
        header, data = image_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        return {"inlineData": {"mimeType": mime_type, "data": data}}
    else:
        # HTTP URL → Vertex AI 使用 fileData（Google AI Studio 需下載後轉 base64）
        return {"fileData": {"mimeType": "image/jpeg", "fileUri": image_url}}


# ─────────────────────────────────────────────
# 6. HTTP 請求 & Response 轉換
# ─────────────────────────────────────────────

def handle_sync_response(url: str, headers: Dict, body: Dict, model: str) -> Dict:
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=body, timeout=600)
        response.raise_for_status()

    return convert_gemini_response_to_openai(response.json(), model)


def convert_gemini_response_to_openai(gemini_resp: Dict, model: str) -> Dict:
    finish_reason_map = {
        "STOP": "stop", "MAX_TOKENS": "length",
        "SAFETY": "content_filter", "RECITATION": "content_filter",
    }

    candidate = gemini_resp["candidates"][0]
    parts = candidate["content"]["parts"]
    text_content = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    reasoning_content = "".join(p.get("text", "") for p in parts if p.get("thought"))

    # 提取 function calls
    tool_calls = None
    fc_parts = [p for p in parts if "functionCall" in p]
    if fc_parts:
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": p["functionCall"]["name"],
                    "arguments": json.dumps(p["functionCall"]["args"]),
                }
            }
            for i, p in enumerate(fc_parts)
        ]

    usage = gemini_resp.get("usageMetadata", {})
    return {
        "id": gemini_resp.get("responseId", ""),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content or None,
                "reasoning_content": reasoning_content or None,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason_map.get(candidate.get("finishReason", "STOP"), "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }
    }


def handle_stream_response(url: str, headers: Dict, body: Dict, model: str):
    """
    SSE streaming：逐行解析 data: {...} chunks
    每個 chunk 都是部分 Gemini response，轉換為 OpenAI delta 格式
    """
    with httpx.Client() as client:
        with client.stream("POST", url, headers=headers, json=body, timeout=600) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    chunk_data = json.loads(line[6:])  # 去除 "data: " 前綴
                    yield convert_gemini_chunk_to_openai_delta(chunk_data, model)


def convert_gemini_chunk_to_openai_delta(chunk: Dict, model: str) -> Dict:
    candidates = chunk.get("candidates", [])
    finish_reason = None
    content_delta = ""

    if candidates:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        content_delta = "".join(p.get("text", "") for p in parts)
        finish_reason_raw = candidate.get("finishReason")
        if finish_reason_raw:
            finish_reason = {"STOP": "stop", "MAX_TOKENS": "length"}.get(finish_reason_raw)

    result = {
        "id": chunk.get("responseId", ""),
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content_delta},
            "finish_reason": finish_reason,
        }]
    }

    # 最後一個 chunk 附帶 usage
    if "usageMetadata" in chunk:
        usage = chunk["usageMetadata"]
        result["usage"] = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }

    return result


# ─────────────────────────────────────────────
# 使用範例
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 一般呼叫
    result = completion(
        model="vertex_ai/gemini-2.0-flash",
        messages=[{"role": "user", "content": "你好！"}],
        temperature=0.7,
        max_tokens=512,
        vertex_project="my-gcp-project",
        vertex_location="us-central1",
        # vertex_credentials="/path/to/service_account.json"
        # 或設環境變數 VERTEXAI_CREDENTIALS、GOOGLE_APPLICATION_CREDENTIALS
    )
    print(result["choices"][0]["message"]["content"])

    # Streaming 呼叫
    for chunk in completion(
        model="vertex_ai/gemini-2.0-flash",
        messages=[{"role": "user", "content": "寫一首詩"}],
        stream=True,
    ):
        print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

---

*分析截止日期：2026-03-28*
*LiteLLM 版本：以 `main` branch 當前 commit 為準*
*對應官方文件：*
- *[Vertex AI Gemini API](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)*
- *[Google AI Studio (Gemini API)](https://ai.google.dev/api/generate-content)*
- *[OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)*
