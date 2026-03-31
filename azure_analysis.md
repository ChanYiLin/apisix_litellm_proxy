# LiteLLM Azure OpenAI GPT 源碼解析

> 本文以 `completion(model="azure/gpt-4", ...)` 為起點，完整追蹤從入口到實際 HTTP 請求發出的每一段邏輯，著重 Azure GPT 路徑。

---

## 目錄

1. [整體執行路徑概覽](#1-整體執行路徑概覽)
2. [入口：`completion()` 函數](#2-入口completion-函數)
3. [Provider 辨識：`get_llm_provider()`](#3-provider-辨識get_llm_provider)
4. [參數處理：`get_optional_params()`](#4-參數處理get_optional_params)
5. [Azure 環境變數解析](#5-azure-環境變數解析)
6. [認證授權機制](#6-認證授權機制)
7. [Request Body 轉換](#7-request-body-轉換)
8. [Azure OpenAI Client 初始化](#8-azure-openai-client-初始化)
9. [實際 HTTP 請求發送](#9-實際-http-請求發送)
10. [Response 處理：轉換為 OpenAI-compatible 格式](#10-response-處理轉換為-openai-compatible-格式)
11. [Streaming 處理](#11-streaming-處理)
12. [錯誤處理](#12-錯誤處理)
13. [附錄：概念簡化版完整執行路徑](#13-附錄概念簡化版完整執行路徑)

---

## 1. 整體執行路徑概覽

```
litellm.completion()                          # main.py:1050
  ↓ 驗證 messages/tools/params
  ↓ get_llm_provider()                        # 辨識出 "azure"
  ↓ get_optional_params()                     # 過濾 + 映射 API 參數
  ↓ 解析 Azure 環境變數 (api_base, api_key…)
  ↓ azure_chat_completions.completion()       # azure/azure.py:187
      ↓ AzureOpenAIConfig().transform_request()   # 組裝 request body
      ↓ get_azure_openai_client()                 # 初始化 SDK client
      ↓ azure_client.chat.completions.with_raw_response.create(**data)
          ↓  HTTP POST → Azure OpenAI REST API
      ↓ response.model_dump()                     # 轉為 dict
      ↓ convert_to_model_response_object()        # 轉為 OpenAI-compatible ModelResponse
```

---

## 2. 入口：`completion()` 函數

**檔案：** [litellm/main.py](litellm/main.py#L1050)

### 2.1 函數簽名

```python
# main.py:1050-1102
def completion(
    model: str,
    messages: List = [],
    timeout: Optional[Union[float, str, httpx.Timeout]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    reasoning_effort: Optional[Literal["none","minimal","low","medium","high","xhigh","default"]] = None,
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    parallel_tool_calls: Optional[bool] = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,       # <-- 涵蓋 custom_llm_provider, api_base, metadata 等
) -> Union[ModelResponse, CustomStreamWrapper]:
```

### 2.2 前置驗證 (main.py:1148-1159)

```python
# 驗證並修正 messages 格式（確保角色值合法、content 型別正確）
messages = validate_and_fix_openai_messages(messages=messages)

# 驗證 tools 格式
tools = validate_and_fix_openai_tools(tools=tools)

# 驗證 tool_choice：確保值為合法 string 或 dict
tool_choice = validate_chat_completion_tool_choice(tool_choice=tool_choice)

# 驗證 stop：確保為 str 或 List[str]，最多 4 個
stop = validate_openai_optional_params(stop=stop)

# 標準化 thinking 參數（camelCase → snake_case）
thinking = validate_and_fix_thinking_param(thinking=thinking)
```

### 2.3 kwargs 解包 (main.py:1224-1310)

`**kwargs` 中包含大量 litellm 專屬控制參數，在此統一解包：

```python
api_base       = kwargs.get("api_base", None)
custom_llm_provider = kwargs.get("custom_llm_provider", None)
headers        = kwargs.get("headers", None) or extra_headers
acompletion    = kwargs.get("acompletion", False)  # 是否為 async 呼叫
max_retries    = kwargs.get("max_retries", None)
ssl_verify     = kwargs.get("ssl_verify", None)
fallbacks      = kwargs.get("fallbacks", None)
metadata       = kwargs.get("metadata", None)
# ... 以及 20+ 個其他控制參數
```

### 2.4 Provider 路由判斷 (main.py:1362-1374)

```python
# 向後相容：舊式 azure=True flag
if kwargs.get("azure", False) is True:
    custom_llm_provider = "azure"

# Azure deployment_id 模式（舊式）
if deployment_id is not None:
    model = deployment_id
    custom_llm_provider = "azure"

# 最終由 get_llm_provider() 確定 provider 並解析 api_base
model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
    model=model,
    custom_llm_provider=custom_llm_provider,
    api_base=api_base,
    api_key=api_key,
)
```

---

## 3. Provider 辨識：`get_llm_provider()`

**檔案：** [litellm/litellm_core_utils/get_llm_provider_logic.py](litellm/litellm_core_utils/get_llm_provider_logic.py#L99)

### 3.1 Azure 辨識邏輯

```python
# get_llm_provider_logic.py:142-209

# 情況 1：model 帶有 "azure/" 前綴，例如 "azure/gpt-4"
if model.split("/", 1)[0] in litellm.provider_list:
    # "azure" 在 provider_list 中
    custom_llm_provider = model.split("/", 1)[0]   # → "azure"
    model = model.split("/", 1)[1]                  # → "gpt-4"（即 deployment name）
    return model, custom_llm_provider, dynamic_api_key, api_base
```

**呼叫範例：**
```python
# 輸入
get_llm_provider(model="azure/my-gpt4-deployment")

# 輸出
# model = "my-gpt4-deployment"
# custom_llm_provider = "azure"
# dynamic_api_key = None
# api_base = None
```

> **注意：** `model` 在此步驟後變成 Azure 的 **deployment name**，而非 OpenAI 的 model name。這是因為 Azure 使用 deployment name 作為路徑參數。

---

## 4. 參數處理：`get_optional_params()`

**檔案：** [litellm/main.py](litellm/main.py#L1478-1518)

### 4.1 收集 optional_param_args

```python
# main.py:1478-1515
optional_param_args = {
    "functions": functions,
    "function_call": function_call,
    "temperature": temperature,
    "top_p": top_p,
    "n": n,
    "stream": stream,
    "stream_options": stream_options,
    "stop": stop,
    "max_tokens": max_tokens,
    "max_completion_tokens": max_completion_tokens,
    "presence_penalty": presence_penalty,
    "frequency_penalty": frequency_penalty,
    "logit_bias": logit_bias,
    "user": user,
    "model": model,
    "custom_llm_provider": custom_llm_provider,  # 用於決定哪些參數被支援
    "response_format": response_format,
    "seed": seed,
    "tools": tools,
    "tool_choice": tool_choice,
    "max_retries": max_retries,
    "logprobs": logprobs,
    "top_logprobs": top_logprobs,
    "api_version": api_version,
    "parallel_tool_calls": parallel_tool_calls,
    "reasoning_effort": reasoning_effort,
    # ...
}

# get_optional_params 內部會呼叫 AzureOpenAIConfig().map_openai_params()
# 只保留 Azure 支援的參數，並做版本檢查
optional_params = get_optional_params(**optional_param_args, **non_default_params)
```

### 4.2 Azure 支援的參數清單

**檔案：** [litellm/llms/azure/chat/gpt_transformation.py](litellm/llms/azure/chat/gpt_transformation.py#L78)

```python
# gpt_transformation.py:78-110
def get_supported_openai_params(self, model: str) -> List[str]:
    return [
        "temperature",
        "n",
        "stream",
        "stream_options",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "tools",
        "tool_choice",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "function_call",
        "functions",
        "top_p",
        "logprobs",
        "top_logprobs",
        "response_format",
        "seed",
        "extra_headers",
        "parallel_tool_calls",
        "prediction",
        "modalities",
        "audio",
        "web_search_options",
        "prompt_cache_key",
        "store",
    ]
```

### 4.3 版本相依的參數驗證

**檔案：** [litellm/llms/azure/chat/gpt_transformation.py](litellm/llms/azure/chat/gpt_transformation.py#L153)

```python
# map_openai_params() 中對 tool_choice 做 api_version 版本檢查
if param == "tool_choice":
    # tool_choice 需要 api_version >= 2023-12-01-preview
    if api_version_year < "2023" or (api_version_year == "2023" and api_version_month < "12"):
        if litellm.drop_params is True:
            pass  # 靜默丟棄
        else:
            raise UnsupportedParamsError(
                message="Azure does not support 'tool_choice', for api_version=..."
            )
    # tool_choice='required' 在 2024-05-01-preview 前不支援
    elif value == "required" and (api_version_year == "2024" and api_version_month <= "05"):
        raise UnsupportedParamsError(...)

# response_format 同樣有 API version + model 雙重限制
elif param == "response_format":
    is_response_format_supported = (
        self._is_response_format_supported_model(model)  # gpt-3.5 不支援
        and self._is_response_format_supported_api_version(year, month)
    )
```

---

## 5. Azure 環境變數解析

**檔案：** [litellm/main.py](litellm/main.py#L1657-1695)

```python
# main.py:1657-1695  (custom_llm_provider == "azure" 分支)

# 1. API Type（通常為 "azure"）
api_type = get_secret("AZURE_API_TYPE") or "azure"

# 2. API Base URL（優先順序：函數參數 > litellm 全域設定 > 環境變數）
api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")
# 例如: "https://my-resource.openai.azure.com"

# 3. API Version（Azure 必須）
api_version = (
    api_version                          # 函數參數優先
    or litellm.api_version               # 全域設定
    or get_secret_str("AZURE_API_VERSION")   # 環境變數
    or litellm.AZURE_DEFAULT_API_VERSION     # 內建預設值
)
# 例如: "2024-02-15-preview"

# 4. API Key（多個 fallback）
api_key = (
    api_key                              # 函數參數
    or litellm.api_key                   # 全域設定
    or litellm.azure_key                 # Azure 專屬全域設定
    or get_secret_str("AZURE_OPENAI_API_KEY")  # 環境變數（優先）
    or get_secret_str("AZURE_API_KEY")         # 環境變數（備用）
)

# 5. Azure AD Token（可選，用於 AAD 認證）
azure_ad_token = (
    optional_params.get("extra_body", {}).pop("azure_ad_token", None)
    or get_secret_str("AZURE_AD_TOKEN")
)

# 6. Azure AD Token Provider（callable，用於動態取得 token）
azure_ad_token_provider = litellm_params.get("azure_ad_token_provider", None)
```

### 環境變數總覽

| 環境變數 | 作用 | 是否必要 |
|---|---|---|
| `AZURE_API_BASE` | Azure OpenAI Endpoint URL | 必要（或函數參數） |
| `AZURE_API_VERSION` | API 版本（例如 `2024-02-15-preview`） | 必要（或有預設） |
| `AZURE_OPENAI_API_KEY` | 主要 API 金鑰 | 必要（或使用 AD token） |
| `AZURE_API_KEY` | 備用 API 金鑰 | 備用 |
| `AZURE_AD_TOKEN` | Azure AD Bearer Token（靜態） | 二擇一 |
| `AZURE_TENANT_ID` | Entra ID 認證用的 Tenant ID | 可選 |
| `AZURE_CLIENT_ID` | Entra ID / OIDC 用的 Client ID | 可選 |
| `AZURE_CLIENT_SECRET` | Entra ID 用的 Client Secret | 可選 |
| `AZURE_USERNAME` | 帳密認證用的 Username | 可選 |
| `AZURE_PASSWORD` | 帳密認證用的 Password | 可選 |
| `AZURE_SCOPE` | OAuth 2.0 Scope | 預設 `https://cognitiveservices.azure.com/.default` |
| `AZURE_AUTHORITY_HOST` | OIDC 認證 Authority URL | 預設 `https://login.microsoftonline.com` |
| `AZURE_API_TYPE` | API 類型 | 預設 `"azure"` |

---

## 6. 認證授權機制

**檔案：** [litellm/llms/azure/common_utils.py](litellm/llms/azure/common_utils.py#L524)

LiteLLM 支援多種 Azure 認證方式，依照以下優先順序決定使用哪種：

### 6.1 認證優先順序

```
1. api_key（直接 API Key 認證）
      ↓ 若無 api_key
2. Entra ID Client Credentials（tenant_id + client_id + client_secret）
      ↓ 若無上述
3. Username / Password（azure_username + azure_password + client_id）
      ↓ 若無上述
4. OIDC Token（azure_ad_token 以 "oidc/" 開頭）
      ↓ 若 litellm.enable_azure_ad_token_refresh == True
5. DefaultAzureCredential（自動探測環境認證）
      ↓ 若無上述
6. Static azure_ad_token（直接設定的 AD bearer token）
```

### 6.2 認證程式碼

```python
# common_utils.py:524-650 (initialize_azure_sdk_client)

# 解析認證所需變數
azure_ad_token_provider = litellm_params.get("azure_ad_token_provider")
azure_ad_token = litellm_params.get("azure_ad_token")

tenant_id = self._resolve_env_var(litellm_params, "tenant_id", "AZURE_TENANT_ID")
client_id = self._resolve_env_var(litellm_params, "client_id", "AZURE_CLIENT_ID")
client_secret = self._resolve_env_var(litellm_params, "client_secret", "AZURE_CLIENT_SECRET")
azure_username = self._resolve_env_var(litellm_params, "azure_username", "AZURE_USERNAME")
azure_password = self._resolve_env_var(litellm_params, "azure_password", "AZURE_PASSWORD")
scope = self._resolve_env_var(litellm_params, "azure_scope", "AZURE_SCOPE") \
        or "https://cognitiveservices.azure.com/.default"

# 方式 2：Entra ID Client Credentials（Service Principal）
if not api_key and azure_ad_token_provider is None and tenant_id and client_id and client_secret:
    azure_ad_token_provider = get_azure_ad_token_from_entra_id(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
    )
    # 內部使用 azure.identity.ClientSecretCredential + get_bearer_token_provider

# 方式 3：Username/Password
if azure_ad_token_provider is None and azure_username and azure_password and client_id:
    azure_ad_token_provider = get_azure_ad_token_from_username_password(
        azure_username=azure_username,
        azure_password=azure_password,
        client_id=client_id,
        scope=scope,
    )
    # 內部使用 azure.identity.UsernamePasswordCredential + get_bearer_token_provider

# 方式 4：OIDC Token（格式：azure_ad_token="oidc/MY_ENV_VAR"）
if azure_ad_token is not None and azure_ad_token.startswith("oidc/"):
    azure_ad_token = get_azure_ad_token_from_oidc(
        azure_ad_token=azure_ad_token,
        azure_client_id=client_id,
        azure_tenant_id=tenant_id,
        scope=scope,
    )
    # 向 login.microsoftonline.com 換取 Access Token

# 方式 5：DefaultAzureCredential
elif not api_key and azure_ad_token_provider is None and litellm.enable_azure_ad_token_refresh:
    azure_ad_token_provider = get_azure_ad_token_provider(azure_scope=scope)
```

### 6.3 認證在 HTTP 請求中的呈現

| 認證方式 | HTTP Header |
|---|---|
| API Key | `api-key: {key}` |
| Azure AD Token（靜態） | `Authorization: Bearer {token}` |
| Azure AD Token Provider（動態） | `Authorization: Bearer {provider()}` |

> Azure OpenAI SDK 會自動選擇正確的 header 格式。

---

## 7. Request Body 轉換

### 7.1 `AzureChatCompletion.completion()` 組裝 data

**檔案：** [litellm/llms/azure/azure.py](litellm/llms/azure/azure.py#L248)

```python
# azure.py:248-255
# 一般 GPT 模型路徑（非 o1/o3 系列）
data = litellm.AzureOpenAIConfig().transform_request(
    model=model,          # deployment name，例如 "my-gpt4-deployment"
    messages=messages,
    optional_params=optional_params,  # 已過濾的 API 參數
    litellm_params=litellm_params,
    headers=headers or {},
)
```

### 7.2 `transform_request()` 實作

**檔案：** [litellm/llms/azure/chat/gpt_transformation.py](litellm/llms/azure/chat/gpt_transformation.py#L250)

```python
# gpt_transformation.py:250-263
def transform_request(
    self,
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    litellm_params: dict,
    headers: dict,
) -> dict:
    # 1. 轉換 messages 格式（Azure 特有需求）
    messages = convert_to_azure_openai_messages(messages)

    # 2. 組裝最終 request body
    return {
        "model": model,           # deployment name
        "messages": messages,     # 轉換後的 messages
        **optional_params,        # temperature, max_tokens, tools, etc.
    }
```

### 7.3 `convert_to_azure_openai_messages()` 細節

**檔案：** [litellm/litellm_core_utils/prompt_templates/factory.py](litellm/litellm_core_utils/prompt_templates/factory.py#L1194)

```python
# factory.py:1177-1207

def _azure_tool_call_invoke_helper(
    function_call_params: ChatCompletionToolCallFunctionChunk,
) -> Optional[ChatCompletionToolCallFunctionChunk]:
    """
    Azure requires 'arguments' to be a string.
    若 arguments 為 None，改為空字串。
    """
    if function_call_params.get("arguments") is None:
        function_call_params["arguments"] = ""  # None → ""
    return function_call_params


def _azure_image_url_helper(content: ChatCompletionImageObject):
    """
    Azure 要求 image_url 必須是 dict，而非裸字串。
    """
    if isinstance(content["image_url"], str):
        content["image_url"] = {"url": content["image_url"]}  # str → {"url": "..."}


def convert_to_azure_openai_messages(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    for m in messages:
        # 1. assistant message 中的 function_call.arguments 必須是字串
        if m["role"] == "assistant":
            function_call = m.get("function_call", None)
            if function_call is not None:
                m["function_call"] = _azure_tool_call_invoke_helper(function_call)

        # 2. user message 中的 image_url 必須是 dict 格式
        if m["role"] == "user" and isinstance(m.get("content"), list):
            for content in m.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image_url":
                    _azure_image_url_helper(content)

    return messages
```

### 7.4 Request Body 轉換範例

**轉換前（標準 OpenAI 格式）：**
```json
{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述這張圖片"},
        {"type": "image_url", "image_url": "https://example.com/img.jpg"}
      ]
    },
    {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_weather",
        "arguments": null
      }
    }
  ],
  "temperature": 0.7,
  "tool_choice": "auto"
}
```

**轉換後（Azure 接受格式）：**
```json
{
  "model": "my-gpt4-deployment",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述這張圖片"},
        {
          "type": "image_url",
          "image_url": {"url": "https://example.com/img.jpg"}
        }
      ]
    },
    {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_weather",
        "arguments": ""
      }
    }
  ],
  "temperature": 0.7,
  "tool_choice": "auto"
}
```

**關鍵差異：**
1. `model` 由模型名稱改為 **deployment name**
2. `image_url` 從字串改為 `{"url": "..."}` dict
3. `function_call.arguments: null` 改為 `""`（空字串）

---

## 8. Azure OpenAI Client 初始化

**檔案：** [litellm/llms/azure/common_utils.py](litellm/llms/azure/common_utils.py#L439)

### 8.1 Client 取得流程

```python
# common_utils.py:439-522
def get_azure_openai_client(
    self,
    api_key, api_base, api_version=None, client=None,
    litellm_params=None, _is_async=False, model=None,
):
    # 1. 若已有 client 物件，直接使用（支援使用者傳入自訂 client）
    if client is not None:
        openai_client = client
        # 更新 api_version（若有動態覆寫）
        if api_version and isinstance(openai_client, (AzureOpenAI, AsyncAzureOpenAI)):
            openai_client._custom_query.setdefault("api-version", api_version)
        return openai_client

    # 2. 嘗試從 in-memory cache 取得（使用 client_initialization_params 為 key）
    cached_client = self.get_cached_openai_client(
        client_initialization_params=client_initialization_params,
        client_type="azure",
    )
    if cached_client:
        return cached_client

    # 3. 初始化新的 client 參數
    azure_client_params = self.initialize_azure_sdk_client(
        litellm_params=litellm_params or {},
        api_key=api_key, api_base=api_base,
        model_name=model, api_version=api_version,
        is_async=_is_async,
    )

    # 4. 判斷使用哪種 client
    if self._is_azure_v1_api_version(api_version):
        # v1 API（api_version in ["v1", "latest", "preview"]）
        # 使用標準 OpenAI client，endpoint 為 {api_base}/openai/v1/
        v1_params = {
            "api_key": azure_client_params.get("api_key"),
            "base_url": f"{api_base}/openai/v1/",
        }
        if _is_async:
            openai_client = AsyncOpenAI(**v1_params)
        else:
            openai_client = OpenAI(**v1_params)
    else:
        # 傳統 Azure API（帶日期的版本，例如 2024-02-15-preview）
        # 使用 AzureOpenAI client
        if _is_async:
            openai_client = AsyncAzureOpenAI(**azure_client_params)
        else:
            openai_client = AzureOpenAI(**azure_client_params)

    # 5. 存入 in-memory cache
    self.set_cached_openai_client(openai_client=openai_client, ...)

    return openai_client
```

### 8.2 AzureOpenAI Client 參數

```python
# common_utils.py:624-650
azure_client_params = {
    "api_key": api_key,                           # API Key（與 azure_ad_token 二選一）
    "azure_endpoint": api_base,                   # https://your-resource.openai.azure.com
    "api_version": api_version,                   # 2024-02-15-preview
    "azure_ad_token": azure_ad_token,             # 靜態 AD token（若有）
    "azure_ad_token_provider": azure_ad_token_provider,  # 動態 token callable（若有）
    "http_client": self._get_sync_http_client(),  # httpx client（含 SSL 設定）
    "max_retries": max_retries,                   # 重試次數
    "timeout": timeout,                           # 請求超時
}
# select_azure_base_url_or_endpoint() 決定用 azure_endpoint 還是 base_url
# （GPT-4 Vision Enhancement 需要 base_url）
azure_client_params = select_azure_base_url_or_endpoint(azure_client_params)
```

---

## 9. 實際 HTTP 請求發送

**檔案：** [litellm/llms/azure/azure.py](litellm/llms/azure/azure.py#L135)

### 9.1 同步請求

```python
# azure.py:310-368  (non-streaming sync path)

# 前置 logging
logging_obj.pre_call(
    input=messages,
    api_key=api_key,
    additional_args={
        "headers": {"api_key": api_key, "azure_ad_token": azure_ad_token},
        "api_version": api_version,
        "api_base": api_base,
        "complete_input_dict": data,  # 完整的 request body
    },
)

# 取得 client
azure_client = self.get_azure_openai_client(
    api_version=api_version, api_base=api_base, api_key=api_key,
    model=model, client=client, _is_async=False, litellm_params=litellm_params,
)

# 發送請求（使用 with_raw_response 以擷取 response headers）
headers, response = self.make_sync_azure_openai_chat_completion_request(
    azure_client=azure_client, data=data, timeout=timeout
)

# azure.py:135-155
def make_sync_azure_openai_chat_completion_request(
    self,
    azure_client: Union[AzureOpenAI, OpenAI],
    data: dict,
    timeout: Union[float, httpx.Timeout],
):
    # with_raw_response 可取得完整 HTTP response headers
    raw_response = azure_client.chat.completions.with_raw_response.create(
        **data, timeout=timeout
    )
    headers = dict(raw_response.headers)  # 保留所有 response headers
    response = raw_response.parse()       # 解析為 ChatCompletion pydantic 物件
    return headers, response
```

### 9.2 實際發出的 HTTP 請求格式

```
POST https://{resource-name}.openai.azure.com/openai/deployments/{deployment-id}/chat/completions?api-version=2024-02-15-preview

Headers:
  api-key: {AZURE_OPENAI_API_KEY}      ← API Key 認證
  Content-Type: application/json
  User-Agent: AsyncAzureOpenAI/...

Body:
{
  "model": "my-gpt4-deployment",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

若使用 Azure AD Token 認證：
```
Headers:
  Authorization: Bearer eyJ0eXAiOiJKV1Qi...   ← AD Token 認證（取代 api-key）
```

### 9.3 異步請求

```python
# azure.py:385-493 (acompletion)

async def acompletion(self, ...):
    azure_client = self.get_azure_openai_client(..., _is_async=True, ...)

    logging_obj.pre_call(...)

    # @track_llm_api_timing() 裝飾器會記錄請求耗時
    headers, response = await self.make_azure_openai_chat_completion_request(
        azure_client=azure_client, data=data, timeout=timeout, logging_obj=logging_obj,
    )

    # 轉為 dict 並轉換格式
    stringified_response = response.model_dump()
    return convert_to_model_response_object(
        response_object=stringified_response,
        model_response_object=model_response,
        _response_headers=headers,
        convert_tool_call_to_json_mode=convert_tool_call_to_json_mode,
    )
```

---

## 10. Response 處理：轉換為 OpenAI-compatible 格式

**檔案：** [litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py](litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py#L447)

### 10.1 Azure 原始 Response 格式

Azure OpenAI API 的回應本身已是 OpenAI-compatible 格式（因為 Azure OpenAI 就是封裝 OpenAI API），但 LiteLLM 仍需做標準化處理：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  },
  "system_fingerprint": "fp_abc123"
}
```

### 10.2 `convert_to_model_response_object()` 轉換邏輯

```python
# convert_dict_to_response.py:447-679

def convert_to_model_response_object(
    response_object: Optional[dict] = None,
    model_response_object: Optional[ModelResponse] = None,
    _response_headers: Optional[dict] = None,
    convert_tool_call_to_json_mode: Optional[bool] = None,
    ...
):
    # 1. 從 response headers 提取 rate limit / metadata
    additional_headers = get_response_headers(_response_headers)
    hidden_params["additional_headers"] = additional_headers

    # 2. 遍歷 choices，建立 litellm Choices 物件
    for idx, choice in enumerate(response_object["choices"]):
        tool_calls = choice["message"].get("tool_calls", None)

        # 3. JSON mode 支援：若 response_format 是 json_schema，
        #    Azure 可能把它包成 tool call；在此展開回 content
        if _should_convert_tool_call_to_json_mode(tool_calls, convert_tool_call_to_json_mode):
            json_mode_content_str = tool_calls[0]["function"].get("arguments")
            message = litellm.Message(content=json_mode_content_str)
            finish_reason = "stop"
        else:
            # 4. 建立標準 Message 物件
            message = Message(
                content=choice["message"].get("content"),
                role=choice["message"]["role"] or "assistant",
                function_call=choice["message"].get("function_call", None),
                tool_calls=tool_calls,
                audio=choice["message"].get("audio", None),
                annotations=choice["message"].get("annotations", None),
                reasoning_content=reasoning_content,
            )
            finish_reason = choice.get("finish_reason")

        # 5. finish_reason 修正：若有 tool_calls 但 finish_reason=stop，改為 tool_calls
        if finish_reason == "stop" and message.tool_calls:
            finish_reason = "tool_calls"

        choice_obj = Choices(
            finish_reason=finish_reason,
            index=idx,
            message=message,
            logprobs=choice.get("logprobs"),
        )
        choice_list.append(choice_obj)

    model_response_object.choices = choice_list

    # 6. usage 填充（token 計數）
    if "usage" in response_object:
        usage_object = litellm.Usage(**response_object["usage"])
        setattr(model_response_object, "usage", usage_object)

    # 7. 時間戳轉換（部分 provider 回傳 float，統一轉為 int）
    if "created" in response_object:
        model_response_object.created = _safe_convert_created_field(response_object["created"])

    # 8. id / model / system_fingerprint 填充
    model_response_object.id = response_object.get("id") or model_response_object.id
    model_response_object.system_fingerprint = response_object.get("system_fingerprint")

    return model_response_object
```

### 10.3 Response Headers 處理

**檔案：** [litellm/llms/azure/common_utils.py](litellm/llms/azure/common_utils.py#L43)

```python
# common_utils.py:43-63
def process_azure_headers(headers: Union[httpx.Headers, dict]) -> dict:
    """
    提取 Azure rate limit headers，並統一加上 "llm_provider-" 前綴
    """
    openai_headers = {}
    # 保留標準 rate limit headers（OpenAI-compatible）
    for h in ["x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
              "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"]:
        if h in headers:
            openai_headers[h] = headers[h]

    # 所有 response headers 加上 "llm_provider-" 前綴，暴露給上層
    llm_response_headers = {
        f"llm_provider-{k}": v for k, v in headers.items()
    }

    return {**llm_response_headers, **openai_headers}
```

### 10.4 最終 ModelResponse 結構

```python
ModelResponse(
    id="chatcmpl-abc123",
    object="chat.completion",
    created=1677858242,
    model="my-gpt4-deployment",   # Azure deployment name
    choices=[
        Choices(
            index=0,
            finish_reason="stop",
            message=Message(
                role="assistant",
                content="Hello! How can I help you?",
                tool_calls=None,
            )
        )
    ],
    usage=Usage(
        prompt_tokens=15,
        completion_tokens=10,
        total_tokens=25,
    ),
    system_fingerprint="fp_abc123",
    _hidden_params={
        "custom_llm_provider": "azure",
        "additional_headers": {
            "x-ratelimit-remaining-requests": "99",
            "llm_provider-x-ms-region": "eastus",
            ...
        }
    }
)
```

---

## 11. Streaming 處理

### 11.1 Streaming 請求發送

**檔案：** [litellm/llms/azure/azure.py](litellm/llms/azure/azure.py#L495)

```python
# azure.py:495-571  (sync streaming)
def streaming(self, logging_obj, api_base, api_key, api_version, data, model, timeout, ...):
    azure_client = self.get_azure_openai_client(..., _is_async=False, ...)

    logging_obj.pre_call(input=data["messages"], ...)

    # data 中已包含 "stream": True
    # with_raw_response.create() 回傳的是 SSE stream iterator
    headers, response = self.make_sync_azure_openai_chat_completion_request(
        azure_client=azure_client, data=data, timeout=timeout
    )

    # 包裝成 CustomStreamWrapper，提供統一的迭代介面
    streamwrapper = CustomStreamWrapper(
        completion_stream=response,      # Iterator[ChatCompletionChunk]
        model=model,
        custom_llm_provider="azure",
        logging_obj=logging_obj,
        stream_options=data.get("stream_options", None),
        _response_headers=process_azure_headers(headers),
    )
    return streamwrapper
```

### 11.2 Azure SSE Stream 格式

Azure 的 streaming response 是 Server-Sent Events（SSE）格式：

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Transfer-Encoding: chunked

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 11.3 `CustomStreamWrapper` 運作機制

**檔案：** [litellm/litellm_core_utils/streaming_handler.py](litellm/litellm_core_utils/streaming_handler.py#L98)

```python
# streaming_handler.py:98-180
class CustomStreamWrapper:
    def __init__(self, completion_stream, model, logging_obj, custom_llm_provider, ...):
        self.completion_stream = completion_stream  # 原始 SSE iterator
        self.model = model
        self.custom_llm_provider = custom_llm_provider

        # stream_options: {"include_usage": True} 時，最後一個 chunk 附加 usage stats
        self.send_stream_usage = self.check_send_stream_usage(stream_options)

        # 保留已回傳的 chunks（用於計算總 token 數）
        self.chunks: List = []

    def __iter__(self):
        return self

    def __next__(self) -> ModelResponseStream:
        # 1. 從底層 iterator 取得下一個 ChatCompletionChunk
        chunk = next(self.completion_stream)

        # 2. 使用 handle_openai_chat_completion_chunk() 解析
        #    （Azure 走 OpenAI SDK，chunk 已是 parsed object）
        parsed = self.handle_openai_chat_completion_chunk(chunk)
        # → {"text": "Hello", "is_finished": False, "finish_reason": None, ...}

        # 3. 建立 ModelResponseStream（OpenAI-compatible streaming chunk）
        model_response = ModelResponseStream(
            id=chunk.id,
            choices=[StreamingChoices(
                index=0,
                delta=Delta(content=parsed["text"], role=...),
                finish_reason=parsed["finish_reason"],
            )],
            model=self.model,
            created=chunk.created,
        )

        # 4. 記錄 chunk（用於最終 usage 計算）
        self.chunks.append(model_response)

        # 5. 無限循環偵測（重複 chunk 保護）
        self.raise_on_model_repetition()

        return model_response
```

### 11.4 Azure Chunk 解析

```python
# streaming_handler.py:510-556
def handle_openai_chat_completion_chunk(self, chunk):
    """
    Azure 使用 OpenAI SDK，chunk 已是 parsed ChatCompletionChunk object
    """
    text = ""
    is_finished = False
    finish_reason = None
    logprobs = None
    usage = None

    if chunk.choices and len(chunk.choices) > 0:
        if chunk.choices[0].delta is not None and chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
        if chunk.choices[0].finish_reason:
            is_finished = True
            finish_reason = chunk.choices[0].finish_reason
        logprobs = getattr(chunk.choices[0], "logprobs", None)

    usage = getattr(chunk, "usage", None)  # stream_options: include_usage=True 時有值

    return {
        "text": text,
        "is_finished": is_finished,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
        "usage": usage,
    }
```

### 11.5 Streaming 輸出格式

使用者迭代 `CustomStreamWrapper` 收到的每個 `ModelResponseStream`：

```python
# 第 1 個 chunk（角色宣告）
ModelResponseStream(
    id="chatcmpl-abc",
    choices=[StreamingChoices(delta=Delta(role="assistant", content=""), finish_reason=None)],
    created=1677858242,
)

# 中間 chunks（內容片段）
ModelResponseStream(
    id="chatcmpl-abc",
    choices=[StreamingChoices(delta=Delta(content="Hello"), finish_reason=None)],
    created=1677858242,
)

# 最後一個 chunk（完成信號）
ModelResponseStream(
    id="chatcmpl-abc",
    choices=[StreamingChoices(delta=Delta(content=""), finish_reason="stop")],
    created=1677858242,
)

# 若 stream_options={"include_usage": True}，則附加 usage chunk
ModelResponseStream(
    id="chatcmpl-abc",
    choices=[],
    usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
)
```

---

## 12. 錯誤處理

**檔案：** [litellm/llms/azure/azure.py](litellm/llms/azure/azure.py#L369)

```python
# azure.py:369-383
except AzureOpenAIError as e:
    raise e  # 直接向上傳遞
except Exception as e:
    # 將 OpenAI SDK 的各種 exception 統一包裝為 AzureOpenAIError
    status_code = getattr(e, "status_code", 500)
    error_headers = getattr(e, "headers", None)
    error_response = getattr(e, "response", None)
    error_body = getattr(e, "body", None)
    if error_headers is None and error_response:
        error_headers = getattr(error_response, "headers", None)
    raise AzureOpenAIError(
        status_code=status_code,
        message=str(e),
        headers=error_headers,
        body=error_body,
    )
```

**常見錯誤對應：**

| Azure 錯誤 | status_code | 原因 |
|---|---|---|
| `AuthenticationError` | 401 | API Key 或 AD Token 無效 |
| `RateLimitError` | 429 | 超過 rate limit |
| `BadRequestError` | 400 | 請求格式錯誤（例如不支援的參數） |
| `NotFoundError` | 404 | Deployment name 不存在 |
| `APITimeoutError` | 408 | 請求超時 |
| `InternalServerError` | 500 | Azure 服務端錯誤 |

---

## 13. 附錄：概念簡化版完整執行路徑

以下為去除所有非 Azure GPT 邏輯後的概念性簡化代碼，完整呈現執行路徑：

```python
# ============================================================
# 概念簡化版：LiteLLM Azure GPT completion 執行路徑
# ============================================================

import os
import httpx
from openai import AzureOpenAI
from pydantic import BaseModel
from typing import List, Optional, Union, Iterator


# ---------- 型別定義（簡化）----------

class Message(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: Optional[list] = None

class Choices(BaseModel):
    index: int
    finish_reason: str
    message: Message

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ModelResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choices]
    usage: Usage


# ---------- Step 1: 入口函數 ----------

def completion(
    model: str,                              # 例如 "azure/my-gpt4-deployment"
    messages: List[dict],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs,
) -> Union[ModelResponse, Iterator]:

    # --- 驗證輸入 ---
    assert model is not None, "model is required"
    messages = validate_messages(messages)  # 確保格式合法

    # --- Step 2: 辨識 Provider ---
    # "azure/my-gpt4-deployment" → provider="azure", model="my-gpt4-deployment"
    if "/" in model and model.split("/")[0] == "azure":
        custom_llm_provider = "azure"
        model = model.split("/", 1)[1]   # deployment name

    # --- Step 3: 解析環境變數（優先順序：參數 > 環境變數）---
    api_base = api_base or os.environ.get("AZURE_API_BASE")
    api_version = api_version or os.environ.get("AZURE_API_VERSION", "2024-02-15-preview")
    api_key = (
        api_key
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_API_KEY")
    )
    azure_ad_token = os.environ.get("AZURE_AD_TOKEN")  # 可選：AD Token 認證

    # --- Step 4: 過濾 + 組裝 optional_params ---
    AZURE_SUPPORTED_PARAMS = {
        "temperature", "top_p", "n", "stream", "stream_options",
        "stop", "max_tokens", "max_completion_tokens", "tools", "tool_choice",
        "presence_penalty", "frequency_penalty", "logit_bias", "user",
        "logprobs", "top_logprobs", "response_format", "seed",
        "parallel_tool_calls", "functions", "function_call",
    }
    optional_params = {}
    all_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        **kwargs,
    }
    for param, value in all_params.items():
        if param in AZURE_SUPPORTED_PARAMS and value is not None:
            optional_params[param] = value

    # --- Step 5: 轉換 messages（Azure 特有格式要求）---
    messages = convert_to_azure_openai_messages(messages)

    # --- Step 6: 組裝 request body ---
    data = {
        "model": model,          # deployment name（不是 OpenAI model name）
        "messages": messages,
        **optional_params,
    }

    # --- Step 7: 初始化 Azure OpenAI Client ---
    azure_client = get_azure_client(
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        azure_ad_token=azure_ad_token,
        is_async=False,
    )

    # --- Step 8: 發送 HTTP 請求 ---
    if stream:
        return handle_streaming(azure_client, data)
    else:
        return handle_non_streaming(azure_client, data, model)


# ---------- Step 5 detail: messages 轉換 ----------

def convert_to_azure_openai_messages(messages: List[dict]) -> List[dict]:
    """
    Azure 要求：
    1. function_call.arguments 必須是字串（不能是 None）
    2. image_url 必須是 dict {"url": "..."} 而非裸字串
    """
    for m in messages:
        # 修正 function_call.arguments
        if m.get("role") == "assistant" and m.get("function_call"):
            if m["function_call"].get("arguments") is None:
                m["function_call"]["arguments"] = ""  # None → ""

        # 修正 image_url 格式
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for content_block in m["content"]:
                if isinstance(content_block, dict) and content_block.get("type") == "image_url":
                    if isinstance(content_block["image_url"], str):
                        content_block["image_url"] = {"url": content_block["image_url"]}

    return messages


# ---------- Step 7 detail: Client 初始化 ----------

def get_azure_client(
    api_key: Optional[str],
    api_base: Optional[str],
    api_version: str,
    azure_ad_token: Optional[str] = None,
    is_async: bool = False,
) -> AzureOpenAI:
    """
    建立 AzureOpenAI SDK client
    - api_key 用於 api-key header
    - azure_ad_token 用於 Authorization: Bearer header
    """
    client_params = {
        "azure_endpoint": api_base,  # https://your-resource.openai.azure.com
        "api_version": api_version,
        # SDK 自動設定 HTTP: POST {api_base}/openai/deployments/{model}/chat/completions
    }

    if api_key:
        client_params["api_key"] = api_key
    elif azure_ad_token:
        client_params["azure_ad_token"] = azure_ad_token

    # 可選：Entra ID 動態 Token（優先於靜態 token）
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if not api_key and tenant_id and client_id and client_secret:
        from azure.identity import ClientSecretCredential, get_bearer_token_provider
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        client_params["azure_ad_token_provider"] = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

    return AzureOpenAI(**client_params)


# ---------- Step 8 detail: Non-Streaming 請求 ----------

def handle_non_streaming(azure_client: AzureOpenAI, data: dict, model: str) -> ModelResponse:
    """
    實際 HTTP 請求：
    POST https://{resource}.openai.azure.com/openai/deployments/{model}/chat/completions?api-version=...
    """
    raw_response = azure_client.chat.completions.with_raw_response.create(**data)
    response_headers = dict(raw_response.headers)   # 保留 rate-limit headers
    response = raw_response.parse()                  # → openai.types.chat.ChatCompletion

    # 轉換為 LiteLLM ModelResponse
    return convert_to_model_response(
        response_dict=response.model_dump(),
        response_headers=response_headers,
        model=model,
    )


# ---------- Step 8 detail: Streaming 請求 ----------

def handle_streaming(azure_client: AzureOpenAI, data: dict) -> Iterator:
    """
    Streaming 請求，回傳 SSE chunk iterator
    """
    # Azure SDK 會處理 SSE 解析，回傳 ChatCompletionChunk iterator
    raw_response = azure_client.chat.completions.with_raw_response.create(**data)
    response_headers = dict(raw_response.headers)
    stream = raw_response.parse()   # → Stream[ChatCompletionChunk]

    # 逐個 chunk 轉換格式
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            yield {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": chunk.created,
                "model": chunk.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": getattr(delta, "role", None),
                        "content": getattr(delta, "content", None),
                        "tool_calls": getattr(delta, "tool_calls", None),
                    },
                    "finish_reason": finish_reason,
                }]
            }


# ---------- Step 10 detail: Response 轉換 ----------

def convert_to_model_response(
    response_dict: dict,
    response_headers: dict,
    model: str,
) -> ModelResponse:
    """
    將 Azure ChatCompletion dict 轉換為 OpenAI-compatible ModelResponse
    """
    choices = []
    for choice_data in response_dict["choices"]:
        tool_calls = choice_data["message"].get("tool_calls")
        finish_reason = choice_data.get("finish_reason", "stop")

        # 修正：有 tool_calls 但 finish_reason=stop → 改為 tool_calls
        if finish_reason == "stop" and tool_calls:
            finish_reason = "tool_calls"

        msg = Message(
            role=choice_data["message"]["role"],
            content=choice_data["message"].get("content"),
            tool_calls=tool_calls,
        )
        choices.append(Choices(
            index=choice_data["index"],
            finish_reason=finish_reason,
            message=msg,
        ))

    usage_data = response_dict.get("usage", {})
    return ModelResponse(
        id=response_dict["id"],
        created=int(response_dict["created"]),  # 確保是 int（部分 provider 回 float）
        model=response_dict.get("model", model),
        choices=choices,
        usage=Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
    )


# ---------- 使用範例 ----------

if __name__ == "__main__":
    import os

    os.environ["AZURE_API_BASE"] = "https://my-resource.openai.azure.com"
    os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"
    os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"

    # Non-streaming
    response = completion(
        model="azure/my-gpt4-deployment",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=100,
    )
    print(response.choices[0].message.content)

    # Streaming
    for chunk in completion(
        model="azure/my-gpt4-deployment",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ):
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
```

---

## 參考資料

- [Azure OpenAI REST API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
- [Azure OpenAI Authentication](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity)
- LiteLLM 原始碼關鍵檔案：
  - [litellm/main.py](litellm/main.py) — 入口與路由
  - [litellm/llms/azure/azure.py](litellm/llms/azure/azure.py) — AzureChatCompletion handler
  - [litellm/llms/azure/chat/gpt_transformation.py](litellm/llms/azure/chat/gpt_transformation.py) — 參數映射與 request 轉換
  - [litellm/llms/azure/common_utils.py](litellm/llms/azure/common_utils.py) — Client 初始化、認證邏輯
  - [litellm/litellm_core_utils/prompt_templates/factory.py](litellm/litellm_core_utils/prompt_templates/factory.py) — Messages 格式轉換
  - [litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py](litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py) — Response 轉換
  - [litellm/litellm_core_utils/streaming_handler.py](litellm/litellm_core_utils/streaming_handler.py) — CustomStreamWrapper
  - [litellm/litellm_core_utils/get_llm_provider_logic.py](litellm/litellm_core_utils/get_llm_provider_logic.py) — Provider 辨識
