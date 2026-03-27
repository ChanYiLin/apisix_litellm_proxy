# LiteLLM 源碼解析：`completion()` → AWS Bedrock Claude Sonnet

> 分析版本基於當前 main branch。聚焦路徑：`bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`

---

## 目錄

1. [整體呼叫流程圖](#1-整體呼叫流程圖)
2. [入口：`completion()` 函數](#2-入口completion-函數)
3. [Provider 辨識：`get_llm_provider()`](#3-provider-辨識get_llm_provider)
4. [參數處理：`get_optional_params()`](#4-參數處理get_optional_params)
5. [Bedrock Route 分發](#5-bedrock-route-分發)
6. [`BedrockConverseLLM.completion()`：主要 Handler](#6-bedrockconversellmcompletion主要-handler)
7. [認證授權：`BaseAWSLLM.get_credentials()`](#7-認證授權baseawsllmget_credentials)
8. [環境變數完整清單](#8-環境變數完整清單)
9. [Region 與 Endpoint 解析](#9-region-與-endpoint-解析)
10. [Request Body 轉換](#10-request-body-轉換)
11. [SigV4 請求簽名](#11-sigv4-請求簽名)
12. [HTTP 請求發送](#12-http-請求發送)
13. [Response 轉換：OpenAI-compatible 格式](#13-response-轉換openai-compatible-格式)
14. [Streaming 處理](#14-streaming-處理)
15. [附錄：概念精簡版執行路徑 Code](#15-附錄概念精簡版執行路徑-code)

---

## 1. 整體呼叫流程圖

```
litellm.completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Hello"}]
)
    │
    ▼
[main.py] completion()
    │  validate messages/tools/params
    │  get_llm_provider()  →  provider="bedrock", model="anthropic.claude-3-5-sonnet-..."
    │  get_optional_params()  →  map OpenAI params to Bedrock params
    │
    ▼
[main.py:3799] BedrockModelInfo.get_bedrock_route(model)
    │  "anthropic.claude-3-5-sonnet-..." in bedrock_converse_models
    │  → route = "converse"
    │
    ▼
[main.py:3802] bedrock_converse_chat_completion.completion(...)
    │  = BedrockConverseLLM.completion()  [converse_handler.py:251]
    │
    ├── 解析 model ID / region prefix
    ├── _get_aws_region_name()
    ├── get_credentials()  →  boto3 Credentials
    ├── get_runtime_endpoint()
    │     → "https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse"
    │
    ├── AmazonConverseConfig()._transform_request()   [converse_transformation.py]
    │     ├── _transform_system_message()   → SystemContentBlock[]
    │     ├── _transform_request_helper()   → inferenceConfig, toolConfig, ...
    │     └── _bedrock_converse_messages_pt()  → MessageBlock[]
    │
    ├── get_request_headers()   [base_aws_llm.py:1221]
    │     └── SigV4Auth.add_auth()  → Authorization, X-Amz-Date, X-Amz-Security-Token
    │
    ├── httpx_client.post(url, headers, data)   ← 實際 HTTP 請求
    │
    └── AmazonConverseConfig()._transform_response()
          ├── ConverseResponseBlock(**response.json())
          ├── _translate_message_content()  → content_str, tool_calls, thinking_blocks
          ├── _transform_usage()   → prompt_tokens, completion_tokens
          └── map_finish_reason()  → "stop" / "tool_calls" / "length"
              → ModelResponse (OpenAI-compatible)
```

---

## 2. 入口：`completion()` 函數

**檔案：** [litellm/main.py](litellm/main.py)

### 2.1 模組載入時預先實例化 Handler（line 281）

```python
# main.py:281 — 模組載入時就建好，避免每次 call 都重建
bedrock_converse_chat_completion = BedrockConverseLLM()
```

### 2.2 `completion()` 入口驗證（line ~1148）

```python
# main.py:1148-1159
### VALIDATE Request ###
if model is None:
    raise ValueError("model param not passed in.")

# 驗證並修正 OpenAI messages 格式
messages = validate_and_fix_openai_messages(messages=messages)
tools = validate_and_fix_openai_tools(tools=tools)
tool_choice = validate_chat_completion_tool_choice(tool_choice=tool_choice)
stop = validate_openai_optional_params(stop=stop)
# camelCase 正規化 e.g. budgetTokens -> budget_tokens
thinking = validate_and_fix_thinking_param(thinking=thinking)
```

### 2.3 kwargs 解包（line ~1224）

```python
# main.py:1224-1253 — 從 kwargs 提取所有控制參數
api_base = kwargs.get("api_base", None)
mock_response = kwargs.get("mock_response", None)
custom_llm_provider = kwargs.get("custom_llm_provider", None)
headers = kwargs.get("headers", None) or extra_headers
```

---

## 3. Provider 辨識：`get_llm_provider()`

**檔案：** [litellm/litellm_core_utils/get_llm_provider_logic.py](litellm/litellm_core_utils/get_llm_provider_logic.py) (line 99)

```python
# 輸入: model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
# 解析: model.split("/", 1) = ["bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"]

if (
    model.split("/", 1)[0] in litellm.provider_list   # "bedrock" 在 provider_list 裡
    and len(model.split("/")) > 1                       # 有 "/" 分隔
):
    custom_llm_provider = model.split("/", 1)[0]        # "bedrock"
    model = model.split("/", 1)[1]                      # "anthropic.claude-3-5-sonnet-20241022-v2:0"
    return model, custom_llm_provider, dynamic_api_key, api_base
```

**結果：**
- `custom_llm_provider = "bedrock"`
- `model = "anthropic.claude-3-5-sonnet-20241022-v2:0"`

---

## 4. 參數處理：`get_optional_params()`

**檔案：** [litellm/utils.py](litellm/utils.py)

`get_optional_params()` 會呼叫 provider 的 `map_openai_params()`，把 OpenAI 格式的參數轉換為 Bedrock Converse API 所需格式。

**對應邏輯在：** [litellm/llms/bedrock/chat/converse_transformation.py:875](litellm/llms/bedrock/chat/converse_transformation.py#L875)

```python
# AmazonConverseConfig.map_openai_params()
def map_openai_params(self, non_default_params, optional_params, model, drop_params):
    for param, value in non_default_params.items():
        if param == "max_tokens" or param == "max_completion_tokens":
            optional_params["maxTokens"] = value          # OpenAI -> Bedrock
        if param == "stop":
            if isinstance(value, str):
                value = [value]                            # str -> List[str]
            optional_params["stopSequences"] = value
        if param == "temperature":
            optional_params["temperature"] = value         # 名稱相同
        if param == "top_p":
            optional_params["topP"] = value               # camelCase
        if param == "tools":
            # 轉換工具定義格式
            self._apply_tool_call_transformation(tools=value, ...)
        if param == "tool_choice":
            optional_params["tool_choice"] = self.map_tool_choice_values(...)
        if param == "thinking":
            optional_params["thinking"] = value            # 傳遞 extended thinking 設定
        if param == "reasoning_effort":
            self._handle_reasoning_effort_parameter(...)   # "low/medium/high" 對應 budget_tokens
        if param == "stream":
            optional_params["stream"] = value
        if param == "response_format":
            # json_mode: 轉成 synthetic tool call 或 native outputConfig
            optional_params = self._translate_response_format_param(...)
        if param == "service_tier":
            # "auto" -> Bedrock "default", "flex/priority" 直接對應
            self._map_service_tier_param(value, optional_params)
```

**參數對照表（OpenAI → Bedrock Converse）：**

| OpenAI 參數 | Bedrock Converse 參數 | 位置 |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | `inferenceConfig.maxTokens` | inferenceConfig |
| `temperature` | `inferenceConfig.temperature` | inferenceConfig |
| `top_p` | `inferenceConfig.topP` | inferenceConfig |
| `stop` | `inferenceConfig.stopSequences` | inferenceConfig |
| `tools` | `toolConfig.tools[]` | toolConfig |
| `tool_choice` | `toolConfig.toolChoice` | toolConfig |
| `thinking` | `additionalModelRequestFields.thinking` | additionalModelRequestFields |
| `stream` | 控制 endpoint 路徑 `/converse-stream` | handler 層 |
| `response_format` | synthetic tool call 或 `outputConfig` | 轉換層 |
| `service_tier` | `serviceTier.type` | top-level |

---

## 5. Bedrock Route 分發

**檔案：** [litellm/main.py:3773](litellm/main.py#L3773) → [litellm/llms/bedrock/common_utils.py:598](litellm/llms/bedrock/common_utils.py#L598)

```python
# main.py:3773-3855
elif custom_llm_provider == "bedrock":

    # [廢棄路徑] 如果用舊的 aws_bedrock_client，提取 credentials 轉換格式
    if "aws_bedrock_client" in optional_params:
        verbose_logger.warning("'aws_bedrock_client' is deprecated ...")
        creds = aws_bedrock_client._get_credentials().get_frozen_credentials()
        optional_params["aws_access_key_id"] = creds.access_key
        ...

    # 決定走哪條路由
    bedrock_route = BedrockModelInfo.get_bedrock_route(model)

    if bedrock_route == "converse":            # ← Claude Sonnet 走這條
        model = model.replace("converse/", "")
        response = bedrock_converse_chat_completion.completion(...)

    elif bedrock_route == "converse_like":     # converse_like/ prefix
        response = base_llm_http_handler.completion(...)

    else:                                      # invoke/ 舊路由（Titan, 舊版 Cohere）
        response = base_llm_http_handler.completion(...)
```

**Route 判斷邏輯：** [litellm/llms/bedrock/common_utils.py:598](litellm/llms/bedrock/common_utils.py#L598)

```python
@staticmethod
def get_bedrock_route(model: str) -> Literal["converse", "invoke", ...]:
    # 1. 先看 model 字串有無明確 prefix
    route_mappings = {
        "invoke/": "invoke",
        "converse_like/": "converse_like",
        "converse/": "converse",
        "agent/": "agent",
        "agentcore/": "agentcore",
    }
    for prefix, route_type in route_mappings.items():
        if prefix in model:
            return route_type

    # 2. Nova spec 特判
    if model.startswith("nova-2/") or model.startswith("nova/"):
        return "converse"

    # 3. 查 litellm.bedrock_converse_models 白名單
    base_model = BedrockModelInfo.get_base_model(model)
    if base_model in litellm.bedrock_converse_models:
        return "converse"   # ← "anthropic.claude-3-5-sonnet-..." 在這個白名單

    return "invoke"         # 老舊 model 才走 invoke
```

---

## 6. `BedrockConverseLLM.completion()`：主要 Handler

**檔案：** [litellm/llms/bedrock/chat/converse_handler.py:251](litellm/llms/bedrock/chat/converse_handler.py#L251)

```python
def completion(self, model, messages, api_base, custom_prompt_dict,
               model_response, encoding, logging_obj, optional_params,
               acompletion, timeout, litellm_params, extra_headers=None,
               client=None, api_key=None):

    ## SETUP ##
    # 從 optional_params 提取控制旗標（pop 後不進入 Bedrock request body）
    stream = optional_params.pop("stream", None)
    stream_chunk_size = optional_params.pop("stream_chunk_size", 1024)
    unencoded_model_id = optional_params.pop("model_id", None)  # ARN 覆蓋
    fake_stream = optional_params.pop("fake_stream", False)
    json_mode = optional_params.get("json_mode", False)

    # 解析 model ID：剝除 routing prefix，取出真正要放入 URL 的 model ID
    _stripped = model
    for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
        if _stripped.startswith(rp):
            _stripped = _stripped[len(rp):]
            break

    # 偵測 model 路徑中嵌入的 region，如 "us-east-1/anthropic.claude-..."
    _potential_region = _stripped.split("/", 1)[0]
    if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
        _region_from_model = _potential_region        # 提取出 region
        _stripped = _stripped.split("/", 1)[1]        # 保留真正的 model ID
        optional_params["aws_region_name"] = _region_from_model

    modelId = self.encode_model_id(model_id=_stripped)
    # encode_model_id 對含特殊字元的 ARN 做 URL encoding
    # 例：arn:aws:bedrock:us-east-1::foundation-model/... → URL encoded

    ### SET REGION NAME ###
    aws_region_name = self._get_aws_region_name(
        optional_params=optional_params, model=model, model_id=unencoded_model_id
    )

    ## CREDENTIALS ##
    # 從 optional_params 彈出 AWS 相關參數（不傳進 Bedrock request）
    aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
    aws_access_key_id = optional_params.pop("aws_access_key_id", None)
    aws_session_token = optional_params.pop("aws_session_token", None)
    aws_role_name = optional_params.pop("aws_role_name", None)
    aws_session_name = optional_params.pop("aws_session_name", None)
    aws_profile_name = optional_params.pop("aws_profile_name", None)
    aws_bedrock_runtime_endpoint = optional_params.pop("aws_bedrock_runtime_endpoint", None)
    aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)
    aws_sts_endpoint = optional_params.pop("aws_sts_endpoint", None)
    aws_external_id = optional_params.pop("aws_external_id", None)
    optional_params.pop("aws_region_name", None)

    # 存入 litellm_params 供 async call 使用
    litellm_params["aws_region_name"] = aws_region_name

    credentials = self.get_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        ...
    )

    ### SET RUNTIME ENDPOINT ###
    endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
        api_base=api_base,
        aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
        aws_region_name=aws_region_name,
    )
    # 根據是否 stream 決定 endpoint
    if stream is True and not fake_stream:
        endpoint_url = f"{endpoint_url}/model/{modelId}/converse-stream"
    else:
        endpoint_url = f"{endpoint_url}/model/{modelId}/converse"

    ## COMPLETION CALL
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers = {"Content-Type": "application/json", **extra_headers}

    ### ROUTING (ASYNC, STREAMING, SYNC)
    if acompletion:
        if stream is True:
            return self.async_streaming(...)    # async streaming
        return self.async_completion(...)       # async non-streaming

    ## SYNC PATH ##

    # 1. 轉換 request body
    _data = litellm.AmazonConverseConfig()._transform_request(
        model=model, messages=messages,
        optional_params=optional_params, litellm_params=litellm_params,
        headers=extra_headers,
    )
    data = json.dumps(_data)

    # 2. SigV4 簽名
    prepped = self.get_request_headers(
        credentials=credentials, aws_region_name=aws_region_name,
        extra_headers=extra_headers, endpoint_url=proxy_endpoint_url,
        data=data, headers=headers, api_key=api_key,
    )

    # 3. 發送 HTTP 請求
    if stream is True:
        completion_stream = make_sync_call(
            client=client, api_base=proxy_endpoint_url,
            headers=prepped.headers, data=data, model=model,
            messages=messages, logging_obj=logging_obj, json_mode=json_mode,
        )
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model, custom_llm_provider="bedrock", logging_obj=logging_obj,
        )

    response = client.post(
        url=proxy_endpoint_url, headers=prepped.headers,
        data=data, logging_obj=logging_obj,
    )

    # 4. 轉換 response
    return litellm.AmazonConverseConfig()._transform_response(
        model=model, response=response, model_response=model_response, ...
    )
```

---

## 7. 認證授權：`BaseAWSLLM.get_credentials()`

**檔案：** [litellm/llms/bedrock/base_aws_llm.py:100](litellm/llms/bedrock/base_aws_llm.py#L100)

### 7.1 參數前處理：`os.environ/` prefix 支援

```python
# base_aws_llm.py:132-141
# LiteLLM 支援用 "os.environ/VAR_NAME" 作為間接參照
for i, param in enumerate(params_to_check):
    if param and param.startswith("os.environ/"):
        # 從環境變數取值，如 aws_access_key_id="os.environ/MY_AWS_KEY"
        _v = get_secret(param)     # get_secret 解析 os.environ/ prefix
        params_to_check[i] = _v
    elif param is None:
        # 如果未傳入，嘗試從環境變數讀取標準 AWS 變數名稱
        key = self.aws_authentication_params[i]   # e.g. "aws_access_key_id"
        if key.upper() in os.environ:
            params_to_check[i] = os.getenv(key.upper())   # AWS_ACCESS_KEY_ID
```

### 7.2 Credentials 解析優先順序

```python
# base_aws_llm.py:200-266
# 優先順序由高到低：

# 1. OIDC / Web Identity Token（EKS IRSA 場景）
if aws_web_identity_token and aws_role_name and aws_session_name:
    credentials = self._auth_with_web_identity_token(
        aws_web_identity_token=aws_web_identity_token,  # JWT token 內容
        aws_role_name=aws_role_name,                    # IAM Role ARN
        aws_session_name=aws_session_name,
        aws_sts_endpoint=aws_sts_endpoint,              # 自訂 STS endpoint
        aws_external_id=aws_external_id,
    )

# 2. IAM Role Assumption（跨帳號）
elif aws_role_name:
    if self._is_already_running_as_role(aws_role_name):
        # 已經是該 role（ECS task role / EC2 instance profile），直接用環境憑證
        credentials = self._auth_with_env_vars()
    else:
        # 呼叫 STS AssumeRole 取得臨時憑證
        credentials = self._auth_with_aws_role(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_role_name=aws_role_name,
            aws_session_name=aws_session_name,
            aws_external_id=aws_external_id,
        )

# 3. Named Profile（~/.aws/credentials 裡的 profile）
elif aws_profile_name:
    credentials = self._auth_with_aws_profile(aws_profile_name)

# 4. Access Key + Session Token
elif aws_access_key_id and aws_secret_access_key and aws_session_token:
    credentials = self._auth_with_aws_session_token(...)

# 5. Access Key + Secret（無 session token）
elif aws_access_key_id and aws_secret_access_key and aws_region_name:
    credentials = self._auth_with_access_key_and_secret_key(...)

# 6. 環境變數兜底（boto3 標準鏈）
else:
    credentials, _cache_ttl = self._auth_with_env_vars()
    # boto3.Session() 自動讀取：
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    # ~/.aws/credentials, ECS task role, EC2 instance metadata 等
```

### 7.3 Credentials 快取

```python
# 對相同的認證參數組合，結果會快取，避免重複呼叫 STS
cache_key = self.get_cache_key(args)           # SHA256(sorted params)
_cached_credentials = self.iam_cache.get_cache(cache_key)
if _cached_credentials:
    return _cached_credentials
# ... resolve credentials ...
self.iam_cache.set_cache(cache_key, credentials, ttl=_cache_ttl)
# TTL: Role assumption = 3540s（59分鐘），靜態 key = 不過期
```

---

## 8. 環境變數完整清單

| 環境變數 | 用途 | 使用位置 |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS 存取金鑰 ID | `base_aws_llm.py:140` → `_auth_with_env_vars()` |
| `AWS_SECRET_ACCESS_KEY` | AWS 秘密存取金鑰 | `base_aws_llm.py:140` → `_auth_with_env_vars()` |
| `AWS_SESSION_TOKEN` | AWS 臨時 session token（STS 用） | `base_aws_llm.py:140` → `_auth_with_env_vars()` |
| `AWS_REGION_NAME` | LiteLLM 優先讀取的 region（非標準） | `base_aws_llm.py:492` |
| `AWS_REGION` | 標準 AWS region 環境變數 | `base_aws_llm.py:501` |
| `AWS_BEDROCK_RUNTIME_ENDPOINT` | 覆蓋預設 endpoint URL（VPC endpoint 等） | `base_aws_llm.py:1121` |
| `AWS_BEARER_TOKEN_BEDROCK` | 若設定，以 Bearer token 取代 SigV4 簽名 | `base_aws_llm.py:1235` |
| `AWS_WEB_IDENTITY_TOKEN_FILE` | IRSA/EKS 用，JWT token 檔案路徑 | boto3 自動讀取 |
| `AWS_ROLE_ARN` | IRSA/EKS 用，要 assume 的 IAM Role ARN | boto3 自動讀取 |

**Region 解析優先順序（高→低）：**
1. `optional_params["aws_region_name"]`（呼叫時直接傳入）
2. model 字串中嵌入的 region，如 `"bedrock/us-east-1/anthropic.claude-..."`
3. `model_id` ARN 中的 region，如 `arn:aws:bedrock:us-east-1::...`
4. `AWS_REGION_NAME` 環境變數
5. `AWS_REGION` 環境變數
6. `boto3.Session().region_name`（讀 `~/.aws/config`）
7. 預設 `"us-west-2"`

---

## 9. Region 與 Endpoint 解析

**檔案：** [litellm/llms/bedrock/base_aws_llm.py:1114](litellm/llms/bedrock/base_aws_llm.py#L1114)

```python
def get_runtime_endpoint(self, api_base, aws_bedrock_runtime_endpoint,
                          aws_region_name, endpoint_type="runtime"):
    env_endpoint = get_secret("AWS_BEDROCK_RUNTIME_ENDPOINT")

    # Endpoint 優先順序（高→低）：
    if api_base is not None:
        endpoint_url = api_base                          # 1. 直接傳入 api_base
    elif aws_bedrock_runtime_endpoint:
        endpoint_url = aws_bedrock_runtime_endpoint      # 2. 參數傳入
    elif env_endpoint:
        endpoint_url = env_endpoint                      # 3. 環境變數
    else:
        endpoint_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"  # 4. 預設

    # 最終 URL 格式（非 stream）：
    # https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse

    # Stream：
    # https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse-stream
```

---

## 10. Request Body 轉換

**檔案：** [litellm/llms/bedrock/chat/converse_transformation.py:1555](litellm/llms/bedrock/chat/converse_transformation.py#L1555)

### 10.1 System Message 提取

```python
# converse_transformation.py:1129 — _transform_system_message()
def _transform_system_message(self, messages, model=None):
    system_prompt_indices = []
    system_content_blocks: List[SystemContentBlock] = []

    for idx, message in enumerate(messages):
        if message["role"] == "system":
            system_prompt_indices.append(idx)
            if isinstance(message["content"], str):
                # 純文字 system prompt
                system_content_blocks.append(SystemContentBlock(text=message["content"]))
                # 如果有 cache_control，附加 cachePoint block
                cache_block = self._get_cache_point_block(message, block_type="system")
                if cache_block:
                    system_content_blocks.append(cache_block)
            elif isinstance(message["content"], list):
                # list 格式（multi-modal system）
                for m in message["content"]:
                    if m.get("type") == "text":
                        system_content_blocks.append(SystemContentBlock(text=m["text"]))

    # 從 messages 列表中移除 system messages
    for idx in reversed(system_prompt_indices):
        messages.pop(idx)

    return messages, system_content_blocks
```

**轉換範例：**

```python
# OpenAI 格式
messages = [
    {"role": "system", "content": "你是一個 Python 專家"},
    {"role": "user", "content": "什麼是 GIL？"}
]

# 轉換後
messages = [{"role": "user", "content": "什麼是 GIL？"}]
system_content_blocks = [SystemContentBlock(text="你是一個 Python 專家")]
```

### 10.2 訊息格式轉換（`_bedrock_converse_messages_pt`）

**檔案：** [litellm/litellm_core_utils/prompt_templates/factory.py](litellm/litellm_core_utils/prompt_templates/factory.py)

將 OpenAI messages 格式轉換為 Bedrock `MessageBlock[]`：

**轉換範例：**

```python
# OpenAI 格式（輸入）
messages = [
    {"role": "user", "content": "分析這張圖"},
    {"role": "assistant", "content": "圖中顯示..."},
    {"role": "user", "content": [
        {"type": "text", "text": "還有嗎？"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
]

# Bedrock Converse 格式（輸出）
bedrock_messages = [
    {
        "role": "user",
        "content": [{"text": "分析這張圖"}]
    },
    {
        "role": "assistant",
        "content": [{"text": "圖中顯示..."}]
    },
    {
        "role": "user",
        "content": [
            {"text": "還有嗎？"},
            {"image": {"format": "jpeg", "source": {"bytes": b"..."}}}
        ]
    }
]
```

### 10.3 完整 Request Body 組成

**檔案：** [litellm/llms/bedrock/chat/converse_transformation.py:1387](litellm/llms/bedrock/chat/converse_transformation.py#L1387)

```python
# _transform_request_helper() 組成完整的 CommonRequestObject

# Step 1: 分離 inferenceConfig / additionalModelRequestFields / metadata
(
    inference_params,              # maxTokens, temperature, topP, stopSequences
    additional_request_params,     # thinking, anthropic_beta, top_k, etc.
    request_metadata,              # requestMetadata KV pairs
    output_config,                 # outputConfig（native structured output）
) = self._prepare_request_params(optional_params, model)

# Step 2: 處理 tools
original_tools = inference_params.pop("tools", [])
bedrock_tools, anthropic_beta_list = self._process_tools_and_beta(
    original_tools, model, headers, additional_request_params
)
# 如果是 Claude，將 anthropic_beta 加入 additionalModelRequestFields

# Step 3: 組成最終 dict
data = {
    "additionalModelRequestFields": additional_request_params,
    "system": system_content_blocks,
    "inferenceConfig": InferenceConfig(**inference_params),
    "toolConfig": ToolConfigBlock(tools=bedrock_tools),   # 若有 tools
    "requestMetadata": request_metadata,                  # 若有 metadata
    "outputConfig": output_config,                        # 若有 native JSON output
}
```

**完整 Bedrock Converse Request Body 範例：**

```json
// 呼叫：completion("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
//   messages=[{"role":"user","content":"解釋 GIL"}],
//   max_tokens=200, temperature=0.7, tools=[...])

{
  "messages": [
    {
      "role": "user",
      "content": [{"text": "解釋 GIL"}]
    }
  ],
  "system": [],
  "inferenceConfig": {
    "maxTokens": 200,
    "temperature": 0.7
  },
  "toolConfig": {
    "tools": [
      {
        "toolSpec": {
          "name": "get_weather",
          "description": "Get weather data",
          "inputSchema": {
            "json": {
              "type": "object",
              "properties": {"location": {"type": "string"}},
              "required": ["location"]
            }
          }
        }
      }
    ],
    "toolChoice": {"auto": {}}
  },
  "additionalModelRequestFields": {}
}
```

### 10.4 Tool 格式轉換（OpenAI → Bedrock）

```python
# OpenAI 格式
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather data",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }
    }
}

# Bedrock Converse 格式（ToolBlock）
{
    "toolSpec": {
        "name": "get_weather",
        "description": "Get weather data",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }
    }
}
```

### 10.5 Extended Thinking（`thinking` 參數）

```python
# 傳入：thinking={"type": "enabled", "budget_tokens": 5000}
# 轉換後放入 additionalModelRequestFields：
{
  "additionalModelRequestFields": {
    "thinking": {
      "type": "enabled",
      "budget_tokens": 5000
    }
  }
}
```

---

## 11. SigV4 請求簽名

**檔案：** [litellm/llms/bedrock/base_aws_llm.py:1221](litellm/llms/bedrock/base_aws_llm.py#L1221)

```python
def get_request_headers(self, credentials, aws_region_name, extra_headers,
                         endpoint_url, data, headers, api_key=None):

    # 特殊路徑：若有 Bearer token，跳過 SigV4，直接用 Bearer
    aws_bearer_token = api_key or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
    if aws_bearer_token:
        headers["Authorization"] = f"Bearer {aws_bearer_token}"
        request = AWSRequest(method="POST", url=endpoint_url, data=data, headers=headers)
    else:
        # 標準 SigV4 路徑

        # 1. 只選取 AWS 相關 headers 進行簽名（避免轉發的客戶端 headers 污染簽名）
        aws_signature_headers = self._filter_headers_for_aws_signature(headers)
        # 只包含：host, content-type, x-amz-*, x-amzn-*

        # 2. 建立 SigV4Auth 並簽名
        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
        request = AWSRequest(
            method="POST",
            url=endpoint_url,
            data=data,
            headers=aws_signature_headers,
        )
        sigv4.add_auth(request)
        # sigv4.add_auth() 新增以下 headers：
        # Authorization: AWS4-HMAC-SHA256 Credential=.../bedrock/aws4_request,
        #                SignedHeaders=content-type;host;x-amz-date,
        #                Signature=<hex>
        # X-Amz-Date: 20240315T120000Z
        # X-Amz-Security-Token: <session_token>  （若使用臨時憑證）

        # 3. 把其餘非 AWS headers 補回（如 Content-Type、自訂 headers）
        for header_name, header_value in headers.items():
            if header_value is not None:
                request.headers[header_name] = header_value

        # 4. 如果 extra_headers 有自訂 Authorization，覆蓋 SigV4 的
        if extra_headers and "Authorization" in extra_headers:
            request.headers["Authorization"] = extra_headers["Authorization"]

    prepped = request.prepare()
    return prepped
```

**最終 HTTP Request Headers 範例：**

```http
POST /model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse HTTP/1.1
Host: bedrock-runtime.us-west-2.amazonaws.com
Content-Type: application/json
X-Amz-Date: 20240315T120000Z
X-Amz-Security-Token: AQoXb3JpZ2lu...（臨時 credentials 才有）
Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20240315/us-west-2/bedrock/aws4_request,
               SignedHeaders=content-type;host;x-amz-date,
               Signature=7638e4f1a9b2c3d4e5f6a7b8c9d0e1f2...
```

### 11.1 `_filter_headers_for_aws_signature()` 白名單

```python
# base_aws_llm.py:1282
def _filter_headers_for_aws_signature(self, headers: dict) -> dict:
    """
    避免轉發的客戶端 headers 破壞 SigV4 簽名計算。
    只有以下 headers 會被納入簽名範圍：
    """
    aws_headers = {
        "host", "content-type", "date",
        "x-amz-date", "x-amz-security-token", "x-amz-content-sha256",
        "x-amz-algorithm", "x-amz-credential",
        "x-amz-signedheaders", "x-amz-signature",
    }
    # 另外：任何 x-amz- 或 x-amzn- 前綴的 header 也納入
```

---

## 12. HTTP 請求發送

**檔案：** [litellm/llms/bedrock/chat/converse_handler.py:485](litellm/llms/bedrock/chat/converse_handler.py#L485)

```python
# Sync 非 streaming：
try:
    response = client.post(
        url=proxy_endpoint_url,          # https://bedrock-runtime.{region}.amazonaws.com/model/{id}/converse
        headers=prepped.headers,         # SigV4 signed headers
        data=data,                       # JSON string
        logging_obj=logging_obj,
    )
    response.raise_for_status()          # 4xx/5xx → BedrockError
except httpx.HTTPStatusError as err:
    raise BedrockError(status_code=err.response.status_code, message=err.response.text)
except httpx.TimeoutException:
    raise BedrockError(status_code=408, message="Timeout error occurred.")
```

HTTP client 使用 `httpx`，通過 `_get_httpx_client()` / `get_async_httpx_client()` 管理。

---

## 13. Response 轉換：OpenAI-compatible 格式

**檔案：** [litellm/llms/bedrock/chat/converse_transformation.py:1920](litellm/llms/bedrock/chat/converse_transformation.py#L1920)

### 13.1 Bedrock Response 原始格式

```json
{
  "output": {
    "message": {
      "role": "assistant",
      "content": [
        {"text": "GIL 是 Python 的全局解釋器鎖..."},
        {
          "toolUse": {
            "toolUseId": "tooluse_abc123",
            "name": "get_weather",
            "input": {"location": "Taipei"}
          }
        },
        {
          "reasoningContent": {
            "reasoningText": {"text": "讓我思考一下...", "signature": "xxx"},
          }
        }
      ]
    }
  },
  "stopReason": "end_turn",
  "usage": {
    "inputTokens": 42,
    "outputTokens": 156,
    "totalTokens": 198,
    "cacheReadInputTokens": 10,
    "cacheWriteInputTokens": 0
  }
}
```

### 13.2 轉換過程

```python
def _transform_response(self, model, response, model_response, stream,
                         logging_obj, optional_params, api_key, data, messages, encoding):

    # 1. 解析 JSON → ConverseResponseBlock（TypedDict）
    completion_response = ConverseResponseBlock(**response.json())

    # 2. 提取 message content
    message = completion_response["output"]["message"]
    # content 是 list，可能包含：text, toolUse, reasoningContent, citationsContent

    (
        content_str,             # 合併所有 text block
        tools,                   # tool_calls list
        reasoningContentBlocks,  # thinking blocks (Claude 3.7+)
        citationsContentBlocks,  # citations (Nova grounding)
    ) = self._translate_message_content(message["content"])

    # 3. 組裝 chat_completion_message
    chat_completion_message = {"role": "assistant"}
    chat_completion_message["content"] = content_str

    # 如果有 extended thinking
    if reasoningContentBlocks:
        chat_completion_message["reasoning_content"] = ...  # str
        chat_completion_message["thinking_blocks"] = ...    # List[ThinkingBlock]

    # 如果有 tool calls
    filtered_tools = self._filter_json_mode_tools(json_mode, tools, chat_completion_message)
    if filtered_tools:
        chat_completion_message["tool_calls"] = filtered_tools

    # 4. Usage 轉換
    usage = self._transform_usage(completion_response["usage"])
    # inputTokens    → prompt_tokens
    # outputTokens   → completion_tokens
    # totalTokens    → total_tokens
    # cacheReadInputTokens  → cache_read_input_tokens（加入 prompt_tokens）
    # cacheWriteInputTokens → cache_creation_input_tokens

    # 5. finish_reason 對應
    # "end_turn"   → "stop"
    # "tool_use"   → "tool_calls"
    # "max_tokens" → "length"
    initial_finish_reason = map_finish_reason(completion_response["stopReason"])

    # 6. 組裝 ModelResponse
    model_response.choices = [
        litellm.Choices(
            finish_reason=returned_finish_reason,
            index=0,
            message=Message(**chat_completion_message),
        )
    ]
    model_response.created = int(time.time())
    model_response.model = model
    setattr(model_response, "usage", usage)

    return model_response
```

### 13.3 轉換前後格式對比

**Bedrock Response → OpenAI-compatible ModelResponse：**

```python
# Bedrock 原始 Response
{
  "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
  "stopReason": "end_turn",
  "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
}

# ↓ _transform_response() ↓

# OpenAI-compatible ModelResponse
ModelResponse(
    id="chatcmpl-xxx",
    choices=[
        Choices(
            finish_reason="stop",       # "end_turn" → "stop"
            index=0,
            message=Message(
                role="assistant",
                content="Hello!",
            )
        )
    ],
    created=1710500000,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    usage=Usage(
        prompt_tokens=10,           # inputTokens
        completion_tokens=5,        # outputTokens
        total_tokens=15,
    )
)
```

**Tool Call 轉換對比：**

```python
# Bedrock toolUse block
{
    "toolUse": {
        "toolUseId": "tooluse_abc123",
        "name": "get_weather",
        "input": {"location": "Taipei"}
    }
}

# ↓ 轉換後 ↓

# OpenAI tool_calls 格式
{
    "tool_calls": [
        {
            "id": "tooluse_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Taipei"}'  # JSON string
            }
        }
    ]
}
```

**Usage 轉換對比：**

```python
# Bedrock usage（有 prompt cache）
{"inputTokens": 30, "outputTokens": 100, "totalTokens": 130,
 "cacheReadInputTokens": 20, "cacheWriteInputTokens": 0}

# OpenAI-compatible Usage
Usage(
    prompt_tokens=50,          # inputTokens(30) + cacheReadInputTokens(20)
    completion_tokens=100,
    total_tokens=130,
    prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=20),
    cache_read_input_tokens=20,
    cache_creation_input_tokens=0,
)
```

---

## 14. Streaming 處理

### 14.1 Streaming Endpoint 與初始化

```python
# converse_handler.py:358
if stream is True and not fake_stream:
    endpoint_url = f"{endpoint_url}/model/{modelId}/converse-stream"
    # ↑ 注意是 converse-stream，不是 converse

# 同步 streaming 路徑
completion_stream = make_sync_call(
    client=client, api_base=proxy_endpoint_url,
    headers=prepped.headers, data=data, model=model,
    messages=messages, logging_obj=logging_obj, json_mode=json_mode,
)
return CustomStreamWrapper(
    completion_stream=completion_stream,
    model=model, custom_llm_provider="bedrock", logging_obj=logging_obj,
)

# 非同步 streaming 路徑（acompletion=True）
return self.async_streaming(...)  # → make_call() → get_async_httpx_client().post()
```

### 14.2 AWS Event Stream 解碼

Bedrock 的 streaming response 使用 **AWS Event Stream** 二進位格式（不是 SSE）。

**檔案：** [litellm/llms/bedrock/chat/invoke_handler.py:1285](litellm/llms/bedrock/chat/invoke_handler.py#L1285)

```python
class AWSEventStreamDecoder:
    def __init__(self, model, json_mode=False):
        from botocore.parsers import EventStreamJSONParser
        self.parser = EventStreamJSONParser()   # botocore 的 event stream 解析器
        self.content_blocks: List = []
        self.tool_calls_index = None
        self.json_mode = json_mode

    def iter_bytes(self, iterator: Iterator[bytes]):
        """給定 bytes iterator，逐一解碼 AWS Event Stream 事件"""
        from botocore.eventstream import EventStreamBuffer
        event_stream_buffer = EventStreamBuffer()
        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                parsed = self.parser.parse(event.to_response_dict(), ...)
                chunk_data = parsed.get("body", {})
                yield self._chunk_parser(chunk_data)
```

### 14.3 Streaming Chunk 事件類型

AWS Bedrock Converse Stream 事件分為幾類：

```python
def converse_chunk_parser(self, chunk_data: dict) -> ModelResponseStream:
    # 1. messageStart 事件
    #    {"messageStart": {"role": "assistant", "conversationId": "xxx"}}
    #    → 初始化 response_id

    # 2. contentBlockStart 事件（新內容塊開始）
    #    {"start": {"toolUse": {"toolUseId": "xxx", "name": "fn_name"}}, "contentBlockIndex": 1}
    if "start" in chunk_data:
        start_obj = ContentBlockStartEvent(**chunk_data["start"])
        tool_use, provider_specific_fields, thinking_blocks = \
            self._handle_converse_start_event(start_obj)

    # 3. contentBlockDelta 事件（增量文字/工具輸入）
    #    {"delta": {"text": "Hello"}, "contentBlockIndex": 0}
    #    {"delta": {"toolUse": {"input": '{"loc'}}, "contentBlockIndex": 1}
    #    {"delta": {"reasoningContent": {"text": "Let me think..."}}, "contentBlockIndex": 2}
    elif "delta" in chunk_data:
        delta_obj = ContentBlockDeltaEvent(**chunk_data["delta"])
        text, tool_use, provider_specific_fields, reasoning_content, thinking_blocks = \
            self._handle_converse_delta_event(delta_obj, content_block_index)

    # 4. contentBlockStop 事件（內容塊結束）
    #    {"contentBlockIndex": 1}（只有 index，沒有 start/delta）
    elif "contentBlockIndex" in chunk_data:
        tool_use = self._handle_converse_stop_event(content_block_index)

    # 5. messageStop 事件
    #    {"stopReason": "end_turn"}
    elif "stopReason" in chunk_data:
        finish_reason = map_finish_reason(chunk_data["stopReason"])

    # 6. metadata 事件（usage stats）
    #    {"usage": {"inputTokens": 42, "outputTokens": 156, "totalTokens": 198}}
    elif "usage" in chunk_data:
        usage = converse_config._transform_usage(chunk_data["usage"])

    # 組裝 OpenAI-compatible ModelResponseStream
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(
                    content=text,
                    role="assistant",
                    tool_calls=[tool_use] if tool_use else None,
                    thinking_blocks=thinking_blocks,
                    reasoning_content=reasoning_content,
                )
            )
        ],
        id=self.response_id,
        model=self.model,
        usage=usage,
    )
```

### 14.4 Streaming 事件序列範例

```
# AWS Event Stream 原始事件序列（對應普通 text 回覆）：

messageStart       → {"messageStart": {"role": "assistant"}}
contentBlockStart  → {"start": {}, "contentBlockIndex": 0}
contentBlockDelta  → {"delta": {"text": "GIL 是"}, "contentBlockIndex": 0}
contentBlockDelta  → {"delta": {"text": " Python 的全局"}, "contentBlockIndex": 0}
contentBlockDelta  → {"delta": {"text": "解釋器鎖"}, "contentBlockIndex": 0}
contentBlockStop   → {"contentBlockIndex": 0}
messageStop        → {"stopReason": "end_turn"}
metadata           → {"usage": {"inputTokens": 15, "outputTokens": 8, "totalTokens": 23}}

# ↓ AWSEventStreamDecoder 逐一轉換 ↓

# OpenAI-compatible streaming chunks：
ModelResponseStream(choices=[StreamingChoices(delta=Delta(content="GIL 是"))])
ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=" Python 的全局"))])
ModelResponseStream(choices=[StreamingChoices(delta=Delta(content="解釋器鎖"))])
ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=""), finish_reason="stop")], usage=...)
```

### 14.5 `fake_stream` 模式

某些 model（如 `meta.llama3-3-70b-instruct-v1:0` + stream mode）不支援真正的 streaming，會啟用 `fake_stream`：

```python
# converse_transformation.py
if "meta.llama3-3-70b-instruct-v1:0" in model and non_default_params.get("stream"):
    optional_params["fake_stream"] = True

# converse_handler.py:make_sync_call()
if fake_stream:
    # 用 /converse（非 /converse-stream）取得完整 response
    model_response = AmazonConverseConfig()._transform_response(...)
    # 然後模擬 streaming，逐 token 分割輸出
    completion_stream = MockResponseIterator(model_response=model_response)
else:
    # 正常 AWS Event Stream decode
    decoder = AWSEventStreamDecoder(model=model, json_mode=json_mode)
    completion_stream = decoder.iter_bytes(response.iter_bytes(chunk_size=stream_chunk_size))
```

---

## 15. 附錄：概念精簡版執行路徑 Code

以下是剝除 litellm 中所有非 AWS Bedrock Claude Sonnet 相關邏輯後，重寫的概念性程式碼，涵蓋整條執行路徑：

```python
"""
概念版：litellm completion() → AWS Bedrock Claude Sonnet
去除所有其他 provider 邏輯，只保留 Bedrock Converse Claude 路徑的核心概念
"""

import json
import os
import time
from typing import Iterator, List, Optional, Union

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


# ─────────────────────────────────────────────────────────────────────────────
# 第一步：入口 completion()
# ─────────────────────────────────────────────────────────────────────────────

def completion(
    model: str,                  # "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    messages: list,              # OpenAI 格式的 messages
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_role_name: Optional[str] = None,
    aws_bedrock_runtime_endpoint: Optional[str] = None,
):
    # 1. 解析 provider 和 model
    provider, model_id = model.split("/", 1)
    # provider = "bedrock"
    # model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert provider == "bedrock"

    # 2. 解析 region
    region = resolve_region(aws_region_name)
    # 優先順序: 直接傳入 → env AWS_REGION_NAME → env AWS_REGION → boto3 → "us-west-2"

    # 3. 取得 AWS Credentials
    credentials = resolve_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        aws_region_name=region,
        aws_role_name=aws_role_name,
    )

    # 4. 建構 endpoint URL
    base_url = aws_bedrock_runtime_endpoint or \
               os.environ.get("AWS_BEDROCK_RUNTIME_ENDPOINT") or \
               f"https://bedrock-runtime.{region}.amazonaws.com"

    # model ID 需要 URL encode（ARN 裡有冒號）
    encoded_model_id = model_id.replace(":", "%3A")

    if stream:
        endpoint_url = f"{base_url}/model/{encoded_model_id}/converse-stream"
    else:
        endpoint_url = f"{base_url}/model/{encoded_model_id}/converse"

    # 5. 轉換 request body（OpenAI → Bedrock Converse 格式）
    request_body = transform_request(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    data = json.dumps(request_body)

    # 6. SigV4 簽名
    headers = sign_request(
        credentials=credentials,
        region=region,
        endpoint_url=endpoint_url,
        data=data,
    )

    # 7. 發送 HTTP 請求
    client = httpx.Client()

    if stream:
        with client.stream("POST", endpoint_url, headers=headers, content=data) as resp:
            resp.raise_for_status()
            return stream_response(resp)
    else:
        resp = client.post(endpoint_url, headers=headers, content=data)
        resp.raise_for_status()
        return transform_response(resp.json(), model_id)


# ─────────────────────────────────────────────────────────────────────────────
# 第二步：Region 解析
# ─────────────────────────────────────────────────────────────────────────────

def resolve_region(aws_region_name: Optional[str]) -> str:
    if aws_region_name:
        return aws_region_name
    # 環境變數 1（LiteLLM 優先）
    if os.environ.get("AWS_REGION_NAME"):
        return os.environ["AWS_REGION_NAME"]
    # 環境變數 2（標準 AWS）
    if os.environ.get("AWS_REGION"):
        return os.environ["AWS_REGION"]
    # boto3 設定（~/.aws/config）
    session = boto3.Session()
    if session.region_name:
        return session.region_name
    return "us-west-2"


# ─────────────────────────────────────────────────────────────────────────────
# 第三步：Credentials 解析
# ─────────────────────────────────────────────────────────────────────────────

def resolve_credentials(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_session_token: Optional[str],
    aws_region_name: str,
    aws_role_name: Optional[str] = None,
):
    """
    優先順序（高→低）：
    1. IAM Role Assumption（STS AssumeRole）
    2. 明確傳入的 access key + secret
    3. 環境變數（boto3 標準鏈：env vars → profile → instance metadata）
    """
    if aws_role_name:
        # 透過 STS 取得臨時憑證
        sts = boto3.client(
            "sts",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        assumed = sts.assume_role(
            RoleArn=aws_role_name,
            RoleSessionName="litellm-session",
        )
        creds = assumed["Credentials"]
        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        ).get_credentials().get_frozen_credentials()

    if aws_access_key_id and aws_secret_access_key:
        # 明確傳入的靜態 key
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region_name,
        ).get_credentials().get_frozen_credentials()

    # 環境變數 / profile / instance metadata（boto3 自動鏈）
    # 讀取：AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    #       ~/.aws/credentials, ECS task role, EC2 instance metadata
    return boto3.Session().get_credentials().get_frozen_credentials()


# ─────────────────────────────────────────────────────────────────────────────
# 第四步：Request Body 轉換（OpenAI → Bedrock Converse）
# ─────────────────────────────────────────────────────────────────────────────

def transform_request(messages: list, max_tokens: int, temperature: float) -> dict:
    """將 OpenAI messages 轉換為 Bedrock Converse API 格式"""

    # 1. 分離 system messages
    system_blocks = []
    user_assistant_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_blocks.append({"text": msg["content"]})  # SystemContentBlock
        else:
            user_assistant_messages.append(msg)

    # 2. 轉換 user/assistant messages
    bedrock_messages = []
    for msg in user_assistant_messages:
        content = msg["content"]
        if isinstance(content, str):
            bedrock_content = [{"text": content}]                # 純文字
        elif isinstance(content, list):
            bedrock_content = transform_content_list(content)    # multi-modal
        bedrock_messages.append({"role": msg["role"], "content": bedrock_content})

    # 3. 組裝完整 request body
    return {
        "messages": bedrock_messages,
        "system": system_blocks,           # system prompt 獨立欄位
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            # 其他可選: "topP", "stopSequences"
        },
        # 如果有 tools：
        # "toolConfig": {"tools": [...], "toolChoice": {"auto": {}}}
        # 如果有 extended thinking：
        # "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 5000}}
    }


def transform_content_list(content: list) -> list:
    """轉換 content list（multi-modal）"""
    bedrock_content = []
    for block in content:
        if block["type"] == "text":
            bedrock_content.append({"text": block["text"]})
        elif block["type"] == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("data:"):
                # base64 inline image
                media_type = url.split(";")[0].split(":")[1]  # e.g. "image/jpeg"
                b64_data = url.split(",")[1]
                import base64
                image_bytes = base64.b64decode(b64_data)
                fmt = media_type.split("/")[1]               # e.g. "jpeg"
                bedrock_content.append({
                    "image": {
                        "format": fmt,
                        "source": {"bytes": image_bytes}
                    }
                })
    return bedrock_content


# ─────────────────────────────────────────────────────────────────────────────
# 第五步：SigV4 簽名
# ─────────────────────────────────────────────────────────────────────────────

def sign_request(credentials, region: str, endpoint_url: str, data: str) -> dict:
    """使用 SigV4 簽名，回傳簽名後的 headers dict"""
    sigv4 = SigV4Auth(credentials, "bedrock", region)
    headers = {"Content-Type": "application/json"}
    request = AWSRequest(method="POST", url=endpoint_url, data=data, headers=headers)
    sigv4.add_auth(request)
    # 簽名後 request.headers 包含：
    # Authorization: AWS4-HMAC-SHA256 Credential=.../bedrock/aws4_request,
    #                SignedHeaders=content-type;host;x-amz-date,
    #                Signature=<hex>
    # X-Amz-Date: 20240315T120000Z
    # X-Amz-Security-Token: <token>  (如果是臨時憑證)
    return dict(request.prepare().headers)


# ─────────────────────────────────────────────────────────────────────────────
# 第六步：Response 轉換（Bedrock → OpenAI-compatible）
# ─────────────────────────────────────────────────────────────────────────────

def transform_response(bedrock_response: dict, model_id: str) -> dict:
    """將 Bedrock Converse response 轉換為 OpenAI-compatible 格式"""

    output_message = bedrock_response["output"]["message"]
    stop_reason = bedrock_response["stopReason"]
    usage = bedrock_response["usage"]

    # 解析 content blocks
    content_str = ""
    tool_calls = []

    for block in output_message["content"]:
        if "text" in block:
            content_str += block["text"]
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append({
                "id": tool_use["toolUseId"],
                "type": "function",
                "function": {
                    "name": tool_use["name"],
                    "arguments": json.dumps(tool_use["input"])
                }
            })

    # finish_reason 對應
    finish_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "guardrail_intervened": "content_filter",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    # 組裝 OpenAI-compatible response
    message = {"role": "assistant", "content": content_str}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage["inputTokens"],
            "completion_tokens": usage["outputTokens"],
            "total_tokens": usage["totalTokens"],
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 第七步：Streaming Response 處理
# ─────────────────────────────────────────────────────────────────────────────

def stream_response(response: httpx.Response) -> Iterator[dict]:
    """
    解碼 AWS Event Stream 格式，轉換為 OpenAI-compatible streaming chunks。
    實際上 LiteLLM 使用 botocore.eventstream.EventStreamBuffer 解碼。
    此處以概念方式呈現事件流轉換邏輯。
    """
    from botocore.eventstream import EventStreamBuffer
    from botocore.parsers import EventStreamJSONParser

    parser = EventStreamJSONParser()
    buffer = EventStreamBuffer()
    response_id = None
    tool_index = -1

    for raw_chunk in response.iter_bytes(chunk_size=1024):
        buffer.add_data(raw_chunk)
        for event in buffer:
            parsed = parser.parse(event.to_response_dict(), ...)
            chunk_data = parsed.get("body", {})

            # 事件類型判斷
            if "messageStart" in chunk_data:
                conversation_id = chunk_data["messageStart"].get("conversationId")
                response_id = f"chatcmpl-{conversation_id}"

            elif "delta" in chunk_data:
                delta = chunk_data["delta"]

                if "text" in delta:
                    # 文字 delta
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": delta["text"]}}]
                    }

                elif "toolUse" in delta:
                    # Tool call 輸入 delta
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": tool_index,
                                    "function": {"arguments": delta["toolUse"]["input"]}
                                }]
                            }
                        }]
                    }

                elif "reasoningContent" in delta:
                    # Extended thinking delta（Claude 3.7+）
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "reasoning_content": delta["reasoningContent"].get("text", "")
                            }
                        }]
                    }

            elif "start" in chunk_data and "toolUse" in chunk_data.get("start", {}):
                # Tool call 開始
                tool_index += 1
                tool_use = chunk_data["start"]["toolUse"]
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": tool_index,
                                "id": tool_use["toolUseId"],
                                "type": "function",
                                "function": {"name": tool_use["name"], "arguments": ""}
                            }]
                        }
                    }]
                }

            elif "stopReason" in chunk_data:
                # 結束事件
                finish_reason_map = {
                    "end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"
                }
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason_map.get(chunk_data["stopReason"], "stop")
                    }]
                }

            elif "usage" in chunk_data:
                # Usage 統計（最後一個事件）
                u = chunk_data["usage"]
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                    "usage": {
                        "prompt_tokens": u["inputTokens"],
                        "completion_tokens": u["outputTokens"],
                        "total_tokens": u["totalTokens"],
                    }
                }


# ─────────────────────────────────────────────────────────────────────────────
# 使用範例
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 非 streaming
    response = completion(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "什麼是 Python 的 GIL？"}],
        max_tokens=200,
        temperature=0.7,
        # aws_region_name 不傳 → 自動讀取 AWS_REGION_NAME / AWS_REGION / boto3
    )
    print(response["choices"][0]["message"]["content"])

    # Streaming
    for chunk in completion(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "解釋 async/await"}],
        stream=True,
    ):
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
```

---

## 關鍵檔案索引

| 功能 | 檔案 | 關鍵 line |
|---|---|---|
| 入口 `completion()` | [litellm/main.py](litellm/main.py) | L1148 |
| Bedrock route 分發 | [litellm/main.py](litellm/main.py) | L3773 |
| Provider 辨識 | [litellm/litellm_core_utils/get_llm_provider_logic.py](litellm/litellm_core_utils/get_llm_provider_logic.py) | L99 |
| Bedrock route 判斷 | [litellm/llms/bedrock/common_utils.py](litellm/llms/bedrock/common_utils.py) | L598 |
| 主要 Converse handler | [litellm/llms/bedrock/chat/converse_handler.py](litellm/llms/bedrock/chat/converse_handler.py) | L251 |
| Credentials 解析 | [litellm/llms/bedrock/base_aws_llm.py](litellm/llms/bedrock/base_aws_llm.py) | L100 |
| Region / Endpoint 解析 | [litellm/llms/bedrock/base_aws_llm.py](litellm/llms/bedrock/base_aws_llm.py) | L466, L1114 |
| SigV4 簽名 | [litellm/llms/bedrock/base_aws_llm.py](litellm/llms/bedrock/base_aws_llm.py) | L1221 |
| Request 轉換 | [litellm/llms/bedrock/chat/converse_transformation.py](litellm/llms/bedrock/chat/converse_transformation.py) | L1555 |
| OpenAI params 對應 | [litellm/llms/bedrock/chat/converse_transformation.py](litellm/llms/bedrock/chat/converse_transformation.py) | L875 |
| Response 轉換 | [litellm/llms/bedrock/chat/converse_transformation.py](litellm/llms/bedrock/chat/converse_transformation.py) | L1920 |
| Usage 轉換 | [litellm/llms/bedrock/chat/converse_transformation.py](litellm/llms/bedrock/chat/converse_transformation.py) | L1657 |
| Streaming decoder | [litellm/llms/bedrock/chat/invoke_handler.py](litellm/llms/bedrock/chat/invoke_handler.py) | L1285 |
| Async streaming | [litellm/llms/bedrock/chat/invoke_handler.py](litellm/llms/bedrock/chat/invoke_handler.py) | L185 |
