# litellm `completion()` 參數解析

> 源碼：`litellm/main.py:1050`
> 函數簽名：`def completion(...) -> Union[ModelResponse, CustomStreamWrapper]`

---

## 一、參數分類總覽

litellm 將 `completion()` 的參數分為以下幾個層次：

| 分類 | 說明 | 傳遞方式 |
|------|------|----------|
| **必填參數** | `model`, `messages` | 具名參數 |
| **OpenAI 標準參數** | 與 OpenAI Chat API 1:1 對應 | 具名參數 |
| **OpenAI v1.0+ 新增** | `response_format`, `tools`, `seed` 等 | 具名參數 |
| **即將棄用的 OpenAI 參數** | `functions`, `function_call` | 具名參數 |
| **API 連接參數** | `base_url`, `api_key`, `api_version` | 具名參數 |
| **LiteLLM 專屬參數** | `thinking`, `shared_session` 等 | 具名參數 |
| **kwargs 額外參數** | 所有未具名的擴充參數 | `**kwargs` |

---

## 二、必填參數

### `model: str`
- **意義**：指定使用的 LLM 模型，格式為 `provider/model-name` 或 OpenAI 相容名稱
- **源碼片段**：
  ```python
  # main.py:1149
  if model is None:
      raise ValueError("model param not passed in.")

  # main.py:1369
  model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
      model=model,
      custom_llm_provider=custom_llm_provider,
      api_base=api_base,
      api_key=api_key,
  )
  ```
- **處理流程**：傳入後先驗證非空，再由 `get_llm_provider()` 解析出實際的 provider 和模型名稱。若設定了 `model_alias_map`，會在此替換為真實模型名稱（`main.py:1356`）。

### `messages: List`
- **意義**：對話歷史，格式遵循 OpenAI 的 `[{"role": "user", "content": "..."}]` 結構
- **源碼片段**：
  ```python
  # main.py:1152
  messages = validate_and_fix_openai_messages(messages=messages)

  # main.py:1304
  messages = get_completion_messages(
      messages=messages,
      ensure_alternating_roles=ensure_alternating_roles or False,
      user_continue_message=user_continue_message,
      assistant_continue_message=assistant_continue_message,
  )
  ```
- **處理流程**：先做格式驗證修正，再依據 `ensure_alternating_roles` 等設定調整訊息結構。

---

## 三、OpenAI 標準參數（`optional_params` 群組）

這些參數最終會被組合進 `optional_param_args` dict，再由 `get_optional_params()` 過濾出當前 provider 支援的子集。

```python
# main.py:1478
optional_param_args = {
    "temperature": temperature,
    "top_p": top_p,
    "n": n,
    "stream": stream,
    ...
}
optional_params = get_optional_params(**optional_param_args, **non_default_params)
```

### `temperature: Optional[float]`
- 控制輸出隨機性，0 = 確定性，2 = 最隨機
- 傳遞給各 provider 時，`get_optional_params()` 會對不支援的 provider 做轉換或丟棄

### `top_p: Optional[float]`
- nucleus sampling 閾值，與 `temperature` 通常擇一使用
- 範圍 0~1，只考慮累積機率前 top_p 的 tokens

### `n: Optional[int]`
- 一次產生幾個候選回應，預設 1
- 用於 mock_completion 時也需傳遞（`main.py:1607`）

### `stream: Optional[bool]`
- 是否啟用串流回應，若為 True 則回傳 `CustomStreamWrapper` 而非 `ModelResponse`

### `stream_options: Optional[dict]`
- 串流模式下的額外選項，例如 `{"include_usage": true}` 取得串流中的 token 用量

### `stop`
- 最多 4 個停止序列，LLM 生成到此即停止
- 源碼：`stop = validate_openai_optional_params(stop=stop)` — 先做格式驗證（`main.py:1157`）

### `max_tokens: Optional[int]`
- 生成回應的最大 token 數量（舊版參數）

### `max_completion_tokens: Optional[int]`
- OpenAI 較新版本使用，包含 reasoning tokens 在內的總 token 上限

### `modalities: Optional[List[ChatCompletionModality]]`
- 指定輸出模態，例如 `["text", "audio"]`，需搭配 `audio` 參數

### `prediction: Optional[ChatCompletionPredictionContentParam]`
- Predicted Output 功能：預先提供部分模型輸出以加速生成（適合文件小幅修改場景）

### `audio: Optional[ChatCompletionAudioParam]`
- 音訊輸出參數，需 `modalities=["audio"]` 時使用

### `presence_penalty: Optional[float]`
- 根據 token 是否已在輸出中出現過，增加生成新話題的傾向（-2.0 ~ 2.0）

### `frequency_penalty: Optional[float]`
- 根據 token 在輸出中的出現頻率，抑制重複（-2.0 ~ 2.0）

### `logit_bias: Optional[dict]`
- 修改特定 token 的出現機率，格式為 `{token_id: bias_value}`

### `user: Optional[str]`
- 終端使用者的唯一識別符，用於 provider 的濫用偵測；也會傳進 `litellm_params` 供 logging 使用（`main.py:1594`）

---

## 四、OpenAI v1.0+ 新增參數

### `reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]]`
- 控制推理模型（o1/o3/Claude 3.7 等）的思考深度
- 源碼：在 Responses API bridge 中有特殊處理：
  ```python
  # main.py:1635
  if isinstance(reasoning_effort, dict) and "summary" in reasoning_effort:
      optional_params = dict(optional_params)
      optional_params["reasoning_effort"] = reasoning_effort
  ```

### `verbosity: Optional[Literal["low", "medium", "high"]]`
- 控制模型回應的詳細程度（部分 provider 支援）

### `response_format: Optional[Union[dict, Type[BaseModel]]]`
- 指定輸出格式，可傳入 `{"type": "json_object"}` 或 Pydantic `BaseModel` 類別
- litellm 會將 BaseModel 自動轉換為 JSON Schema

### `seed: Optional[int]`
- 設定隨機種子，有助於產生可重現的輸出（OpenAI Beta 功能）

### `tools: Optional[List]`
- function calling / tool use 的工具定義列表
- 源碼：先做驗證 `tools = validate_and_fix_openai_tools(tools=tools)`（`main.py:1153`），再檢查是否有 MCP tools（`main.py:1165`）

### `tool_choice: Optional[Union[str, dict]]`
- 控制模型如何選擇工具，`"auto"`, `"none"`, `"required"` 或指定特定工具
- 源碼：`tool_choice = validate_chat_completion_tool_choice(tool_choice=tool_choice)`（`main.py:1155`）

### `parallel_tool_calls: Optional[bool]`
- 是否允許模型在單次回應中同時呼叫多個工具

### `logprobs: Optional[bool]`
- 是否回傳每個輸出 token 的 log probability

### `top_logprobs: Optional[int]`
- 每個 token position 回傳機率最高的前 N 個 tokens（需 `logprobs=True`，最大 5）

### `web_search_options: Optional[OpenAIWebSearchOptions]`
- OpenAI 的網頁搜尋工具選項（搭配 Responses API bridge 使用）
- 源碼：`responses_api_bridge_check(model=model, ..., web_search_options=web_search_options)`

---

## 五、即將棄用的參數

### `functions: Optional[List]`
### `function_call: Optional[str]`
- OpenAI 舊版 function calling API，已被 `tools` / `tool_choice` 取代
- litellm 仍支援，並在不支援的 provider 上可透過 `litellm.add_function_to_prompt=True` 轉為 prompt 注入：
  ```python
  # main.py:1530
  if litellm.add_function_to_prompt and optional_params.get("functions_unsupported_model"):
      messages = function_call_prompt(messages=messages, functions=functions_unsupported_model)
  ```

---

## 六、API 連接參數

### `base_url: Optional[str]` / `api_base` (kwargs)
- API 端點的 base URL，`base_url` 是新版別名，內部統一轉換：
  ```python
  # main.py:1343
  if base_url is not None:
      api_base = base_url
  ```

### `api_version: Optional[str]`
- API 版本，主要用於 Azure OpenAI（例如 `"2024-02-01"`）
- 源碼：Azure 路徑中的 fallback 鏈：
  ```python
  # main.py:1674
  api_version = (
      api_version
      or litellm.api_version
      or get_secret_str("AZURE_API_VERSION")
      or litellm.AZURE_DEFAULT_API_VERSION
  )
  ```

### `api_key: Optional[str]`
- 覆蓋環境變數中的 API key，優先級高於 env var

### `model_list: Optional[list]`
- 傳入多個 deployment 的設定（api_base + api_key 等），觸發 batch completion：
  ```python
  # main.py:1351
  if model_list is not None:
      deployments = [m["litellm_params"] for m in model_list if m["model_name"] == model]
      return litellm.batch_completion_models(deployments=deployments, **args)
  ```

### `extra_headers: Optional[dict]`
- 附加到請求的 HTTP headers，與 `headers` kwargs 合併：
  ```python
  # main.py:1241
  headers = kwargs.get("headers", None) or extra_headers
  if extra_headers is not None:
      headers.update(extra_headers)
  ```

### `deployment_id`
- Azure 部署 ID，設定後自動將 `custom_llm_provider` 設為 `"azure"`（`main.py:1366`）

### `service_tier: Optional[str]`
- OpenAI 的服務層級，例如 `"default"` 或 `"flex"`

### `safety_identifier: Optional[str]`
- 部分 provider（如 Google）的安全識別符

---

## 七、LiteLLM 專屬具名參數

### `thinking: Optional[AnthropicThinkingParam]`
- Anthropic Claude 的 extended thinking 設定，例如 `{"type": "enabled", "budget_tokens": 5000}`
- 源碼：`thinking = validate_and_fix_thinking_param(thinking=thinking)`（`main.py:1159`），支援 camelCase 正規化

### `shared_session: Optional[ClientSession]`
- 共用的 aiohttp `ClientSession`，避免每次請求建立新連線

### `enable_json_schema_validation: Optional[bool]`
- 覆蓋全域的 `litellm.enable_json_schema_validation` 設定，對單次請求生效

---

## 八、`**kwargs` 擴充參數（重要）

這些參數沒有出現在函數簽名中，透過 `kwargs` 傳遞，在函數內 `locals()` 後解包：

```python
# main.py:1162
args = locals()
# ...以下逐一從 kwargs 中取出
api_base = kwargs.get("api_base", None)
mock_response = kwargs.get("mock_response", None)
custom_llm_provider = kwargs.get("custom_llm_provider", None)
```

### 除錯 / 測試
| 參數 | 說明 |
|------|------|
| `mock_response` | 直接回傳模擬回應，繞過真實 API 呼叫 |
| `mock_tool_calls` | 模擬工具呼叫回應 |
| `mock_timeout` | 模擬請求逾時 |
| `mock_delay` | 模擬延遲時間 |
| `verbose` | 啟用詳細 logging |
| `logger_fn` | 自定義 logging 函數 |
| `no-log` | 停用此次請求的 logging |

### Provider 控制
| 參數 | 說明 |
|------|------|
| `custom_llm_provider` | 強制指定 provider（如 `"bedrock"`, `"ollama"`） |
| `organization` | OpenAI organization ID |
| `ssl_verify` | 是否驗證 SSL 憑證 |
| `provider_specific_header` | 特定 provider 的 header 設定 |

### 成本追蹤
| 參數 | 說明 |
|------|------|
| `input_cost_per_token` | 自定義輸入 token 費率 |
| `output_cost_per_token` | 自定義輸出 token 費率 |
| `input_cost_per_second` | 按秒計費模型的輸入費率 |
| `output_cost_per_second` | 按秒計費模型的輸出費率 |

```python
# main.py:1412
if (input_cost_per_token is not None and output_cost_per_token is not None) or input_cost_per_second is not None:
    litellm.register_model({f"{custom_llm_provider}/{model}": _build_custom_pricing_entry(...)})
```

### Retry / Fallback
| 參數 | 說明 |
|------|------|
| `max_retries` / `num_retries` | 最大重試次數 |
| `cooldown_time` | 重試冷卻時間（秒） |
| `fallbacks` | fallback 模型列表，觸發時呼叫 `completion_with_fallbacks()` |
| `context_window_fallback_dict` | context window 超過時的 fallback 模型 |

```python
# main.py:1348
fallbacks = fallbacks or litellm.model_fallbacks
if fallbacks is not None:
    return completion_with_fallbacks(**args)
```

### 訊息處理
| 參數 | 說明 |
|------|------|
| `ensure_alternating_roles` | 強制訊息角色交替（user/assistant） |
| `user_continue_message` | 角色不交替時自動插入的 user 訊息 |
| `assistant_continue_message` | 角色不交替時自動插入的 assistant 訊息 |
| `supports_system_message` | 若為 False，system message 會轉換為 user message |
| `litellm_system_prompt` | 自動在 messages 前插入的系統提示 |

### 自定義 Prompt Template
| 參數 | 說明 |
|------|------|
| `initial_prompt_value` | Prompt 開頭文字 |
| `roles` | 角色名稱對應表（用於非 ChatML 格式的模型） |
| `final_prompt_value` | Prompt 結尾文字 |
| `bos_token` | Beginning of sentence token |
| `eos_token` | End of sentence token |

```python
# main.py:1425
custom_prompt_dict = {}
if initial_prompt_value or roles or final_prompt_value or bos_token or eos_token:
    custom_prompt_dict = {model: {...}}
```

### Prompt Management
| 參數 | 說明 |
|------|------|
| `prompt_id` | 從 prompt management 系統載入的 prompt ID |
| `prompt_variables` | prompt template 的變數替換 |
| `prompt_label` | prompt 標籤 |
| `prompt_version` | prompt 版本號 |

### 觀測性 / Tracing
| 參數 | 說明 |
|------|------|
| `metadata` | 附加到 logging 的任意 dict（如 prompt 版本、標籤） |
| `litellm_trace_id` | 自定義 trace ID |
| `litellm_session_id` | 對話 session ID |
| `litellm_metadata` | LiteLLM 系統內部 metadata |
| `model_info` | 模型資訊，傳遞給 logging |
| `proxy_server_request` | Proxy server 的原始請求物件 |

### Azure 專屬
| 參數 | 說明 |
|------|------|
| `azure_ad_token` | Azure AD token（取代 api_key） |
| `azure_ad_token_provider` | 動態取得 Azure AD token 的 callable |
| `tenant_id` | Azure tenant ID |
| `client_id` | Azure client ID |
| `client_secret` | Azure client secret |
| `azure_username` | Azure username |
| `azure_password` | Azure password |
| `azure_scope` | Azure OAuth scope |

### 速率限制
| 參數 | 說明 |
|------|------|
| `tpm` | tokens per minute 限制 |
| `rpm` | requests per minute 限制 |

### 其他
| 參數 | 說明 |
|------|------|
| `client` | 傳入已建立的 OpenAI/AsyncOpenAI client，避免重複初始化 |
| `acompletion` | 內部用：標記這是 async 呼叫（由 `acompletion()` 設定） |
| `drop_params` | 自動丟棄特定 provider 不支援的參數 |
| `additional_drop_params` | 額外要丟棄的參數列表 |
| `base_model` | 用於計費的基底模型名稱（fine-tuned 模型場景） |
| `hf_model_name` | HuggingFace 模型名稱 |
| `preset_cache_key` | 預設的 cache key |
| `merge_reasoning_content_in_choices` | 是否將 reasoning content 合併進 choices |
| `litellm_request_debug` | 啟用請求除錯資訊 |

---

## 九、參數的內部處理流程

```
completion() 呼叫
    │
    ├─ 1. 驗證 (validate_and_fix_*)
    │      messages, tools, tool_choice, stop, thinking
    │
    ├─ 2. MCP 工具偵測
    │      → 若含 MCP tools，轉交 acompletion_with_mcp()
    │
    ├─ 3. kwargs 解包
    │      args = locals() 快照整個本地變數
    │      api_base, mock_response, custom_llm_provider, metadata...
    │
    ├─ 4. Prompt Management Hooks
    │      prompt_id + litellm_logging_obj → 可動態替換 model/messages
    │
    ├─ 5. Fallback / Model List 路由
    │      fallbacks → completion_with_fallbacks()
    │      model_list → batch_completion_models()
    │
    ├─ 6. Provider 解析
    │      get_llm_provider(model, custom_llm_provider, api_base, api_key)
    │      → (model, custom_llm_provider, dynamic_api_key, api_base)
    │
    ├─ 7. Responses API Bridge 檢查
    │      某些模型（含 web_search_options 或 reasoning_effort）
    │      → 轉發至 responses_api_bridge.completion()
    │
    ├─ 8. optional_params 組裝
    │      get_optional_params(**optional_param_args)
    │      → 過濾出當前 provider 支援的參數子集
    │
    ├─ 9. litellm_params 組裝
    │      get_litellm_params(acompletion, api_key, metadata, model_info...)
    │      → LiteLLM 內部用的控制參數
    │
    ├─ 10. Mock 回應處理
    │       mock_response / mock_tool_calls / mock_timeout
    │       → 直接回傳，不呼叫真實 API
    │
    └─ 11. Provider 路由
           custom_llm_provider == "azure" → azure 處理邏輯
           custom_llm_provider == "openai" → openai 處理邏輯
           ... (每個 provider 各自的 API 呼叫)
```

---

## 十、`optional_params` vs `litellm_params` 的區別

| | `optional_params` | `litellm_params` |
|--|-------------------|-----------------|
| **內容** | 傳遞給 LLM provider 的參數 | LiteLLM 內部控制參數 |
| **例子** | `temperature`, `tools`, `stream` | `api_key`, `metadata`, `model_info` |
| **provider 過濾** | 是，`get_optional_params()` 只保留支援的參數 | 否，全部保留 |
| **傳遞方向** | 送往 provider API | 用於 logging、routing、caching |
| **源碼** | `main.py:1516` | `main.py:1541` |

---

*生成自 litellm/main.py，分析日期：2026-03-26*
