# LiteLLM 三大 Provider 功能相容性深度解析

> 分析對象：AWS Bedrock Claude、GCP Vertex AI Gemini、Azure OpenAI GPT
> 分析日期：2026-03-31
> LiteLLM 版本：main branch (commit bdf4acc)

---

## 一、架構概覽：completion() 呼叫鏈

```
使用者呼叫 litellm.completion()
         │
         ▼
litellm/main.py
  └─ get_optional_params()        ← 依 provider 呼叫對應 Config 的 map_openai_params()
         │
         ▼
  各 Provider Config.map_openai_params()
  └─ 過濾/轉換 OpenAI 參數 → provider-specific 參數
         │
         ▼
  各 Provider Handler (transform_request / _transform_request_helper)
  └─ 組裝最終請求 Body
         │
         ▼
  HTTP 呼叫到真實 LLM Endpoint
```

---

## 二、AWS Bedrock Claude（Converse API）

### 2.1 核心轉換邏輯

**主要檔案**：[`litellm/llms/bedrock/chat/converse_transformation.py`](litellm/llms/bedrock/chat/converse_transformation.py)

#### 2.1.1 支援的 Inference Config 欄位（`inferenceConfig`）

```python
# converse_transformation.py:123-158
# AmazonConverseConfig 定義了哪些參數會進入 inferenceConfig
class AmazonConverseConfig(BaseConfig):
    maxTokens: Optional[int]       # ← OpenAI: max_tokens / max_completion_tokens
    stopSequences: Optional[List[str]]  # ← OpenAI: stop
    temperature: Optional[int]     # ← OpenAI: temperature
    topP: Optional[int]            # ← OpenAI: top_p
    topK: Optional[int]            # ← 非 OpenAI 標準！模型特定參數
```

#### 2.1.2 Config Blocks（頂層特殊欄位）

```python
# converse_transformation.py:152-158
# 這些是 Bedrock Converse 獨有的頂層 block，無 OpenAI 對應
@classmethod
def get_config_blocks(cls) -> dict:
    return {
        "guardrailConfig": GuardrailConfigBlock,    # ← 內容安全過濾
        "performanceConfig": PerformanceConfigBlock, # ← 效能最佳化
        "serviceTier": ServiceTierBlock,             # ← 服務等級
    }
```

#### 2.1.3 get_supported_openai_params()：宣告支援哪些 OpenAI 參數

```python
# converse_transformation.py:503-584
def get_supported_openai_params(self, model: str) -> List[str]:
    supported_params = [
        "max_tokens",            # → inferenceConfig.maxTokens
        "max_completion_tokens", # → inferenceConfig.maxTokens（同上）
        "stream",
        "stream_options",
        "stop",                  # → inferenceConfig.stopSequences
        "temperature",           # → inferenceConfig.temperature
        "top_p",                 # → inferenceConfig.topP
        "extra_headers",
        "response_format",       # → outputConfig（原生）或 tool-call fallback
        "requestMetadata",       # → 頂層 requestMetadata（Bedrock 專屬）
        "service_tier",          # → serviceTier config block
        "parallel_tool_calls",   # → 僅 Claude 4.5+ 支援
    ]

    # tools / tool_choice：依模型能力動態加入
    # 僅 anthropic / mistral / cohere / llama / nova 系列支援
    if base_model.startswith("anthropic") or base_model.startswith("amazon.nova") ...:
        supported_params.append("tools")

    # Nova 模型才支援 web_search_options（對應 nova_grounding systemTool）
    if base_model.startswith("amazon.nova"):
        supported_params.append("web_search_options")

    # thinking / reasoning_effort：僅 Claude 3.7+、Claude 4+ 支援
    # 注意：三種 model family 有不同的轉換邏輯（見下方 2.1.4）
    if "claude-3-7" in model or "claude-sonnet-4" in model or "claude-opus-4" in model ...:
        supported_params.append("thinking")
        supported_params.append("reasoning_effort")

    return supported_params

    # ⚠️ 以下 OpenAI 標準參數「不在」supported_params：
    # n, logit_bias, user, logprobs, top_logprobs,
    # frequency_penalty, presence_penalty, seed,
    # function_call（legacy）, functions（legacy）
    # → 這些參數在 map_openai_params() 時不會被加入 optional_params，直接被丟棄
```

#### 2.1.4 reasoning_effort 的三路轉換

```python
# converse_transformation.py:435-482
def _handle_reasoning_effort_parameter(
    self, model: str, reasoning_effort: str, optional_params: dict
) -> None:
    if "gpt-oss" in model:
        # GPT-OSS on Bedrock：直接保留 reasoning_effort
        # 最終進入 additionalModelRequestFields
        optional_params["reasoning_effort"] = reasoning_effort

    elif self._is_nova_2_model(model):
        # Nova 2：轉換為 reasoningConfig 結構
        # 進入 additionalModelRequestFields.reasoningConfig
        optional_params.update({
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": reasoning_effort  # "low"/"medium"/"high"
            }
        })

    else:
        # Anthropic Claude：轉換為 thinking 參數
        # 進入 additionalModelRequestFields.thinking
        optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=reasoning_effort, model=model
        )
```

#### 2.1.5 _prepare_request_params()：參數分桶核心邏輯

```python
# converse_transformation.py:1184-1265
def _prepare_request_params(self, optional_params: dict, model: str):
    supported_converse_params = list(
        AmazonConverseConfig.__annotations__.keys()  # maxTokens, stopSequences, ...
    ) + ["top_k"]
    supported_tool_call_params = ["tools", "tool_choice"]
    supported_config_params = list(self.get_config_blocks().keys())  # guardrailConfig, ...
    total_supported_params = (
        supported_converse_params + supported_tool_call_params + supported_config_params
    )

    inference_params.pop("json_mode", None)    # LiteLLM 內部參數，Bedrock 不認識
    inference_params.pop("output_config", None) # snake_case 版本，Bedrock 只接受 camelCase

    # 額外提取：不進入分桶流程
    request_metadata = inference_params.pop("requestMetadata", None)
    output_config     = inference_params.pop("outputConfig", None)

    # 分桶 ①：已知參數 → inferenceConfig
    inference_params = {
        k: v for k, v in inference_params.items()
        if k in total_supported_params
    }
    # 分桶 ②：未知參數 → additionalModelRequestFields（直接轉給底層模型）
    additional_request_params = {
        k: v for k, v in inference_params.items()
        if k not in total_supported_params
    }

    # 清理 LiteLLM 內部參數（如 MCP 相關欄位）
    additional_request_params = filter_internal_params(additional_request_params)
    additional_request_params = filter_exceptions_from_params(additional_request_params)
    additional_request_params.pop("parallel_tool_calls", None)  # Bedrock 本身不接受此欄位

    return inference_params, additional_request_params, request_metadata, output_config
```

#### 2.1.6 _transform_request_helper()：最終 Request Body 組裝

```python
# converse_transformation.py:1387-1495
def _transform_request_helper(self, model, system_content_blocks, optional_params, ...):
    # 驗證：如果 messages 有 tool_call blocks，但沒有傳 tools= 參數 → 報錯或自動加 dummy tool
    if "tools" not in optional_params and has_tool_call_blocks(messages):
        raise litellm.UnsupportedParamsError(...)  # 或 modify_params=True 時自動修正

    # 組裝最終 Body
    data: CommonRequestObject = {
        # 分桶 ②：未識別參數，Bedrock 把這些直接穿透給底層模型
        "additionalModelRequestFields": additional_request_params,
        "system": system_content_blocks,
        # 分桶 ①：標準推論參數
        "inferenceConfig": self._transform_inference_params(inference_params),
        # top_k 的處理是模型特定的：
        # - Anthropic → additionalModelRequestFields.top_k
        # - Nova      → inferenceConfig.topK
    }

    # 可選的頂層 Bedrock 欄位
    if bedrock_tool_config is not None:
        data["toolConfig"] = bedrock_tool_config    # tools + tool_choice 的轉換結果
    if request_metadata is not None:
        data["requestMetadata"] = request_metadata  # Bedrock 特定 metadata
    if output_config is not None:
        data["outputConfig"] = output_config        # 原生結構化輸出（Claude 4.5+ 等）

    # 讀取 guardrailConfig / performanceConfig / serviceTier
    for config_name, config_class in self.get_config_blocks().items():
        config_value = inference_params.pop(config_name, None)
        if config_value is not None:
            data[config_name] = config_class(**config_value)

    return data
```

#### 2.1.7 tool_choice 的限制

```python
# converse_transformation.py:586-615
def map_tool_choice_values(self, model, tool_choice, drop_params):
    if tool_choice == "none":
        # ⚠️ Bedrock Converse 不支援 "none"
        # 必須設定 litellm.drop_params=True 才不會報錯
        raise UnsupportedParamsError("Bedrock doesn't support tool_choice='none'")

    elif tool_choice == "required":
        return ToolChoiceValuesBlock(any={})   # → Bedrock: {"any": {}}

    elif tool_choice == "auto":
        return ToolChoiceValuesBlock(auto={})  # → Bedrock: {"auto": {}}

    elif isinstance(tool_choice, dict):
        # 指定特定 function → Bedrock: {"tool": {"name": "..."}}
        ...
```

#### 2.1.8 web_search_options 的限制

```python
# converse_transformation.py:357-386
def _map_web_search_options(self, web_search_options: dict, model: str):
    if "nova" not in model.lower():
        # ⚠️ 非 Nova 模型完全不支援 web search
        # web_search_options 會被靜默忽略（返回 None）
        verbose_logger.debug(f"web_search_options passed but model {model} is not a Nova model.")
        return None

    # Nova 模型：只支援開啟/關閉，不支援 search_context_size 或 user_location
    return BedrockToolBlock(systemTool={"name": "nova_grounding"})
```

#### 2.1.9 被封鎖的 Anthropic Beta Headers（≠ 功能本身被封鎖）

```python
# converse_transformation.py:88-92
# 以下 Anthropic beta HEADERS 會被自動過濾，不代表功能本身不支援
UNSUPPORTED_BEDROCK_CONVERSE_BETA_PATTERNS = [
    "advanced-tool-use",       # Bedrock Converse 不支援進階工具功能
    "prompt-caching",          # Anthropic 的 prompt-caching beta HEADER 不支援
                               # ⚠️ 但 prompt caching 功能本身支援，見 2.1.10
    "compact-2026-01-12",      # compact beta 功能不支援 Converse/ConverseStream
]
```

> **重要區分**：`prompt-caching` 被封鎖的是 **Anthropic 的 beta header 傳遞方式**，
> 而 Bedrock Converse API 有自己的 `cachePoint` block 機制實現相同功能。

---

#### 2.1.10 Prompt Caching（透過 `cachePoint` 機制）

Bedrock Converse API 使用 `cachePoint` block（非 Anthropic `cache_control`）實現 prompt caching。LiteLLM 提供兩種使用方式：

**方式一：直接在 messages 中標記 `cache_control`**

LiteLLM 會在 `_get_cache_point_block()` 方法中將 OpenAI 格式的 `cache_control` 轉換成 Bedrock 的 `cachePoint` block：

```python
# converse_transformation.py:1102-1127
# 將 OpenAI cache_control 轉換為 Bedrock cachePoint
def _get_cache_point_block(self, message_block, block_type, model=None):
    cache_control = message_block.get("cache_control", None)
    if cache_control is None:
        return None

    cache_point = CachePointBlock(type="default")

    # Claude 4.5+ on Bedrock 支援 TTL（"5m" 或 "1h"）
    if isinstance(cache_control, dict) and "ttl" in cache_control:
        ttl = cache_control["ttl"]
        if ttl in ["5m", "1h"] and model is not None:
            if is_claude_4_5_on_bedrock(model):   # ← 僅 Claude 4.5/4.6 系列支援 TTL
                cache_point["ttl"] = ttl

    if block_type == "system":
        return SystemContentBlock(cachePoint=cache_point)   # 系統訊息 cachePoint
    else:
        return ContentBlock(cachePoint=cache_point)         # 對話訊息 cachePoint
```

此方法被調用於三個位置：
- **system messages**：`_transform_system_message()` (converse_transformation.py:1141, 1152)
- **user messages（list content）**：`_bedrock_converse_messages_pt()` (factory.py:4818-4827)
- **user messages（string content）**：`_bedrock_converse_messages_pt()` (factory.py:4831-4838)
- **tool result messages**：另有單獨的 cachePoint block 追加邏輯 (factory.py:4873-4889)

**方式二：`cache_control_injection_points` 參數（LiteLLM 專屬便利功能）**

```python
# integrations/anthropic_cache_control_hook.py
# AnthropicCacheControlHook 在 completion() 前自動注入 cache_control

# 支援兩種 injection point 類型（types/integrations/anthropic_cache_control_hook.py）:

# 1. message 層級：注入到特定 message
class CacheControlMessageInjectionPoint(TypedDict):
    location: Literal["message"]
    role: Optional[Literal["user", "system", "assistant"]]  # 依 role 注入
    index: Optional[Union[int, str]]                        # 依 index 注入（支援負數）
    control: Optional[ChatCompletionCachedContent]          # 預設 type="ephemeral"

# 2. tool_config 層級：在 tools 清單末尾加入 cachePoint（Bedrock 專屬）
class CacheControlToolConfigInjectionPoint(TypedDict):
    location: Literal["tool_config"]
```

Hook 執行邏輯：
- `location: "message"` → pre-process 階段注入 `cache_control` 到指定 message，
  再由 `_get_cache_point_block()` 轉換為 Bedrock `cachePoint` block
- `location: "tool_config"` → 在 `_transform_request_helper()` 中直接追加
  `{"cachePoint": {"type": "default"}}` 到 tools 清單末尾

```python
# converse_transformation.py:1449-1457
# tool_config injection point 的實現
cache_injection_points = additional_request_params.pop(
    "cache_control_injection_points", None
)
if cache_injection_points and len(bedrock_tools) > 0:
    for point in cache_injection_points:
        if point.get("location") == "tool_config":
            bedrock_tools.append({"cachePoint": {"type": "default"}})
            break
```

**TTL 支援情況**：

```python
# common_utils.py:483-511
# 只有 Claude 4.5+ 系列支援 TTL（"5m" 或 "1h"）
def is_claude_4_5_on_bedrock(model: str) -> bool:
    claude_4_5_patterns = [
        "sonnet-4.5", "haiku-4.5", "opus-4.5",   # Claude 4.5 系列
        "sonnet-4.6", "opus-4.6",                  # Claude 4.6 系列
        # ... 各種命名格式
    ]
    return any(pattern in model.lower() for pattern in claude_4_5_patterns)
```

| 模型 | cachePoint 支援 | TTL 支援 |
|---|---|---|
| Claude 3.x (Bedrock) | ✅ default type | ❌ 無 TTL |
| Claude 4.5+ (Bedrock) | ✅ default type | ✅ "5m" / "1h" |

---

### 2.2 Bedrock Claude 功能支援總表

| OpenAI 參數 | Bedrock 對應 | 支援狀態 | 備註 |
|---|---|---|---|
| `max_tokens` | `inferenceConfig.maxTokens` | ✅ 支援 | |
| `max_completion_tokens` | `inferenceConfig.maxTokens` | ✅ 支援 | 同上 |
| `temperature` | `inferenceConfig.temperature` | ✅ 支援 | |
| `top_p` | `inferenceConfig.topP` | ✅ 支援 | |
| `stop` | `inferenceConfig.stopSequences` | ✅ 支援 | |
| `stream` | Converse streaming | ✅ 支援 | |
| `tools` | `toolConfig.tools` | ✅ 部分支援 | 依模型，非所有模型 |
| `tool_choice` | `toolConfig.toolChoice` | ✅ 部分支援 | 不支援 `"none"` |
| `response_format` | `outputConfig` 或工具 fallback | ✅ 部分支援 | 依模型版本 |
| `thinking` | `additionalModelRequestFields.thinking` | ✅ 部分支援 | 僅 Claude 3.7+ |
| `reasoning_effort` | 依模型不同轉換 | ✅ 部分支援 | 僅特定模型 |
| `web_search_options` | `nova_grounding` systemTool | ✅ 部分支援 | 僅 Nova 系列 |
| `top_k` | model-specific | ✅ 部分支援 | 非標準 OpenAI 參數 |
| `parallel_tool_calls` | Claude 4.5+ 特殊處理 | ✅ 部分支援 | 僅 Claude 4.5+ |
| `requestMetadata` | 頂層 `requestMetadata` | ✅ Bedrock 專屬 | 非 OpenAI 標準 |
| `guardrailConfig` | 頂層 `guardrailConfig` | ✅ Bedrock 專屬 | 非 OpenAI 標準 |
| `performanceConfig` | 頂層 `performanceConfig` | ✅ Bedrock 專屬 | 非 OpenAI 標準 |
| `n` | 無 | ❌ 不支援 | 直接丟棄 |
| `logit_bias` | 無 | ❌ 不支援 | 直接丟棄 |
| `user` | 無 | ❌ 不支援 | 直接丟棄 |
| `logprobs` | 無 | ❌ 不支援 | 直接丟棄 |
| `top_logprobs` | 無 | ❌ 不支援 | 直接丟棄 |
| `frequency_penalty` | 無 | ❌ 不支援 | 直接丟棄 |
| `presence_penalty` | 無 | ❌ 不支援 | 直接丟棄 |
| `seed` | 無 | ❌ 不支援 | 直接丟棄 |
| `function_call` | 無 | ❌ 不支援 | Legacy，直接丟棄 |
| `tool_choice="none"` | 無 | ❌ 不支援 | 需 `drop_params=True` |
| Prompt Caching | Converse `cachePoint` block | ✅ 支援 | 透過 `cache_control` 或 `cache_control_injection_points`；Claude 4.5+ 支援 TTL |

---

## 三、GCP Vertex AI Gemini

### 3.1 核心轉換邏輯

**主要檔案**：
- [`litellm/llms/vertex_ai/gemini/transformation.py`](litellm/llms/vertex_ai/gemini/transformation.py)
- [`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`](litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py)

#### 3.1.1 GenerationConfig 型別定義（對應 inferenceConfig）

```python
# litellm/types/llms/vertex_ai.py:215-234
# 這個 TypedDict 決定哪些參數可以進入 generationConfig
class GenerationConfig(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: float               # ← 非 OpenAI 標準，但 Gemini 支援
    candidate_count: int       # ← 對應 OpenAI: n
    max_output_tokens: int     # ← 對應 OpenAI: max_tokens
    stop_sequences: List[str]  # ← 對應 OpenAI: stop
    presence_penalty: float    # ← 僅非 preview 模型支援
    frequency_penalty: float   # ← 僅非 preview 模型支援
    response_mime_type: Literal["text/plain", "application/json"]
    response_schema: dict      # ← 對應 OpenAI: response_format.json_schema
    response_json_schema: dict
    seed: int                  # ← 對應 OpenAI: seed
    responseLogprobs: bool
    logprobs: int              # ← 對應 OpenAI: logprobs
    responseModalities: List[GeminiResponseModalities]  # ← 對應 OpenAI: modalities
    imageConfig: GeminiImageConfig
    thinkingConfig: GeminiThinkingConfig  # ← 對應 OpenAI: thinking/reasoning_effort
    mediaResolution: str       # ← Gemini 2.x 圖片解析度，非 OpenAI 標準
    speechConfig: SpeechConfig # ← 對應 OpenAI: audio
```

#### 3.1.2 get_supported_openai_params()

```python
# vertex_and_google_ai_studio_gemini.py:299-330
def get_supported_openai_params(self, model: str) -> List[str]:
    supported_params = [
        "temperature",           # → generationConfig.temperature
        "top_p",                 # → generationConfig.top_p
        "max_tokens",            # → generationConfig.max_output_tokens
        "max_completion_tokens", # → generationConfig.max_output_tokens
        "stream",
        "tools",                 # → 頂層 tools
        "functions",             # → 頂層 tools（legacy）
        "tool_choice",           # → 頂層 toolConfig（透過 map_tool_choice_values）
        "response_format",       # → generationConfig.response_mime_type + response_schema
        "n",                     # → generationConfig.candidate_count
        "stop",                  # → generationConfig.stop_sequences
        "extra_headers",
        "seed",                  # → generationConfig.seed
        "logprobs",              # → generationConfig.logprobs
        "top_logprobs",          # ⚠️ 列入支援但 GenerationConfig 無此欄位（可能被濾除）
        "modalities",            # → generationConfig.responseModalities
        "audio",                 # → generationConfig.speechConfig
        "parallel_tool_calls",   # → 映射到 Gemini 等效行為
        "web_search_options",    # → 頂層 tools.googleSearch（丟棄 user_location 等）
        "include_server_side_tool_invocations",  # → toolConfig.includeServerSideToolInvocations
    ]

    # frequency_penalty / presence_penalty：僅非 preview 模型才加入
    if self._supports_penalty_parameters(model):
        supported_params.extend(["frequency_penalty", "presence_penalty"])

    # reasoning_effort / thinking：僅 Gemini 2.5-pro 等推理模型
    if supports_reasoning(model):
        supported_params.append("reasoning_effort")  # → generationConfig.thinkingConfig
        supported_params.append("thinking")          # → generationConfig.thinkingConfig

    # ⚠️ 以下 OpenAI 標準參數不在清單中：
    # logit_bias, user, function_call（legacy）
    # → 在 map_openai_params 時被丟棄

    return supported_params
```

#### 3.1.3 _transform_request_body()：核心 Body 組裝

```python
# transformation.py:653-773
def _transform_request_body(messages, model, optional_params, custom_llm_provider, ...):

    # Step 1：處理 system message → 抽取為頂層 system_instruction
    system_instructions, messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    # Step 2：response_schema 的降級處理
    if "response_schema" in optional_params:
        if not get_supports_response_schema(model):
            # ⚠️ 不支援 response_schema 的舊模型：改為在 prompt 中注入 schema 文字
            messages.append({"role": "user", "content": response_schema_prompt(...)})
            optional_params.pop("response_schema")  # 從參數中移除

    # Step 3：提取不進入 generationConfig 的特殊參數
    tools      = optional_params.pop("tools", None)         # → 頂層 tools
    tool_choice = optional_params.pop("tool_choice", None)  # → 頂層 toolConfig
    include_server_side_tool_invocations = optional_params.pop(..., False)
    safety_settings = optional_params.pop("safety_settings", None)  # → 頂層 safetySettings
    optional_params.pop("output_config", None)              # ⚠️ 直接丟棄，Vertex AI 不支援
    labels = optional_params.pop("labels", None)            # → 頂層 labels（僅 Vertex AI）

    # Step 4：過濾參數，只保留 GenerationConfig 有定義的欄位
    config_fields = GenerationConfig.__annotations__.keys()
    filtered_params = {
        k: v for k, v in optional_params.items()
        if _get_equivalent_key(k, set(config_fields))  # 同時支援 snake_case 和 camelCase
    }
    # ⚠️ 這個過濾會靜默丟棄所有 GenerationConfig 沒有定義的參數！
    generation_config = GenerationConfig(**filtered_params)

    # Step 5：Gemini 2.x 特殊：從 messages 的 image detail 提取全局 mediaResolution
    if "gemini-2" in model:
        max_media_resolution = _extract_max_media_resolution_from_messages(messages)
        if max_media_resolution:
            generation_config["mediaResolution"] = ...  # 全局媒體解析度

    # Step 6：組裝最終 Body
    data = RequestBody(contents=content)
    if system_instructions:      data["system_instruction"] = system_instructions
    if tools:                    data["tools"] = tools
    if tool_choice:              data["toolConfig"] = tool_choice
    if safety_settings:          data["safetySettings"] = safety_settings
    if generation_config:        data["generationConfig"] = generation_config
    if cached_content:           data["cachedContent"] = cached_content
    # labels 只加給 Vertex AI，不加給 Google AI Studio (gemini provider)
    if labels and custom_llm_provider != LlmProviders.GEMINI:
        data["labels"] = labels

    # Step 7：extra_body 的參數直接淺層合併進 data（允許傳遞任意 Gemini 原生參數）
    _pop_and_merge_extra_body(data, optional_params)

    return data
```

#### 3.1.4 tool_choice 映射

```python
# vertex_and_google_ai_studio_gemini.py:332-355
def map_tool_choice_values(self, model, tool_choice):
    if tool_choice == "none":
        # OpenAI "none" → Gemini mode="NONE"
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
    elif tool_choice == "required":
        # OpenAI "required" → Gemini mode="ANY"
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
    elif tool_choice == "auto":
        # OpenAI "auto" → Gemini mode="AUTO"
        return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
    elif isinstance(tool_choice, dict):
        # 指定特定 function → Gemini mode="ANY" + allowed_function_names
        name = tool_choice.get("function", {}).get("name", "")
        return ToolConfig(
            functionCallingConfig=FunctionCallingConfig(
                mode="ANY", allowed_function_names=[name]
            )
        )
    else:
        raise UnsupportedParamsError(...)  # 其他值不支援
```

#### 3.1.5 web_search_options 的處理

```python
# vertex_and_google_ai_studio_gemini.py:357-363
def _map_web_search_options(self, value: dict) -> Tools:
    # ⚠️ Google Gemini 不支援 user_location 或 search_context_size
    # 無論傳什麼，都只是開啟 Google Search，options 被完全忽略
    return Tools(googleSearch={})
```

---

### 3.2 Vertex AI Gemini 功能支援總表

| OpenAI 參數 | Gemini 對應 | 支援狀態 | 備註 |
|---|---|---|---|
| `max_tokens` | `generationConfig.max_output_tokens` | ✅ 支援 | |
| `max_completion_tokens` | `generationConfig.max_output_tokens` | ✅ 支援 | |
| `temperature` | `generationConfig.temperature` | ✅ 支援 | |
| `top_p` | `generationConfig.top_p` | ✅ 支援 | |
| `n` | `generationConfig.candidate_count` | ✅ 支援 | |
| `stop` | `generationConfig.stop_sequences` | ✅ 支援 | |
| `seed` | `generationConfig.seed` | ✅ 支援 | |
| `logprobs` | `generationConfig.logprobs` | ✅ 支援 | |
| `stream` | Gemini streaming | ✅ 支援 | |
| `tools` | 頂層 `tools` | ✅ 支援 | |
| `tool_choice` | 頂層 `toolConfig` | ✅ 支援 | 所有值均支援 |
| `response_format` | `generationConfig.response_schema` | ✅ 支援 | 不支援的舊模型降級為 prompt |
| `modalities` | `generationConfig.responseModalities` | ✅ 支援 | |
| `audio` | `generationConfig.speechConfig` | ✅ 支援 | |
| `web_search_options` | `tools.googleSearch` | ✅ 部分支援 | 忽略 user_location / search_context_size |
| `frequency_penalty` | `generationConfig.frequency_penalty` | ✅ 部分支援 | preview 模型不支援 |
| `presence_penalty` | `generationConfig.presence_penalty` | ✅ 部分支援 | preview 模型不支援 |
| `reasoning_effort` | `generationConfig.thinkingConfig` | ✅ 部分支援 | 僅 Gemini 2.5-pro 等 |
| `thinking` | `generationConfig.thinkingConfig` | ✅ 部分支援 | 僅推理模型 |
| `top_k` | `generationConfig.top_k` | ✅ Gemini 專屬 | 非 OpenAI 標準 |
| `safety_settings` | 頂層 `safetySettings` | ✅ Gemini 專屬 | 非 OpenAI 標準 |
| `top_logprobs` | 無對應 | ⚠️ 列出但可能被濾除 | GenerationConfig 無此欄位 |
| `logit_bias` | 無 | ❌ 不支援 | 直接丟棄 |
| `user` | 無 | ❌ 不支援 | 直接丟棄 |
| `function_call` | 無 | ❌ 不支援 | Legacy，直接丟棄 |
| `output_config` | 無 | ❌ 不支援 | 明確 pop 丟棄（L711） |
| `web_search_options.user_location` | 無 | ❌ 不支援 | 被忽略 |
| `web_search_options.search_context_size` | 無 | ❌ 不支援 | 被忽略 |
| Prompt Caching | `cachedContent` | ✅ Gemini 專屬 | 透過 context caching 機制 |

---

## 四、Azure OpenAI GPT

### 4.1 核心轉換邏輯

**主要檔案**：[`litellm/llms/azure/chat/gpt_transformation.py`](litellm/llms/azure/chat/gpt_transformation.py)

#### 4.1.1 AzureOpenAIConfig 定義

```python
# gpt_transformation.py:29-54
class AzureOpenAIConfig(BaseConfig):
    """
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#chat-completions
    """
    # 這些欄位與 OpenAI 完全相同，Azure 幾乎是 OpenAI API 的超集
    frequency_penalty: Optional[int]
    function_call: Optional[Union[str, dict]]   # Legacy function calling
    functions: Optional[list]                   # Legacy function calling
    logit_bias: Optional[dict]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[int]
    stop: Optional[Union[str, list]]
    temperature: Optional[int]
    top_p: Optional[int]
```

#### 4.1.2 get_supported_openai_params()：Azure 支援幾乎全部 OpenAI 參數

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
        "logit_bias",          # ← Bedrock/Gemini 不支援，Azure 支援
        "user",                # ← Bedrock/Gemini 不支援，Azure 支援
        "function_call",       # Legacy，但 Azure 支援
        "functions",           # Legacy，但 Azure 支援
        "tools",
        "tool_choice",
        "top_p",
        "logprobs",
        "top_logprobs",        # ← Bedrock/Gemini 不支援，Azure 支援
        "response_format",
        "seed",
        "extra_headers",
        "parallel_tool_calls",
        "prediction",          # ← 投機解碼（Speculative Decoding），Azure 特有
        "modalities",
        "audio",
        "web_search_options",
        "prompt_cache_key",    # ← Azure 特有
        "store",               # ← Azure 特有
        # ⚠️ 沒有：top_k, thinking, reasoning_effort, guardrailConfig
    ]
```

#### 4.1.3 map_openai_params()：API 版本相依的參數驗證

```python
# gpt_transformation.py:153-248
def map_openai_params(self, non_default_params, optional_params, model, drop_params, api_version=""):
    api_version_times = api_version.split("-")  # e.g. "2024-08-01-preview" → ["2024", "08", "01"]
    api_version_year, api_version_month, api_version_day = api_version_times[:3]

    for param, value in non_default_params.items():

        if param == "tool_choice":
            # ⚠️ tool_choice 需要 API version >= 2023-12-01-preview
            if api_version < "2023-12-01":
                if not litellm.drop_params:
                    raise UnsupportedParamsError(
                        f"Azure 不支援 tool_choice，api_version={api_version}，"
                        f"請升級到 2023-12-01-preview 或更新版本"
                    )
                # drop_params=True 時靜默跳過
            # ⚠️ tool_choice='required' 在 API version <= 2024-05 不支援
            elif value == "required" and api_version_year == "2024" and api_version_month <= "05":
                if not litellm.drop_params:
                    raise UnsupportedParamsError(
                        f"Azure 不支援 tool_choice='required'，api_version={api_version}"
                    )
            else:
                optional_params["tool_choice"] = value

        elif param == "response_format" and isinstance(value, dict):
            # ⚠️ response_format 限制：
            # 1. GPT-3.5 系列不支援
            # 2. 需要 API version >= 2024-08 (approx)
            is_model_supported = self._is_response_format_supported_model(model)
            # _is_response_format_supported_model 對 gpt-3.5 / gpt-35 返回 False
            is_version_supported = self._is_response_format_supported_api_version(
                api_version_year, api_version_month
            )
            optional_params = self._add_response_format_to_tools(
                optional_params=optional_params,
                value=value,
                is_response_format_supported=(is_model_supported and is_version_supported)
            )

        elif param in supported_openai_params:
            optional_params[param] = value  # 其他參數直接通過

    return optional_params
```

#### 4.1.4 transform_request()：最簡化的 Body 組裝

```python
# gpt_transformation.py:250-263
def transform_request(self, model, messages, optional_params, litellm_params, headers):
    # Azure 使用 OpenAI SDK，只需做極少的轉換
    messages = convert_to_azure_openai_messages(messages)
    return {
        "model": model,
        "messages": messages,
        **optional_params,  # ← 所有已通過驗證的參數直接展開，無需名稱轉換
    }
    # 注意：實際 HTTP 調用是透過 OpenAI SDK，response 轉換也在 azure.py 中
    # transform_response 在這個 class 故意拋出 NotImplementedError
```

---

### 4.2 Azure OpenAI 功能支援總表

| OpenAI 參數 | Azure 對應 | 支援狀態 | 備註 |
|---|---|---|---|
| `temperature` | `temperature` | ✅ 完整支援 | |
| `n` | `n` | ✅ 完整支援 | |
| `stream` / `stream_options` | 同上 | ✅ 完整支援 | |
| `stop` | `stop` | ✅ 完整支援 | |
| `max_tokens` | `max_tokens` | ✅ 完整支援 | |
| `max_completion_tokens` | `max_completion_tokens` | ✅ 完整支援 | |
| `tools` | `tools` | ✅ 完整支援 | |
| `tool_choice` | `tool_choice` | ✅ 條件支援 | 需 API version >= 2023-12-01 |
| `tool_choice="required"` | 同上 | ✅ 條件支援 | 需 API version > 2024-05 |
| `presence_penalty` | `presence_penalty` | ✅ 完整支援 | |
| `frequency_penalty` | `frequency_penalty` | ✅ 完整支援 | |
| `logit_bias` | `logit_bias` | ✅ 完整支援 | Bedrock/Gemini 不支援 |
| `user` | `user` | ✅ 完整支援 | Bedrock/Gemini 不支援 |
| `top_p` | `top_p` | ✅ 完整支援 | |
| `logprobs` | `logprobs` | ✅ 完整支援 | |
| `top_logprobs` | `top_logprobs` | ✅ 完整支援 | Bedrock/Gemini 不支援 |
| `response_format` | `response_format` | ✅ 條件支援 | 不支援 GPT-3.5；需夠新的 API version |
| `seed` | `seed` | ✅ 完整支援 | |
| `parallel_tool_calls` | `parallel_tool_calls` | ✅ 完整支援 | |
| `prediction` | `prediction` | ✅ Azure 特有 | 投機解碼 |
| `modalities` | `modalities` | ✅ 完整支援 | |
| `audio` | `audio` | ✅ 完整支援 | |
| `web_search_options` | `web_search_options` | ✅ 完整支援 | |
| `function_call` | `function_call` | ✅ 支援（Legacy） | |
| `functions` | `functions` | ✅ 支援（Legacy） | |
| `prompt_cache_key` | `prompt_cache_key` | ✅ Azure 特有 | |
| `store` | `store` | ✅ Azure 特有 | |
| `top_k` | 無 | ❌ 不支援 | OpenAI/Azure 本就無此參數 |
| `thinking` | 無 | ❌ 不支援 | Azure GPT 不支援 |
| `reasoning_effort` | 無（GPT 版） | ❌ 不支援 | o-series 用 o_series_transformation |
| `guardrailConfig` | 無 | ❌ 不適用 | Bedrock 特有 |
| `safety_settings` | 無 | ❌ 不適用 | Gemini 特有 |
| `response_format` + GPT-3.5 | 無 | ❌ 不支援 | 模型限制 |

---

## 五、三大 Provider 功能對照總表

| 功能/參數 | Bedrock Claude | Vertex Gemini | Azure GPT |
|---|---|---|---|
| `max_tokens` | ✅ `inferenceConfig.maxTokens` | ✅ `max_output_tokens` | ✅ 直接傳遞 |
| `temperature` | ✅ `inferenceConfig.temperature` | ✅ `generationConfig.temperature` | ✅ 直接傳遞 |
| `top_p` | ✅ `inferenceConfig.topP` | ✅ `generationConfig.top_p` | ✅ 直接傳遞 |
| `top_k` | ✅ model-specific | ✅ `generationConfig.top_k` | ❌ 不存在 |
| `stop` | ✅ `stopSequences` | ✅ `stop_sequences` | ✅ 直接傳遞 |
| `n` | ❌ 丟棄 | ✅ `candidate_count` | ✅ 直接傳遞 |
| `stream` | ✅ | ✅ | ✅ |
| `tools` | ✅ 部分模型 | ✅ | ✅ |
| `tool_choice` | ✅ 不支援 `"none"` | ✅ 全部值 | ✅ API 版本相依 |
| `response_format` | ✅ 部分模型 | ✅ 降級 fallback | ✅ 部分模型/版本 |
| `seed` | ❌ 丟棄 | ✅ | ✅ |
| `logprobs` | ❌ 丟棄 | ✅ | ✅ |
| `top_logprobs` | ❌ 丟棄 | ⚠️ 可能被濾除 | ✅ |
| `logit_bias` | ❌ 丟棄 | ❌ 丟棄 | ✅ |
| `user` | ❌ 丟棄 | ❌ 丟棄 | ✅ |
| `frequency_penalty` | ❌ 丟棄 | ✅ 部分模型 | ✅ |
| `presence_penalty` | ❌ 丟棄 | ✅ 部分模型 | ✅ |
| `parallel_tool_calls` | ✅ Claude 4.5+ | ✅ | ✅ |
| `thinking` | ✅ Claude 3.7+/4 | ✅ 推理模型 | ❌ |
| `reasoning_effort` | ✅ model-specific | ✅ 推理模型 | ❌（GPT）|
| `web_search_options` | ✅ Nova 系列（限制多） | ✅（忽略大部分 options） | ✅ |
| `guardrailConfig` | ✅ Bedrock 專屬 | ❌ | ❌ |
| `safety_settings` | ❌ | ✅ Gemini 專屬 | ❌ |
| Prompt Caching | ✅ Converse `cachePoint` block（`cache_control` 或 `cache_control_injection_points`）| ✅ cachedContent | ✅ |
| `function_call`（Legacy） | ❌ | ❌ | ✅ |

---

## 六、完整呼叫範例

### 6.1 AWS Bedrock Claude 完整範例

```python
import litellm

# 允許不支援的參數自動丟棄，避免報錯
litellm.drop_params = True

response = litellm.completion(
    # Provider 路由：bedrock/ 前綴
    model="bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0",

    messages=[
        {"role": "system", "content": "你是一個專業助理"},
        {"role": "user",   "content": "請分析這段程式碼的安全性問題"}
    ],

    # === 標準 inferenceConfig 參數（完整支援）===
    max_tokens=2048,           # → inferenceConfig.maxTokens
    temperature=0.7,           # → inferenceConfig.temperature
    top_p=0.9,                 # → inferenceConfig.topP
    stop=["END", "STOP"],      # → inferenceConfig.stopSequences

    # === Tool Calling（Claude 系列支援）===
    tools=[{
        "type": "function",
        "function": {
            "name": "scan_code",
            "description": "掃描程式碼安全性",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string"}
                }
            }
        }
    }],
    tool_choice="auto",        # → toolConfig.toolChoice.auto
                               # ⚠️ "none" 不支援，會報 UnsupportedParamsError

    # === Extended Thinking（Claude 3.7+/4 專屬）===
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # Bedrock 最低 1024，會自動 clamp
    },

    # === Bedrock 專屬參數（非 OpenAI 標準）===
    guardrailConfig={          # 內容過濾
        "guardrailIdentifier": "my-guardrail-id",
        "guardrailVersion":    "1",
        "trace":               "enabled"
    },
    requestMetadata={          # 請求追蹤 metadata（最多 16 組）
        "project": "my-project",
        "environment": "prod"
    },

    # === 不支援的 OpenAI 參數（drop_params=True 時靜默丟棄）===
    # n=3,                     # ❌ Bedrock 不支援多個 completions → 丟棄
    # logit_bias={},           # ❌ 不支援 → 丟棄
    # seed=42,                 # ❌ 不支援 → 丟棄
    # frequency_penalty=0.5,   # ❌ 不支援 → 丟棄
    # presence_penalty=0.5,    # ❌ 不支援 → 丟棄
)

print(response.choices[0].message.content)
```

**最終送出給 Bedrock Converse API 的 Body 結構：**
```json
{
  "modelId": "anthropic.claude-sonnet-4-5-20251001-v1:0",
  "system": [{"text": "你是一個專業助理"}],
  "messages": [{"role": "user", "content": [{"text": "請分析..."}]}],
  "inferenceConfig": {
    "maxTokens": 2048,
    "temperature": 0.7,
    "topP": 0.9,
    "stopSequences": ["END", "STOP"]
  },
  "toolConfig": {
    "tools": [{"toolSpec": {"name": "scan_code", "description": "...", "inputSchema": {...}}}],
    "toolChoice": {"auto": {}}
  },
  "additionalModelRequestFields": {
    "thinking": {"type": "enabled", "budget_tokens": 5000}
  },
  "guardrailConfig": {
    "guardrailIdentifier": "my-guardrail-id",
    "guardrailVersion": "1",
    "trace": "enabled"
  },
  "requestMetadata": {
    "project": "my-project",
    "environment": "prod"
  }
}
```

---

### 6.2 GCP Vertex AI Gemini 完整範例

```python
import litellm

litellm.drop_params = True

response = litellm.completion(
    # Provider 路由：vertex_ai/ 前綴
    model="vertex_ai/gemini-2.5-pro-preview-05-06",

    messages=[
        {"role": "system", "content": "你是一個程式碼審查助理"},
        {"role": "user",   "content": "請審查這段 Python 程式碼"}
    ],

    # === 標準 generationConfig 參數（完整支援）===
    max_tokens=4096,           # → generationConfig.max_output_tokens
    temperature=0.3,           # → generationConfig.temperature
    top_p=0.95,                # → generationConfig.top_p
    stop=["```"],              # → generationConfig.stop_sequences
    seed=42,                   # → generationConfig.seed
    n=1,                       # → generationConfig.candidate_count
    logprobs=5,                # → generationConfig.logprobs

    # === Gemini 原生（非 OpenAI 標準）參數 ===
    top_k=40,                  # → generationConfig.top_k（Gemini 特有）

    # === Tool Calling ===
    tools=[{
        "type": "function",
        "function": {
            "name": "analyze_security",
            "description": "分析程式碼安全性",
            "parameters": {
                "type": "object",
                "properties": {"vulnerability": {"type": "string"}}
            }
        }
    }],
    tool_choice="auto",        # → toolConfig.functionCallingConfig.mode="AUTO"
                               # "required" → mode="ANY"
                               # "none" → mode="NONE"（Gemini 完整支援三種值）

    # === Response Format（結構化輸出）===
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CodeReview",
            "schema": {
                "type": "object",
                "properties": {
                    "issues": {"type": "array"},
                    "severity": {"type": "string"}
                }
            }
        }
    },  # → generationConfig.response_mime_type="application/json"
        #   + generationConfig.response_schema={...}

    # === Extended Thinking（Gemini 2.5-pro 支援）===
    thinking={"type": "enabled", "budget_tokens": 8000},
    # 或使用 OpenAI 相容的 reasoning_effort
    # reasoning_effort="high",  # → generationConfig.thinkingConfig

    # === Gemini 特有安全設定（非 OpenAI 標準）===
    safety_settings=[          # → 頂層 safetySettings
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ],

    # === Web Search（Google Search grounding）===
    web_search_options={},     # → tools.googleSearch={}
                               # ⚠️ user_location / search_context_size 會被忽略

    # === Vertex AI 標籤（僅 Vertex AI，非 AI Studio）===
    labels={"team": "backend", "cost_center": "eng"},

    # === 懲罰參數（非 preview 模型才支援）===
    frequency_penalty=0.2,     # → generationConfig.frequency_penalty
    presence_penalty=0.1,      # → generationConfig.presence_penalty

    # 傳遞 Vertex AI 認證資訊
    vertex_project="my-gcp-project",
    vertex_location="us-central1",

    # === 不支援的 OpenAI 參數（drop_params=True 時靜默丟棄）===
    # logit_bias={},           # ❌ 不支援 → 丟棄
    # user="user-123",         # ❌ 不支援 → 丟棄
    # function_call="auto",    # ❌ Legacy，不支援 → 丟棄
)

print(response.choices[0].message.content)
```

**最終送出給 Gemini generateContent API 的 Body 結構：**
```json
{
  "system_instruction": {"parts": [{"text": "你是一個程式碼審查助理"}]},
  "contents": [{"role": "user", "parts": [{"text": "請審查這段 Python 程式碼"}]}],
  "tools": [
    {
      "functionDeclarations": [{
        "name": "analyze_security",
        "description": "分析程式碼安全性",
        "parameters": {...}
      }]
    },
    {"googleSearch": {}}
  ],
  "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
  "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
  "generationConfig": {
    "max_output_tokens": 4096,
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 40,
    "stop_sequences": ["```"],
    "seed": 42,
    "candidate_count": 1,
    "logprobs": 5,
    "response_mime_type": "application/json",
    "response_schema": {...},
    "thinkingConfig": {"thinkingBudget": 8000},
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1
  },
  "labels": {"team": "backend", "cost_center": "eng"}
}
```

---

### 6.3 Azure OpenAI GPT 完整範例

```python
import litellm

response = litellm.completion(
    # Provider 路由：azure/ 前綴
    model="azure/gpt-4o",

    messages=[
        {"role": "system", "content": "你是一個程式碼助理"},
        {"role": "user",   "content": "請寫一個排序演算法"}
    ],

    # === 完整支援的 OpenAI 標準參數 ===
    max_tokens=2048,           # 直接傳遞
    temperature=0.7,
    top_p=0.9,
    stop=["END"],
    n=1,                       # ✅ Azure 支援（Bedrock 不支援）
    seed=42,                   # ✅ Azure 支援（Bedrock 不支援）
    logprobs=True,
    top_logprobs=5,            # ✅ Azure 支援（Bedrock/Gemini 不支援）
    logit_bias={"50256": -100},# ✅ Azure 支援（Bedrock/Gemini 不支援）
    user="user-123",           # ✅ Azure 支援（Bedrock/Gemini 不支援）
    frequency_penalty=0.5,     # ✅ Azure 完整支援（Bedrock 不支援；Gemini 部分支援）
    presence_penalty=0.3,

    # === Tool Calling ===
    tools=[{
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "寫程式碼",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "algorithm": {"type": "string"}
                }
            }
        }
    }],
    # ✅ tool_choice 支援所有值（需 API version >= 2023-12-01-preview）
    tool_choice="auto",

    # === Structured Output ===
    response_format={          # ✅ 需要 API version >= 2024-08 且非 GPT-3.5
        "type": "json_schema",
        "json_schema": {
            "name": "CodeOutput",
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "explanation": {"type": "string"}
                }
            }
        }
    },

    # === Azure 特有參數 ===
    prediction={               # Speculative Decoding（投機解碼）加速
        "type": "content",
        "content": "def sort_array(arr):"
    },

    # Azure 認證
    api_key="your-azure-api-key",
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-08-01-preview",  # ⚠️ API version 影響功能支援

    # === 不支援的參數（Azure GPT 無對應）===
    # top_k=40,                # ❌ Azure/OpenAI 本就沒有此參數
    # thinking={...},          # ❌ Azure GPT 不支援推理模式
    # guardrailConfig={...},   # ❌ Bedrock 特有，Azure 不支援
    # safety_settings=[...],   # ❌ Gemini 特有，Azure 不支援
)

print(response.choices[0].message.content)
```

**最終送出給 Azure OpenAI API 的 Body 結構（幾乎等同 OpenAI）：**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "你是一個程式碼助理"},
    {"role": "user",   "content": "請寫一個排序演算法"}
  ],
  "max_tokens": 2048,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": ["END"],
  "n": 1,
  "seed": 42,
  "logprobs": true,
  "top_logprobs": 5,
  "logit_bias": {"50256": -100},
  "user": "user-123",
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "tools": [{"type": "function", "function": {...}}],
  "tool_choice": "auto",
  "response_format": {"type": "json_schema", "json_schema": {...}},
  "prediction": {"type": "content", "content": "def sort_array(arr):"}
}
```

---

### 6.4 OpenAI-Compatible Request → litellm.completion() 參數對應

> **情境**：使用者透過 OpenAI-compatible 格式送 request 進來（例如用 OpenAI Python SDK 打自建 gateway），
> gateway 拆解 request body 後呼叫 `litellm.completion()`。
> 哪些參數可以直接從 request 傳入？哪些不行？

#### 架構流程

```
User (OpenAI SDK / raw HTTP)
        │
        │  POST /v1/chat/completions
        │  Body: { model, messages, temperature, ...extra_body fields }
        ▼
Gateway / LiteLLM Proxy
        │  _read_request_body() → raw dict（任何欄位都接受，不過濾）
        │  add_litellm_data_to_request() → 加入 metadata、auth 資訊
        │  route_request() → 去掉 proxy-internal 欄位
        ▼
litellm.acompletion(**data)   ← 整個 dict unpack 傳入
        │
        ▼
Provider-specific handler（Bedrock / Gemini / Azure ...）
```

**關鍵設計**：LiteLLM proxy 採用「透傳」設計——request body 的所有欄位都原封不動地變成 `**kwargs` 傳入 `litellm.acompletion()`，不做白名單過濾。

---

#### 類別一：標準 OpenAI Chat Completions 參數（可直接從 request 送）

這些欄位是 OpenAI Chat Completions API 的正式欄位，直接放在 request body 頂層即可：

| 欄位 | Bedrock | Gemini | Azure | 備註 |
|------|---------|--------|-------|------|
| `model` | ✅ | ✅ | ✅ | 加上 `bedrock/` / `vertex_ai/` / `azure/` 前綴 |
| `messages` | ✅ | ✅ | ✅ | |
| `max_tokens` | ✅ | ✅ | ✅ | |
| `temperature` | ✅ | ✅ | ✅ | |
| `top_p` | ✅ | ✅ | ✅ | |
| `stop` | ✅ | ✅ | ✅ | |
| `stream` | ✅ | ✅ | ✅ | |
| `tools` | ✅ | ✅ | ✅ | |
| `tool_choice` | ⚠️ `"none"` 不支援 | ✅ 全值支援 | ✅ | |
| `response_format` | ⚠️ 部分 Claude 支援 | ✅ | ✅ | |
| `n` | ❌ 丟棄 | ✅ | ✅ | |
| `seed` | ❌ 丟棄 | ✅ | ✅ | |
| `frequency_penalty` | ❌ 丟棄 | ⚠️ 非 preview 才支援 | ✅ | |
| `presence_penalty` | ❌ 丟棄 | ⚠️ 非 preview 才支援 | ✅ | |
| `logprobs` | ❌ 丟棄 | ✅ | ✅ | |
| `top_logprobs` | ❌ 丟棄 | ❌ | ✅ | |
| `logit_bias` | ❌ 丟棄 | ❌ | ✅ | |
| `user` | ❌ 丟棄 | ❌ | ✅ | |

---

#### 類別二：非標準參數（需要透過 `extra_body` 送）

這些欄位不是 OpenAI 官方 spec 的一部分，但因為 LiteLLM proxy 採透傳設計，
**只要透過 OpenAI SDK 的 `extra_body` 傳入，就會被合併到 request body 頂層，
最終以 `**kwargs` 的形式傳入 `litellm.acompletion()`**。

| 欄位 | 目標 Provider | 說明 |
|------|--------------|------|
| `thinking` | Bedrock Claude 3.7+/4、Gemini 2.5 | Extended Thinking；litellm 有 named param 直接對應 |
| `reasoning_effort` | Gemini 2.5 | OpenAI o-series 風格，litellm 有對應 |
| `guardrailConfig` | Bedrock | 內容過濾；作為 `**kwargs` 傳入 converse handler |
| `requestMetadata` | Bedrock | 請求追蹤 metadata |
| `safety_settings` | Gemini | Vertex AI 安全設定 |
| `top_k` | Gemini | generationConfig.top_k |
| `labels` | Vertex AI | GCP resource labels |
| `web_search_options` | Gemini、Bedrock | Google Search / Nova Search grounding |
| `cache_control_injection_points` | Bedrock | Prompt Caching 注入點（`AnthropicCacheControlHook` 處理） |
| `prediction` | Azure | Speculative Decoding |

**使用範例：OpenAI SDK 打自建 gateway**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://my-gateway/v1",
    api_key="my-gateway-key"
)

# Bedrock Claude 帶 extended thinking + guardrail + prompt caching
response = client.chat.completions.create(
    model="bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0",
    messages=[
        {"role": "system", "content": "你是一個安全助理"},
        {"role": "user",   "content": "分析以下程式碼"}
    ],
    max_tokens=2048,
    temperature=0.7,
    extra_body={
        # Provider-specific 參數透過 extra_body 傳入
        "thinking": {
            "type": "enabled",
            "budget_tokens": 5000
        },
        "guardrailConfig": {
            "guardrailIdentifier": "my-guardrail-id",
            "guardrailVersion": "1"
        },
        "cache_control_injection_points": [
            {"location": "message", "index": 0}   # 在第一條 message 注入 cache_control
        ]
    }
)
```

**OpenAI SDK 實際送出的 HTTP body（`extra_body` 會被 SDK 合併到頂層）：**
```json
{
  "model": "bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0",
  "messages": [...],
  "max_tokens": 2048,
  "temperature": 0.7,
  "thinking": {"type": "enabled", "budget_tokens": 5000},
  "guardrailConfig": {"guardrailIdentifier": "my-guardrail-id", "guardrailVersion": "1"},
  "cache_control_injection_points": [{"location": "message", "index": 0}]
}
```

**Gateway 收到後，等效的 litellm.completion() 呼叫：**
```python
litellm.completion(
    model="bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0",
    messages=[...],
    max_tokens=2048,
    temperature=0.7,
    thinking={"type": "enabled", "budget_tokens": 5000},
    guardrailConfig={"guardrailIdentifier": "my-guardrail-id", "guardrailVersion": "1"},
    cache_control_injection_points=[{"location": "message", "index": 0}]
)
```

---

#### 類別三：無法從 request 送入，只能在 server 端設定

這些參數涉及**認證憑證**或**全局行為設定**，放在 request body 中是不安全或無效的，
必須在 gateway server 端設定（環境變數、litellm config YAML、或程式碼初始化）：

| 參數 | 設定方式 | 說明 |
|------|---------|------|
| `aws_access_key_id` / `aws_secret_access_key` | 環境變數 `AWS_ACCESS_KEY_ID` 等 | Bedrock 認證，不應放在 request 中 |
| `vertex_project` / `vertex_location` | 環境變數或 litellm config | Vertex AI GCP 專案設定 |
| `api_key` / `api_base` / `api_version` | 環境變數或 litellm config | Azure 連線設定 |
| `drop_params` | `litellm.drop_params = True`（global） | 是否靜默丟棄不支援的參數 |
| `fallbacks` | Router config YAML | 跨 provider 的 fallback 邏輯 |
| `num_retries` | `litellm.num_retries = 3`（global）或 Router config | 全局重試次數 |
| `cache` | `litellm.cache = Cache(type="redis", ...)` | 快取後端設定 |
| `success_callback` / `failure_callback` | `litellm.success_callback = [...]`（global） | 觀測性 hook 設定 |

**Config YAML 範例（在 gateway 啟動時載入）：**
```yaml
model_list:
  - model_name: bedrock-claude         # 對使用者暴露的 model alias
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0
      aws_region_name: us-east-1
      # 認證由 IAM role 或 env var 提供

  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-2.5-pro-preview-05-06
      vertex_project: my-gcp-project
      vertex_location: us-central1

  - model_name: azure-gpt4
    litellm_params:
      model: azure/gpt-4o
      api_base: https://my-resource.openai.azure.com
      api_version: "2024-08-01-preview"
      api_key: os.environ/AZURE_API_KEY   # 從環境變數讀取

litellm_settings:
  drop_params: true          # 全局設定
  num_retries: 3
  fallbacks:
    - {"bedrock-claude": ["gemini-pro"]}  # Bedrock 失敗時 fallback 到 Gemini
```

---

#### 總結：三類參數對應表

```
┌────────────────────────────────┬──────────────────────────────┬────────────────────────────┐
│ 類別                           │ 怎麼送                       │ 範例                       │
├────────────────────────────────┼──────────────────────────────┼────────────────────────────┤
│ 標準 OpenAI params             │ request body 頂層欄位         │ temperature, max_tokens,   │
│                                │ （直接用 openai SDK 傳）      │ tools, response_format     │
├────────────────────────────────┼──────────────────────────────┼────────────────────────────┤
│ Provider-specific / LiteLLM    │ extra_body（OpenAI SDK）      │ thinking, guardrailConfig, │
│ 擴充參數                       │ 或 request body 額外欄位      │ safety_settings,           │
│                                │                              │ cache_control_injection_   │
│                                │                              │ points                     │
├────────────────────────────────┼──────────────────────────────┼────────────────────────────┤
│ 認證 / 全局設定                │ Server-side 環境變數          │ aws_access_key_id,         │
│ （不應放在 request 中）         │ 或 litellm config YAML        │ vertex_project, api_key,   │
│                                │ 或程式碼初始化                │ drop_params, fallbacks     │
└────────────────────────────────┴──────────────────────────────┴────────────────────────────┘
```

---

## 七、關鍵差異總結與選型建議

### 7.1 三大痛點

**痛點一：`n` 多次 completion**
- Azure ✅、Gemini ✅（`candidate_count`）、Bedrock ❌（直接丟棄）
- 在 Bedrock 上要多個輸出只能多次呼叫

**痛點二：`logit_bias` token 機率控制**
- 僅 Azure ✅，Bedrock ❌，Gemini ❌
- 依賴此功能的應用無法遷移至 Bedrock/Gemini

**痛點三：推理/思考模式（Thinking）**
- Bedrock Claude 3.7+/4 ✅（`thinking` param）
- Gemini 2.5-pro ✅（`thinkingConfig`）
- Azure GPT ❌（o-series 透過不同的 transformation class 處理）

### 7.2 使用 LiteLLM 時的注意事項

1. **設定 `drop_params=True`**：避免傳遞不支援的參數時整個呼叫失敗
   ```python
   litellm.drop_params = True  # 全局設定
   # 或
   litellm.completion(..., drop_params=True)  # 單次設定
   ```

2. **Bedrock 的 `tool_choice="none"` 陷阱**：必須用 `drop_params=True` 或避免傳遞

3. **Azure 的 API version 相依性**：同一個 endpoint 因 api_version 不同會有不同的功能支援，建議使用 `2024-08-01-preview` 或更新版本

4. **Gemini preview 模型的 penalty 參數**：`frequency_penalty`/`presence_penalty` 在 preview 模型上不支援，會在 `get_supported_openai_params` 時就被排除

5. **傳遞 Provider 特有參數**：可以直接在 `completion()` 中傳遞，未被識別的參數會進入 `additionalModelRequestFields`（Bedrock）或透過 `extra_body` 合併（Gemini）

6. **Bedrock Prompt Caching 的正確用法**：
   - Bedrock **不使用** Anthropic 的 `prompt-caching` beta header（會被過濾掉）
   - 要啟用 Bedrock prompt caching，使用 `cache_control` 在 message 中標記，或使用 `cache_control_injection_points` 參數
   - Claude 4.5/4.6 系列支援 TTL（`"5m"` 或 `"1h"`）；其他 Claude 模型只支援預設 TTL
   ```python
   # 方式一：在 messages 中直接標記
   messages = [
       {"role": "system", "content": "很長的系統提示...", "cache_control": {"type": "ephemeral"}},
       {"role": "user",   "content": "問題"}
   ]

   # 方式二：透過 cache_control_injection_points 自動注入
   litellm.completion(
       model="bedrock/anthropic.claude-sonnet-4-5-20251001-v1:0",
       messages=messages,
       cache_control_injection_points=[
           {"location": "message", "index": 0},                        # 在第一條 message 注入
           {"location": "tool_config"},                                  # 在 tools 末尾注入
       ],
   )
   ```

### 7.3 跨 Provider 遷移備忘

```
從 Azure 遷移到 Bedrock：
  移除：n, logit_bias, user, logprobs, top_logprobs, seed, frequency_penalty, presence_penalty
  注意：tool_choice="none" 需改為 drop_params=True

從 Azure 遷移到 Gemini：
  移除：logit_bias, user, function_call（legacy）
  注意：top_logprobs 可能不可靠；web_search_options 的細節參數失效

從 Bedrock 遷移到 Gemini：
  幾乎無損，兩者支援集合相似
  注意：web_search_options 在兩者都有限制（Bedrock 僅 Nova，Gemini 忽略細節）
  注意：guardrailConfig（Bedrock 特有）需改用 Gemini 的 safety_settings
```
