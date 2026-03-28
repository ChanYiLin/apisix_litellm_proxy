from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "claude-sonnet-4-5"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream_options: Optional[Dict[str, Any]] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="allow")
