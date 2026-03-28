import litellm
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional

from db import init_db, reload_cache, get_instance_config
from models import ChatCompletionRequest

litellm.drop_params = True

app = FastAPI(title="LiteLLM Multi-Provider Proxy")

_OPTIONAL_PARAMS = [
    "temperature", "max_tokens", "top_p", "stop",
    "tools", "tool_choice", "stream_options",
    "n", "presence_penalty", "frequency_penalty", "user",
]


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/reload")
async def admin_reload():
    await reload_cache()
    return {"status": "cache reloaded"}


def _build_litellm_kwargs(cfg: dict, body: ChatCompletionRequest) -> dict:
    provider = cfg["provider"]
    kwargs: dict = {
        "messages": [m.model_dump(exclude_none=True) for m in body.messages],
        "stream": body.stream or False,
    }

    match provider:
        case "bedrock":
            kwargs.update({
                "model":           f"bedrock/{cfg['model_id']}",
                "api_base":        cfg["bedrock_base_url"],
                "api_key":         cfg["bedrock_api_key"],
                "aws_region_name": cfg["aws_region_name"],
            })
        case "gemini":
            kwargs.update({
                "model":   f"gemini/{cfg['model_id']}",
                "api_key": cfg["gemini_api_key"],
            })
            if cfg.get("gemini_api_base"):
                kwargs["api_base"] = cfg["gemini_api_base"]
        case "vertex_ai":
            kwargs.update({
                "model":              f"vertex_ai/{cfg['model_id']}",
                "vertex_project":     cfg["vertex_project"],
                "vertex_location":    cfg["vertex_location"],
                "vertex_credentials": cfg["vertex_credentials"],
            })
            if cfg.get("vertex_api_base"):
                kwargs["api_base"] = cfg["vertex_api_base"]
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    for param in _OPTIONAL_PARAMS:
        val = getattr(body, param, None)
        if val is not None:
            kwargs[param] = val

    return kwargs


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    x_litellm_instance: Optional[str] = Header(None, alias="X-LiteLLM-Instance"),
):
    if not x_litellm_instance:
        raise HTTPException(status_code=400, detail="Missing X-LiteLLM-Instance header")

    cfg = await get_instance_config(x_litellm_instance)
    litellm_kwargs = _build_litellm_kwargs(cfg, body)

    try:
        response = await litellm.acompletion(**litellm_kwargs)
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except litellm.APIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if body.stream:
        return StreamingResponse(
            _stream_generator(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(response.model_dump())


async def _stream_generator(litellm_stream):
    async for chunk in litellm_stream:
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
