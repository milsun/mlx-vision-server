#!/usr/bin/env python3
"""MLX-VLM Vision Server — Gemma 4 E4B with OpenAI-compatible /v1/chat/completions.

Loads the model lazily on first request.
Supports: text, single/multi image, image URLs, base64 data URIs.
"""

import argparse
import os
import sys

# expose the mlx-vlm server via uvicorn
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="MLX-VLM Vision Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    parser.add_argument("--model", default=os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "models", "gemma4-e4b-4bit"),
    ))
    args = parser.parse_args()

    os.environ["MLX_VLM_MODEL_PATH"] = args.model

    print(f"Starting MLX-VLM server on {args.host}:{args.port}")
    print(f"Default model (if not specified in request): {args.model}")
    print(f"Endpoints:")
    print(f"  POST /v1/chat/completions  (OpenAI-compatible)")
    print(f"  POST /generate              (native)")
    print(f"  POST /chat                  (native chat)")
    print(f"  GET  /health")
    print(f"  POST /unload")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=1,
        timeout_keep_alive=300,
    )


# ── Build the FastAPI app ──────────────────────────────────────────
import base64
import gc
import io
import json
import traceback
import uuid
from datetime import datetime
from typing import List, Optional, Union

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlx_vlm.generate import DEFAULT_MAX_TOKENS, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

app = FastAPI(title="MLX-VLM Vision Server", version="0.1.0")

_model_cache: dict = {}
DEFAULT_MODEL = os.environ.get("MLX_VLM_MODEL_PATH", "")

# ── Helpers ────────────────────────────────────────────────────────

def _load_model(model_path: str):
    global _model_cache
    cache_key = (model_path, None)
    if _model_cache.get("cache_key") == cache_key:
        return _model_cache["model"], _model_cache["processor"], _model_cache["config"]
    if _model_cache:
        _unload()
    print(f"Loading model from: {model_path}")
    model, processor = load(model_path, adapter_path=None, trust_remote_code=True)
    config = model.config
    _model_cache = {"cache_key": cache_key, "model": model, "processor": processor, "config": config}
    print("Model loaded.")
    return model, processor, config


def _unload():
    global _model_cache
    if _model_cache:
        print(f"Unloading model: {_model_cache.get('model_path')}")
    _model_cache.clear()
    gc.collect()
    mx.clear_cache()


def _decode_image(raw: bytes) -> str:
    """Convert raw image bytes to a base64 data URI that MLX-VLM can load."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        fmt = img.format or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


# ── OpenAI-compatible schemas ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="system, user, assistant")
    content: Union[str, list] = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model_config = {"populate_by_name": True, "extra": "allow"}

    model: str = ""
    messages: List[ChatMessage]
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, alias="max_completion_tokens")
    temperature: float = 0.5
    top_p: float = 1.0
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# ── v1/chat/completions ────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    req = ChatCompletionRequest(**body)

    model_path = req.model or DEFAULT_MODEL
    if not model_path:
        raise HTTPException(status_code=400, detail="No model specified")

    model, processor, config = _load_model(model_path)

    # parse messages → text + images
    chat_messages: list[dict] = []
    images: list[str] = []

    for msg in req.messages:
        role = msg.role
        if isinstance(msg.content, str):
            chat_messages.append({"role": role, "content": msg.content})
        elif isinstance(msg.content, list):
            # OpenAI multimodal content block array
            text_parts: list[str] = []
            for part in msg.content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("text", "input_text"):
                        text_parts.append(part.get("text", ""))
                    elif ptype in ("image_url", "input_image"):
                        url = part.get("image_url", "")
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        # strip file:// prefix
                        if url.startswith("file://"):
                            url = url[7:]
                        # handle base64 data URIs directly
                        if url.startswith("data:"):
                            images.append(url)
                        elif url.startswith("http://") or url.startswith("https://"):
                            images.append(url)
                        elif os.path.isfile(url):
                            with open(url, "rb") as f:
                                data_uri = _decode_image(f.read())
                            if data_uri:
                                images.append(data_uri)
                        else:
                            images.append(url)
            chat_messages.append({"role": role, "content": "\n".join(text_parts)})

    if not chat_messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = apply_chat_template(processor, config, chat_messages, num_images=len(images),
                                  enable_thinking=False)

    generated_at = int(datetime.now().timestamp())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        async def stream_gen():
            prompt_tps = gen_tps = peak_mem = 0
            prompt_tokens = gen_tokens = 0
            try:
                for chunk in stream_generate(
                    model=model, processor=processor, prompt=prompt,
                    image=images if images else None,
                    temperature=req.temperature, max_tokens=req.max_tokens,
                    top_p=req.top_p,
                ):
                    if chunk and hasattr(chunk, "text"):
                        delta = {"role": "assistant", "content": chunk.text}
                        data = {
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": generated_at, "model": model_path,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        prompt_tps = getattr(chunk, "prompt_tps", 0)
                        gen_tps = getattr(chunk, "generation_tps", 0)
                        peak_mem = getattr(chunk, "peak_memory", 0)
                        prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                        gen_tokens = getattr(chunk, "generation_tokens", 0)
                print(f"[{response_id}] prompt={prompt_tokens}t @ {prompt_tps:.1f} t/s, "
                      f"gen={gen_tokens}t @ {gen_tps:.1f} t/s, "
                      f"peak_mem={peak_mem:.2f} GB")
                yield "data: [DONE]\n\n"
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                gc.collect()
                mx.clear_cache()

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # non-streaming
    result = generate(
        model=model, processor=processor, prompt=prompt,
        image=images if images else None,
        temperature=req.temperature, max_tokens=req.max_tokens,
        top_p=req.top_p, verbose=False,
    )
    prompt_tps = getattr(result, "prompt_tps", 0)
    gen_tps = getattr(result, "generation_tps", 0)
    peak_mem = getattr(result, "peak_memory", 0)
    print(f"[{response_id}] prompt={getattr(result, 'prompt_tokens', 0)}t @ {prompt_tps:.1f} t/s, "
          f"gen={getattr(result, 'generation_tokens', 0)}t @ {gen_tps:.1f} t/s, "
          f"peak_mem={peak_mem:.2f} GB")
    gc.collect()
    mx.clear_cache()

    usage = ChatCompletionUsage(
        prompt_tokens=getattr(result, "prompt_tokens", 0),
        completion_tokens=getattr(result, "generation_tokens", 0),
        total_tokens=getattr(result, "total_tokens", 0),
    )

    return ChatCompletionResponse(
        id=response_id, created=generated_at, model=model_path,
        choices=[ChatCompletionChoice(
            message={"role": "assistant", "content": result.text},
        )],
        usage=usage,
    )


# ── Health ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "loaded_model": _model_cache.get("model_path", None),
    }


@app.post("/unload")
async def unload():
    was = _model_cache.get("model_path", None)
    _unload()
    return {"status": "unloaded", "was": was}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model"}] if _model_cache else [],
    }


if __name__ == "__main__":
    main()
