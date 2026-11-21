from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from engine import GenerationStep, NaiveEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

_DONE = object()

engine: NaiveEngine | None = None
gpu_executor: ThreadPoolExecutor | None = None
engine_lock = asyncio.Lock()
default_temperature: float = 0.0


def load_server_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    bench_cfg = cfg.get("benchmarks", {})

    model_name = os.getenv(
        "CUSTOM_MODEL",
        model_cfg.get("name", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    )
    dtype = os.getenv(
        "CUSTOM_DTYPE",
        model_cfg.get("dtype", "bfloat16"),
    )
    max_model_len = int(
        os.getenv(
            "CUSTOM_MAX_MODEL_LEN",
            model_cfg.get("max_model_len", 9216),
        )
    )
    temperature = float(
        os.getenv(
            "CUSTOM_TEMPERATURE",
            bench_cfg.get("temperature", 0.0),
        )
    )
    device = os.getenv("CUSTOM_DEVICE", "cuda")

    return {
        "model_name": model_name,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "default_temperature": temperature,
        "device": device,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine, gpu_executor, default_temperature
    cfg = load_server_config()
    default_temperature = cfg["default_temperature"]
    gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage3-gpu")
    loop = asyncio.get_running_loop()

    def _init_and_warmup() -> NaiveEngine:
        eng = NaiveEngine(
            model_name=cfg["model_name"],
            dtype=cfg["dtype"],
            device=cfg["device"],
            max_model_len=cfg["max_model_len"],
        )
        eng.warmup()
        return eng

    engine = await loop.run_in_executor(gpu_executor, _init_and_warmup)
    try:
        yield
    finally:
        engine = None
        gpu_executor.shutdown(wait=True)
        gpu_executor = None


app = FastAPI(
    title="Custom Serving - Stage 3: Parallel Decode (Split-K)",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    ignore_eos: bool = False


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    model_id = (
        engine.model_name
        if engine is not None
        else load_server_config()["model_name"]
    )
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom-serving",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest) -> Any:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is not initialized.")

    if req.model != engine.model_name:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Loaded model is '{engine.model_name}'.",
        )

    effective_max_tokens = (
        req.max_completion_tokens
        if req.max_completion_tokens is not None
        else (req.max_tokens if req.max_tokens is not None else 256)
    )
    effective_temp = (
        req.temperature if req.temperature is not None else default_temperature
    )
    include_usage = bool(
        req.stream_options and req.stream_options.get("include_usage")
    )

    active_engine = engine
    active_executor = gpu_executor

    if req.stream:

        async def event_generator() -> AsyncIterator[str]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            loop = asyncio.get_running_loop()

            async with engine_lock:
                step_iter = await loop.run_in_executor(
                    active_executor,
                    active_engine.generate_stream,
                    req.messages,
                    effective_max_tokens,
                    effective_temp,
                    req.ignore_eos,
                )

                first_chunk = True
                prompt_tokens = 0
                completion_tokens = 0

                try:
                    while True:
                        step = await loop.run_in_executor(
                            active_executor, next, step_iter, _DONE
                        )
                        if step is _DONE:
                            break

                        assert isinstance(step, GenerationStep)
                        prompt_tokens = step.prompt_tokens
                        completion_tokens = step.completion_tokens

                        if first_chunk:
                            delta: dict[str, Any] = {
                                "role": "assistant",
                                "content": step.text_delta,
                            }
                            first_chunk = False
                        elif step.token_id is None and step.text_delta == "":
                            delta = {}
                        else:
                            delta = {"content": step.text_delta}

                        chunk_payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": active_engine.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "logprobs": None,
                                    "finish_reason": step.finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"

                    if include_usage:
                        usage_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": active_engine.model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"

                    yield "data: [DONE]\n\n"
                finally:
                    close_fn = getattr(step_iter, "close", None)
                    if callable(close_fn):
                        try:
                            await loop.run_in_executor(active_executor, close_fn)
                        except RuntimeError:
                            close_fn()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _generate_full() -> tuple[str, str | None, int, int]:
        pieces: list[str] = []
        finish_reason: str | None = "length"
        prompt_tokens = 0
        completion_tokens = 0
        step_iter = active_engine.generate_stream(
            req.messages,
            effective_max_tokens,
            effective_temp,
            req.ignore_eos,
        )
        try:
            for step in step_iter:
                if step.text_delta:
                    pieces.append(step.text_delta)
                if step.finish_reason is not None:
                    finish_reason = step.finish_reason
                prompt_tokens = step.prompt_tokens
                completion_tokens = step.completion_tokens
        finally:
            close_fn = getattr(step_iter, "close", None)
            if callable(close_fn):
                close_fn()
        return "".join(pieces), finish_reason, prompt_tokens, completion_tokens

    loop = asyncio.get_running_loop()
    async with engine_lock:
        full_text, finish_reason, prompt_tokens, completion_tokens = (
            await loop.run_in_executor(active_executor, _generate_full)
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": active_engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
