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

from engine import PagedEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

engine: PagedEngine | None = None
gpu_executor: ThreadPoolExecutor | None = None
scheduler_task: asyncio.Task[None] | None = None
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
    num_kv_blocks = int(os.getenv("CUSTOM_NUM_KV_BLOCKS", "4096"))
    max_batch_size = int(os.getenv("CUSTOM_MAX_BATCH_SIZE", "64"))

    return {
        "model_name": model_name,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "default_temperature": temperature,
        "device": device,
        "num_kv_blocks": num_kv_blocks,
        "max_batch_size": max_batch_size,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine, gpu_executor, scheduler_task, default_temperature
    cfg = load_server_config()
    default_temperature = cfg["default_temperature"]
    gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage5-gpu")
    loop = asyncio.get_running_loop()

    def _init_and_warmup() -> PagedEngine:
        eng = PagedEngine(
            model_name=cfg["model_name"],
            dtype=cfg["dtype"],
            device=cfg["device"],
            max_model_len=cfg["max_model_len"],
            num_blocks=cfg["num_kv_blocks"],
            max_batch_size=cfg["max_batch_size"],
        )
        eng.warmup()
        return eng

    engine = await loop.run_in_executor(gpu_executor, _init_and_warmup)
    scheduler_task = asyncio.create_task(engine.run_scheduler_loop(gpu_executor))
    try:
        yield
    finally:
        if engine is not None:
            engine.stop()
        if scheduler_task is not None:
            scheduler_task.cancel()
            await asyncio.gather(scheduler_task, return_exceptions=True)
            scheduler_task = None
        engine = None
        if gpu_executor is not None:
            gpu_executor.shutdown(wait=True)
            gpu_executor = None


app = FastAPI(
    title="Custom Serving - Stage 5: Operator Fusion",
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

    if req.stream:

        async def event_generator() -> AsyncIterator[str]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            first_chunk = True
            prompt_tokens = 0
            completion_tokens = 0

            async for step in active_engine.generate_stream_async(
                req.messages,
                effective_max_tokens,
                effective_temp,
                req.ignore_eos,
            ):
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

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    pieces: list[str] = []
    finish_reason: str | None = "length"
    prompt_tokens = 0
    completion_tokens = 0
    async for step in active_engine.generate_stream_async(
        req.messages,
        effective_max_tokens,
        effective_temp,
        req.ignore_eos,
    ):
        if step.text_delta:
            pieces.append(step.text_delta)
        if step.finish_reason is not None:
            finish_reason = step.finish_reason
        prompt_tokens = step.prompt_tokens
        completion_tokens = step.completion_tokens

    full_text = "".join(pieces)

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
