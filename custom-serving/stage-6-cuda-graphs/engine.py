"""Stage 6 generation engine with Bucketed CUDA Graphs for Decode, Fused Triton Ops, Paged KV Cache (BlockAllocator), and Iteration-Level Continuous Batching."""

from __future__ import annotations

import asyncio
import collections
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from model import (
    KV_BLOCK_SIZE,
    DecodeGraphRunner,
    PagedKVCache,
    Qwen2Model,
    resolve_model_dir,
)
from transformers import AutoTokenizer

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class GenerationStep:
    """Single incremental emission from ``PagedEngine.generate_stream_async``."""

    token_id: int | None
    text_delta: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def _extract_eos_ids(value: Any) -> set[int]:
    if isinstance(value, int):
        return {value}
    if isinstance(value, list):
        return {int(v) for v in value if isinstance(v, int)}
    return set()


def _resolve_eos_token_ids(model_dir: Path, tokenizer: Any) -> set[int]:
    eos_ids: set[int] = set()
    for filename in ("generation_config.json", "config.json"):
        cfg_path = model_dir / filename
        if cfg_path.is_file():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            eos_ids.update(_extract_eos_ids(data.get("eos_token_id")))
    eos_ids.update(_extract_eos_ids(getattr(tokenizer, "eos_token_id", None)))
    return eos_ids


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


class BlockAllocator:
    """Manages fixed-size physical KV cache blocks across all transformer layers."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        if num_layers <= 0 or num_blocks <= 0 or block_size <= 0:
            raise ValueError(
                f"num_layers ({num_layers}), num_blocks ({num_blocks}), and "
                f"block_size ({block_size}) must all be positive."
            )
        self.num_layers = int(num_layers)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = _resolve_dtype(dtype)
        self.device = torch.device(device)

        self.kv_cache: PagedKVCache = [
            torch.empty(
                (2, self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        self.free_blocks: collections.deque[int] = collections.deque(range(self.num_blocks))
        self.allocated: set[int] = set()

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        num_blocks: int = 4096,
        block_size: int = KV_BLOCK_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> "BlockAllocator":
        return cls(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=_resolve_dtype(dtype),
            device=device,
        )

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def blocks_needed_for(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_needed: int) -> bool:
        return num_needed <= len(self.free_blocks)

    def allocate(self, num_needed: int) -> list[int]:
        if num_needed < 0:
            raise ValueError(f"num_needed must be non-negative, got {num_needed}")
        if num_needed == 0:
            return []
        if num_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Out of KV cache blocks: requested {num_needed}, "
                f"available {len(self.free_blocks)} / {self.num_blocks}"
            )
        block_ids: list[int] = []
        for _ in range(num_needed):
            bid = self.free_blocks.popleft()
            self.allocated.add(bid)
            block_ids.append(bid)
        return block_ids

    def free(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        seen: set[int] = set()
        for bid in block_ids:
            if bid < 0 or bid >= self.num_blocks:
                raise ValueError(f"Invalid block ID {bid}; expected 0 <= bid < {self.num_blocks}")
            if bid in seen or bid not in self.allocated:
                raise ValueError(f"Double-free or unallocated block ID {bid}")
            seen.add(bid)
        for bid in block_ids:
            self.allocated.remove(bid)
            self.free_blocks.append(bid)


@dataclass
class SequenceRequest:
    """Per-request state tracked across prefill, continuous batching decode, and preemption."""

    request_id: str
    prompt_ids: list[int]
    pending_prefill_ids: list[int]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    output_queue: asyncio.Queue[GenerationStep | Exception | None]
    block_ids: list[int] = field(default_factory=list)
    context_len: int = 0
    next_token_id: int | None = None
    generated_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    cancelled: bool = False
    finished: bool = False


class PagedEngine:
    """Continuous batching generation engine backed by Bucketed CUDA Graphs for Decode, Fused Triton Ops, Paged KV Cache, and Triton Split-K Decode (Stage 6)."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_model_len: int = 9216,
        num_blocks: int = 4096,
        block_size: int = KV_BLOCK_SIZE,
        max_batch_size: int = 64,
        max_prefill_tokens: int = 4096,
        allocator: BlockAllocator | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)
        self.max_model_len = int(max_model_len)
        self.max_batch_size = int(max_batch_size)
        self.max_prefill_tokens = int(max_prefill_tokens)

        model_dir = resolve_model_dir(model_name)
        self.model = Qwen2Model.from_pretrained(str(model_dir), dtype=self.dtype, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.eos_token_ids = _resolve_eos_token_ids(model_dir, self.tokenizer)

        if allocator is not None:
            self.allocator = allocator
        else:
            self.allocator = BlockAllocator.from_model_config(
                self.model.config,
                num_blocks=num_blocks,
                block_size=block_size,
                dtype=self.dtype,
                device=self.device,
            )
        self.num_blocks = self.allocator.num_blocks
        self.block_size = self.allocator.block_size
        self.graph_runner = DecodeGraphRunner(
            self.model,
            self.allocator.kv_cache,
            max_model_len=self.max_model_len,
            max_batch_size=self.max_batch_size,
            block_size=self.block_size,
            device=self.device,
        )

        self._request_queue: asyncio.Queue[SequenceRequest | None] | None = None
        self._queue_loop: asyncio.AbstractEventLoop | None = None
        self._stop_requested: bool = False

    def _get_request_queue(self) -> asyncio.Queue[SequenceRequest | None]:
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._request_queue is None or (loop is not None and self._queue_loop is not loop):
            self._request_queue = asyncio.Queue()
            self._queue_loop = loop
        return self._request_queue

    @property
    def request_queue(self) -> asyncio.Queue[SequenceRequest | None]:
        return self._get_request_queue()

    def warmup(self) -> None:
        """Compile all Triton prefill/decode specializations, capture CUDA decode graphs, and reclaim temporary blocks."""
        with torch.inference_mode():
            # (a) L=64 prefill + 1-token decode
            blocks_a: list[int] = []
            try:
                blocks_a = self.allocator.allocate(self.allocator.blocks_needed_for(65))
                bt_a = torch.tensor([blocks_a], dtype=torch.int32, device=self.device)
                dummy_64 = torch.arange(1, 65, dtype=torch.long, device=self.device).unsqueeze(0)
                ctx_64 = torch.tensor([64], dtype=torch.int32, device=self.device)
                _ = self.model(
                    dummy_64,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_a,
                    context_lengths=ctx_64,
                    last_token_only=True,
                    max_context_len=64,
                )
                next_a = torch.tensor([[65]], dtype=torch.long, device=self.device)
                ctx_65 = torch.tensor([65], dtype=torch.int32, device=self.device)
                _ = self.model(
                    next_a,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_a,
                    context_lengths=ctx_65,
                    last_token_only=True,
                    max_context_len=65,
                )
            finally:
                if blocks_a:
                    self.allocator.free(blocks_a)

            # (b) L=192 prefill + 1-token decode (ensures num_splits > 1 compiles Split-K + combine kernels)
            blocks_b: list[int] = []
            try:
                blocks_b = self.allocator.allocate(self.allocator.blocks_needed_for(193))
                bt_b = torch.tensor([blocks_b], dtype=torch.int32, device=self.device)
                dummy_192 = torch.arange(1, 193, dtype=torch.long, device=self.device).unsqueeze(0)
                ctx_192 = torch.tensor([192], dtype=torch.int32, device=self.device)
                _ = self.model(
                    dummy_192,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_b,
                    context_lengths=ctx_192,
                    last_token_only=True,
                    max_context_len=192,
                )
                next_b = torch.tensor([[193]], dtype=torch.long, device=self.device)
                ctx_193 = torch.tensor([193], dtype=torch.int32, device=self.device)
                _ = self.model(
                    next_b,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_b,
                    context_lengths=ctx_193,
                    last_token_only=True,
                    max_context_len=193,
                )
            finally:
                if blocks_b:
                    self.allocator.free(blocks_b)

            # (c) L=8 prefill (BLOCK_M=16) + B=2 batched decode (B>1 strides)
            blocks_c0: list[int] = []
            blocks_c1: list[int] = []
            try:
                need_c = self.allocator.blocks_needed_for(9)
                blocks_c0 = self.allocator.allocate(need_c)
                blocks_c1 = self.allocator.allocate(need_c)
                bt_c = torch.tensor([blocks_c0, blocks_c1], dtype=torch.int32, device=self.device)
                dummy_8 = (
                    torch.arange(1, 9, dtype=torch.long, device=self.device)
                    .unsqueeze(0)
                    .expand(2, -1)
                    .contiguous()
                )
                ctx_8 = torch.tensor([8, 8], dtype=torch.int32, device=self.device)
                _ = self.model(
                    dummy_8,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_c,
                    context_lengths=ctx_8,
                    last_token_only=True,
                    max_context_len=8,
                )
                next_c = torch.tensor([[9], [9]], dtype=torch.long, device=self.device)
                ctx_9 = torch.tensor([9, 9], dtype=torch.int32, device=self.device)
                _ = self.model(
                    next_c,
                    kv_cache=self.allocator.kv_cache,
                    block_table=bt_c,
                    context_lengths=ctx_9,
                    last_token_only=True,
                    max_context_len=9,
                )
            finally:
                if blocks_c1:
                    self.allocator.free(blocks_c1)
                if blocks_c0:
                    self.allocator.free(blocks_c0)

            # (d) Eager B=16 decode warmup (compiles SINGLE_SPLIT=True)
            warm_b = min(16, self.max_batch_size)
            if warm_b >= 16 and self.allocator.can_allocate(warm_b):
                blocks_d: list[list[int]] = []
                try:
                    for _ in range(warm_b):
                        blocks_d.append(self.allocator.allocate(1))
                    bt_d = torch.tensor(blocks_d, dtype=torch.int32, device=self.device)
                    next_d = torch.ones((warm_b, 1), dtype=torch.long, device=self.device)
                    ctx_d = torch.ones((warm_b,), dtype=torch.int32, device=self.device)
                    _ = self.model(
                        next_d,
                        kv_cache=self.allocator.kv_cache,
                        block_table=bt_d,
                        context_lengths=ctx_d,
                        last_token_only=True,
                        max_context_len=self.max_model_len,
                    )
                finally:
                    for blk_list in reversed(blocks_d):
                        self.allocator.free(blk_list)

            # (e) Capture all decode CUDA graph buckets using a temporary physical block
            temp_blocks: list[int] = []
            try:
                if self.allocator.can_allocate(1):
                    temp_blocks = self.allocator.allocate(1)
                self.graph_runner.capture_all(temp_block_ids=temp_blocks or None)
            finally:
                if temp_blocks:
                    self.allocator.free(temp_blocks)

        assert self.allocator.num_free_blocks == self.allocator.num_blocks, (
            f"Warmup leaked KV blocks: {self.allocator.num_free_blocks} free != {self.allocator.num_blocks} total"
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def format_prompt(self, messages: list[dict[str, Any]]) -> list[int]:
        """Normalize chat messages and tokenize with the model's chat template."""
        norm_messages = [
            {"role": msg["role"], "content": _normalize_content(msg.get("content"))}
            for msg in messages
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            norm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def _free_seq_blocks(self, seq: SequenceRequest) -> None:
        if seq.block_ids:
            blocks = list(seq.block_ids)
            seq.block_ids.clear()
            self.allocator.free(blocks)

    def _build_block_table(self, seqs: list[SequenceRequest]) -> torch.Tensor:
        max_blocks = max((len(s.block_ids) for s in seqs), default=1)
        max_blocks = max(max_blocks, 1)
        rows = [
            s.block_ids + [0] * (max_blocks - len(s.block_ids))
            for s in seqs
        ]
        return torch.tensor(rows, dtype=torch.int32, device=self.device)

    def _sample_batch_tokens(
        self,
        logits: torch.Tensor,
        seqs: list[SequenceRequest],
    ) -> list[int]:
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        if all(s.temperature <= 0.0 for s in seqs):
            next_tokens = torch.argmax(logits, dim=-1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)
            for idx, seq in enumerate(seqs):
                if seq.temperature > 0.0:
                    probs = F.softmax(logits[idx].float() / seq.temperature, dim=-1)
                    next_tokens[idx] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # Single D2H transfer for the entire batch (no per-request .item() syncs)
        return [int(t) for t in next_tokens.tolist()]

    def _run_prefill_batch(self, batch: list[SequenceRequest]) -> list[int]:
        prefill_len = len(batch[0].pending_prefill_ids)
        input_ids = torch.tensor(
            [s.pending_prefill_ids for s in batch],
            dtype=torch.long,
            device=self.device,
        )
        block_table = self._build_block_table(batch)
        context_lengths = torch.full(
            (len(batch),),
            prefill_len,
            dtype=torch.int32,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.nvtx.range_push("prefill_step")
        try:
            with torch.inference_mode():
                logits = self.model(
                    input_ids,
                    kv_cache=self.allocator.kv_cache,
                    block_table=block_table,
                    context_lengths=context_lengths,
                    last_token_only=True,
                    max_context_len=prefill_len,
                )
                return self._sample_batch_tokens(logits, batch)
        finally:
            if self.device.type == "cuda":
                torch.cuda.nvtx.range_pop()

    def _run_decode_batch(self, batch: list[SequenceRequest]) -> list[int]:
        input_ids_list: list[int] = []
        for s in batch:
            if s.next_token_id is None:
                raise RuntimeError(f"Sequence {s.request_id} missing next_token_id in decode step")
            input_ids_list.append(s.next_token_id)
        block_table_rows = [s.block_ids for s in batch]
        ctx_list = [s.context_len for s in batch]
        if self.device.type == "cuda":
            torch.cuda.nvtx.range_push("decode_step")
        try:
            with torch.inference_mode():
                logits = self.graph_runner.replay(
                    actual_batch_size=len(batch),
                    input_ids_list=input_ids_list,
                    block_table_rows=block_table_rows,
                    ctx_lens_list=ctx_list,
                )
                return self._sample_batch_tokens(logits, batch)
        finally:
            if self.device.type == "cuda":
                torch.cuda.nvtx.range_pop()

    def _process_sampled_token(self, seq: SequenceRequest, tok: int) -> None:
        prompt_len = len(seq.prompt_ids)
        if not seq.ignore_eos and tok in self.eos_token_ids:
            full_text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            flushed = full_text[len(seq.emitted_text) :]
            seq.emitted_text = full_text
            seq.finished = True
            seq.output_queue.put_nowait(
                GenerationStep(
                    token_id=None,
                    text_delta=flushed,
                    finish_reason="stop",
                    prompt_tokens=prompt_len,
                    completion_tokens=len(seq.generated_ids),
                )
            )
            seq.output_queue.put_nowait(None)
            return

        seq.generated_ids.append(tok)
        full_text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
        total_seq_len = prompt_len + len(seq.generated_ids)
        is_last_allowed = (len(seq.generated_ids) >= seq.max_tokens) or (
            total_seq_len >= self.max_model_len
        )

        if is_last_allowed:
            delta = full_text[len(seq.emitted_text) :]
            seq.emitted_text = full_text
            seq.finished = True
            seq.output_queue.put_nowait(
                GenerationStep(
                    token_id=tok,
                    text_delta=delta,
                    finish_reason="length",
                    prompt_tokens=prompt_len,
                    completion_tokens=len(seq.generated_ids),
                )
            )
            seq.output_queue.put_nowait(None)
            return

        candidate = full_text.rstrip("\ufffd") if full_text.endswith("\ufffd") else full_text
        if len(candidate) >= len(seq.emitted_text):
            delta = candidate[len(seq.emitted_text) :]
            seq.emitted_text = candidate
        else:
            delta = ""

        seq.next_token_id = tok
        seq.output_queue.put_nowait(
            GenerationStep(
                token_id=tok,
                text_delta=delta,
                finish_reason=None,
                prompt_tokens=prompt_len,
                completion_tokens=len(seq.generated_ids),
            )
        )

    def stop(self) -> None:
        """Signal the background scheduler loop to exit cleanly."""
        self._stop_requested = True
        if (
            self._request_queue is not None
            and self._queue_loop is not None
            and not self._queue_loop.is_closed()
        ):
            try:
                self._request_queue.put_nowait(None)
            except RuntimeError:
                pass

    async def run_scheduler_loop(
        self,
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Iteration-level continuous batching loop interleaving equal-length prefill and batched decode steps."""
        self._stop_requested = False
        waiting: collections.deque[SequenceRequest] = collections.deque()
        running: list[SequenceRequest] = []
        loop = asyncio.get_running_loop()
        queue = self.request_queue

        async def _run_gpu(fn: Any, *args: Any) -> Any:
            if gpu_executor is not None:
                return await loop.run_in_executor(gpu_executor, fn, *args)
            return fn(*args)

        try:
            while not self._stop_requested:
                # Reclaim any cancelled or finished sequences before checking idle state
                if waiting:
                    kept_waiting: collections.deque[SequenceRequest] = collections.deque()
                    for req in waiting:
                        if req.cancelled or req.finished:
                            self._free_seq_blocks(req)
                        else:
                            kept_waiting.append(req)
                    waiting = kept_waiting

                if running:
                    kept_running: list[SequenceRequest] = []
                    for req in running:
                        if req.cancelled or req.finished:
                            self._free_seq_blocks(req)
                        else:
                            kept_running.append(req)
                    running = kept_running

                # 0% CPU spin when idle: await first request on request_queue
                if not waiting and not running:
                    item = await queue.get()
                    if item is None or self._stop_requested:
                        break
                    if not item.cancelled and not item.finished:
                        waiting.append(item)

                # Non-blockingly drain all immediately available arrivals
                while not queue.empty():
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        self._stop_requested = True
                        break
                    if not item.cancelled and not item.finished:
                        waiting.append(item)

                if self._stop_requested:
                    break

                # Re-filter any newly drained requests that were immediately cancelled
                if waiting:
                    kept_waiting = collections.deque()
                    for req in waiting:
                        if req.cancelled or req.finished:
                            self._free_seq_blocks(req)
                        else:
                            kept_waiting.append(req)
                    waiting = kept_waiting

                if not waiting and not running:
                    continue

                # 1. Prefill Admission
                admitted_prefill = False
                if waiting and len(running) < self.max_batch_size:
                    # Handle max_tokens <= 0 or prefill_len >= max_model_len edge cases without GPU forward
                    while waiting:
                        head = waiting[0]
                        prefill_len = len(head.pending_prefill_ids)
                        if head.max_tokens <= 0 or prefill_len >= self.max_model_len or prefill_len == 0:
                            waiting.popleft()
                            self._free_seq_blocks(head)
                            head.finished = True
                            head.output_queue.put_nowait(
                                GenerationStep(
                                    token_id=None,
                                    text_delta="",
                                    finish_reason="length",
                                    prompt_tokens=len(head.prompt_ids),
                                    completion_tokens=len(head.generated_ids),
                                )
                            )
                            head.output_queue.put_nowait(None)
                        else:
                            break

                    if waiting and len(running) < self.max_batch_size:
                        first = waiting[0]
                        target_len = len(first.pending_prefill_ids)
                        blocks_per_seq = self.allocator.blocks_needed_for(target_len)

                        if blocks_per_seq > self.allocator.num_blocks:
                            waiting.popleft()
                            first.finished = True
                            first.output_queue.put_nowait(
                                RuntimeError(
                                    f"Request requires {blocks_per_seq} KV blocks, exceeding total pool "
                                    f"capacity of {self.allocator.num_blocks} blocks."
                                )
                            )
                            first.output_queue.put_nowait(None)
                            continue

                        if self.allocator.can_allocate(blocks_per_seq):
                            prefill_batch: list[SequenceRequest] = []
                            max_admit = self.max_batch_size - len(running)

                            while (
                                waiting
                                and len(prefill_batch) < max_admit
                                and len(waiting[0].pending_prefill_ids) == target_len
                                and waiting[0].max_tokens > 0
                                and len(waiting[0].pending_prefill_ids) < self.max_model_len
                                and (
                                    len(prefill_batch) == 0
                                    or (len(prefill_batch) + 1) * target_len <= self.max_prefill_tokens
                                )
                                and self.allocator.can_allocate(blocks_per_seq)
                            ):
                                req = waiting.popleft()
                                if req.cancelled:
                                    self._free_seq_blocks(req)
                                    req.finished = True
                                    continue
                                req.block_ids = self.allocator.allocate(blocks_per_seq)
                                req.context_len = target_len
                                prefill_batch.append(req)

                            if prefill_batch:
                                admitted_prefill = True
                                try:
                                    sampled_ids = await _run_gpu(self._run_prefill_batch, prefill_batch)
                                except Exception as exc:
                                    for seq in prefill_batch:
                                        self._free_seq_blocks(seq)
                                        seq.finished = True
                                        seq.output_queue.put_nowait(exc)
                                        seq.output_queue.put_nowait(None)
                                else:
                                    for seq, tok in zip(prefill_batch, sampled_ids):
                                        seq.pending_prefill_ids = []
                                        if seq.cancelled:
                                            self._free_seq_blocks(seq)
                                            seq.finished = True
                                            continue
                                        self._process_sampled_token(seq, tok)
                                        if seq.finished or seq.cancelled:
                                            self._free_seq_blocks(seq)
                                        else:
                                            running.append(seq)

                # 2. Batched Decode Step
                if not admitted_prefill and running:
                    idx = 0
                    while idx < len(running):
                        seq = running[idx]
                        if seq.cancelled or seq.finished:
                            self._free_seq_blocks(seq)
                            running.pop(idx)
                            continue

                        needed_total = self.allocator.blocks_needed_for(seq.context_len + 1)
                        delta_blocks = needed_total - len(seq.block_ids)

                        # Recomputation preemption when free physical blocks are exhausted
                        while delta_blocks > 0 and not self.allocator.can_allocate(delta_blocks):
                            if len(running) > 1:
                                victim = running.pop()
                                self._free_seq_blocks(victim)
                                victim.pending_prefill_ids = list(victim.prompt_ids) + list(
                                    victim.generated_ids
                                )
                                victim.context_len = 0
                                victim.next_token_id = None
                                waiting.appendleft(victim)
                                if victim is seq:
                                    break
                            else:
                                break

                        if idx >= len(running) or running[idx] is not seq:
                            # seq itself was preempted from the tail of running
                            continue

                        if delta_blocks > 0 and not self.allocator.can_allocate(delta_blocks):
                            self._free_seq_blocks(seq)
                            seq.finished = True
                            running.pop(idx)
                            seq.output_queue.put_nowait(
                                RuntimeError("Out of KV cache blocks during decode step.")
                            )
                            seq.output_queue.put_nowait(None)
                            continue

                        if delta_blocks > 0:
                            seq.block_ids.extend(self.allocator.allocate(delta_blocks))
                        seq.context_len += 1
                        idx += 1

                    if running:
                        try:
                            sampled_ids = await _run_gpu(self._run_decode_batch, running)
                        except Exception as exc:
                            for seq in running:
                                self._free_seq_blocks(seq)
                                seq.finished = True
                                seq.output_queue.put_nowait(exc)
                                seq.output_queue.put_nowait(None)
                            running.clear()
                        else:
                            next_running: list[SequenceRequest] = []
                            for seq, tok in zip(running, sampled_ids):
                                if seq.cancelled:
                                    self._free_seq_blocks(seq)
                                    seq.finished = True
                                    continue
                                self._process_sampled_token(seq, tok)
                                if seq.finished or seq.cancelled:
                                    self._free_seq_blocks(seq)
                                else:
                                    next_running.append(seq)
                            running = next_running

                await asyncio.sleep(0)
        finally:
            while not queue.empty():
                try:
                    leftover = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if leftover is not None:
                    self._free_seq_blocks(leftover)
            for seq in waiting:
                self._free_seq_blocks(seq)
            for seq in running:
                self._free_seq_blocks(seq)
            self._stop_requested = False

    async def generate_stream_async(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> AsyncIterator[GenerationStep]:
        """Enqueue a chat request into the continuous batching scheduler and stream incremental steps."""
        prompt_ids = self.format_prompt(messages)
        output_queue: asyncio.Queue[GenerationStep | Exception | None] = asyncio.Queue()
        req = SequenceRequest(
            request_id=uuid.uuid4().hex,
            prompt_ids=prompt_ids,
            pending_prefill_ids=list(prompt_ids),
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            output_queue=output_queue,
        )
        self.request_queue.put_nowait(req)
        try:
            while True:
                item = await output_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            req.cancelled = True

    async def generate_tokens_async(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> list[int]:
        """Generate token IDs asynchronously through the continuous batching scheduler."""
        if max_tokens <= 0 or len(input_ids) >= self.max_model_len:
            return []
        output_queue: asyncio.Queue[GenerationStep | Exception | None] = asyncio.Queue()
        req = SequenceRequest(
            request_id=uuid.uuid4().hex,
            prompt_ids=list(input_ids),
            pending_prefill_ids=list(input_ids),
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            output_queue=output_queue,
        )
        self.request_queue.put_nowait(req)
        tokens: list[int] = []
        try:
            while True:
                item = await output_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                if item.token_id is not None:
                    tokens.append(item.token_id)
            return tokens
        finally:
            req.cancelled = True

    def generate_tokens(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> Iterator[int]:
        """Synchronous convenience wrapper around ``generate_tokens_async``."""

        async def _run() -> list[int]:
            sched = asyncio.create_task(self.run_scheduler_loop())
            try:
                return await self.generate_tokens_async(
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    ignore_eos=ignore_eos,
                )
            finally:
                self.stop()
                sched.cancel()
                await asyncio.gather(sched, return_exceptions=True)
                self._stop_requested = False

        yield from asyncio.run(_run())


NaiveEngine = PagedEngine
