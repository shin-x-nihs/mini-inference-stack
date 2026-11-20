"""Naive Stage 0 generation engine without KV caching."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from model import Qwen2Model, resolve_model_dir
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
    """Single incremental emission from ``NaiveEngine.generate_stream``."""

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


class NaiveEngine:
    """Recomputes full sequence forward passes on every decode step (Stage 0 baseline)."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_model_len: int = 9216,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)
        self.max_model_len = max_model_len

        model_dir = resolve_model_dir(model_name)
        self.model = Qwen2Model.from_pretrained(str(model_dir), dtype=self.dtype, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.eos_token_ids = _resolve_eos_token_ids(model_dir, self.tokenizer)

    def warmup(self) -> None:
        """Run a 2-token dummy forward pass outside NVTX step ranges."""
        dummy = torch.tensor([[1, 2]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            _ = self.model(dummy, last_token_only=True)
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

    def _sample_next_token(
        self,
        seq: torch.Tensor,
        step: int,
        temperature: float,
    ) -> tuple[torch.Tensor, int]:
        range_name = "prefill_step" if step == 0 else "decode_step"
        if self.device.type == "cuda":
            torch.cuda.nvtx.range_push(range_name)
        try:
            with torch.inference_mode():
                logits = self.model(seq, last_token_only=True)[:, -1, :]
                if temperature <= 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits.float() / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())
        finally:
            if self.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
        return next_token, token_id

    def generate_tokens(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> Iterator[int]:
        """Yield generated token IDs autoregressively without KV caching."""
        seq = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        for step in range(max_tokens):
            if seq.shape[1] >= self.max_model_len:
                break
            next_token, token_id = self._sample_next_token(seq, step, temperature)
            if not ignore_eos and token_id in self.eos_token_ids:
                break
            seq = torch.cat([seq, next_token], dim=1)
            yield token_id

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> Iterator[GenerationStep]:
        """Stream UTF-8-safe incremental text deltas for a chat message list."""
        input_ids = self.format_prompt(messages)
        prompt_len = len(input_ids)
        if max_tokens <= 0 or prompt_len >= self.max_model_len:
            yield GenerationStep(
                token_id=None,
                text_delta="",
                finish_reason="length",
                prompt_tokens=prompt_len,
                completion_tokens=0,
            )
            return

        seq = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_ids: list[int] = []
        emitted_text = ""

        for step in range(max_tokens):
            if seq.shape[1] >= self.max_model_len:
                break

            next_token, tok = self._sample_next_token(seq, step, temperature)
            if not ignore_eos and tok in self.eos_token_ids:
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                flushed = full_text[len(emitted_text) :]
                yield GenerationStep(
                    token_id=None,
                    text_delta=flushed,
                    finish_reason="stop",
                    prompt_tokens=prompt_len,
                    completion_tokens=len(generated_ids),
                )
                return

            seq = torch.cat([seq, next_token], dim=1)
            generated_ids.append(tok)
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            is_last_allowed = (step == max_tokens - 1) or (seq.shape[1] >= self.max_model_len)

            if is_last_allowed:
                delta = full_text[len(emitted_text) :]
                emitted_text = full_text
                yield GenerationStep(
                    token_id=tok,
                    text_delta=delta,
                    finish_reason="length",
                    prompt_tokens=prompt_len,
                    completion_tokens=len(generated_ids),
                )
                return

            candidate = full_text.rstrip("\ufffd") if full_text.endswith("\ufffd") else full_text
            if len(candidate) >= len(emitted_text):
                delta = candidate[len(emitted_text) :]
                emitted_text = candidate
            else:
                delta = ""

            yield GenerationStep(
                token_id=tok,
                text_delta=delta,
                finish_reason=None,
                prompt_tokens=prompt_len,
                completion_tokens=len(generated_ids),
            )
