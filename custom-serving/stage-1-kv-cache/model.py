"""From-scratch Stage 1 Qwen2.5 model implementation in pure PyTorch with KV cache.

Stage 1 characteristics:
- Naive per-layer PyTorch KV cache (post-RoPE unexpanded K/V tensors concatenated along seq dim).
- No FlashAttention / ``F.scaled_dot_product_attention`` (explicit QK^T / softmax / V).
- No ``transformers`` model classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

LayerKVCache = tuple[torch.Tensor, torch.Tensor]
KVCache = list[LayerKVCache]


def resolve_model_dir(model_name_or_path: str) -> Path:
    """Resolve a local directory path or cached HuggingFace snapshot directory."""
    local_dir = Path(model_name_or_path)
    if local_dir.is_dir():
        return local_dir
    return Path(snapshot_download(model_name_or_path, local_files_only=True))


@dataclass
class Qwen2Config:
    """Configuration for Qwen2.5 causal language models."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    head_dim: int = 0

    def __post_init__(self) -> None:
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> Qwen2Config:
        """Load configuration fields from a HuggingFace ``config.json`` file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hidden_size = int(data["hidden_size"])
        num_attention_heads = int(data["num_attention_heads"])
        head_dim = int(data.get("head_dim") or (hidden_size // num_attention_heads))
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(data["num_key_value_heads"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            head_dim=head_dim,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization matching Qwen2 cast order."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the negative first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors ``[B, H, S, D]``."""
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos_b) + (rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (rotate_half(k) * sin_b)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputed float32 RoPE cos/sin cache with dynamic sequence extension."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("cos_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self._build_cache(max_position_embeddings, device=torch.device("cpu"))

    def _build_cache(self, max_seq_len: int, device: torch.device) -> None:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_pos = position_offset + seq_len
        if end_pos > self.cos_cached.shape[0] or self.cos_cached.device != device:
            target_len = max(end_pos, self.cos_cached.shape[0])
            self._build_cache(target_len, device=device)
        return (
            self.cos_cached[position_offset:end_pos].to(dtype=dtype, device=device),
            self.sin_cached[position_offset:end_pos].to(dtype=dtype, device=device),
        )


class NaiveAttention(nn.Module):
    """Explicit GQA multi-head attention with naive PyTorch KV cache and no SDPA."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor | None,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present_key_value: LayerKVCache = (k, v)

        if hidden_states.is_cuda:
            torch.cuda.nvtx.range_push("attn_compute")
        try:
            if self.num_kv_groups > 1:
                k_attn = k.repeat_interleave(self.num_kv_groups, dim=1)
                v_attn = v.repeat_interleave(self.num_kv_groups, dim=1)
            else:
                k_attn = k
                v_attn = v

            scores = torch.matmul(q, k_attn.transpose(-2, -1)) * self.scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v_attn)

            out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.num_heads * self.head_dim)
        finally:
            if hidden_states.is_cuda:
                torch.cuda.nvtx.range_pop()
        return self.o_proj(out), present_key_value


class SwiGLUMLP(nn.Module):
    """Qwen2 SwiGLU feed-forward network."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Layer(nn.Module):
    """Single pre-norm Qwen2 transformer decoder block."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NaiveAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor | None,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            causal_mask,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_key_value


class Qwen2Model(nn.Module):
    """Full Qwen2.5 causal language model with naive PyTorch KV cache."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: KVCache | None = None,
        last_token_only: bool = False,
    ) -> tuple[torch.Tensor, KVCache]:
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} KV cache layers, got {len(past_key_values)}"
            )

        _, seq_len = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            position_offset=past_len,
        )
        if seq_len == 1:
            causal_mask: torch.Tensor | None = None
        else:
            causal_mask = torch.triu(
                torch.ones(
                    (seq_len, past_len + seq_len),
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
                diagonal=past_len + 1,
            ).view(1, 1, seq_len, past_len + seq_len)

        present_key_values: KVCache = []
        for idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[idx]
            hidden_states, layer_present = layer(
                hidden_states,
                cos,
                sin,
                causal_mask,
                past_key_value=layer_past,
            )
            present_key_values.append(layer_present)

        hidden_states = self.norm(hidden_states)
        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        return self.lm_head(hidden_states), present_key_values

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> Qwen2Model:
        """Load weights from local safetensors files and move the model to ``(device, dtype)``."""
        local_dir = resolve_model_dir(model_name_or_path)
        config = Qwen2Config.from_json(local_dir / "config.json")
        model = cls(config)

        shard_paths = sorted(local_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No .safetensors files found in {local_dir}")

        state_dict: dict[str, torch.Tensor] = {}
        for shard_path in shard_paths:
            for key, tensor in load_file(str(shard_path), device="cpu").items():
                mapped_key = key.removeprefix("model.")
                if mapped_key in state_dict:
                    raise RuntimeError(f"Duplicate weight key {mapped_key!r} across safetensors shards")
                state_dict[mapped_key] = tensor

        # Cast model to target dtype on CPU first so load_state_dict copies directly in target dtype.
        model = model.to(dtype=dtype)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
        missing = set(incompatible.missing_keys) - allowed_missing
        unexpected = set(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Weight loading mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

        target_device = torch.device(device)
        model = model.to(device=target_device, dtype=dtype)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
            if model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr():
                raise RuntimeError("Failed to tie lm_head.weight to embed_tokens.weight")

        # Keep rotary cache in float32 on the target device after parent module dtype cast.
        model.rotary_emb._build_cache(config.max_position_embeddings, device=target_device)
        model.eval()
        return model
