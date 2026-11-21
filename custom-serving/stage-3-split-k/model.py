"""From-scratch Stage 3 Qwen2.5 model implementation with Split-K (FlashDecode) Triton attention.

Stage 3 characteristics:
- Prefill (``q_len > 1``): retains Stage 2's tiled FlashAttention-2 forward kernel
  (``_flash_attention_fwd``) with analytical causal masking and fp32 online softmax.
- Decode (``q_len == 1``): parallelizes attention across KV sequence splits using
  ``_flash_attention_decode_splitk_fwd`` and reduces per-split log-sum-exp accumulators
  in ``_flash_attention_splitk_combine``.
- Native GQA indexing directly into unexpanded ``[B, num_kv_heads, kv_len, head_dim]``
  KV tensors (no ``repeat_interleave`` HBM duplication, no ``torch.matmul`` in attention).
- Zero-copy output layout written directly in ``[B, q_len, num_heads, head_dim]`` strides.
- No ``F.scaled_dot_product_attention`` and no ``transformers`` model classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

LayerKVCache = tuple[torch.Tensor, torch.Tensor]
KVCache = list[LayerKVCache]

MAX_SPLITK_SPLITS = 32
_SM_COUNT_CACHE: dict[int, int] = {}


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


def _get_sm_count(device: torch.device) -> int:
    """Return and cache the multiprocessor count for a CUDA device."""
    device_idx = 0 if device.index is None else int(device.index)
    sm_count = _SM_COUNT_CACHE.get(device_idx)
    if sm_count is None:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        _SM_COUNT_CACHE[device_idx] = sm_count
    return sm_count


def _choose_num_splits(
    bsz: int,
    num_heads: int,
    kv_len: int,
    block_n: int,
    device: torch.device,
) -> int:
    """Select runtime split-K factor in ``[1, MAX_SPLITK_SPLITS]`` to saturate SMs."""
    max_useful_splits = max(1, triton.cdiv(kv_len, block_n))
    # Avoid splitting tiny <=2-tile caches where combine overhead outweighs parallelism.
    if max_useful_splits <= 2:
        return 1
    total_heads = max(1, bsz * num_heads)
    sm_count = _get_sm_count(device)
    target_ctas = 3 * sm_count if max_useful_splits >= 16 else 2 * sm_count
    splits_for_sms = triton.cdiv(target_ctas, total_heads)
    splits_for_work = triton.cdiv(max_useful_splits, 5)
    num_splits = max(splits_for_sms, splits_for_work)
    return max(1, min(MAX_SPLITK_SPLITS, max_useful_splits, num_splits))


@triton.jit
def _flash_attention_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    q_len,
    kv_len,
    past_len,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward-pass FlashAttention-2 kernel with native GQA indexing and fp32 online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // NUM_HEADS
    q_head_idx = pid_bh % NUM_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_base = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    if BLOCK_D == HEAD_DIM:
        q_mask = offs_m[:, None] < q_len
    else:
        q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    qk_scale = sm_scale * 1.4426950408889634

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    hi = tl.minimum(kv_len, past_len + start_m + BLOCK_M)
    max_full = tl.minimum(kv_len, past_len + start_m)
    full_end = (max_full // BLOCK_N) * BLOCK_N

    for start_n in range(0, full_end, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if BLOCK_D == HEAD_DIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            d_mask = offs_d[None, :] < HEAD_DIM
            k = tl.load(k_ptrs, mask=d_mask, other=0.0)
            v = tl.load(v_ptrs, mask=d_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    for start_n in range(full_end, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if BLOCK_D == HEAD_DIM:
            kv_mask = offs_n[:, None] < kv_len
        else:
            kv_mask = (offs_n[:, None] < kv_len) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
        causal_mask = (offs_n[None, :] <= (past_len + offs_m)[:, None]) & (offs_n[None, :] < kv_len)
        qk = tl.where(causal_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    if BLOCK_D == HEAD_DIM:
        o_mask = offs_m[:, None] < q_len
    else:
        o_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


@triton.jit(do_not_specialize=["kv_len", "num_splits", "stride_pm_bh", "stride_pl_bh"])
def _flash_attention_decode_splitk_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    Partial_acc_ptr,
    Partial_m_ptr,
    Partial_l_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_pa_bh,
    stride_pa_s,
    stride_pa_d,
    stride_pm_bh,
    stride_pm_s,
    stride_pl_bh,
    stride_pl_s,
    kv_len,
    num_splits,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Split-K decode attention forward kernel with native GQA and vectorized reductions."""
    pid_bh = tl.program_id(0)
    pid_split = tl.program_id(1)

    batch_idx = pid_bh // NUM_HEADS
    q_head_idx = pid_bh % NUM_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_ptrs = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh + offs_d * stride_qd
    if BLOCK_D == HEAD_DIM:
        q = tl.load(q_ptrs).to(tl.float32)
    else:
        q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)
    q_scaled = q * (sm_scale * 1.4426950408889634)

    num_tiles = tl.cdiv(kv_len, BLOCK_N)
    base_tiles = num_tiles // num_splits
    rem = num_tiles % num_splits
    start_tile = pid_split * base_tiles + tl.minimum(pid_split, rem)
    n_tiles = base_tiles + tl.where(pid_split < rem, 1, 0)
    split_start = start_tile * BLOCK_N
    split_end = tl.minimum(kv_len, (start_tile + n_tiles) * BLOCK_N)

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh

    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if BLOCK_D == HEAD_DIM:
            kv_mask = offs_n[:, None] < split_end
        else:
            kv_mask = (offs_n[:, None] < split_end) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        qk = tl.sum(k.to(tl.float32) * q_scaled[None, :], axis=1)
        qk = tl.where(offs_n < split_end, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    tl.store(Partial_m_ptr + pid_bh * stride_pm_bh + pid_split * stride_pm_s, m_i)
    tl.store(Partial_l_ptr + pid_bh * stride_pl_bh + pid_split * stride_pl_s, l_i)
    pa_ptrs = (
        Partial_acc_ptr
        + pid_bh * stride_pa_bh
        + pid_split * stride_pa_s
        + offs_d * stride_pa_d
    )
    if BLOCK_D == HEAD_DIM:
        tl.store(pa_ptrs, acc)
    else:
        tl.store(pa_ptrs, acc, mask=offs_d < HEAD_DIM)


@triton.jit(do_not_specialize=["num_splits", "stride_pm_bh", "stride_pl_bh"])
def _flash_attention_splitk_combine(
    Partial_acc_ptr,
    Partial_m_ptr,
    Partial_l_ptr,
    O_ptr,
    stride_pa_bh,
    stride_pa_s,
    stride_pa_d,
    stride_pm_bh,
    stride_pm_s,
    stride_pl_bh,
    stride_pl_s,
    stride_ob,
    stride_oh,
    stride_od,
    num_splits,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce per-split log-sum-exp partial outputs into final ``[B, 1, H, D]`` output."""
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // NUM_HEADS
    q_head_idx = pid_bh % NUM_HEADS

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    s_mask = offs_s < num_splits

    m_s = tl.load(
        Partial_m_ptr + pid_bh * stride_pm_bh + offs_s * stride_pm_s,
        mask=s_mask,
        other=float("-inf"),
    )
    l_s = tl.load(
        Partial_l_ptr + pid_bh * stride_pl_bh + offs_s * stride_pl_s,
        mask=s_mask,
        other=0.0,
    )

    valid = s_mask & (l_s > 0.0)
    m_s_safe = tl.where(valid, m_s, float("-inf"))
    m_global = tl.max(m_s_safe, axis=0)
    scale_s = tl.where(valid, tl.math.exp2(m_s_safe - m_global), 0.0)
    l_global = tl.sum(l_s * scale_s, axis=0)

    pa_ptrs = (
        Partial_acc_ptr
        + pid_bh * stride_pa_bh
        + offs_s[:, None] * stride_pa_s
        + offs_d[None, :] * stride_pa_d
    )
    if BLOCK_D == HEAD_DIM:
        pa_mask = valid[:, None]
    else:
        pa_mask = valid[:, None] & (offs_d[None, :] < HEAD_DIM)
    partial_acc_tile = tl.load(pa_ptrs, mask=pa_mask, other=0.0)
    out_acc = tl.sum(partial_acc_tile * scale_s[:, None], axis=0) / tl.where(
        l_global == 0.0, 1.0, l_global
    )

    o_ptrs = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh + offs_d * stride_od
    if BLOCK_D == HEAD_DIM:
        tl.store(o_ptrs, out_acc.to(O_ptr.dtype.element_ty))
    else:
        tl.store(o_ptrs, out_acc.to(O_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)


def flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    past_len: int = 0,
    num_splits_override: int | None = None,
) -> torch.Tensor:
    """Dispatch decode (Split-K) or prefill (FlashAttention-2) and return ``[B, q_len, H * D]``."""
    bsz, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    num_kv_groups = num_heads // num_kv_heads

    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()

    if q_len == 0:
        return torch.empty((bsz, 0, num_heads * head_dim), device=q.device, dtype=q.dtype)

    block_d = triton.next_power_of_2(head_dim)

    if q_len == 1:
        decode_block_n = 64
        num_splits = (
            num_splits_override
            if num_splits_override is not None
            else _choose_num_splits(bsz, num_heads, kv_len, decode_block_n, q.device)
        )
        bh = bsz * num_heads
        partial_acc = torch.empty((bh, num_splits, head_dim), device=q.device, dtype=torch.float32)
        partial_m = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
        out = torch.empty((bsz, 1, num_heads, head_dim), device=q.device, dtype=q.dtype)

        _flash_attention_decode_splitk_fwd[(bh, num_splits)](
            q,
            k,
            v,
            partial_acc,
            partial_m,
            partial_l,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_l.stride(0),
            partial_l.stride(1),
            kv_len,
            num_splits,
            NUM_HEADS=num_heads,
            NUM_KV_GROUPS=num_kv_groups,
            HEAD_DIM=head_dim,
            BLOCK_N=decode_block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=1,
        )
        _flash_attention_splitk_combine[(bh,)](
            partial_acc,
            partial_m,
            partial_l,
            out,
            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_l.stride(0),
            partial_l.stride(1),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            num_splits,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_S=MAX_SPLITK_SPLITS,
            BLOCK_D=block_d,
            num_warps=2,
            num_stages=1,
        )
        return out.view(bsz, 1, num_heads * head_dim)

    out = torch.empty((bsz, q_len, num_heads, head_dim), device=q.device, dtype=q.dtype)
    block_m = 64 if q_len > 16 else 16
    block_n = 64
    grid = (triton.cdiv(q_len, block_m), bsz * num_heads)

    _flash_attention_fwd[grid](
        q,
        k,
        v,
        out,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        q_len,
        kv_len,
        past_len,
        NUM_HEADS=num_heads,
        NUM_KV_GROUPS=num_kv_groups,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out.view(bsz, q_len, num_heads * head_dim)


class SplitKAttention(nn.Module):
    """Split-K FlashDecode GQA attention for decode and FlashAttention-2 for prefill."""

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
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        bsz, seq_len, _ = hidden_states.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[2]

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
            out = flash_attention_fwd(q, k, v, self.scale, past_len=past_len)
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
        self.self_attn = SplitKAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        residual = hidden_states
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_key_value


class Qwen2Model(nn.Module):
    """Full Qwen2.5 causal language model with Split-K Triton FlashDecode and KV cache."""

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

        present_key_values: KVCache = []
        for idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[idx]
            hidden_states, layer_present = layer(
                hidden_states,
                cos,
                sin,
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
