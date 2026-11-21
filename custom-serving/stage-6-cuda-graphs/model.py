"""From-scratch Stage 6 Qwen2.5 model implementation with Bucketed CUDA Graphs for Decode.

Stage 6 characteristics (building on Stage 5):
- Capture-safe ``Qwen2Model.forward``: zero host-device syncs and static-shape decode
  execution when ``max_context_len`` is passed as a Python ``int``.
- Padded-lane safety: clamps ``positions`` at ``min=0`` so padded lanes with
  ``context_lengths[pad] == 0`` safely read RoPE row 0 while ``_reshape_and_cache``
  skips KV writes (``pos == -1``) and ``_flash_attention_decode_splitk_fwd`` produces
  finite zero outputs without NaNs or out-of-bounds reads.
- Bucketed ``DecodeGraphRunner``: pre-allocates static device buffers for batch buckets
  ``(1, 2, 4, 8, 16, 32, 64)`` (``<= max_batch_size``), warms up both
  ``SINGLE_SPLIT=False`` and ``SINGLE_SPLIT=True`` Triton specializations, captures
  each bucket with ``torch.cuda.graph()``, and replays with padded-lane zero-filling
  or falls back to eager execution when uncaptured / oversized / on CPU.
- Retains Stage 5 fused Triton RMSNorm, RoPE, SwiGLU, Paged KV Cache, and Split-K
  attention kernels.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

KV_BLOCK_SIZE = 16
PagedKVCache = list[torch.Tensor]
GRAPH_BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

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


def _to_2d_rows(t: torch.Tensor, row_dim: int) -> torch.Tensor:
    """Return a 2D ``[num_rows, row_dim]`` view with unit last-dim stride, avoiding copies when possible."""
    if t.ndim == 2 and t.stride(-1) == 1:
        return t
    if t.ndim == 3 and t.shape[1] == 1 and t.stride(-1) == 1:
        return t[:, 0, :]
    if t.is_contiguous():
        return t.view(-1, row_dim)
    return t.contiguous().view(-1, row_dim)


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "stride_x_row",
        "stride_y_row",
        "stride_r_row",
        "stride_ro_row",
    ]
)
def _rms_norm_fwd(
    X_ptr,
    W_ptr,
    Y_ptr,
    R_ptr,
    RO_ptr,
    stride_x_row,
    stride_y_row,
    stride_r_row,
    stride_ro_row,
    num_rows,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Single-pass RMSNorm forward kernel with optional fused residual addition."""
    row = tl.program_id(0)
    if row >= num_rows:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HIDDEN_SIZE

    x_fp32 = tl.load(X_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r_fp32 = tl.load(R_ptr + row * stride_r_row + offs, mask=mask, other=0.0).to(tl.float32)
        res_out = (x_fp32 + r_fp32).to(RO_ptr.dtype.element_ty)
        tl.store(RO_ptr + row * stride_ro_row + offs, res_out, mask=mask)
        x_fp32 = res_out.to(tl.float32)

    mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / HIDDEN_SIZE
    inv_rms = tl.math.rsqrt(mean_sq + eps)
    x_norm = (x_fp32 * inv_rms).to(Y_ptr.dtype.element_ty)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(Y_ptr.dtype.element_ty)
    tl.store(Y_ptr + row * stride_y_row + offs, w * x_norm, mask=mask)


def rms_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm (and optional residual fusion) via Triton on CUDA or PyTorch on CPU."""
    if residual is not None and residual.shape != x.shape:
        raise ValueError(f"residual shape {residual.shape} must match x shape {x.shape}")

    if x.numel() == 0:
        y_empty = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        if residual is not None:
            return y_empty, torch.empty(x.shape, device=x.device, dtype=x.dtype)
        return y_empty

    if not x.is_cuda:
        if residual is not None:
            res_out = (x.to(torch.float32) + residual.to(torch.float32)).to(x.dtype)
            norm_in = res_out
        else:
            res_out = None
            norm_in = x
        input_dtype = norm_in.dtype
        x_fp32 = norm_in.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + eps)
        y = weight.to(input_dtype) * x_norm.to(input_dtype)
        if res_out is not None:
            return y, res_out
        return y

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = _to_2d_rows(x, hidden_size)
    num_rows = x_2d.shape[0]
    y = torch.empty(orig_shape, device=x.device, dtype=x.dtype)
    y_2d = y.view(-1, hidden_size)

    if residual is not None:
        r_2d = _to_2d_rows(residual, hidden_size)
        res_out = torch.empty(orig_shape, device=x.device, dtype=x.dtype)
        ro_2d = res_out.view(-1, hidden_size)
        r_ptr = r_2d
        ro_ptr = ro_2d
        stride_r_row = r_2d.stride(0)
        stride_ro_row = ro_2d.stride(0)
        has_residual = True
    else:
        res_out = None
        r_ptr = x_2d
        ro_ptr = y_2d
        stride_r_row = 0
        stride_ro_row = 0
        has_residual = False

    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_size >= 1024 else 4
    _rms_norm_fwd[(num_rows,)](
        x_2d,
        weight,
        y_2d,
        r_ptr,
        ro_ptr,
        x_2d.stride(0),
        y_2d.stride(0),
        stride_r_row,
        stride_ro_row,
        num_rows,
        eps,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        HAS_RESIDUAL=has_residual,
        num_warps=num_warps,
    )
    if res_out is not None:
        return y, res_out
    return y


class RMSNorm(nn.Module):
    """Fused Root Mean Square Layer Normalization matching Qwen2 cast order."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_fwd(x, self.weight, self.eps, residual=residual)


@triton.jit(
    do_not_specialize=[
        "bsz",
        "seq_len",
        "stride_qb",
        "stride_qh",
        "stride_qs",
        "stride_qd",
        "stride_kb",
        "stride_kh",
        "stride_ks",
        "stride_kd",
        "stride_cb",
        "stride_cs",
        "stride_cd",
        "stride_sb",
        "stride_ss",
        "stride_sd",
    ]
)
def _rope_embedding_fwd(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_cb,
    stride_cs,
    stride_cd,
    stride_sb,
    stride_ss,
    stride_sd,
    bsz,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    """Single-launch in-place RoPE kernel rotating both Q and K ``[B, H, S, D]``."""
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_bs // seq_len
    s = pid_bs % seq_len
    if b >= bsz:
        return

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF_DIM

    c_base = Cos_ptr + b * stride_cb + s * stride_cs
    s_base = Sin_ptr + b * stride_sb + s * stride_ss
    c1 = tl.load(c_base + offs * stride_cd, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(c_base + (HALF_DIM + offs) * stride_cd, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(s_base + offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(s_base + (HALF_DIM + offs) * stride_sd, mask=mask, other=0.0).to(tl.float32)

    if pid_h < NUM_Q_HEADS:
        x_base = Q_ptr + b * stride_qb + pid_h * stride_qh + s * stride_qs
        x1 = tl.load(x_base + offs * stride_qd, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + (HALF_DIM + offs) * stride_qd, mask=mask, other=0.0).to(tl.float32)
        out1 = x1 * c1 - x2 * s1
        out2 = x2 * c2 + x1 * s2
        tl.store(x_base + offs * stride_qd, out1.to(Q_ptr.dtype.element_ty), mask=mask)
        tl.store(x_base + (HALF_DIM + offs) * stride_qd, out2.to(Q_ptr.dtype.element_ty), mask=mask)
    else:
        kv_h = pid_h - NUM_Q_HEADS
        x_base = K_ptr + b * stride_kb + kv_h * stride_kh + s * stride_ks
        x1 = tl.load(x_base + offs * stride_kd, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + (HALF_DIM + offs) * stride_kd, mask=mask, other=0.0).to(tl.float32)
        out1 = x1 * c1 - x2 * s1
        out2 = x2 * c2 + x1 * s2
        tl.store(x_base + offs * stride_kd, out1.to(K_ptr.dtype.element_ty), mask=mask)
        tl.store(x_base + (HALF_DIM + offs) * stride_kd, out2.to(K_ptr.dtype.element_ty), mask=mask)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply in-place rotary position embeddings to query and key tensors ``[B, H, S, D]``."""
    bsz, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    if bsz == 0 or seq_len == 0 or (num_q_heads + num_kv_heads) == 0:
        return q, k

    half_dim = head_dim // 2
    if not q.is_cuda:
        if cos.ndim == 3:
            cos_b = cos.unsqueeze(1)
            sin_b = sin.unsqueeze(1)
        else:
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
        c1 = cos_b[..., :half_dim].to(torch.float32)
        c2 = cos_b[..., half_dim:].to(torch.float32)
        s1 = sin_b[..., :half_dim].to(torch.float32)
        s2 = sin_b[..., half_dim:].to(torch.float32)
        for t in (q, k):
            t1 = t[..., :half_dim].to(torch.float32)
            t2 = t[..., half_dim:].to(torch.float32)
            out1 = (t1 * c1 - t2 * s1).to(t.dtype)
            out2 = (t2 * c2 + t1 * s2).to(t.dtype)
            t[..., :half_dim].copy_(out1)
            t[..., half_dim:].copy_(out2)
        return q, k

    stride_cb = (cos.stride(0) if cos.shape[0] > 1 else 0) if cos.ndim == 3 else 0
    stride_sb = (sin.stride(0) if sin.shape[0] > 1 else 0) if sin.ndim == 3 else 0
    stride_cs = cos.stride(-2)
    stride_cd = cos.stride(-1)
    stride_ss = sin.stride(-2)
    stride_sd = sin.stride(-1)

    block_half = triton.next_power_of_2(half_dim)
    grid = (bsz * seq_len, num_q_heads + num_kv_heads)
    _rope_embedding_fwd[grid](
        q,
        k,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        stride_cb,
        stride_cs,
        stride_cd,
        stride_sb,
        stride_ss,
        stride_sd,
        bsz,
        seq_len,
        NUM_Q_HEADS=num_q_heads,
        HALF_DIM=half_dim,
        BLOCK_HALF=block_half,
        num_warps=2,
    )
    return q, k


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
        positions: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        max_pos_hint: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.numel() == 0:
            empty = torch.empty((*positions.shape, self.head_dim), dtype=dtype, device=device)
            return empty, empty

        needed_len = (
            (max_pos_hint + 1)
            if max_pos_hint is not None
            else (int(positions.max().item()) + 1)
        )
        if needed_len > self.cos_cached.shape[0] or self.cos_cached.device != device:
            target_len = max(needed_len, self.cos_cached.shape[0])
            self._build_cache(target_len, device=device)

        return (
            self.cos_cached[positions].to(dtype=dtype),
            self.sin_cached[positions].to(dtype=dtype),
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
    max_kv_len: int,
    block_n: int,
    device: torch.device,
) -> int:
    """Select runtime split-K factor in ``[1, MAX_SPLITK_SPLITS]`` for paged decode."""
    max_useful_splits = max(1, triton.cdiv(max_kv_len, block_n))
    total_heads = max(1, bsz * num_heads)
    sm_count = _get_sm_count(device)
    # Collapse to 1 split when <=2 tiles or when batch*heads already saturates SMs.
    if max_useful_splits <= 2 or total_heads >= 2 * sm_count:
        return 1
    target_ctas = 3 * sm_count if max_useful_splits >= 16 else 2 * sm_count
    splits_for_sms = triton.cdiv(target_ctas, total_heads)
    splits_for_work = triton.cdiv(max_useful_splits, 5)
    num_splits = max(splits_for_sms, splits_for_work)
    return max(1, min(MAX_SPLITK_SPLITS, max_useful_splits, num_splits))


@triton.jit(do_not_specialize=["q_len", "max_blocks_per_seq", "stride_bt_b"])
def _reshape_and_cache(
    K_ptr,
    V_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Context_lengths_ptr,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_kc_blk,
    stride_kc_off,
    stride_kc_h,
    stride_kc_d,
    stride_vc_blk,
    stride_vc_off,
    stride_vc_h,
    stride_vc_d,
    stride_bt_b,
    stride_bt_blk,
    q_len,
    max_blocks_per_seq,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Write newly projected K/V tokens ``[B, num_kv_heads, q_len, D]`` into paged KV cache."""
    pid_bs = tl.program_id(0)
    kv_h = tl.program_id(1)

    b = pid_bs // q_len
    s = pid_bs % q_len

    ctx_len = tl.load(Context_lengths_ptr + b)
    pos = (ctx_len - q_len) + s
    logical_block_idx = pos // BLOCK_SIZE
    block_offset = pos % BLOCK_SIZE

    valid_blk = (pos >= 0) & (logical_block_idx < max_blocks_per_seq)
    physical_block_idx = tl.load(
        Block_table_ptr + b * stride_bt_b + logical_block_idx * stride_bt_blk,
        mask=valid_blk,
        other=0,
    )

    offs_d = tl.arange(0, BLOCK_D)
    k_ptrs = K_ptr + b * stride_kb + kv_h * stride_kh + s * stride_ks + offs_d * stride_kd
    v_ptrs = V_ptr + b * stride_vb + kv_h * stride_vh + s * stride_vs + offs_d * stride_vd
    kc_ptrs = (
        K_cache_ptr
        + physical_block_idx * stride_kc_blk
        + block_offset * stride_kc_off
        + kv_h * stride_kc_h
        + offs_d * stride_kc_d
    )
    vc_ptrs = (
        V_cache_ptr
        + physical_block_idx * stride_vc_blk
        + block_offset * stride_vc_off
        + kv_h * stride_vc_h
        + offs_d * stride_vc_d
    )

    if BLOCK_D == HEAD_DIM:
        k_val = tl.load(k_ptrs, mask=valid_blk, other=0.0)
        v_val = tl.load(v_ptrs, mask=valid_blk, other=0.0)
        tl.store(kc_ptrs, k_val, mask=valid_blk)
        tl.store(vc_ptrs, v_val, mask=valid_blk)
    else:
        mask = valid_blk & (offs_d < HEAD_DIM)
        k_val = tl.load(k_ptrs, mask=mask, other=0.0)
        v_val = tl.load(v_ptrs, mask=mask, other=0.0)
        tl.store(kc_ptrs, k_val, mask=mask)
        tl.store(vc_ptrs, v_val, mask=mask)


def reshape_and_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
) -> None:
    """Scatter newly projected ``k, v`` ``[B, num_kv_heads, q_len, D]`` into paged KV cache."""
    bsz, num_kv_heads, q_len, head_dim = k.shape
    if bsz == 0 or q_len == 0:
        return

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    block_size = k_cache.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    max_blocks_per_seq = block_table.shape[1]

    grid = (bsz * q_len, num_kv_heads)
    _reshape_and_cache[grid](
        k,
        v,
        k_cache,
        v_cache,
        block_table,
        context_lengths,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        q_len,
        max_blocks_per_seq,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_D=block_d,
        num_warps=2,
        num_stages=1,
    )


@triton.jit(do_not_specialize=["q_len", "max_blocks_per_seq", "stride_bt_b"])
def _flash_attention_fwd(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Context_lengths_ptr,
    O_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kc_blk,
    stride_kc_off,
    stride_kc_h,
    stride_kc_d,
    stride_vc_blk,
    stride_vc_off,
    stride_vc_h,
    stride_vc_d,
    stride_bt_b,
    stride_bt_blk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    q_len,
    max_blocks_per_seq,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Paged FlashAttention-2 forward kernel with native GQA and fp32 online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // NUM_HEADS
    q_head_idx = pid_bh % NUM_HEADS
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    kv_len = tl.load(Context_lengths_ptr + batch_idx)
    past_len = kv_len - q_len

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_base = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    o_base = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh
    bt_base = Block_table_ptr + batch_idx * stride_bt_b

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
        if BLOCK_N == BLOCK_SIZE:
            logical_block_idx = start_n // BLOCK_SIZE
            valid_blk = (start_n < kv_len) & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=valid_blk,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx * stride_kc_blk
                + offs_n_base[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx * stride_vc_blk
                + offs_n_base[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )
        else:
            logical_block_idx = offs_n // BLOCK_SIZE
            block_offset = offs_n % BLOCK_SIZE
            valid_n = (offs_n < kv_len) & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=valid_n,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx[:, None] * stride_kc_blk
                + block_offset[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx[:, None] * stride_vc_blk
                + block_offset[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )

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
        if BLOCK_N == BLOCK_SIZE:
            logical_block_idx = start_n // BLOCK_SIZE
            valid_blk = (start_n < kv_len) & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=valid_blk,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx * stride_kc_blk
                + offs_n_base[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx * stride_vc_blk
                + offs_n_base[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )
        else:
            logical_block_idx = offs_n // BLOCK_SIZE
            block_offset = offs_n % BLOCK_SIZE
            valid_n = (offs_n < kv_len) & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=valid_n,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx[:, None] * stride_kc_blk
                + block_offset[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx[:, None] * stride_vc_blk
                + block_offset[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )

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


@triton.jit(
    do_not_specialize=[
        "num_splits",
        "max_blocks_per_seq",
        "stride_bt_b",
        "stride_pm_bh",
        "stride_pl_bh",
    ]
)
def _flash_attention_decode_splitk_fwd(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Context_lengths_ptr,
    Partial_acc_ptr,
    Partial_m_ptr,
    Partial_l_ptr,
    O_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kc_blk,
    stride_kc_off,
    stride_kc_h,
    stride_kc_d,
    stride_vc_blk,
    stride_vc_off,
    stride_vc_h,
    stride_vc_d,
    stride_bt_b,
    stride_bt_blk,
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
    max_blocks_per_seq,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINGLE_SPLIT: tl.constexpr,
):
    """Paged Split-K decode attention forward kernel with heterogeneous context lengths."""
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

    kv_len = tl.load(Context_lengths_ptr + batch_idx)
    if SINGLE_SPLIT:
        split_start = 0
        split_end = kv_len
    else:
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

    bt_base = Block_table_ptr + batch_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + offs_n_base
        valid_n = offs_n < split_end
        if BLOCK_N == BLOCK_SIZE:
            logical_block_idx = start_n // BLOCK_SIZE
            valid_blk = (start_n < split_end) & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=valid_blk,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx * stride_kc_blk
                + offs_n_base[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx * stride_vc_blk
                + offs_n_base[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )
        else:
            logical_block_idx = offs_n // BLOCK_SIZE
            block_offset = offs_n % BLOCK_SIZE
            bt_mask = valid_n & (logical_block_idx < max_blocks_per_seq)
            physical_block_idx = tl.load(
                bt_base + logical_block_idx * stride_bt_blk,
                mask=bt_mask,
                other=0,
            )
            k_ptrs = (
                K_cache_ptr
                + physical_block_idx[:, None] * stride_kc_blk
                + block_offset[:, None] * stride_kc_off
                + kv_head_idx * stride_kc_h
                + offs_d[None, :] * stride_kc_d
            )
            v_ptrs = (
                V_cache_ptr
                + physical_block_idx[:, None] * stride_vc_blk
                + block_offset[:, None] * stride_vc_off
                + kv_head_idx * stride_vc_h
                + offs_d[None, :] * stride_vc_d
            )

        if BLOCK_D == HEAD_DIM:
            kv_mask = valid_n[:, None]
        else:
            kv_mask = valid_n[:, None] & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        qk = tl.sum(k.to(tl.float32) * q_scaled[None, :], axis=1)
        qk = tl.where(valid_n, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    if SINGLE_SPLIT:
        out_vec = acc / tl.where(l_i == 0.0, 1.0, l_i)
        o_ptrs = O_ptr + batch_idx * stride_ob + q_head_idx * stride_oh + offs_d * stride_od
        if BLOCK_D == HEAD_DIM:
            tl.store(o_ptrs, out_vec.to(O_ptr.dtype.element_ty))
        else:
            tl.store(o_ptrs, out_vec.to(O_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)
    else:
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
    m_diff = tl.where(valid, m_s_safe - m_global, float("-inf"))
    scale_s = tl.where(valid, tl.math.exp2(m_diff), 0.0)
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


def paged_flash_attention_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    scale: float,
    max_context_len: int | None = None,
    num_splits_override: int | None = None,
) -> torch.Tensor:
    """Dispatch paged decode (Split-K) or prefill (FlashAttention-2) and return ``[B, q_len, H * D]``."""
    bsz, num_heads, q_len, head_dim = q.shape
    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    _, block_size, num_kv_heads, _ = k_cache.shape
    num_kv_groups = num_heads // num_kv_heads
    max_blocks_per_seq = block_table.shape[1]

    if q.stride(-1) != 1:
        q = q.contiguous()

    if bsz == 0 or q_len == 0:
        return torch.empty((bsz, q_len, num_heads * head_dim), device=q.device, dtype=q.dtype)

    block_d = triton.next_power_of_2(head_dim)

    if q_len == 1:
        decode_block_n = 64
        resolved_max_kv_len = (
            max_context_len if max_context_len is not None else int(context_lengths.max().item())
        )
        num_splits = (
            num_splits_override
            if num_splits_override is not None
            else _choose_num_splits(bsz, num_heads, resolved_max_kv_len, decode_block_n, q.device)
        )
        bh = bsz * num_heads
        out = torch.empty((bsz, 1, num_heads, head_dim), device=q.device, dtype=q.dtype)

        if num_splits == 1:
            _flash_attention_decode_splitk_fwd[(bh, 1)](
                q,
                k_cache,
                v_cache,
                block_table,
                context_lengths,
                out,
                out,
                out,
                out,
                scale,
                q.stride(0),
                q.stride(1),
                q.stride(3),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(3),
                block_table.stride(0),
                block_table.stride(1),
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                out.stride(0),
                out.stride(2),
                out.stride(3),
                1,
                max_blocks_per_seq,
                NUM_HEADS=num_heads,
                NUM_KV_GROUPS=num_kv_groups,
                HEAD_DIM=head_dim,
                BLOCK_SIZE=block_size,
                BLOCK_N=decode_block_n,
                BLOCK_D=block_d,
                SINGLE_SPLIT=True,
                num_warps=2,
                num_stages=2,
            )
            return out.view(bsz, 1, num_heads * head_dim)

        partial_acc = torch.empty((bh, num_splits, head_dim), device=q.device, dtype=torch.float32)
        partial_m = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)

        _flash_attention_decode_splitk_fwd[(bh, num_splits)](
            q,
            k_cache,
            v_cache,
            block_table,
            context_lengths,
            partial_acc,
            partial_m,
            partial_l,
            out,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(3),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
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
            max_blocks_per_seq,
            NUM_HEADS=num_heads,
            NUM_KV_GROUPS=num_kv_groups,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            BLOCK_N=decode_block_n,
            BLOCK_D=block_d,
            SINGLE_SPLIT=False,
            num_warps=2,
            num_stages=2,
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
    block_n = 32
    grid = (triton.cdiv(q_len, block_m), bsz * num_heads)

    _flash_attention_fwd[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        context_lengths,
        out,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        q_len,
        max_blocks_per_seq,
        NUM_HEADS=num_heads,
        NUM_KV_GROUPS=num_kv_groups,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out.view(bsz, q_len, num_heads * head_dim)


class SplitKAttention(nn.Module):
    """Paged KV Split-K FlashDecode GQA attention for decode and FlashAttention-2 for prefill."""

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
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lengths: torch.Tensor,
        max_context_len: int | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        reshape_and_cache(k, v, kv_cache, block_table, context_lengths)

        if hidden_states.is_cuda:
            torch.cuda.nvtx.range_push("attn_compute")
        try:
            out = paged_flash_attention_fwd(
                q,
                kv_cache,
                block_table,
                context_lengths,
                self.scale,
                max_context_len=max_context_len,
            )
        finally:
            if hidden_states.is_cuda:
                torch.cuda.nvtx.range_pop()
        return self.o_proj(out)


@triton.jit(do_not_specialize=["num_rows", "stride_gu_row", "stride_out_row"])
def _swiglu_fwd(
    GateUp_ptr,
    Out_ptr,
    stride_gu_row,
    stride_out_row,
    num_rows,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D-tiled SwiGLU forward kernel over concatenated ``[gate, up]`` activations."""
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= num_rows:
        return

    offs = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < INTERMEDIATE_SIZE

    gu_base = GateUp_ptr + row * stride_gu_row
    gate = tl.load(gu_base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_base + INTERMEDIATE_SIZE + offs, mask=mask, other=0.0).to(tl.float32)

    val = gate * tl.sigmoid(gate) * up
    tl.store(
        Out_ptr + row * stride_out_row + offs,
        val.to(Out_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_fwd(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """Compute ``SiLU(gate) * up`` from concatenated ``[..., 2 * intermediate_size]`` tensor."""
    if gate_up.numel() == 0:
        return torch.empty(
            (*gate_up.shape[:-1], intermediate_size),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )

    if not gate_up.is_cuda:
        gate = gate_up[..., :intermediate_size].to(torch.float32)
        up = gate_up[..., intermediate_size:].to(torch.float32)
        return (F.silu(gate) * up).to(gate_up.dtype)

    gate_up_2d = _to_2d_rows(gate_up, 2 * intermediate_size)
    num_rows = gate_up_2d.shape[0]
    out = torch.empty(
        (*gate_up.shape[:-1], intermediate_size),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    out_2d = out.view(-1, intermediate_size)
    block_n = 512
    grid = (num_rows, triton.cdiv(intermediate_size, block_n))
    _swiglu_fwd[grid](
        gate_up_2d,
        out_2d,
        gate_up_2d.stride(0),
        out_2d.stride(0),
        num_rows,
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


class SwiGLUMLP(nn.Module):
    """Qwen2 SwiGLU feed-forward network with fused ``gate_up_proj`` and Triton SwiGLU."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_fwd(self.gate_up_proj(x), self.intermediate_size))


class Qwen2Layer(nn.Module):
    """Single pre-norm Qwen2 transformer decoder block with fused residual threading."""

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
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lengths: torch.Tensor,
        max_context_len: int | None = None,
        residual: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
        else:
            normed, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            normed,
            cos,
            sin,
            kv_cache,
            block_table,
            context_lengths,
            max_context_len=max_context_len,
        )
        normed_mlp, residual = self.post_attention_layernorm(attn_out, residual)
        mlp_out = self.mlp(normed_mlp)
        if return_residual:
            return mlp_out, residual
        return residual + mlp_out


class Qwen2Model(nn.Module):
    """Full Qwen2.5 causal language model with Paged KV Cache and Split-K Triton FlashDecode."""

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
        kv_cache: PagedKVCache,
        block_table: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_only: bool = False,
        max_context_len: int | None = None,
    ) -> torch.Tensor:
        if len(kv_cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} KV cache layers, got {len(kv_cache)}"
            )

        _, seq_len = input_ids.shape
        resolved_max_ctx_len = (
            max_context_len
            if max_context_len is not None
            else (int(context_lengths.max().item()) if context_lengths.numel() > 0 else 0)
        )
        start_pos = context_lengths - seq_len
        positions = (
            start_pos[:, None]
            + torch.arange(seq_len, device=input_ids.device, dtype=torch.int32)[None, :]
        ).clamp(min=0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            positions,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            max_pos_hint=max(0, resolved_max_ctx_len - 1),
        )

        residual: torch.Tensor | None = None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                cos,
                sin,
                kv_cache[idx],
                block_table,
                context_lengths,
                max_context_len=resolved_max_ctx_len,
                residual=residual,
                return_residual=True,
            )

        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
            if residual is not None:
                residual = residual[:, -1:, :]
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual=residual)
        else:
            hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

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
        mlp_buffers: dict[str, dict[str, torch.Tensor]] = {}
        for shard_path in shard_paths:
            for key, tensor in load_file(str(shard_path), device="cpu").items():
                mapped_key = key.removeprefix("model.")
                if mapped_key.endswith(".mlp.gate_proj.weight"):
                    prefix = mapped_key.removesuffix("gate_proj.weight")
                    pair = mlp_buffers.setdefault(prefix, {})
                    if "gate" in pair:
                        raise RuntimeError(
                            f"Duplicate weight key {mapped_key!r} across safetensors shards"
                        )
                    pair["gate"] = tensor
                elif mapped_key.endswith(".mlp.up_proj.weight"):
                    prefix = mapped_key.removesuffix("up_proj.weight")
                    pair = mlp_buffers.setdefault(prefix, {})
                    if "up" in pair:
                        raise RuntimeError(
                            f"Duplicate weight key {mapped_key!r} across safetensors shards"
                        )
                    pair["up"] = tensor
                else:
                    if mapped_key in state_dict:
                        raise RuntimeError(
                            f"Duplicate weight key {mapped_key!r} across safetensors shards"
                        )
                    state_dict[mapped_key] = tensor

        for prefix, pair in mlp_buffers.items():
            if "gate" not in pair or "up" not in pair:
                raise RuntimeError(
                    f"Incomplete MLP weights for {prefix!r}: found {sorted(pair.keys())}"
                )
            fused_key = f"{prefix}gate_up_proj.weight"
            if fused_key in state_dict:
                raise RuntimeError(
                    f"Duplicate weight key {fused_key!r} across safetensors shards"
                )
            state_dict[fused_key] = torch.cat((pair["gate"], pair["up"]), dim=0)
        mlp_buffers.clear()

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


@dataclass
class _GraphBucket:
    """Static device buffers and captured CUDA graph state for a single batch bucket."""

    batch_size: int
    input_ids: torch.Tensor
    block_table: torch.Tensor
    context_lengths: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    static_logits: torch.Tensor | None = None


def _normalize_decode_input_ids(
    input_ids_list: Sequence[int] | Sequence[Sequence[int]],
    actual_batch_size: int,
) -> list[int]:
    if len(input_ids_list) != actual_batch_size:
        raise ValueError(
            f"input_ids_list length {len(input_ids_list)} != actual_batch_size {actual_batch_size}"
        )
    if actual_batch_size == 0:
        return []
    first = input_ids_list[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        normalized: list[int] = []
        for row in input_ids_list:
            assert isinstance(row, Sequence)
            if len(row) != 1:
                raise ValueError(f"Expected decode row length 1, got {len(row)}")
            normalized.append(int(row[0]))
        return normalized
    return [int(tok) for tok in input_ids_list]  # type: ignore[arg-type]


class DecodeGraphRunner:
    """Bucketed CUDA Graph capture and replay runner for fixed-shape ``q_len == 1`` decode."""

    def __init__(
        self,
        model: Qwen2Model,
        kv_cache: PagedKVCache,
        max_model_len: int,
        max_batch_size: int = 64,
        block_size: int = KV_BLOCK_SIZE,
        device: str | torch.device | None = None,
    ) -> None:
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.model = model
        self.kv_cache = kv_cache
        self.max_model_len = int(max_model_len)
        self.max_batch_size = int(max_batch_size)
        self.block_size = int(block_size)
        self.max_blocks_per_seq = max(1, triton.cdiv(self.max_model_len, self.block_size))

        if device is not None:
            self.device = torch.device(device)
        elif kv_cache:
            self.device = kv_cache[0].device
        else:
            self.device = next(model.parameters()).device

        self.buckets: tuple[int, ...] = tuple(
            b for b in GRAPH_BATCH_BUCKETS if b <= self.max_batch_size
        )
        self.max_bucket_size: int = max(self.buckets) if self.buckets else 0
        self.is_captured: bool = False
        self._buckets: dict[int, _GraphBucket] = {}

        if self.device.type == "cuda":
            for b in self.buckets:
                self._buckets[b] = _GraphBucket(
                    batch_size=b,
                    input_ids=torch.zeros((b, 1), dtype=torch.int64, device=self.device),
                    block_table=torch.zeros(
                        (b, self.max_blocks_per_seq), dtype=torch.int32, device=self.device
                    ),
                    context_lengths=torch.zeros((b,), dtype=torch.int32, device=self.device),
                )

    def select_bucket(self, actual_batch_size: int) -> int | None:
        """Return the smallest captured bucket ``>= actual_batch_size``, or ``None``."""
        for b in self.buckets:
            if b >= actual_batch_size:
                return b
        return None

    def capture_all(self, temp_block_ids: Sequence[int] | None = None) -> None:
        """Warm up Triton kernels (``SINGLE_SPLIT=False/True``) and capture all CUDA graph buckets."""
        if self.device.type != "cuda" or not self.buckets:
            self.is_captured = False
            return

        temp_block = int(temp_block_ids[0]) if temp_block_ids else None

        def _prepare_static_buffers(state: _GraphBucket) -> None:
            state.input_ids.zero_()
            state.block_table.zero_()
            if temp_block is not None:
                state.block_table[:, 0].fill_(temp_block)
                state.context_lengths.fill_(1)
            else:
                state.context_lengths.zero_()

        with torch.inference_mode():
            # Eager warmup passes to JIT-compile both SINGLE_SPLIT=False (small B)
            # and SINGLE_SPLIT=True (large B, e.g. B=16) before stream capture.
            warmup_buckets: list[int] = [self.buckets[0]]
            large_buckets = [b for b in self.buckets if b >= 10]
            if large_buckets:
                if large_buckets[0] not in warmup_buckets:
                    warmup_buckets.append(large_buckets[0])
            elif self.buckets[-1] not in warmup_buckets:
                warmup_buckets.append(self.buckets[-1])

            for wb in warmup_buckets:
                w_state = self._buckets[wb]
                _prepare_static_buffers(w_state)
                _ = self.model(
                    w_state.input_ids,
                    kv_cache=self.kv_cache,
                    block_table=w_state.block_table,
                    context_lengths=w_state.context_lengths,
                    last_token_only=True,
                    max_context_len=self.max_model_len,
                )

            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream(self.device))

            for b in self.buckets:
                state = self._buckets[b]
                _prepare_static_buffers(state)

                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        _ = self.model(
                            state.input_ids,
                            kv_cache=self.kv_cache,
                            block_table=state.block_table,
                            context_lengths=state.context_lengths,
                            last_token_only=True,
                            max_context_len=self.max_model_len,
                        )
                torch.cuda.current_stream(self.device).wait_stream(capture_stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    out = self.model(
                        state.input_ids,
                        kv_cache=self.kv_cache,
                        block_table=state.block_table,
                        context_lengths=state.context_lengths,
                        last_token_only=True,
                        max_context_len=self.max_model_len,
                    )
                state.graph = graph
                state.static_logits = out[:, -1, :]

            torch.cuda.current_stream(self.device).wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)

        self.is_captured = True

    def _run_eager(
        self,
        actual_batch_size: int,
        flat_input_ids: list[int],
        block_table_rows: Sequence[Sequence[int]],
        ctx_lens_list: Sequence[int],
    ) -> torch.Tensor:
        with torch.inference_mode():
            input_ids = torch.tensor(
                [[tok] for tok in flat_input_ids],
                dtype=torch.int64,
                device=self.device,
            )
            max_cols = max((len(row) for row in block_table_rows), default=1)
            max_cols = max(max_cols, 1)
            padded_bt = [
                list(row) + [0] * (max_cols - len(row)) for row in block_table_rows
            ]
            block_table = torch.tensor(padded_bt, dtype=torch.int32, device=self.device)
            ctx_list = [int(c) for c in ctx_lens_list]
            context_lengths = torch.tensor(ctx_list, dtype=torch.int32, device=self.device)
            max_ctx_len = max(ctx_list, default=0)
            logits = self.model(
                input_ids,
                kv_cache=self.kv_cache,
                block_table=block_table,
                context_lengths=context_lengths,
                last_token_only=True,
                max_context_len=max_ctx_len,
            )
            return logits[:, -1, :]

    def replay(
        self,
        actual_batch_size: int,
        input_ids_list: Sequence[int] | Sequence[Sequence[int]],
        block_table_rows: Sequence[Sequence[int]],
        ctx_lens_list: Sequence[int],
    ) -> torch.Tensor:
        """Replay a captured decode bucket for ``actual_batch_size`` rows, or fall back to eager."""
        if actual_batch_size < 0:
            raise ValueError(f"actual_batch_size must be non-negative, got {actual_batch_size}")
        if actual_batch_size == 0:
            return torch.empty(
                (0, self.model.config.vocab_size),
                dtype=self.model.embed_tokens.weight.dtype,
                device=self.device,
            )

        flat_input_ids = _normalize_decode_input_ids(input_ids_list, actual_batch_size)
        if len(block_table_rows) != actual_batch_size:
            raise ValueError(
                f"block_table_rows length {len(block_table_rows)} != actual_batch_size {actual_batch_size}"
            )
        if len(ctx_lens_list) != actual_batch_size:
            raise ValueError(
                f"ctx_lens_list length {len(ctx_lens_list)} != actual_batch_size {actual_batch_size}"
            )

        bucket_size = self.select_bucket(actual_batch_size)
        if (
            self.device.type != "cuda"
            or not self.is_captured
            or bucket_size is None
        ):
            return self._run_eager(
                actual_batch_size, flat_input_ids, block_table_rows, ctx_lens_list
            )

        state = self._buckets[bucket_size]
        if state.graph is None or state.static_logits is None:
            return self._run_eager(
                actual_batch_size, flat_input_ids, block_table_rows, ctx_lens_list
            )

        max_cols = max((len(row) for row in block_table_rows), default=0)
        if max_cols > self.max_blocks_per_seq:
            raise ValueError(
                f"block_table row length {max_cols} exceeds max_blocks_per_seq {self.max_blocks_per_seq}"
            )

        state.input_ids[:actual_batch_size, 0].copy_(
            torch.tensor(flat_input_ids, dtype=torch.int64)
        )
        if actual_batch_size < bucket_size:
            state.input_ids[actual_batch_size:].zero_()

        if max_cols > 0:
            padded_bt = [
                list(row) + [0] * (max_cols - len(row)) for row in block_table_rows
            ]
            state.block_table[:actual_batch_size, :max_cols].copy_(
                torch.tensor(padded_bt, dtype=torch.int32)
            )
        if actual_batch_size < bucket_size:
            state.block_table[actual_batch_size:].zero_()

        state.context_lengths[:actual_batch_size].copy_(
            torch.tensor([int(c) for c in ctx_lens_list], dtype=torch.int32)
        )
        if actual_batch_size < bucket_size:
            state.context_lengths[actual_batch_size:].zero_()

        state.graph.replay()
        return state.static_logits[:actual_batch_size]
