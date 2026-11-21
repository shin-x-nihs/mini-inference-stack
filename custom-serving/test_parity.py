from __future__ import annotations

import argparse
import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

CUSTOM_SERVING_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUSTOM_SERVING_DIR.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_EXTRA_STAGE4_PROMPTS: tuple[str, ...] = (
    "Write a Python function that reverses a list in place.",
    "Implement a Python class for an LRU cache using collections.OrderedDict with get and put methods and type annotations.",
    "Write a complete Python script that implements binary search on a sorted list of integers, includes docstrings explaining time and space complexity, and runs unit tests verifying edge cases such as empty lists, single-element lists, and duplicate values.",
)


def _stage_sort_key(stage_dir: Path) -> tuple[int, str]:
    match = re.match(r"^stage-(\d+)\b", stage_dir.name)
    stage_idx = int(match.group(1)) if match else 999_999
    return (stage_idx, stage_dir.name)


def discover_stages(
    custom_serving_dir: Path = CUSTOM_SERVING_DIR,
    stage: str = "all",
) -> list[Path]:
    if stage == "all":
        stages = [
            d.resolve()
            for d in custom_serving_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("stage-")
            and (d / "model.py").is_file()
            and (d / "engine.py").is_file()
        ]
        stages.sort(key=_stage_sort_key)
        if not stages:
            raise FileNotFoundError(
                f"No implemented stage-* directories containing model.py and engine.py "
                f"found under {custom_serving_dir}"
            )
        return stages

    candidate = (custom_serving_dir / stage).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Stage directory not found: {candidate}")
    if not (candidate / "model.py").is_file() or not (candidate / "engine.py").is_file():
        raise FileNotFoundError(
            f"Stage directory {candidate} must contain both model.py and engine.py"
        )
    return [candidate]


def load_stage_modules(stage_dir: Path) -> tuple[ModuleType, ModuleType]:
    stage_dir = stage_dir.resolve()
    model_path = (stage_dir / "model.py").resolve()
    engine_path = (stage_dir / "engine.py").resolve()
    if not model_path.is_file() or not engine_path.is_file():
        raise FileNotFoundError(
            f"Stage directory {stage_dir} must contain both model.py and engine.py"
        )

    orig_sys_path = list(sys.path)
    had_model = "model" in sys.modules
    orig_model = sys.modules.get("model")
    had_engine = "engine" in sys.modules
    orig_engine = sys.modules.get("engine")

    try:
        sys.modules.pop("model", None)
        sys.modules.pop("engine", None)
        sys.path.insert(0, str(stage_dir))

        model_spec = importlib.util.spec_from_file_location("model", model_path)
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"Could not create module spec for {model_path}")
        model_mod = importlib.util.module_from_spec(model_spec)
        sys.modules["model"] = model_mod
        model_spec.loader.exec_module(model_mod)

        engine_spec = importlib.util.spec_from_file_location("engine", engine_path)
        if engine_spec is None or engine_spec.loader is None:
            raise ImportError(f"Could not create module spec for {engine_path}")
        engine_mod = importlib.util.module_from_spec(engine_spec)
        sys.modules["engine"] = engine_mod
        engine_spec.loader.exec_module(engine_mod)

        assert model_mod.__file__ is not None
        assert engine_mod.__file__ is not None
        assert Path(model_mod.__file__).resolve() == model_path
        assert Path(engine_mod.__file__).resolve() == engine_path
        return model_mod, engine_mod
    finally:
        sys.path[:] = orig_sys_path
        if had_model and orig_model is not None:
            sys.modules["model"] = orig_model
        else:
            sys.modules.pop("model", None)
        if had_engine and orig_engine is not None:
            sys.modules["engine"] = orig_engine
        else:
            sys.modules.pop("engine", None)


def _default_attn_impl(stage_dir: Path) -> str:
    return "eager" if _stage_sort_key(stage_dir)[0] < 2 else "sdpa"


async def _run_with_scheduler(engine: object, work_coro: object) -> object:
    sched_task = asyncio.create_task(engine.run_scheduler_loop())  # type: ignore[attr-defined]
    work_task = asyncio.create_task(work_coro)  # type: ignore[arg-type]
    try:
        done, _ = await asyncio.wait(
            {work_task, sched_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sched_task in done and work_task not in done:
            work_task.cancel()
            await asyncio.gather(work_task, return_exceptions=True)
            exc = sched_task.exception()
            if exc is not None:
                raise exc
            raise RuntimeError(
                "Scheduler loop exited unexpectedly before work completed."
            )
        return await work_task
    finally:
        engine.stop()  # type: ignore[attr-defined]
        sched_task.cancel()
        await asyncio.gather(sched_task, return_exceptions=True)


def _compute_hf_reference(
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    input_tensor: torch.Tensor,
    prompt_len: int,
    num_tokens: int,
    pad_token_id: int | None,
    attn_impl: str,
    extra_prompt_ids_list: list[list[int]] | None = None,
) -> tuple[torch.Tensor, list[int], list[list[int]]]:
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    try:
        with torch.inference_mode():
            hf_prefill_logits = (
                hf_model(input_tensor, use_cache=False).logits[:, -1, :].float().cpu()
            )
            gen_cfg = GenerationConfig(
                do_sample=False,
                max_new_tokens=num_tokens,
                min_new_tokens=num_tokens,
                repetition_penalty=1.0,
                use_cache=(attn_impl == "sdpa"),
                eos_token_id=None,
                pad_token_id=pad_token_id,
            )
            hf_out = hf_model.generate(
                input_tensor,
                generation_config=gen_cfg,
            )

            extra_hf_ids: list[list[int]] = []
            if extra_prompt_ids_list:
                for extra_ids in extra_prompt_ids_list:
                    extra_tensor = torch.tensor(
                        [extra_ids], dtype=torch.long, device=device
                    )
                    extra_out = hf_model.generate(
                        extra_tensor,
                        generation_config=gen_cfg,
                    )
                    e_ids: list[int] = extra_out[0, len(extra_ids) :].tolist()
                    assert len(e_ids) == num_tokens, (
                        f"Expected {num_tokens} extra HF ({attn_impl}) tokens, got {len(e_ids)}"
                    )
                    extra_hf_ids.append(e_ids)

        hf_ids: list[int] = hf_out[0, prompt_len:].tolist()
        assert len(hf_ids) == num_tokens, (
            f"Expected {num_tokens} HF ({attn_impl}) tokens, got {len(hf_ids)}"
        )
        return hf_prefill_logits, hf_ids, extra_hf_ids
    finally:
        del hf_model
        torch.cuda.empty_cache()


def _verify_stage5_fused_ops(
    stage_name: str,
    model_mod: ModuleType,
    model: object,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
) -> None:
    import torch.nn.functional as F
    import triton
    from safetensors import safe_open

    atol = (
        3e-2
        if torch_dtype == torch.bfloat16
        else (5e-3 if torch_dtype == torch.float16 else 1e-5)
    )
    rtol = (
        2e-2
        if torch_dtype == torch.bfloat16
        else (5e-3 if torch_dtype == torch.float16 else 1e-5)
    )

    inter_size = model.config.intermediate_size  # type: ignore[attr-defined]
    hidden_size = model.config.hidden_size  # type: ignore[attr-defined]
    model_dir = model_mod.resolve_model_dir(model_name)
    shard_paths = sorted(model_dir.glob("*.safetensors"))
    assert shard_paths, f"[{stage_name}] No .safetensors files in {model_dir}"

    ref_gate: dict[int, torch.Tensor] = {}
    ref_up: dict[int, torch.Tensor] = {}
    layer_pat = re.compile(
        r"^(?:model\.)?layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.weight$"
    )
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                m = layer_pat.match(key)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                proj_kind = m.group(2)
                t = sf.get_tensor(key).to(dtype=torch_dtype)
                if proj_kind == "gate_proj":
                    ref_gate[layer_idx] = t
                else:
                    ref_up[layer_idx] = t

    layers = model.layers  # type: ignore[attr-defined]
    assert len(ref_gate) == len(layers) and len(ref_up) == len(layers), (
        f"[{stage_name}] Missing MLP reference weights in safetensors: "
        f"gate={len(ref_gate)}, up={len(ref_up)}, layers={len(layers)}"
    )
    gu_cpu = None
    for layer_idx, layer in enumerate(layers):
        assert not hasattr(layer.mlp, "gate_proj") and not hasattr(
            layer.mlp, "up_proj"
        ), f"[{stage_name}] Layer {layer_idx} unexpectedly kept unfused gate_proj/up_proj"
        gu_cpu = layer.mlp.gate_up_proj.weight.detach().cpu()
        assert tuple(gu_cpu.shape) == (2 * inter_size, hidden_size), (
            f"[{stage_name}] Layer {layer_idx} gate_up_proj shape {tuple(gu_cpu.shape)} "
            f"!= {(2 * inter_size, hidden_size)}"
        )
        assert torch.equal(gu_cpu[:inter_size], ref_gate[layer_idx]), (
            f"[{stage_name}] Layer {layer_idx} gate_up_proj[:{inter_size}] != gate_proj.weight"
        )
        assert torch.equal(gu_cpu[inter_size:], ref_up[layer_idx]), (
            f"[{stage_name}] Layer {layer_idx} gate_up_proj[{inter_size}:] != up_proj.weight"
        )
    del ref_gate, ref_up, gu_cpu

    with torch.inference_mode():
        # Verify _rms_norm_fwd (HAS_RESIDUAL=False and HAS_RESIDUAL=True)
        layer0_norm = layers[0].input_layernorm
        weight = layer0_norm.weight
        eps = float(model.config.rms_norm_eps)  # type: ignore[attr-defined]
        block_size = triton.next_power_of_2(hidden_size)
        num_warps = 8 if block_size >= 1024 else 4

        x = torch.randn((2, 17, hidden_size), device=device, dtype=torch_dtype)
        r = torch.randn((2, 17, hidden_size), device=device, dtype=torch_dtype)
        x_2d = x.view(-1, hidden_size)
        num_rows = x_2d.shape[0]
        y = torch.empty_like(x)
        y_2d = y.view(-1, hidden_size)

        model_mod._rms_norm_fwd[(num_rows,)](
            x_2d,
            weight,
            y_2d,
            x_2d,
            y_2d,
            x_2d.stride(0),
            y_2d.stride(0),
            0,
            0,
            num_rows,
            eps,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SIZE=block_size,
            HAS_RESIDUAL=False,
            num_warps=num_warps,
        )
        x_f = x.float()
        ref_y = weight.to(torch_dtype) * (
            x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        ).to(torch_dtype)
        assert torch.allclose(y, ref_y, atol=atol, rtol=rtol), (
            f"[{stage_name}] _rms_norm_fwd (HAS_RESIDUAL=False) max diff "
            f"{(y - ref_y).abs().max().item():.6f}"
        )

        r_2d = r.view(-1, hidden_size)
        res_out = torch.empty_like(x)
        ro_2d = res_out.view(-1, hidden_size)
        y_res = torch.empty_like(x)
        y_res_2d = y_res.view(-1, hidden_size)
        model_mod._rms_norm_fwd[(num_rows,)](
            x_2d,
            weight,
            y_res_2d,
            r_2d,
            ro_2d,
            x_2d.stride(0),
            y_res_2d.stride(0),
            r_2d.stride(0),
            ro_2d.stride(0),
            num_rows,
            eps,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SIZE=block_size,
            HAS_RESIDUAL=True,
            num_warps=num_warps,
        )
        ref_res = (x.float() + r.float()).to(torch_dtype)
        norm_in = ref_res.float()
        ref_y_res = weight.to(torch_dtype) * (
            norm_in * torch.rsqrt(norm_in.pow(2).mean(-1, keepdim=True) + eps)
        ).to(torch_dtype)
        assert torch.allclose(res_out, ref_res, atol=atol, rtol=rtol), (
            f"[{stage_name}] _rms_norm_fwd residual max diff "
            f"{(res_out - ref_res).abs().max().item():.6f}"
        )
        assert torch.allclose(y_res, ref_y_res, atol=atol, rtol=rtol), (
            f"[{stage_name}] _rms_norm_fwd (HAS_RESIDUAL=True) max diff "
            f"{(y_res - ref_y_res).abs().max().item():.6f}"
        )
        assert isinstance(layer0_norm(x), torch.Tensor)
        assert isinstance(layer0_norm(x, residual=r), tuple)
        del x, r, x_2d, y, y_2d, x_f, ref_y, r_2d, res_out, ro_2d, y_res, y_res_2d, ref_res, norm_in, ref_y_res

        # Verify _rope_embedding_fwd (non-contiguous S>1 .transpose(1, 2) and S=1)
        def _eager_rope(
            t: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor
        ) -> torch.Tensor:
            half = t.shape[-1] // 2
            cos_b = (
                cos_t.unsqueeze(1)
                if cos_t.ndim == 3
                else cos_t.unsqueeze(0).unsqueeze(0)
            )
            sin_b = (
                sin_t.unsqueeze(1)
                if sin_t.ndim == 3
                else sin_t.unsqueeze(0).unsqueeze(0)
            )
            t_rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            return (t.float() * cos_b.float() + t_rot.float() * sin_b.float()).to(
                t.dtype
            )

        num_q = model.config.num_attention_heads  # type: ignore[attr-defined]
        num_kv = model.config.num_key_value_heads  # type: ignore[attr-defined]
        head_dim = model.config.head_dim  # type: ignore[attr-defined]
        half_dim = head_dim // 2
        block_half = triton.next_power_of_2(half_dim)

        for bsz, seq_len in ((2, 19), (3, 1)):
            q = torch.randn(
                (bsz, seq_len, num_q, head_dim), device=device, dtype=torch_dtype
            ).transpose(1, 2)
            k = torch.randn(
                (bsz, seq_len, num_kv, head_dim), device=device, dtype=torch_dtype
            ).transpose(1, 2)
            if seq_len > 1:
                assert not q.is_contiguous() and not k.is_contiguous()

            positions = (
                torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(0)
                + torch.arange(bsz, device=device, dtype=torch.int32).unsqueeze(1) * 5
                + 7
            )
            cos, sin = model.rotary_emb(  # type: ignore[attr-defined]
                positions, dtype=torch_dtype, device=torch.device(device)
            )
            q_ref = _eager_rope(q.clone(), cos, sin)
            k_ref = _eager_rope(k.clone(), cos, sin)

            model_mod._rope_embedding_fwd[(bsz * seq_len, num_q + num_kv)](
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
                cos.stride(0) if cos.ndim == 3 and cos.shape[0] > 1 else 0,
                cos.stride(-2),
                cos.stride(-1),
                sin.stride(0) if sin.ndim == 3 and sin.shape[0] > 1 else 0,
                sin.stride(-2),
                sin.stride(-1),
                bsz,
                seq_len,
                NUM_Q_HEADS=num_q,
                HALF_DIM=half_dim,
                BLOCK_HALF=block_half,
                num_warps=2,
            )
            assert torch.allclose(q, q_ref, atol=atol, rtol=rtol), (
                f"[{stage_name}] _rope_embedding_fwd Q (B={bsz}, S={seq_len}) max diff "
                f"{(q - q_ref).abs().max().item():.6f}"
            )
            assert torch.allclose(k, k_ref, atol=atol, rtol=rtol), (
                f"[{stage_name}] _rope_embedding_fwd K (B={bsz}, S={seq_len}) max diff "
                f"{(k - k_ref).abs().max().item():.6f}"
            )
            del q, k, positions, cos, sin, q_ref, k_ref

        # Verify _swiglu_fwd
        gu = torch.randn((2, 17, 2 * inter_size), device=device, dtype=torch_dtype)
        gu_2d = gu.view(-1, 2 * inter_size)
        num_rows = gu_2d.shape[0]
        out = torch.empty((2, 17, inter_size), device=device, dtype=torch_dtype)
        out_2d = out.view(-1, inter_size)
        block_n = 512
        grid = (num_rows, triton.cdiv(inter_size, block_n))
        model_mod._swiglu_fwd[grid](
            gu_2d,
            out_2d,
            gu_2d.stride(0),
            out_2d.stride(0),
            num_rows,
            INTERMEDIATE_SIZE=inter_size,
            BLOCK_N=block_n,
            num_warps=4,
        )
        ref_out = (
            F.silu(gu[..., :inter_size].float()) * gu[..., inter_size:].float()
        ).to(torch_dtype)
        assert torch.allclose(out, ref_out, atol=atol, rtol=rtol), (
            f"[{stage_name}] _swiglu_fwd max diff {(out - ref_out).abs().max().item():.6f}"
        )
        del gu, gu_2d, out, out_2d, ref_out

    print(
        f"[{stage_name}] PASS: gate_up_proj weight slicing ({len(layers)} layers) "
        f"and fused Triton kernels (_rms_norm_fwd, _rope_embedding_fwd, _swiglu_fwd) verified!"
    )


def run_parity(
    stage: str = "all",
    num_tokens: int = 32,
    logit_tol: float = 0.25,
    attn_impl: str = "auto",
) -> None:
    stage_dirs = discover_stages(CUSTOM_SERVING_DIR, stage)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    bench_cfg = cfg.get("benchmarks", {})

    model_name: str = model_cfg.get("name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    dtype_str: str = model_cfg.get("dtype", "bfloat16")
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.bfloat16)
    prompt_text: str = bench_cfg.get(
        "prompt",
        "Write a clean, well-documented Python module implementing a high-performance "
        "matrix multiplication benchmark.",
    ).strip()

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids: list[int] = tokenizer.encode(
        formatted_prompt, add_special_tokens=False
    )
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    stage_impls = {
        stage_dir: (
            _default_attn_impl(stage_dir) if attn_impl == "auto" else attn_impl
        )
        for stage_dir in stage_dirs
    }
    needs_extra_batch_refs = any(
        _stage_sort_key(d)[0] >= 4
        or "BlockAllocator" in (d / "engine.py").read_text(encoding="utf-8")
        for d in stage_dirs
    )
    extra_prompt_ids_list: list[list[int]] = []
    if needs_extra_batch_refs:
        # Build 3 extra prompts of distinct lengths (~18, ~45, ~90 tokens)
        short_raw_ids = tokenizer.encode(_EXTRA_STAGE4_PROMPTS[0], add_special_tokens=False)
        for idx, extra_text in enumerate(_EXTRA_STAGE4_PROMPTS):
            if idx == 0 and len(short_raw_ids) >= 8:
                extra_prompt_ids_list.append(short_raw_ids)
            else:
                extra_fmt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": extra_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                extra_prompt_ids_list.append(
                    tokenizer.encode(extra_fmt, add_special_tokens=False)
                )

    print(f"Model:        {model_name} ({dtype_str})")
    print(f"Prompt chars: {len(prompt_text)} | Prompt tokens: {len(prompt_ids)}")
    print(f"Stages:       {[d.name for d in stage_dirs]}")

    # 1. Compute each required HuggingFace reference once before loading custom stages.
    hf_refs: dict[str, tuple[torch.Tensor, list[int], list[list[int]]]] = {}
    for impl_name in dict.fromkeys(stage_impls.values()):
        hf_refs[impl_name] = _compute_hf_reference(
            model_name=model_name,
            torch_dtype=torch_dtype,
            device=device,
            input_tensor=input_tensor,
            prompt_len=len(prompt_ids),
            num_tokens=num_tokens,
            pad_token_id=tokenizer.eos_token_id,
            attn_impl=impl_name,
            extra_prompt_ids_list=extra_prompt_ids_list if needs_extra_batch_refs else None,
        )

    # 2. Evaluate each requested custom stage sequentially
    for stage_dir in stage_dirs:
        stage_name = stage_dir.name
        impl_name = stage_impls[stage_dir]
        hf_prefill_logits, hf_ids, extra_hf_ids = hf_refs[impl_name]
        print(f"\n--- Verifying {stage_name} ---")
        print(f"[{stage_name}] Reference: HuggingFace attn_implementation={impl_name}")
        model_mod, engine_mod = load_stage_modules(stage_dir)
        Qwen2Model = model_mod.Qwen2Model
        NaiveEngine = engine_mod.NaiveEngine
        is_paged_stage = hasattr(engine_mod, "BlockAllocator")

        engine = None
        try:
            if is_paged_stage:
                engine = NaiveEngine(
                    model_name=model_name,
                    dtype=dtype_str,
                    device=device,
                    num_blocks=256,
                )
            else:
                engine = NaiveEngine(
                    model_name=model_name,
                    dtype=dtype_str,
                    device=device,
                )
            assert isinstance(engine.model, Qwen2Model), (
                f"[{stage_name}] Expected engine.model to be an instance of "
                f"{Qwen2Model}, got {type(engine.model)}"
            )

            if hasattr(engine.model.layers[0].mlp, "gate_up_proj"):
                _verify_stage5_fused_ops(
                    stage_name=stage_name,
                    model_mod=model_mod,
                    model=engine.model,
                    model_name=model_name,
                    torch_dtype=torch_dtype,
                    device=device,
                )

            if hasattr(engine, "graph_runner"):
                engine.warmup()
                assert engine.graph_runner.is_captured, (
                    f"[{stage_name}] Expected DecodeGraphRunner.is_captured=True after warmup()"
                )
                assert engine.allocator.num_free_blocks == engine.allocator.num_blocks, (
                    f"[{stage_name}] Expected all {engine.allocator.num_blocks} blocks free after "
                    f"warmup(), got {engine.allocator.num_free_blocks}"
                )
                print(
                    f"[{stage_name}] Warmup + CUDA graph capture verified "
                    f"(buckets={engine.graph_runner.buckets}, "
                    f"{engine.allocator.num_free_blocks}/{engine.allocator.num_blocks} blocks free)"
                )

            if is_paged_stage:
                allocator = engine.allocator
                assert allocator.num_free_blocks == allocator.num_blocks, (
                    f"[{stage_name}] Expected {allocator.num_blocks} free blocks initially, "
                    f"got {allocator.num_free_blocks}"
                )
                needed_blocks = allocator.blocks_needed_for(len(prompt_ids))
                allocated_blocks = allocator.allocate(needed_blocks)
                try:
                    block_table = torch.tensor(
                        [allocated_blocks], dtype=torch.int32, device=device
                    )
                    context_lengths = torch.tensor(
                        [len(prompt_ids)], dtype=torch.int32, device=device
                    )
                    with torch.inference_mode():
                        logits_tensor = engine.model(
                            input_tensor,
                            kv_cache=allocator.kv_cache,
                            block_table=block_table,
                            context_lengths=context_lengths,
                            last_token_only=True,
                            max_context_len=len(prompt_ids),
                        )
                    custom_prefill_logits = logits_tensor[:, -1, :].float().cpu()
                    del logits_tensor
                finally:
                    allocator.free(allocated_blocks)
                assert allocator.num_free_blocks == allocator.num_blocks, (
                    f"[{stage_name}] Expected all {allocator.num_blocks} blocks free after "
                    f"prefill check, got {allocator.num_free_blocks}"
                )
            else:
                with torch.inference_mode():
                    out = engine.model(input_tensor, last_token_only=True)

                if isinstance(out, tuple):
                    logits_tensor, kv = out
                    model_config = engine.model.config
                    assert isinstance(kv, (list, tuple)), (
                        f"[{stage_name}] Expected KV cache list/tuple, got {type(kv)}"
                    )
                    assert len(kv) == model_config.num_hidden_layers, (
                        f"[{stage_name}] Expected {model_config.num_hidden_layers} KV layers, "
                        f"got {len(kv)}"
                    )
                    expected_kv_shape = (
                        1,
                        model_config.num_key_value_heads,
                        len(prompt_ids),
                        model_config.head_dim,
                    )
                    for layer_idx, layer_kv in enumerate(kv):
                        assert isinstance(layer_kv, tuple) and len(layer_kv) == 2, (
                            f"[{stage_name}] Layer {layer_idx} KV entry must be a (k, v) tuple"
                        )
                        k, v = layer_kv
                        assert tuple(k.shape) == expected_kv_shape, (
                            f"[{stage_name}] Layer {layer_idx} k shape {tuple(k.shape)} "
                            f"!= {expected_kv_shape}"
                        )
                        assert tuple(v.shape) == expected_kv_shape, (
                            f"[{stage_name}] Layer {layer_idx} v shape {tuple(v.shape)} "
                            f"!= {expected_kv_shape}"
                        )
                    del kv
                else:
                    logits_tensor = out

                custom_prefill_logits = logits_tensor[:, -1, :].float().cpu()
                del logits_tensor, out

            max_abs_logit_diff = float(
                (hf_prefill_logits - custom_prefill_logits).abs().max().item()
            )
            print(
                f"[{stage_name}] Single-step prefill max-abs logit diff: "
                f"{max_abs_logit_diff:.6f} (tol={logit_tol})"
            )
            assert max_abs_logit_diff <= logit_tol, (
                f"[{stage_name}] Prefill max-abs logit diff {max_abs_logit_diff:.6f} "
                f"exceeded tolerance {logit_tol}"
            )

            if is_paged_stage:
                async def _run_single_async() -> list[int]:
                    return await _run_with_scheduler(
                        engine,
                        engine.generate_tokens_async(
                            prompt_ids,
                            max_tokens=num_tokens,
                            temperature=0.0,
                            ignore_eos=True,
                        ),
                    )  # type: ignore[return-value]

                custom_ids = asyncio.run(_run_single_async())
                assert engine.allocator.num_free_blocks == engine.allocator.num_blocks, (
                    f"[{stage_name}] Expected all {engine.allocator.num_blocks} blocks free "
                    f"after single-request generation, got {engine.allocator.num_free_blocks}"
                )
            else:
                custom_ids = list(
                    engine.generate_tokens(
                        prompt_ids,
                        max_tokens=num_tokens,
                        temperature=0.0,
                        ignore_eos=True,
                    )
                )

            assert len(custom_ids) == num_tokens, (
                f"[{stage_name}] Expected {num_tokens} custom tokens, got {len(custom_ids)}"
            )

            if custom_ids != hf_ids:
                print(
                    f"MISMATCH DETECTED between custom {stage_name} and HuggingFace {impl_name}:"
                )
                for step_idx, (c_id, h_id) in enumerate(zip(custom_ids, hf_ids)):
                    marker = "==" if c_id == h_id else "!="
                    c_tok = repr(tokenizer.decode([c_id], skip_special_tokens=False))
                    h_tok = repr(tokenizer.decode([h_id], skip_special_tokens=False))
                    print(
                        f"  Step {step_idx:02d}: custom={c_id:6d} ({c_tok}) "
                        f"{marker} hf={h_id:6d} ({h_tok})"
                    )
                raise AssertionError(
                    f"[{stage_name}] Greedy token parity failed: "
                    f"custom_ids={custom_ids} != hf_ids={hf_ids}"
                )

            decoded_output = tokenizer.decode(custom_ids, skip_special_tokens=False)
            print(
                f"[{stage_name}] PASS: 100% greedy token parity verified across "
                f"all {num_tokens} tokens!"
            )
            print(f"[{stage_name}] Token IDs: {custom_ids}")
            print(f"[{stage_name}] Decoded output:\n{decoded_output}")

            if is_paged_stage:
                batch_prompt_ids = [prompt_ids, *extra_prompt_ids_list]
                batch_hf_refs = [hf_ids, *extra_hf_ids]
                batch_cases: list[tuple[str, list[list[int]], list[list[int]]]] = [
                    ("", batch_prompt_ids, batch_hf_refs)
                ]
                if hasattr(engine, "graph_runner") and len(batch_prompt_ids) >= 3:
                    bucket_for_3 = engine.graph_runner.select_bucket(3)
                    assert bucket_for_3 == 4, (
                        f"[{stage_name}] Expected select_bucket(3) == 4, got {bucket_for_3}"
                    )
                    batch_cases.append(
                        (
                            f" (padded into bucket B={bucket_for_3})",
                            batch_prompt_ids[:3],
                            batch_hf_refs[:3],
                        )
                    )

                for case_label, case_prompts, case_refs in batch_cases:
                    prompt_lens = [len(p) for p in case_prompts]
                    print(
                        f"[{stage_name}] Verifying concurrent B={len(case_prompts)}{case_label} "
                        f"heterogeneous-length batched parity (prompt lengths={prompt_lens})..."
                    )

                    async def _run_batched_async(
                        prompts_to_run: list[list[int]] = case_prompts,
                    ) -> list[list[int]]:
                        async def _gather_all() -> list[list[int]]:
                            return list(
                                await asyncio.gather(
                                    *(
                                        engine.generate_tokens_async(
                                            p_ids,
                                            max_tokens=num_tokens,
                                            temperature=0.0,
                                            ignore_eos=True,
                                        )
                                        for p_ids in prompts_to_run
                                    )
                                )
                            )

                        return await _run_with_scheduler(engine, _gather_all())  # type: ignore[return-value]

                    batched_outputs = asyncio.run(_run_batched_async())
                    assert engine.allocator.num_free_blocks == engine.allocator.num_blocks, (
                        f"[{stage_name}] Expected all {engine.allocator.num_blocks} blocks free "
                        f"after B={len(case_prompts)} batched generation, got "
                        f"{engine.allocator.num_free_blocks}"
                    )
                    for req_idx, (b_ids, ref_ids, p_len) in enumerate(
                        zip(batched_outputs, case_refs, prompt_lens)
                    ):
                        assert len(b_ids) == num_tokens, (
                            f"[{stage_name}] Request {req_idx} (len={p_len}) expected "
                            f"{num_tokens} tokens, got {len(b_ids)}"
                        )
                        if b_ids != ref_ids:
                            raise AssertionError(
                                f"[{stage_name}] Concurrent B={len(case_prompts)} parity failed "
                                f"on request {req_idx} (prompt_len={p_len}): "
                                f"custom={b_ids} != hf={ref_ids}"
                            )
                    print(
                        f"[{stage_name}] PASS: Concurrent B={len(case_prompts)}{case_label} heterogeneous "
                        f"batched parity (32/32 greedy tokens x {len(case_prompts)} requests) "
                        f"and 100% block reclamation ({engine.allocator.num_free_blocks}/"
                        f"{engine.allocator.num_blocks} free) verified!"
                    )
        finally:
            if engine is not None:
                del engine
            torch.cuda.empty_cache()


def test_greedy_parity() -> None:
    run_parity(stage="all", num_tokens=32, logit_tol=0.25, attn_impl="auto")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify custom-serving stage(s) against HuggingFace reference."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage directory name under custom-serving/ (e.g. stage-0-baseline, "
        "stage-1-kv-cache, stage-2-tiled-attn, stage-3-split-k, stage-4-paged-kv, "
        "stage-5-fused-ops, stage-6-cuda-graphs) or 'all' (default: all).",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=32,
        help="Number of greedy tokens to generate and compare (default: 32).",
    )
    parser.add_argument(
        "--logit-tol",
        type=float,
        default=0.25,
        help="Maximum allowed single-step prefill abs logit diff (default: 0.25).",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        choices=["auto", "eager", "sdpa"],
        default="auto",
        help="HuggingFace reference attn_implementation: 'auto' uses eager for "
        "Stages 0-1 and sdpa for Stage 2+ (default: auto).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_parity(
        stage=args.stage,
        num_tokens=args.num_tokens,
        logit_tol=args.logit_tol,
        attn_impl=args.attn_impl,
    )
