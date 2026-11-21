from __future__ import annotations

import argparse
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


def run_parity(
    stage: str = "all",
    num_tokens: int = 32,
    logit_tol: float = 0.25,
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

    print(f"Model:        {model_name} ({dtype_str})")
    print(f"Prompt chars: {len(prompt_text)} | Prompt tokens: {len(prompt_ids)}")
    print(f"Stages:       {[d.name for d in stage_dirs]}")

    # 1. Reference: HuggingFace AutoModelForCausalLM with eager attention and no KV cache
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    with torch.inference_mode():
        hf_prefill_logits = (
            hf_model(input_tensor, use_cache=False).logits[:, -1, :].float().cpu()
        )
        gen_cfg = GenerationConfig(
            do_sample=False,
            max_new_tokens=num_tokens,
            min_new_tokens=num_tokens,
            repetition_penalty=1.0,
            use_cache=False,
            eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        hf_out = hf_model.generate(
            input_tensor,
            generation_config=gen_cfg,
        )

    hf_ids: list[int] = hf_out[0, len(prompt_ids) :].tolist()
    del hf_out
    del hf_model
    torch.cuda.empty_cache()

    assert len(hf_ids) == num_tokens, (
        f"Expected {num_tokens} HF tokens, got {len(hf_ids)}"
    )

    # 2. Evaluate each requested custom stage sequentially
    for stage_dir in stage_dirs:
        stage_name = stage_dir.name
        print(f"\n--- Verifying {stage_name} ---")
        model_mod, engine_mod = load_stage_modules(stage_dir)
        Qwen2Model = model_mod.Qwen2Model
        NaiveEngine = engine_mod.NaiveEngine

        engine = None
        try:
            engine = NaiveEngine(
                model_name=model_name,
                dtype=dtype_str,
                device=device,
            )
            assert isinstance(engine.model, Qwen2Model), (
                f"[{stage_name}] Expected engine.model to be an instance of "
                f"{Qwen2Model}, got {type(engine.model)}"
            )

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
                    f"MISMATCH DETECTED between custom {stage_name} and HuggingFace eager:"
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
        finally:
            if engine is not None:
                del engine
            torch.cuda.empty_cache()


def test_greedy_parity() -> None:
    run_parity(stage="all", num_tokens=32, logit_tol=0.25)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify custom-serving stage(s) against HuggingFace eager reference."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage directory name under custom-serving/ (e.g. stage-0-baseline, "
        "stage-1-kv-cache, stage-2-tiled-attn) or 'all' (default: all).",
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_parity(
        stage=args.stage,
        num_tokens=args.num_tokens,
        logit_tol=args.logit_tol,
    )
