from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from engine import NaiveEngine
from model import Qwen2Model

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def test_greedy_parity() -> None:
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

    num_tokens = 32
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
    del hf_model
    torch.cuda.empty_cache()

    # 2. Custom Stage 0 NaiveEngine (backed by from-scratch Qwen2Model)
    engine = NaiveEngine(
        model_name=model_name,
        dtype=dtype_str,
        device=device,
    )
    assert isinstance(engine.model, Qwen2Model)

    with torch.inference_mode():
        custom_prefill_logits = (
            engine.model(input_tensor, last_token_only=True)[:, -1, :].float().cpu()
        )

    max_abs_logit_diff = float(
        (hf_prefill_logits - custom_prefill_logits).abs().max().item()
    )
    print(f"Single-step prefill max-abs logit diff: {max_abs_logit_diff:.6f}")

    custom_ids = list(
        engine.generate_tokens(
            prompt_ids,
            max_tokens=num_tokens,
            temperature=0.0,
            ignore_eos=True,
        )
    )

    assert len(hf_ids) == num_tokens, (
        f"Expected {num_tokens} HF tokens, got {len(hf_ids)}"
    )
    assert len(custom_ids) == num_tokens, (
        f"Expected {num_tokens} custom tokens, got {len(custom_ids)}"
    )

    if custom_ids != hf_ids:
        print("MISMATCH DETECTED between custom Stage 0 and HuggingFace eager:")
        for step_idx, (c_id, h_id) in enumerate(zip(custom_ids, hf_ids)):
            marker = "==" if c_id == h_id else "!="
            c_tok = repr(tokenizer.decode([c_id], skip_special_tokens=False))
            h_tok = repr(tokenizer.decode([h_id], skip_special_tokens=False))
            print(
                f"  Step {step_idx:02d}: custom={c_id:6d} ({c_tok}) "
                f"{marker} hf={h_id:6d} ({h_tok})"
            )
        raise AssertionError(
            f"Greedy token parity failed: custom_ids={custom_ids} != hf_ids={hf_ids}"
        )

    decoded_output = tokenizer.decode(custom_ids, skip_special_tokens=False)
    print(f"PASS: 100% greedy token parity verified across all {num_tokens} tokens!")
    print(f"Token IDs: {custom_ids}")
    print(f"Decoded output:\n{decoded_output}")


if __name__ == "__main__":
    test_greedy_parity()
