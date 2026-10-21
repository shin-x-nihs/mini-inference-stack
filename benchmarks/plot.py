#!/usr/bin/env python3
import glob
import json
import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#00F5D4", "#FF4D6D", "#FFB703", "#3A86FF", "#8AC926", "#B5179E"]
BG_DARK = "#0D1117"
PANEL_BG = "#161B22"
GRID_COLOR = "#30363D"
TEXT_COLOR = "#E6EDF3"

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "legend.facecolor": BG_DARK,
    "legend.edgecolor": GRID_COLOR,
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
})


def load_stage_mode(stage: str, prefix: str, x_key: str) -> pd.DataFrame:
    pattern = os.path.join("benchmarks", "results", stage, f"{prefix}*.json")
    rows = []
    for path in glob.glob(pattern):
        match = re.search(rf"{prefix}([0-9.]+)\.json$", os.path.basename(path))
        if not match:
            continue
        x_val = float(match.group(1))
        if x_val.is_integer():
            x_val = int(x_val)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        tpot_p50 = float(d["p50_tpot_ms"])
        rows.append({
            "stage": stage,
            x_key: x_val,
            "model": d.get("model_id", ""),
            "completed": f"{d['completed']}/{d['num_prompts']}",
            "output_tok_s": round(float(d["output_throughput"]), 1),
            "total_tok_s": round(float(d["total_token_throughput"]), 1),
            "interactivity_tok_s_user": round(1000.0 / tpot_p50, 1) if tpot_p50 > 0 else 0.0,
            "ttft_p50_ms": round(float(d["p50_ttft_ms"]), 1),
            "ttft_p99_ms": round(float(d["p99_ttft_ms"]), 1),
            "tpot_p50_ms": round(tpot_p50, 2),
            "tpot_p99_ms": round(float(d["p99_tpot_ms"]), 2),
            "itl_p50_ms": round(float(d["p50_itl_ms"]), 2),
            "itl_p99_ms": round(float(d["p99_itl_ms"]), 2),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(x_key).reset_index(drop=True)


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_dashboard(dfs: dict[str, pd.DataFrame], x_key: str, x_label: str, title_prefix: str, out_png: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200)
    model_name = next(iter(dfs.values()))["model"].iloc[0]
    fig.suptitle(
        f"INFERENCE BENCHMARK // {title_prefix.upper()}  [{model_name}]",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax_pareto, ax_tput = axes[0, 0], axes[0, 1]
    ax_ttft, ax_decode = axes[1, 0], axes[1, 1]

    for idx, (stage, df) in enumerate(dfs.items()):
        color = PALETTE[idx % len(PALETTE)]
        x = df[x_key]

        ax_pareto.plot(
            df["interactivity_tok_s_user"],
            df["output_tok_s"],
            marker="o",
            linewidth=2.4,
            color=color,
            label=stage,
        )
        for _, r in df.iterrows():
            ax_pareto.annotate(
                f"{x_key[0].upper()}={r[x_key]}",
                (r["interactivity_tok_s_user"], r["output_tok_s"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                color=color,
            )

        ax_tput.plot(x, df["output_tok_s"], marker="o", linewidth=2.4, color=color, label=f"{stage} (Output)")
        ax_tput.plot(x, df["total_tok_s"], marker=".", linestyle=":", linewidth=1.5, color=color, alpha=0.7, label=f"{stage} (Total)")

        ax_ttft.plot(x, df["ttft_p50_ms"], marker="s", linewidth=2.2, color=color, label=f"{stage} P50")
        ax_ttft.plot(x, df["ttft_p99_ms"], marker="s", linestyle="--", linewidth=1.8, color=color, alpha=0.75, label=f"{stage} P99")

        ax_decode.plot(x, df["tpot_p50_ms"], marker="^", linewidth=2.2, color=color, label=f"{stage} TPOT P50")
        ax_decode.plot(x, df["itl_p99_ms"], marker="v", linestyle="--", linewidth=1.8, color=color, alpha=0.75, label=f"{stage} ITL P99")

    ax_pareto.set_title("Pareto Frontier: Throughput vs. Interactivity", fontweight="bold")
    ax_pareto.set_xlabel("User Interactivity (tok/s/user = 1000 / P50 TPOT)")
    ax_pareto.set_ylabel("Aggregate Output Throughput (tok/s)")

    ax_tput.set_title(f"System Throughput vs. {x_label}", fontweight="bold")
    ax_tput.set_xlabel(x_label)
    ax_tput.set_ylabel("Tokens / sec")

    ax_ttft.set_title(f"Prefill & Queue Latency (TTFT) vs. {x_label}", fontweight="bold")
    ax_ttft.set_xlabel(x_label)
    ax_ttft.set_ylabel("TTFT Latency (ms) [Log Scale]")
    ax_ttft.set_yscale("log")

    ax_decode.set_title(f"Decode & Tail Jitter (TPOT / ITL) vs. {x_label}", fontweight="bold")
    ax_decode.set_xlabel(x_label)
    ax_decode.set_ylabel("Token Latency (ms)")

    for ax in axes.flat:
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, facecolor=BG_DARK)
    plt.close(fig)


def main() -> int:
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["stage-0-vllm"]
    out_dir = (
        os.path.join("benchmarks", "results", stages[0])
        if len(stages) == 1
        else os.path.join("benchmarks", "results", "comparisons")
    )
    os.makedirs(out_dir, exist_ok=True)

    modes = [
        ("bench_c", "concurrency", "Concurrency (In-Flight Requests)", "Closed-Loop Saturation", "closed_loop"),
        ("bench_qps", "qps", "Arrival Rate (QPS)", "Open-Loop Queue Buildup", "open_loop"),
    ]

    md_sections = []
    for prefix, x_key, x_label, title_prefix, file_tag in modes:
        dfs = {s: load_stage_mode(s, prefix, x_key) for s in stages}
        dfs = {s: df for s, df in dfs.items() if not df.empty}
        if not dfs:
            continue

        combined_df = pd.concat(dfs.values(), ignore_index=True)
        md_table = to_markdown_table(combined_df)
        section = f"### {title_prefix} ({', '.join(stages)})\n\n{md_table}\n"
        md_sections.append(section)
        print("\n" + section)

        out_png = os.path.join(out_dir, f"{file_tag}_dashboard.png")
        render_dashboard(dfs, x_key, x_label, title_prefix, out_png)
        print(f"Saved plot: {out_png}")

    if md_sections:
        md_path = os.path.join(out_dir, "summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_sections))
        print(f"Saved summary table: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())