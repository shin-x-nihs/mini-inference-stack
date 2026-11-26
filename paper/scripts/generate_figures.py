#!/usr/bin/env python3
"""Generate publication-grade 2-column LaTeX figures for the mini-inference-stack paper.

Reads benchmark & profiling JSONs directly from benchmarks/results/<stage>/ using the
exact keys, units, and roofline ceiling math from benchmarks/plot_dashboard.py.

Outputs:
  - paper/latex/figures/fig1_pareto.pdf
  - paper/latex/figures/fig2_context_scaling.pdf
  - paper/latex/figures/fig3_nsys_bottleneck.pdf
  - paper/latex/figures/fig4_roofline.pdf
  - paper/scripts/preview/fig{1..4}_*.png  (300 DPI inspection previews)
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
OUT_PDF_DIR = REPO_ROOT / "paper" / "latex" / "figures"
OUT_PNG_DIR = REPO_ROOT / "paper" / "scripts" / "preview"

# ---------------------------------------------------------------------------
# Publication Matplotlib Styling (2-column IEEE/ACM/MLSys aesthetic)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "legend.title_fontsize": 7.6,
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "pdf.fonttype": 42,   # TrueType fonts embedded in PDF
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# Colorblind-friendly, high-contrast academic palette on white background
# Custom track (S0 -> S6): Warm-to-Teal progression, solid lines, circles/squares
# vLLM track (S0 -> S2): Cool Navy/Purple/Magenta, dashed lines, diamonds/triangles
STAGE_STYLE: dict[str, dict] = {
    "stage-0-custom": {"label": "S0-C (PyTorch)",         "short": "S0-C", "color": "#c0392b", "ls": "-",  "marker": "o", "ms": 3.8, "lw": 1.25, "z": 3},
    "stage-1-custom": {"label": "S1-C (+KV Cache)",       "short": "S1-C", "color": "#d95f02", "ls": "-",  "marker": "s", "ms": 3.8, "lw": 1.25, "z": 3},
    "stage-2-custom": {"label": "S2-C (+FlashAttn-2)",    "short": "S2-C", "color": "#e6ab02", "ls": "-",  "marker": "^", "ms": 4.0, "lw": 1.25, "z": 3},
    "stage-3-custom": {"label": "S3-C (+Split-K)",        "short": "S3-C", "color": "#a6761d", "ls": "-",  "marker": "v", "ms": 4.0, "lw": 1.25, "z": 3},
    "stage-4-custom": {"label": "S4-C (+Paged/ContBatch)","short": "S4-C", "color": "#66a61e", "ls": "-",  "marker": "P", "ms": 4.3, "lw": 1.45, "z": 4},
    "stage-5-custom": {"label": "S5-C (+Fused Triton)",   "short": "S5-C", "color": "#1b9e77", "ls": "-",  "marker": "X", "ms": 4.3, "lw": 1.55, "z": 5},
    "stage-6-custom": {"label": "S6-C (+CUDA Graphs)",    "short": "S6-C", "color": "#0072b2", "ls": "-",  "marker": "o", "ms": 4.8, "lw": 2.00, "z": 7},
    "stage-0-vllm":   {"label": "S0-vLLM (Baseline)",     "short": "S0-vLLM", "color": "#1f2937", "ls": "--", "marker": "D", "ms": 4.2, "lw": 1.65, "z": 6},
    "stage-1-vllm":   {"label": "S1-vLLM (SpecDec)",      "short": "S1-vLLM", "color": "#7570b3", "ls": ":",  "marker": "d", "ms": 4.2, "lw": 1.35, "z": 4},
    "stage-2-vllm":   {"label": "S2-vLLM (PrefixCache)",  "short": "S2-vLLM", "color": "#cc79a7", "ls": "-.", "marker": "D", "ms": 4.4, "lw": 1.75, "z": 6},
}

STAGE_ORDER = [
    "stage-0-custom",
    "stage-1-custom",
    "stage-2-custom",
    "stage-3-custom",
    "stage-4-custom",
    "stage-5-custom",
    "stage-6-custom",
    "stage-0-vllm",
    "stage-1-vllm",
    "stage-2-vllm",
]


def extract_int(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def effective_tflops(k: dict) -> float:
    """Exact effective_tflops logic from benchmarks/plot_dashboard.py."""
    tflops = k.get("attained_tflops", 0.0) or 0.0
    if tflops <= 0.0:
        ai = k.get("arithmetic_intensity_flop_byte", 0.0) or 0.0
        dram = k.get("achieved_dram_gbps", 0.0) or 0.0
        if ai > 0.0 and dram > 0.0:
            tflops = (ai * dram) / 1000.0
    return tflops


def load_all_data() -> dict[str, dict]:
    """Load pareto, context, nsys, and ncu JSONs for all stages."""
    data: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        sdir = RESULTS_DIR / stage
        if not sdir.exists():
            continue
        entry: dict = {"pareto": [], "context": [], "nsys": None, "ncu": None}

        for f in sorted(sdir.glob("pareto_c*.json"), key=lambda p: extract_int(p.name)):
            j = json.loads(f.read_text(encoding="utf-8"))
            entry["pareto"].append({
                "concurrency": j.get("max_concurrency", extract_int(f.name)),
                "throughput": j.get("output_throughput", 0.0),
                "p50_itl": j.get("p50_itl_ms", 0.0),
                "p90_itl": j.get("p90_itl_ms", 0.0),
                "p99_itl": j.get("p99_itl_ms", 0.0),
                "mean_tpot": j.get("mean_tpot_ms", 0.0),
            })

        for f in sorted(sdir.glob("context_l*.json"), key=lambda p: extract_int(p.name)):
            j = json.loads(f.read_text(encoding="utf-8"))
            entry["context"].append({
                "length": extract_int(f.name),
                "ttft": j.get("mean_ttft_ms", 0.0),
                "tpot": j.get("mean_tpot_ms", 0.0),
            })

        nsys_f = sdir / "nsys_metrics.json"
        if nsys_f.exists():
            entry["nsys"] = json.loads(nsys_f.read_text(encoding="utf-8"))

        ncu_f = sdir / "ncu_metrics.json"
        if ncu_f.exists():
            ncu_j = json.loads(ncu_f.read_text(encoding="utf-8"))
            for k in ncu_j.get("kernels", []):
                k["effective_tflops"] = effective_tflops(k)
            entry["ncu"] = ncu_j

        data[stage] = entry
    return data


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_PDF_DIR / f"{stem}.pdf"
    png_path = OUT_PNG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Figure 1: Throughput vs. Latency Pareto Frontier (P50 ITL & P99 ITL)
# ---------------------------------------------------------------------------
def make_fig1_pareto(data: dict[str, dict]) -> tuple[Path, Path]:
    fig, (ax_p50, ax_p99) = plt.subplots(1, 2, figsize=(7.16, 3.05), sharey=True)
    fig.subplots_adjust(wspace=0.08, top=0.77, bottom=0.15, left=0.08, right=0.985)

    for ax, metric, title in [
        (ax_p50, "p50_itl", "(a) Throughput vs. P50 ITL"),
        (ax_p99, "p99_itl", "(b) Throughput vs. P99 ITL (Tail)"),
    ]:
        for stage in STAGE_ORDER:
            st = STAGE_STYLE[stage]
            pts = data[stage]["pareto"]
            xs = [p["throughput"] for p in pts]
            ys = [p[metric] for p in pts]
            ax.plot(
                xs, ys,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markersize=st["ms"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=st["label"],
                zorder=st["z"],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(11, 4300)
        ax.set_ylim(3.15, 560.0)
        ax.set_xlabel("System Output Throughput (tok/s)")
        ax.set_title(title, pad=5)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:g}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}" if y >= 10 else f"{y:g}"))
        ax.set_xticks([15, 50, 150, 500, 1500, 3000])
        ax.set_yticks([5, 10, 20, 40, 80, 160, 350])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Concurrency endpoints on S6-C
        s6_pts = data["stage-6-custom"]["pareto"]
        ax.annotate(
            "C=1", xy=(s6_pts[0]["throughput"], s6_pts[0][metric]),
            xytext=(s6_pts[0]["throughput"] * 0.52, s6_pts[0][metric] * 0.85),
            fontsize=6.6, color=STAGE_STYLE["stage-6-custom"]["color"], fontweight="bold",
            ha="center",
        )

    ax_p50.set_ylabel("Inter-Token Latency — ITL (ms)")

    # Panel (a) annotations
    s6_c64 = data["stage-6-custom"]["pareto"][-1]
    s0v_c64 = data["stage-0-vllm"]["pareto"][-1]
    s2v_c64 = data["stage-2-vllm"]["pareto"][-1]

    ax_p50.annotate(
        "S6-C leads S0-vLLM (C=1..64)\n(1,771 tok/s, 16.8 ms @ C=64)",
        xy=(s6_c64["throughput"], s6_c64["p50_itl"]),
        xytext=(140, 3.38),
        fontsize=6.15, fontweight="bold", color=STAGE_STYLE["stage-6-custom"]["color"],
        ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-6-custom"]["color"], lw=0.85, connectionstyle="arc3,rad=0.18"),
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=STAGE_STYLE["stage-6-custom"]["color"], lw=0.6, alpha=0.95),
    )
    ax_p50.annotate(
        "S2-vLLM (C=64, h≈1)\n2,874 tok/s\n(prefix cache)",
        xy=(s2v_c64["throughput"], s2v_c64["p50_itl"]),
        xytext=(2150, 72),
        fontsize=6.5, fontweight="bold", color="#9e3b73",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#9e3b73", lw=0.85, connectionstyle="arc3,rad=-0.10"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cc79a7", lw=0.6, alpha=0.95),
    )
    ax_p50.annotate(
        "Single-req\n(S0–S3)",
        xy=(40, 22), xytext=(24, 150),
        fontsize=6.5, color="#555555", ha="center",
        arrowprops=dict(arrowstyle="->", color="#777777", lw=0.7),
    )

    # Panel (b) annotations
    ax_p99.annotate(
        "S0-vLLM P99 spike\n259 ms (C=64)",
        xy=(s0v_c64["throughput"], s0v_c64["p99_itl"]),
        xytext=(115, 290),
        fontsize=6.5, fontweight="bold", color="#1f2937",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#1f2937", lw=0.85, connectionstyle="arc3,rad=-0.10"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f2937", lw=0.6, alpha=0.95),
    )
    ax_p99.annotate(
        "S6-C (C=64):\n27.6 ms P99",
        xy=(s6_c64["throughput"], s6_c64["p99_itl"]),
        xytext=(2250, 125),
        fontsize=6.5, fontweight="bold", color=STAGE_STYLE["stage-6-custom"]["color"],
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-6-custom"]["color"], lw=0.85, connectionstyle="arc3,rad=0.14"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STAGE_STYLE["stage-6-custom"]["color"], lw=0.6, alpha=0.95),
    )
    # Direction indicator in bottom-right of Panel (b)
    ax_p99.text(
        0.97, 0.03, "Better (lower ITL, higher tok/s) ↘",
        transform=ax_p99.transAxes, fontsize=6.5, style="italic",
        color="#444444", ha="right", va="bottom",
    )

    # Shared top legend (2 clean rows, 5 cols)
    handles, labels = ax_p50.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=5,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#cccccc",
        framealpha=0.95,
        columnspacing=0.95,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.35,
    )

    return save_figure(fig, "fig1_pareto")


# ---------------------------------------------------------------------------
# Figure 2: Context Length Scaling (TTFT & TPOT)
# ---------------------------------------------------------------------------
def make_fig2_context_scaling(data: dict[str, dict]) -> tuple[Path, Path]:
    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(7.16, 3.05))
    fig.subplots_adjust(wspace=0.22, top=0.77, bottom=0.15, left=0.08, right=0.985)

    lengths_ticks = [128, 256, 512, 1024, 2048, 4096, 8192]
    lengths_labels = ["128", "256", "512", "1k", "2k", "4k", "8k"]

    for stage in STAGE_ORDER:
        st = STAGE_STYLE[stage]
        pts = data[stage]["context"]
        xs = [p["length"] for p in pts]
        ttfts = [p["ttft"] for p in pts]
        tpots = [p["tpot"] for p in pts]

        ax_ttft.plot(
            xs, ttfts,
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=st["ms"],
            markeredgecolor="white", markeredgewidth=0.45,
            label=st["label"], zorder=st["z"],
        )
        ax_tpot.plot(
            xs, tpots,
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=st["ms"],
            markeredgecolor="white", markeredgewidth=0.45,
            label=st["label"], zorder=st["z"],
        )

    for ax in (ax_ttft, ax_tpot):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(105, 10200)
        ax.set_xticks(lengths_ticks)
        ax.set_xticklabels(lengths_labels)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel (a): TTFT
    ax_ttft.set_ylim(28, 2600)
    ax_ttft.set_yticks([30, 60, 120, 250, 500, 1000, 2000])
    ax_ttft.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax_ttft.set_xlabel("Input Prompt Length $L$ (tokens)")
    ax_ttft.set_ylabel("Mean TTFT (ms)")
    ax_ttft.set_title("(a) Prefill Scaling: Mean TTFT vs. $L$", pad=5)

    # Annotations on Panel (a)
    ax_ttft.annotate(
        "S1-vLLM: 1,753 ms",
        xy=(8192, 1752.7), xytext=(1550, 1550),
        fontsize=6.6, fontweight="bold", color=STAGE_STYLE["stage-1-vllm"]["color"],
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-1-vllm"]["color"], lw=0.8),
    )
    ax_ttft.annotate(
        "S6-C: 629 ms\n(vs. S0-vLLM 674 ms)",
        xy=(8192, 629.3), xytext=(680, 620),
        fontsize=6.5, fontweight="bold", color=STAGE_STYLE["stage-6-custom"]["color"],
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-6-custom"]["color"], lw=0.85, connectionstyle="arc3,rad=0.10"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=STAGE_STYLE["stage-6-custom"]["color"], lw=0.6, alpha=0.95),
    )
    ax_ttft.annotate(
        "S2-vLLM: 198 ms\n(3.4× faster, h≈1\nprefix cache)",
        xy=(8192, 198.3), xytext=(4100, 56),
        fontsize=6.5, fontweight="bold", color="#9e3b73",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#9e3b73", lw=0.85, connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cc79a7", lw=0.6, alpha=0.95),
    )

    # Panel (b): TPOT
    ax_tpot.set_ylim(3.5, 360)
    ax_tpot.set_yticks([5, 10, 20, 40, 80, 160, 300])
    ax_tpot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax_tpot.set_xlabel("Context Sequence Length $L$ (tokens)")
    ax_tpot.set_ylabel("Mean TPOT (ms/tok)")
    ax_tpot.set_title("(b) Decode Scaling: Mean TPOT vs. $L$", pad=5)

    # Annotations on Panel (b)
    ax_tpot.annotate(
        "S0-C (no KV cache):\n266 ms/tok at $L$=2k\n(full-seq recompute)",
        xy=(2048, 265.51), xytext=(2500, 108),
        fontsize=6.5, fontweight="bold", color=STAGE_STYLE["stage-0-custom"]["color"],
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-0-custom"]["color"], lw=0.85, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STAGE_STYLE["stage-0-custom"]["color"], lw=0.6, alpha=0.95),
    )
    ax_tpot.annotate(
        "S1–S4-C flat (~20–24 ms)",
        xy=(4096, 22.75), xytext=(2400, 43),
        fontsize=6.5, color="#4d7c0f", fontweight="bold",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#66a61e", lw=0.8),
    )
    ax_tpot.annotate(
        "S6-C: 7.5–8.6 ms/tok (vs. S0-vLLM 9.0–10.7 ms)",
        xy=(2048, 8.31), xytext=(1024, 4.1),
        fontsize=6.4, fontweight="bold", color=STAGE_STYLE["stage-6-custom"]["color"],
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=STAGE_STYLE["stage-6-custom"]["color"], lw=0.85, connectionstyle="arc3,rad=-0.08"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=STAGE_STYLE["stage-6-custom"]["color"], lw=0.6, alpha=0.95),
    )

    handles, labels = ax_ttft.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=5,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#cccccc",
        framealpha=0.95,
        columnspacing=0.95,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.35,
    )

    return save_figure(fig, "fig2_context_scaling")


# ---------------------------------------------------------------------------
# Figure 3: Nsight Systems Bottleneck Shift (CPU Launch + GPU Active/Idle)
# ---------------------------------------------------------------------------
def make_fig3_nsys_bottleneck(data: dict[str, dict]) -> tuple[Path, Path]:
    stages = [
        "stage-1-custom",
        "stage-2-custom",
        "stage-3-custom",
        "stage-4-custom",
        "stage-5-custom",
        "stage-6-custom",
        "stage-0-vllm",
    ]
    x_labels = ["S1-C", "S2-C", "S3-C", "S4-C", "S5-C", "S6-C", "S0-vLLM*"]
    x = np.arange(len(stages), dtype=float)
    # Slight gap before vLLM reference bar
    x[-1] += 0.35

    calls = [data[s]["nsys"]["cpu_launch_calls"] for s in stages]
    launch_ms = [data[s]["nsys"]["cpu_launch_overhead_ms"] for s in stages]
    act_pct = [data[s]["nsys"]["gpu_active_pct"] for s in stages]
    idle_pct = [data[s]["nsys"]["gpu_idle_pct"] for s in stages]
    dec_step = [data[s]["nsys"]["decode_step_avg_ms"] for s in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))
    fig.subplots_adjust(wspace=0.47, top=0.73, bottom=0.15, left=0.075, right=0.925)

    bar_w = 0.56

    # --- Panel (a): CPU Launch Calls (log) + CUDA Launch API Time (ms) ---
    bar_colors = ["#4c78a8"] * 6 + ["#374151"]
    ax1.bar(x, calls, width=bar_w, color=bar_colors, edgecolor="#1f2937", linewidth=0.55, alpha=0.88, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(150, 380000)
    ax1.set_yticks([200, 1000, 5000, 25000, 100000])
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y/1000)}k" if y >= 1000 else f"{int(y)}"))
    ax1.yaxis.set_minor_locator(ticker.NullLocator())
    ax1.set_ylabel("CPU Launch Calls (log scale)", color="#2b547e")
    ax1.tick_params(axis="y", colors="#2b547e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=18, ha="right")
    ax1.set_title("(a) Host Dispatch & Launch API Time", pad=6)
    ax1.spines["top"].set_visible(False)

    # Bar value labels
    for xi, c in zip(x, calls):
        lbl = f"{c/1000:.1f}k" if c >= 1000 else f"{c}"
        ax1.text(xi, c * 1.18, lbl, ha="center", va="bottom", fontsize=6.3, color="#1f2937", fontweight="bold")

    # Secondary axis: CUDA Launch API Time (ms)
    ax1_r = ax1.twinx()
    ax1_r.plot(
        x, launch_ms,
        color="#d95f02", marker="o", markersize=4.5, linewidth=1.6,
        markeredgecolor="white", markeredgewidth=0.5, zorder=5,
        label="Launch API Time (ms)",
    )
    ax1_r.set_ylim(0, 1020)
    ax1_r.set_yticks([0, 200, 400, 600, 800, 1000])
    ax1_r.set_ylabel("CUDA Launch API Time (ms)", color="#b44d00")
    ax1_r.tick_params(axis="y", colors="#b44d00")
    ax1_r.grid(False)
    ax1_r.spines["top"].set_visible(False)

    # Vertical separator between Custom and vLLM
    sep_x = (x[5] + x[6]) / 2.0
    ax1.axvline(sep_x, color="#888888", linestyle=":", linewidth=0.9, zorder=2)

    # Callouts for 3.03x and 50.6x drops + S6 blocking graph calls
    ax1.annotate(
        "3.03× drop\n(Fusion)",
        xy=(x[4], calls[4]), xytext=(x[3] - 0.15, 2200),
        fontsize=6.5, fontweight="bold", color="#0f766e", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=0.85, connectionstyle="arc3,rad=0.18"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#0f766e", lw=0.6, alpha=0.95),
    )
    ax1.annotate(
        "50.6× drop\n(63 Graphs)",
        xy=(x[5], calls[5]), xytext=(x[5] - 1.18, 350),
        fontsize=6.5, fontweight="bold", color="#0072b2", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#0072b2", lw=0.85, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#0072b2", lw=0.6, alpha=0.95),
    )
    ax1_r.annotate(
        "285 ms\n(blocking\ngraph calls)",
        xy=(x[5], launch_ms[5]), xytext=(x[5] + 0.35, 620),
        fontsize=6.1, fontweight="bold", color="#b44d00", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#b44d00", lw=0.8, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#d95f02", lw=0.6, alpha=0.95),
        zorder=9,
    )

    # Compact legend cleanly above title for Panel (a)
    h1 = [
        Patch(facecolor="#4c78a8", edgecolor="#1f2937", linewidth=0.5, label="Launch Calls (Custom)"),
        Patch(facecolor="#374151", edgecolor="#1f2937", linewidth=0.5, label="Launch Calls (vLLM)"),
        Line2D([0], [0], color="#d95f02", marker="o", ms=4.2, lw=1.5, label="Launch API Time (ms)"),
    ]
    ax1.legend(
        handles=h1, loc="lower center", bbox_to_anchor=(0.50, 1.13),
        ncol=2, frameon=True, facecolor="#fafafa", edgecolor="#cccccc",
        fontsize=6.7, handlelength=1.4, columnspacing=0.8, borderpad=0.3,
    )

    # --- Panel (b): GPU Active vs. Idle (%) + Mean Decode Step (ms) ---
    ax2.bar(x, act_pct, width=bar_w, color="#1b9e77", edgecolor="#1f2937", linewidth=0.5, alpha=0.9, label="GPU Active (%)", zorder=3)
    ax2.bar(x, idle_pct, width=bar_w, bottom=act_pct, color="#e5e7eb", edgecolor="#6b7280", linewidth=0.5, hatch="///", alpha=0.9, label="GPU Idle (%)", zorder=3)
    ax2.set_ylim(0, 126)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("GPU Timeline Share (%)", color="#136f53")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=18, ha="right")
    ax2.set_title("(b) GPU Utilization & Decode Latency", pad=6)
    ax2.spines["top"].set_visible(False)
    ax2.axvline(sep_x, color="#888888", linestyle=":", linewidth=0.9, zorder=2)

    # Idle % labels on key transition bars S4, S5, S6, S0-vLLM
    idle_label_y = {3: 88.5, 4: 85.5, 5: 102.0, 6: 102.0}
    for idx in [3, 4, 5, 6]:
        ip = idle_pct[idx]
        y_pos = idle_label_y[idx]
        va = "center" if idx in (3, 4) else "bottom"
        ax2.text(
            x[idx], y_pos, f"{ip:.1f}%",
            ha="center", va=va, fontsize=6.3, fontweight="bold",
            color="#991b1b" if idx in (3, 4, 5) else "#374151",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85) if idx in (3, 4) else None,
            zorder=6,
        )

    # Secondary axis: Mean Decode Step (ms)
    ax2_r = ax2.twinx()
    ax2_r.plot(
        x, dec_step,
        color="#111827", marker="s", markersize=4.3, linewidth=1.65,
        markeredgecolor="white", markeredgewidth=0.5, zorder=7,
        label="Decode Step (ms)",
    )
    ax2_r.set_ylim(0, 34)
    ax2_r.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax2_r.set_ylabel("Mean Decode Step (ms/tok)", color="#111827")
    ax2_r.grid(False)
    ax2_r.spines["top"].set_visible(False)

    # Annotate decode step drop at S6-C (8.6 ms) vs S0-vLLM* (10.7 ms)
    ax2_r.annotate(
        "8.6 ms",
        xy=(x[5], dec_step[5]), xytext=(x[5] - 0.15, 4.0),
        fontsize=6.6, fontweight="bold", color="#111827", ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="#111827", lw=0.75),
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#111827", lw=0.55, alpha=0.94),
        zorder=8,
    )
    ax2.text(
        (x[0] + x[4]) / 2.0, 115.5,
        "Idle: 53.5% → 29.0% → 9.0%",
        fontsize=6.4, fontweight="bold", color="#991b1b", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.16", fc="#fff5f5", ec="#dc2626", lw=0.55, alpha=0.95),
        zorder=8,
    )

    h2 = [
        Patch(facecolor="#1b9e77", edgecolor="#1f2937", linewidth=0.5, label="GPU Active (%)"),
        Patch(facecolor="#e5e7eb", edgecolor="#6b7280", linewidth=0.5, hatch="///", label="GPU Idle (%)"),
        Line2D([0], [0], color="#111827", marker="s", ms=4.0, lw=1.5, label="Decode Step (ms)"),
    ]
    ax2.legend(
        handles=h2, loc="lower center", bbox_to_anchor=(0.50, 1.13),
        ncol=2, frameon=True, facecolor="#fafafa", edgecolor="#cccccc",
        fontsize=6.7, handlelength=1.4, columnspacing=0.8, borderpad=0.3,
    )

    return save_figure(fig, "fig3_nsys_bottleneck")


# ---------------------------------------------------------------------------
# Figure 4: Blackwell RTX 5080 Hardware Roofline (NCU Layer 0)
# ---------------------------------------------------------------------------
def make_fig4_roofline(data: dict[str, dict]) -> tuple[Path, Path]:
    # Extract hardware ceilings using exact logic from plot_dashboard.py
    peak_bw_gbps = 896.0
    peak_compute_tflops = 40.0
    for s in STAGE_ORDER:
        ncu = data.get(s, {}).get("ncu")
        if ncu and "hardware_ceilings" in ncu:
            c = ncu["hardware_ceilings"]
            peak_bw_gbps = c.get("estimated_peak_dram_gbps", peak_bw_gbps)
            peak_compute_tflops = c.get("estimated_peak_tensor_tflops", peak_compute_tflops)
            break

    peak_compute_gflops = peak_compute_tflops * 1000.0  # 40,000 GFLOP/s
    ridge_ai = peak_compute_gflops / peak_bw_gbps      # 44.64 FLOP/B

    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14)

    ai_min, ai_max = 0.12, 3800.0
    y_min, y_max = 35.0, 85000.0

    # Shaded Memory-Bound vs Compute-Bound regions
    ax.axvspan(ai_min, ridge_ai, color="#eff6ff", alpha=0.55, zorder=0)
    ax.axvspan(ridge_ai, ai_max, color="#fefce8", alpha=0.45, zorder=0)
    ax.axvline(ridge_ai, color="#6b7280", linestyle=":", linewidth=0.95, zorder=1)

    # Roofline ceiling curve (GFLOP/s)
    ceil_x = np.array([ai_min, ridge_ai, ai_max])
    ceil_y = np.array([peak_bw_gbps * ai_min, peak_compute_gflops, peak_compute_gflops])
    ax.plot(ceil_x, ceil_y, color="#b91c1c", linestyle="--", linewidth=1.7, zorder=2)

    # Ceiling labels
    ax.text(
        0.52, 0.52 * peak_bw_gbps * 1.48,
        f"DRAM BW Ceiling: {peak_bw_gbps:.0f} GB/s",
        fontsize=6.9, fontweight="bold", color="#991b1b",
        rotation=31.0, rotation_mode="anchor", ha="left", va="bottom",
    )
    ax.text(
        68.0, peak_compute_gflops * 1.15,
        f"Ref. BF16 Baseline: {peak_compute_tflops:.0f} TFLOP/s (config.yaml)",
        fontsize=6.9, fontweight="bold", color="#991b1b", ha="left", va="bottom",
    )
    ax.text(
        ridge_ai * 1.12, 48.0,
        f"Ridge Point\n{ridge_ai:.1f} FLOP/B",
        fontsize=6.6, color="#4b5563", ha="left", va="bottom",
    )
    ax.text(0.16, 48.0, "Memory-Bound Region", fontsize=6.9, style="italic", color="#1e40af", alpha=0.85)
    ax.text(380.0, 48.0, "Compute-Bound Region", fontsize=6.9, style="italic", color="#854d0e", alpha=0.85)

    # Representative stages to display cleanly without 7-way overplotting
    rep_stages = [
        ("stage-1-custom", "S1-C (Eager+KV)",    "#d95f02"),
        ("stage-2-custom", "S2-C (FlashAttn-2)", "#d4a017"),
        ("stage-3-custom", "S3-C (Split-K)",     "#1b9e77"),
        ("stage-6-custom", "S6-C (Fused+Graph)", "#0072b2"),
        ("stage-0-vllm",   "S0-vLLM (Ref)",      "#1f2937"),
    ]

    # Kernel categories & shapes
    def classify_kernel(k: dict) -> str | None:
        lbl = k.get("label", "").lower()
        phase = k.get("phase", "").lower()
        if "combine" in lbl:
            return None
        if phase == "prefill" and ("linear" in lbl or "gemm" in lbl):
            return "prefill_linear"
        if phase == "prefill" and ("attention" in lbl or "flash" in lbl):
            return "prefill_attn"
        if phase == "decode" and ("attention" in lbl or "flash" in lbl):
            return "decode_attn"
        if phase == "decode" and ("linear" in lbl or "gemv" in lbl or "gemm" in lbl):
            return "decode_linear"
        if phase == "prefill" and ("swiglu" in lbl or "rmsnorm" in lbl):
            return "prefill_elem"
        return None

    kernel_markers = {
        "prefill_linear": ("s", 54, "Prefill Linear (GEMM)"),
        "prefill_attn":   ("*", 92, "Prefill Attention"),
        "decode_attn":    ("^", 58, "Decode Attention"),
        "decode_linear":  ("D", 44, "Decode Linear (GEMV)"),
        "prefill_elem":   ("o", 26, "Prefill RMSNorm / SwiGLU"),
    }

    # Collect coordinates for transition arrows
    coords: dict[tuple[str, str], tuple[float, float, float, float]] = {}

    for stage_idx, (stage, _slabel, scolor) in enumerate(rep_stages):
        ncu = data[stage]["ncu"]
        for k in ncu.get("kernels", []):
            cat = classify_kernel(k)
            if cat is None:
                continue
            ai = k["arithmetic_intensity_flop_byte"]
            gflops = k["effective_tflops"] * 1000.0
            if ai <= 0 or gflops <= 0:
                continue
            coords[(stage, cat)] = (ai, gflops, k.get("execution_time_us", 0.0), k.get("achieved_dram_gbps", 0.0))
            marker, msize, _ = kernel_markers[cat]
            # Slight size hierarchy so overlapping points remain concentric
            scale = 1.18 - 0.08 * stage_idx
            alpha = 0.72 if cat == "prefill_elem" else 0.95
            ax.scatter(
                ai, gflops,
                s=msize * scale,
                c=scolor,
                marker=marker,
                edgecolors="white" if cat != "prefill_elem" else "#333333",
                linewidths=0.65 if cat != "prefill_elem" else 0.45,
                alpha=alpha,
                zorder=5 + stage_idx if cat != "prefill_elem" else 4,
            )

    # Arrow 1: Stage 1 -> Stage 2 Prefill Attention (FlashAttention-2: 11.8x AI jump)
    s1_pa = coords[("stage-1-custom", "prefill_attn")]
    s2_pa = coords[("stage-2-custom", "prefill_attn")]
    ax.annotate(
        "",
        xy=(s2_pa[0] * 0.88, s2_pa[1] * 0.96),
        xytext=(s1_pa[0] * 1.12, s1_pa[1] * 0.96),
        arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=1.35, connectionstyle="arc3,rad=-0.18"),
        zorder=10,
    )
    ax.text(
        480, 6200,
        "S1→S2 FlashAttn-2:\n135→1,598 FLOP/B (vs. S1 BMM)\n>4,100 µs total → 978 µs",
        fontsize=6.6, fontweight="bold", color="#92400e", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d97706", lw=0.65, alpha=0.94),
        zorder=11,
    )

    # Arrow 2: Stage 2 -> Stage 3 Decode Attention (Split-K: 3.3x combined speedup)
    s2_da = coords[("stage-2-custom", "decode_attn")]
    s3_da = coords[("stage-3-custom", "decode_attn")]
    ax.annotate(
        "",
        xy=(s3_da[0] * 1.15, s3_da[1] * 1.10),
        xytext=(s2_da[0] * 0.86, s2_da[1] * 0.92),
        arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=1.35, connectionstyle="arc3,rad=-0.22"),
        zorder=10,
    )
    ax.text(
        34, 155,
        "S2→S3 Split-K Decode:\n3.3× faster (129→39.7 µs)\nS2 AI inflated by BLOCK_M=16",
        fontsize=6.5, fontweight="bold", color="#0f766e", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1b9e77", lw=0.65, alpha=0.95),
        zorder=11,
    )

    # Cluster labels
    ax.annotate(
        "Prefill Linear GEMM\n(25.9–27.3 TFLOP/s)",
        xy=(961, 27209), xytext=(220, 26500),
        fontsize=6.5, fontweight="bold", color="#1f2937", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#4b5563", lw=0.75),
    )
    ax.annotate(
        "Decode Linear GEMV\n(AI≈1.0, ~291–325 GB/s)",
        xy=(1.0, 323), xytext=(0.16, 4600),
        fontsize=6.5, fontweight="bold", color="#1f2937", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#4b5563", lw=0.75, connectionstyle="arc3,rad=0.16"),
    )
    ax.annotate(
        "S0-vLLM Decode\nAttn (26.0 µs)",
        xy=(66.67, 5525), xytext=(15.5, 2900),
        fontsize=6.4, fontweight="bold", color="#1f2937", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#1f2937", lw=0.75, connectionstyle="arc3,rad=-0.12"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Attained Performance (GFLOP/s)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:g}"))
    ax.set_xticks([0.2, 1, 5, 20, 100, 500, 2000])
    ax.set_yticks([50, 200, 1000, 5000, 20000, 40000])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Dual compact legend above plot: Row 1 = Stage Colors, Row 2 = Kernel Shapes
    stage_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=col, markeredgecolor="white", markersize=6, label=lbl)
        for _, lbl, col in rep_stages
    ]
    kernel_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#555555", markeredgecolor="white", markersize=6.2 if m != "*" else 8.0, label=lbl)
        for m, _, lbl in kernel_markers.values()
    ]

    leg1 = fig.legend(
        handles=stage_handles, loc="upper center", bbox_to_anchor=(0.54, 1.005),
        ncol=5, frameon=False, fontsize=7.0, handletextpad=0.25, columnspacing=0.85,
    )
    fig.legend(
        handles=kernel_handles, loc="upper center", bbox_to_anchor=(0.54, 0.945),
        ncol=5, frameon=False, fontsize=6.9, handletextpad=0.25, columnspacing=0.75,
    )
    fig.add_artist(leg1)

    return save_figure(fig, "fig4_roofline")


# ---------------------------------------------------------------------------
# Figure 0: 7-Stage Additive Progression & vLLM Reference Track Overview
# ---------------------------------------------------------------------------
def make_fig0_progression() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.16, 1.92))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 27)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)

    # Track B header label
    ax.text(
        0.6, 25.5, "Track B: Custom Serving Engine (Strictly Additive, Stages 0 → 6)",
        fontsize=7.3, fontweight="bold", color="#111827", ha="left", va="center",
    )

    # Regime brackets above boxes
    regimes = [
        (0.6, 42.2, "Regime 1: Algorithmic & Memory Hierarchy", "#b45309", "#fffbeb"),
        (43.2, 70.8, "Regime 2: Occupancy & Batching", "#3f6212", "#f7fee7"),
        (71.8, 99.4, "Regime 3: Host-Dispatch Elim.", "#075985", "#f0f9ff"),
    ]
    for x0, x1, rlabel, rcol, rbg in regimes:
        rect = FancyBboxPatch(
            (x0, 20.6), x1 - x0, 3.0,
            boxstyle="round,pad=0.15,rounding_size=0.45",
            facecolor=rbg, edgecolor=rcol, linewidth=0.65, alpha=0.9, zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2.0, 22.1, rlabel,
            fontsize=6.3, fontweight="bold", color=rcol, ha="center", va="center", zorder=2,
        )

    # 7 stage boxes
    stages = [
        ("S0 Naive",      "Eager PyTorch",   "14.9 tok/s (C=1)",  "O(S²) decode",      "#c0392b", "#fef2f2"),
        ("S1 KV Cache",   "Memoize (K,V)",   "37.3 tok/s (C=1)",  "54.3% GPU idle",    "#d95f02", "#fff7ed"),
        ("S2 Triton FA-2","Online Softmax",  "44.0 tok/s (C=1)",  "−91% attn bytes",   "#c99200", "#fefce8"),
        ("S3 Split-K",    "Parallel Decode", "45.0 tok/s (C=1)",  "3.3× dec attn",     "#8c6216", "#faf5eb"),
        ("S4 Paged+Batch","Cont. Batching",  "1,260 tok/s (C=64)","91× lower TTFT",    "#4d7c0f", "#f4fce3"),
        ("S5 Fused Ops",  "RMS/RoPE/SwiGLU", "68 / 1,486 tok/s",  "3.0× fewer kern.",  "#0f766e", "#f0fdfa"),
        ("S6 CUDA Graphs","Bucketed B=1..64","113 / 1,771 tok/s", "9.0% GPU idle",     "#0072b2", "#eff6ff"),
    ]

    n = len(stages)
    margin = 0.6
    gap = 1.35
    box_w = (100.0 - 2 * margin - (n - 1) * gap) / n
    box_y = 9.2
    box_h = 10.6

    for i, (title, sub, metric, badge, col, bg) in enumerate(stages):
        x0 = margin + i * (box_w + gap)
        xc = x0 + box_w / 2.0

        # Main card
        card = FancyBboxPatch(
            (x0, box_y), box_w, box_h,
            boxstyle="round,pad=0.12,rounding_size=0.55",
            facecolor=bg, edgecolor=col, linewidth=1.15 if i == 6 else 0.85, zorder=2,
        )
        ax.add_patch(card)

        # Colored header strip
        hdr = FancyBboxPatch(
            (x0, box_y + box_h - 3.0), box_w, 3.0,
            boxstyle="round,pad=0.12,rounding_size=0.55",
            facecolor=col, edgecolor=col, linewidth=0.5, zorder=3,
        )
        ax.add_patch(hdr)
        ax.text(xc, box_y + box_h - 1.5, title, fontsize=6.5, fontweight="bold", color="white", ha="center", va="center", zorder=4)

        # Body lines
        ax.text(xc, box_y + 5.7, sub, fontsize=6.0, color="#1f2937", ha="center", va="center", zorder=4)
        ax.text(xc, box_y + 3.4, metric, fontsize=6.1, fontweight="bold", color="#111827", ha="center", va="center", zorder=4)
        ax.text(xc, box_y + 1.35, badge, fontsize=5.75, style="italic", color=col, fontweight="bold", ha="center", va="center", zorder=4)

        # Connecting arrow to next stage
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x0 + box_w + gap - 0.12, box_y + box_h / 2.0),
                xytext=(x0 + box_w + 0.12, box_y + box_h / 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#4b5563", lw=1.0, mutation_scale=7.5),
                zorder=5,
            )

    # Bottom bar: Track A (vLLM 0.31.0 Reference)
    ref_bg = FancyBboxPatch(
        (0.6, 0.7), 98.8, 6.7,
        boxstyle="round,pad=0.12,rounding_size=0.55",
        facecolor="#f8fafc", edgecolor="#64748b", linewidth=0.8, linestyle="--", zorder=1,
    )
    ax.add_patch(ref_bg)
    ax.text(
        1.8, 5.85, "Track A: vLLM 0.31.0 Reference (Feature Isolation — Each Branched from S0-vLLM)",
        fontsize=6.8, fontweight="bold", color="#334155", ha="left", va="center", zorder=2,
    )
    ax.text(
        98.4, 5.85, "S6-C matches/exceeds S0-vLLM (n=1)",
        fontsize=6.4, fontweight="bold", color="#0072b2", ha="right", va="center", zorder=2,
    )

    vllm_items = [
        (1.8,  30.6, "S0-vLLM Baseline",           "87.8 (C=1) | 1,232 (C=64) | 11.6% idle", "#1f2937", "#ffffff"),
        (33.8, 32.0, "S1-vLLM SpecDec (0.5B, K=5)","106.0 (C=1) | 872 (C=64) | 35.3% idle",  "#7570b3", "#f5f3ff"),
        (67.2, 31.2, "S2-vLLM Prefix Cache (h≈1)", "80.4 (C=1) | 2,874 (C=64) | −75% TTFT", "#9e3b73", "#fdf2f8"),
    ]
    for vx, vw, vtitle, vdesc, vcol, vbg in vllm_items:
        vc = FancyBboxPatch(
            (vx, 1.35), vw, 3.4,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            facecolor=vbg, edgecolor=vcol, linewidth=0.8, zorder=2,
        )
        ax.add_patch(vc)
        ax.text(vx + 0.9, 3.7, vtitle, fontsize=6.25, fontweight="bold", color=vcol, ha="left", va="center", zorder=3)
        ax.text(vx + 0.9, 2.15, vdesc, fontsize=5.85, color="#374151", ha="left", va="center", zorder=3)

    return save_figure(fig, "fig0_progression")


def main() -> None:
    warnings.filterwarnings("error", category=UserWarning)
    data = load_all_data()
    print(f"Loaded {len(data)} stages from {RESULTS_DIR}")

    outputs = [
        ("fig0_progression", make_fig0_progression()),
        ("fig1_pareto", make_fig1_pareto(data)),
        ("fig2_context_scaling", make_fig2_context_scaling(data)),
        ("fig3_nsys_bottleneck", make_fig3_nsys_bottleneck(data)),
        ("fig4_roofline", make_fig4_roofline(data)),
    ]
    for name, (pdf_p, png_p) in outputs:
        print(f"  [OK] {name}: {pdf_p} ({pdf_p.stat().st_size:,} B) | preview: {png_p}")


if __name__ == "__main__":
    main()
