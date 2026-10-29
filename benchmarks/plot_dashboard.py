import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
import plotly.graph_objects as go

# --- Red, White & Black Palette (SemiAnalysis-inspired) ---
BG_PAGE = "#0a0a0a"
BG_CARD = "#121212"
GRID_COLOR = "#222222"
TEXT_MAIN = "#ffffff"
TEXT_MUTED = "#a0a0a0"

STAGE_COLORS = [
    "#ff2a2a",  # Crimson Red (Primary)
    "#ffffff",  # Pure White (Secondary)
    "#ff7b54",  # Coral Red
    "#888888",  # Steel Grey
    "#b30000",  # Deep Blood Red
]


def extract_int(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def padded_range(values: list[float], pad: float = 1.15) -> list[float]:
    valid = [v for v in values if v is not None and v > 0]
    return [0.0, max(valid) * pad] if valid else [0.0, 1.0]


def canonical_kernel_info(label: str, phase: str) -> tuple[str, str, bool, int]:
    """
    Maps raw stage kernel labels (e.g. 'Decode Linear GEMV' vs 'Decode Linear GEMM')
    to a unified (canonical_group, short_plot_label, is_core, sort_order).
    """
    lbl = label.lower()
    p = phase.capitalize() if phase else "Other"
    p_offset = 0 if p == "Prefill" else 100

    if "linear" in lbl or "gemm" in lbl or "gemv" in lbl:
        return (f"{p} Linear (GEMM/GEMV)", f"{p} Linear", True, p_offset + 1)
    if "combine" in lbl:
        return (f"{p} Attention Combine", f"{p} Attn Combine", False, p_offset + 3)
    if "attention" in lbl or "flash" in lbl:
        return (f"{p} Attention", f"{p} Attn", True, p_offset + 2)
    if "rmsnorm" in lbl or "norm" in lbl:
        return (f"{p} Fused RMSNorm", f"{p} RMSNorm", False, p_offset + 4)
    if "swiglu" in lbl or "act_and_mul" in lbl or "silu" in lbl:
        return (f"{p} SwiGLU Activation", f"{p} SwiGLU", False, p_offset + 5)
    if "rope" in lbl or "rotary" in lbl:
        return (f"{p} RoPE Embedding", f"{p} RoPE", False, p_offset + 6)
    if "reshape_and_cache" in lbl or "cache" in lbl:
        return (f"{p} KV Cache Write", f"{p} KV Cache", False, p_offset + 7)
    return (label, label, False, p_offset + 50)


def effective_tflops(k: dict) -> float:
    """
    Returns attained_tflops, or reconstructs it from (AI * achieved_dram_gbps) / 1000
    when JSON rounding truncated small non-zero values (like Decode KV Cache Write) to 0.0.
    """
    tflops = k.get("attained_tflops", 0.0) or 0.0
    if tflops <= 0.0:
        ai = k.get("arithmetic_intensity_flop_byte", 0.0) or 0.0
        dram = k.get("achieved_dram_gbps", 0.0) or 0.0
        if ai > 0.0 and dram > 0.0:
            tflops = (ai * dram) / 1000.0
    return tflops


def load_stage_data(stage_dir: Path) -> dict:
    data = {
        "name": stage_dir.name,
        "pareto": [],
        "context": [],
        "nsys": None,
        "ncu": None,
    }

    # 1. Load Pareto files (pareto_c*.json)
    for f in sorted(stage_dir.glob("pareto_c*.json"), key=lambda p: extract_int(p.name)):
        with open(f, "r", encoding="utf-8") as fp:
            j = json.load(fp)
            data["pareto"].append({
                "concurrency": j.get("max_concurrency", extract_int(f.name)),
                "throughput": j.get("output_throughput", 0.0),
                "p99_itl": j.get("p99_itl_ms", 0.0),
                "p90_itl": j.get("p90_itl_ms", 0.0),
                "p50_itl": j.get("p50_itl_ms", 0.0),
                "mean_tpot": j.get("mean_tpot_ms", 0.0),
            })

    # 2. Load Context files (context_l*.json)
    for f in sorted(stage_dir.glob("context_l*.json"), key=lambda p: extract_int(p.name)):
        with open(f, "r", encoding="utf-8") as fp:
            j = json.load(fp)
            tpot = j.get("mean_tpot_ms", 0.0)
            data["context"].append({
                "length": extract_int(f.name),
                "ttft": j.get("mean_ttft_ms", 0.0),
                "tpot": tpot,
                "gen_speed": (1000.0 / tpot) if tpot > 0 else 0.0,
            })

    # 3. Load Nsight Systems metrics
    nsys_file = stage_dir / "nsys_metrics.json"
    if nsys_file.exists():
        with open(nsys_file, "r", encoding="utf-8") as fp:
            data["nsys"] = json.load(fp)

    # 4. Load Nsight Compute metrics
    ncu_file = stage_dir / "ncu_metrics.json"
    if ncu_file.exists():
        with open(ncu_file, "r", encoding="utf-8") as fp:
            ncu_data = json.load(fp)
            for k in ncu_data.get("kernels", []):
                group, short_lbl, is_core, order = canonical_kernel_info(
                    k.get("label", ""), k.get("phase", "")
                )
                k["canonical_group"] = group
                k["short_label"] = short_lbl
                k["is_core"] = is_core
                k["sort_order"] = order
                k["effective_tflops"] = effective_tflops(k)
            data["ncu"] = ncu_data

    return data


def base_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=17, color=TEXT_MAIN, family="Inter, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MUTED, family="Inter, monospace", size=12),
        margin=dict(l=60, r=30, t=82, b=55),
        height=435,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN, size=11),
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=TEXT_MAIN)),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            linecolor="#333333",
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=TEXT_MAIN)),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            linecolor="#333333",
        ),
    )


def dropdown_style() -> dict:
    return dict(
        type="dropdown",
        direction="down",
        x=1.0,
        y=1.11,
        xanchor="right",
        yanchor="top",
        bgcolor="#1a1a1a",
        bordercolor="#ff2a2a",
        borderwidth=1,
        font=dict(color=TEXT_MAIN, size=11),
    )


def build_plot1_pareto(stages: list[dict]) -> go.Figure:
    fig = go.Figure(layout=base_layout(
        "Throughput vs. Latency (System Capacity)",
        "System Throughput (tokens/s)",
        "ITL P99 (ms)"
    ))

    metrics = [
        ("ITL P99 (ms)", "p99_itl"),
        ("ITL P90 (ms)", "p90_itl"),
        ("ITL P50 (ms)", "p50_itl"),
        ("TPOT Mean (ms/token)", "mean_tpot"),
    ]

    for idx, s in enumerate(stages):
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        pts = s["pareto"]
        fig.add_trace(go.Scatter(
            x=[p["throughput"] for p in pts],
            y=[p["p99_itl"] for p in pts],
            text=[f"C={p['concurrency']}" for p in pts],
            textposition="top center",
            mode="lines+markers+text",
            name=s["name"],
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(width=1, color=BG_CARD)),
            hovertemplate="<b>%{fullData.name}</b> (%{text})<br>Throughput: %{x:.1f} tok/s<br>Latency: %{y:.2f} ms<extra></extra>",
        ))

    all_x = [p["throughput"] for s in stages for p in s["pareto"]]
    fig.update_xaxes(range=padded_range(all_x))
    fig.update_yaxes(range=padded_range([p["p99_itl"] for s in stages for p in s["pareto"]]))

    buttons = []
    for label, key in metrics:
        y_arrays = [[p[key] for p in s["pareto"]] for s in stages]
        flat_y = [val for arr in y_arrays for val in arr]
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"y": y_arrays},
                {"yaxis.title.text": label, "yaxis.range": padded_range(flat_y)}
            ],
        ))

    fig.update_layout(updatemenus=[{**dropdown_style(), "buttons": buttons}])
    return fig


def build_plot2_prefill(stages: list[dict]) -> go.Figure:
    fig = go.Figure(layout=base_layout(
        "Prompt Size vs. Time to First Token",
        "Input Prompt Length (tokens)",
        "Time to First Token - TTFT (ms)"
    ))

    for idx, s in enumerate(stages):
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        pts = s["context"]
        fig.add_trace(go.Scatter(
            x=[p["length"] for p in pts],
            y=[p["ttft"] for p in pts],
            mode="lines+markers",
            name=s["name"],
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            hovertemplate="<b>%{fullData.name}</b><br>Prompt: %{x} tok<br>TTFT: %{y:.2f} ms<extra></extra>",
        ))

    all_x = [p["length"] for s in stages for p in s["context"]]
    all_y = [p["ttft"] for s in stages for p in s["context"]]
    if all_x:
        fig.update_xaxes(range=[0, max(all_x) * 1.08])
    fig.update_yaxes(range=padded_range(all_y))
    return fig


def build_plot3_decode(stages: list[dict]) -> go.Figure:
    fig = go.Figure(layout=base_layout(
        "Context Length vs. Generation Speed",
        "Context Sequence Length (tokens)",
        "TPOT (ms/token)"
    ))

    metrics = [
        ("TPOT (ms/token)", "tpot"),
        ("Generation Speed (tokens/s/req)", "gen_speed"),
    ]

    for idx, s in enumerate(stages):
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        pts = s["context"]
        fig.add_trace(go.Scatter(
            x=[p["length"] for p in pts],
            y=[p["tpot"] for p in pts],
            mode="lines+markers",
            name=s["name"],
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            hovertemplate="<b>%{fullData.name}</b><br>Context: %{x} tok<br>Value: %{y:.2f}<extra></extra>",
        ))

    all_x = [p["length"] for s in stages for p in s["context"]]
    if all_x:
        fig.update_xaxes(range=[0, max(all_x) * 1.08])
    fig.update_yaxes(range=padded_range([p["tpot"] for s in stages for p in s["context"]]))

    buttons = []
    for label, key in metrics:
        y_arrays = [[p[key] for p in s["context"]] for s in stages]
        flat_y = [val for arr in y_arrays for val in arr]
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"y": y_arrays},
                {"yaxis.title.text": label, "yaxis.range": padded_range(flat_y)}
            ],
        ))

    fig.update_layout(updatemenus=[{**dropdown_style(), "buttons": buttons}])
    return fig


def build_plot4_roofline(stages: list[dict]) -> tuple[go.Figure, str, str]:
    layout = base_layout(
        "",
        "Arithmetic Intensity (FLOP/Byte)",
        "Attained Performance (TFLOP/s)"
    )
    layout["margin"] = dict(l=60, r=30, t=36, b=50)
    layout["height"] = 365
    layout["legend"]["y"] = 1.03

    fig = go.Figure(layout=layout)

    peak_bw_gbps = 512.0
    peak_compute_tflops = 100.0
    for s in stages:
        if s["ncu"] and "hardware_ceilings" in s["ncu"]:
            c = s["ncu"]["hardware_ceilings"]
            peak_bw_gbps = c.get("estimated_peak_dram_gbps", peak_bw_gbps)
            peak_compute_tflops = c.get("estimated_peak_tensor_tflops", peak_compute_tflops)
            break

    ridge_ai = (peak_compute_tflops * 1000.0) / peak_bw_gbps
    ai_min, ai_max = 0.02, 4096.0
    ceil_x = [ai_min, ridge_ai, ai_max]
    ceil_y = [(peak_bw_gbps * ai_min) / 1000.0, peak_compute_tflops, peak_compute_tflops]

    # Trace 0: Fixed Hardware Ceiling Line
    fig.add_trace(go.Scatter(
        x=ceil_x,
        y=ceil_y,
        mode="lines",
        name=f"Hardware Ceiling ({peak_bw_gbps:.0f} GB/s | {peak_compute_tflops:.0f} TFLOP/s)",
        line=dict(color="#ff2a2a", width=2, dash="dash"),
        hoverinfo="skip",
    ))

    ncu_stages = [s for s in stages if s["ncu"]]
    all_tflops = []
    group_orders = {}
    js_stage_kernels = []

    # Concentric marker radii so overlapping stages form a visible bullseye (outer ring + inner center)
    concentric_sizes = [14, 8, 5, 4]

    for idx, s in enumerate(ncu_stages):
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        marker_size = concentric_sizes[idx] if idx < len(concentric_sizes) else 4

        valid_k = [
            k for k in s["ncu"].get("kernels", [])
            if k.get("arithmetic_intensity_flop_byte", 0) > 0 and k.get("effective_tflops", 0) > 0
        ]

        stage_k_list = []
        for k in valid_k:
            tflops_val = k["effective_tflops"]
            all_tflops.append(tflops_val)
            g = k["canonical_group"]
            group_orders[g] = min(group_orders.get(g, 999), k["sort_order"])
            stage_k_list.append({
                "x": k["arithmetic_intensity_flop_byte"],
                "y": tflops_val,
                "label": k["label"],
                "phase": k.get("phase", ""),
                "phase_lower": k.get("phase", "").lower(),
                "group": g,
                "is_core": bool(k.get("is_core")),
                "exec_us": k.get("execution_time_us", 0.0),
                "dram_gbps": k.get("achieved_dram_gbps", 0.0),
                "symbol": "diamond" if k.get("phase") == "Prefill" else "circle",
            })

        js_stage_kernels.append({
            "stage": s["name"],
            "kernels": stage_k_list,
        })

        # Default initial view: Core Kernels (Attn & Linear)
        core_ks = [k for k in stage_k_list if k["is_core"]]
        fig.add_trace(go.Scatter(
            x=[k["x"] for k in core_ks],
            y=[k["y"] for k in core_ks],
            text=[k["label"] for k in core_ks],
            customdata=[[k["phase"], k["exec_us"], k["dram_gbps"], k["label"], s["name"]] for k in core_ks],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=10, color=TEXT_MUTED),
            name=s["name"],
            marker=dict(
                size=marker_size,
                color=color,
                symbol=[k["symbol"] for k in core_ks],
                line=dict(width=1.2, color="#000000"),
            ),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Stage: <b>%{customdata[4]}</b> (%{customdata[0]})<br>"
                "AI: %{x:.2f} FLOP/B<br>"
                "Perf: %{y:.4f} TFLOP/s<br>"
                "DRAM: %{customdata[2]:.1f} GB/s<br>"
                "Duration: %{customdata[1]:.2f} us<extra></extra>"
            ),
        ))

    # Fixed X and Y ranges across ALL kernels (including reconstructed low-TFLOP/s kernels)
    min_y = min(all_tflops) * 0.4 if all_tflops else 0.00005
    fig.update_xaxes(
        type="log",
        range=[math.log10(ai_min), math.log10(ai_max)],
        autorange=False,
    )
    fig.update_yaxes(
        type="log",
        range=[math.log10(min_y), math.log10(peak_compute_tflops * 1.8)],
        autorange=False,
    )

    ordered_groups = [g for g, _ in sorted(group_orders.items(), key=lambda x: x[1])]
    kernel_options_html = (
        '<option value="core" selected>Core Kernels (Attn & Linear)</option>'
        '<option value="all">All Kernels</option>'
        + "".join(f'<option value="{g}">Kernel: {g}</option>' for g in ordered_groups)
    )
    stage_options_html = (
        '<option value="all" selected>All Stages</option>'
        + "".join(f'<option value="{s["name"]}">{s["name"]}</option>' for s in ncu_stages)
    )

    controls_html = f"""
    <div class="plot4-header-bar">
        <div class="plot4-title">Hardware Roofline (Compute vs. Memory)</div>
        <div class="filter-controls">
            <div class="pill-group" id="roofline-phase-pills">
                <button class="pill active" data-phase="all" onclick="setRooflinePhase('all')">All Phases</button>
                <button class="pill" data-phase="prefill" onclick="setRooflinePhase('prefill')">Prefill</button>
                <button class="pill" data-phase="decode" onclick="setRooflinePhase('decode')">Decode</button>
            </div>
            <select id="roofline-kernel-select" class="dark-select" onchange="filterRooflinePlot()">
                {kernel_options_html}
            </select>
            <select id="roofline-stage-select" class="dark-select" onchange="filterRooflinePlot()">
                {stage_options_html}
            </select>
        </div>
    </div>
    """

    return fig, controls_html, json.dumps(js_stage_kernels)


def build_tables_html(stages: list[dict]) -> str:
    # Table A: Nsight Systems Timeline Overview
    nsys_rows = ""
    for s in stages:
        n = s["nsys"]
        if not n:
            continue

        wall_ms = n.get("wall_time_ms", 0.0)
        num_k = n.get("num_kernels")
        wall_sub = f' <span class="sub">({num_k:,} kernels)</span>' if num_k is not None else ""

        act_pct = n.get("gpu_active_pct", 0.0)
        act_ms = n.get("gpu_active_ms")
        act_sub = f' <span class="sub">({act_ms:.1f} ms)</span>' if act_ms is not None else ""

        idle_pct = n.get("gpu_idle_pct", 0.0)
        idle_ms = n.get("gpu_idle_ms")
        idle_sub = f' <span class="sub">({idle_ms:.1f} ms)</span>' if idle_ms is not None else ""

        launch_ms = n.get("cpu_launch_overhead_ms")
        if launch_ms is not None:
            calls = n.get("cpu_launch_calls", 0)
            graphs = n.get("cuda_graph_launches", 0)
            sub_parts = [f"{calls:,} calls"] + ([f"{graphs} graphs"] if graphs else [])
            launch_cell = f'{launch_ms:.2f} ms <span class="sub">({", ".join(sub_parts)})</span>'
        else:
            launch_cell = "-"

        sync_ms = n.get("cuda_sync_wait_ms", 0.0)

        pref_ms = n.get("prefill_duration_ms", 0.0)
        pref_pct = n.get("prefill_wall_pct")
        pref_sub = f' <span class="sub">({pref_pct:.1f}%)</span>' if pref_pct is not None else ""

        dec_step = n.get("decode_step_avg_ms", 0.0)
        dec_tot = n.get("decode_total_ms")
        dec_sub = f' <span class="sub">({dec_tot:.1f} ms tot.)</span>' if dec_tot is not None else ""

        kb = n.get("kernel_time_breakdown")
        if kb:
            lin_p = kb.get("linear_pct", 0.0)
            att_p = kb.get("attention_pct", 0.0)
            oth_p = kb.get("norm_act_other_pct", 0.0)
            lin_m = kb.get("linear_ms", 0.0)
            att_m = kb.get("attention_ms", 0.0)
            oth_m = kb.get("norm_act_other_ms", 0.0)
            kb_cell = (
                f"{lin_p:.1f}% / {att_p:.1f}% / {oth_p:.1f}% "
                f'<span class="sub">({lin_m:.1f} / {att_m:.1f} / {oth_m:.1f} ms)</span>'
            )
        else:
            kb_cell = "-"

        nsys_rows += f"""
        <tr>
            <td class="stage-cell">{s['name']}</td>
            <td>{wall_ms:.2f}{wall_sub}</td>
            <td><span class="highlight-red">{act_pct:.1f}%</span>{act_sub}</td>
            <td>{idle_pct:.1f}%{idle_sub}</td>
            <td>{launch_cell}</td>
            <td>{sync_ms:.2f}</td>
            <td>{pref_ms:.2f} ms{pref_sub}</td>
            <td>{dec_step:.2f} ms/step{dec_sub}</td>
            <td>{kb_cell}</td>
        </tr>"""

    # Table B: Nsight Compute Kernel Details (Grouped by Kernel for direct cross-stage comparison)
    flat_entries = []
    group_orders = {}

    for s_idx, s in enumerate(stages):
        ncu = s["ncu"]
        if not ncu:
            continue
        ceilings = ncu.get("hardware_ceilings", {})
        peak_dram = ceilings.get("estimated_peak_dram_gbps", 512.0)
        peak_tc = ceilings.get("estimated_peak_tensor_tflops", 100.0)

        for k in ncu.get("kernels", []):
            dram_pct = k.get("dram_utilization_pct", 0.0)
            if dram_pct == 0.0 and peak_dram > 0:
                dram_pct = (k.get("achieved_dram_gbps", 0.0) / peak_dram) * 100.0

            comp_pct = k.get("compute_utilization_pct", 0.0)
            if comp_pct == 0.0 and peak_tc > 0:
                comp_pct = (k.get("attained_tflops", 0.0) / peak_tc) * 100.0

            group = k.get("canonical_group", k.get("label", ""))
            order = k.get("sort_order", 999)
            group_orders[group] = min(group_orders.get(group, 999), order)

            flat_entries.append({
                "stage": s["name"],
                "stage_idx": s_idx,
                "group": group,
                "order": order,
                "is_core": "true" if k.get("is_core") else "false",
                "phase": k.get("phase", "").lower(),
                "phase_label": k.get("phase", ""),
                "label": k.get("label", ""),
                "exec_us": k.get("execution_time_us", 0.0),
                "dram_pct": dram_pct,
                "dram_gbps": k.get("achieved_dram_gbps", 0.0),
                "comp_pct": comp_pct,
                "comp_tflops": k.get("attained_tflops", 0.0),
                "ai": k.get("arithmetic_intensity_flop_byte", 0.0),
                "warps": k.get("active_warps_pct", 0.0),
                "regs": k.get("registers_per_thread", 0),
                "spills": k.get("local_spills_bytes", 0),
            })

    # Sort by Canonical Kernel first, Stage second
    flat_entries.sort(key=lambda x: (x["order"], x["group"], x["stage_idx"]))

    ncu_rows = ""
    prev_group = None
    for item in flat_entries:
        group_divider = "group-start" if (prev_group is not None and item["group"] != prev_group) else ""
        prev_group = item["group"]
        phase_badge = f"<span class='badge badge-{item['phase']}'>{item['phase_label']}</span>"
        stage_dot = STAGE_COLORS[item["stage_idx"] % len(STAGE_COLORS)]

        ncu_rows += f"""
        <tr class="ncu-row {group_divider}"
            data-stage="{item['stage']}"
            data-phase="{item['phase']}"
            data-group="{item['group']}"
            data-core="{item['is_core']}">
            <td class="stage-cell"><span class="stage-dot" style="background:{stage_dot};"></span>{item['stage']}</td>
            <td>{phase_badge} {item['label']}</td>
            <td>{item['exec_us']:.2f}</td>
            <td>{item['dram_pct']:.1f}% <span class="sub">({item['dram_gbps']:.1f} GB/s)</span></td>
            <td>{item['comp_pct']:.1f}% <span class="sub">({item['comp_tflops']:.2f} TFLOP/s)</span></td>
            <td class="ai-cell">{item['ai']:.2f}</td>
            <td>{item['warps']:.1f}%</td>
            <td>{item['regs']}</td>
            <td>{item['spills']}</td>
        </tr>"""

    ordered_groups = [g for g, _ in sorted(group_orders.items(), key=lambda x: x[1])]
    kernel_options_html = (
        '<option value="core" selected>Core Kernels (Attn & Linear)</option>'
        '<option value="all">All Kernels</option>'
        + "".join(f'<option value="{g}">Kernel: {g}</option>' for g in ordered_groups)
    )
    stage_options_html = (
        '<option value="all" selected>All Stages</option>'
        + "".join(f'<option value="{s["name"]}">{s["name"]}</option>' for s in stages)
    )

    return f"""
    <div class="tables-container">
        <div class="card table-card">
            <h2>Table A: Timeline Overview (Nsight Systems)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Wall Time (ms)</th>
                        <th>GPU Active Time (%)</th>
                        <th>GPU Idle (%)</th>
                        <th>CPU Launch Overhead</th>
                        <th>CUDA Sync (ms)</th>
                        <th>Prefill Duration</th>
                        <th>Decode Duration</th>
                        <th>Kernel Time (Lin / Attn / Oth)</th>
                    </tr>
                </thead>
                <tbody>{nsys_rows or '<tr><td colspan="9">No nsys_metrics.json found</td></tr>'}</tbody>
            </table>
        </div>

        <div class="card table-card">
            <div class="table-header-bar">
                <h2>Table B: Kernel Details (Nsight Compute)</h2>
                <div class="filter-controls">
                    <div class="pill-group" id="ncu-phase-pills">
                        <button class="pill active" data-phase="all" onclick="setNcuPhase('all')">All Phases</button>
                        <button class="pill" data-phase="prefill" onclick="setNcuPhase('prefill')">Prefill</button>
                        <button class="pill" data-phase="decode" onclick="setNcuPhase('decode')">Decode</button>
                    </div>
                    <select id="ncu-kernel-select" class="dark-select" onchange="filterNcuTable()">
                        {kernel_options_html}
                    </select>
                    <select id="ncu-stage-select" class="dark-select" onchange="filterNcuTable()">
                        {stage_options_html}
                    </select>
                </div>
            </div>
            <table id="ncu-table">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Kernel Name</th>
                        <th>Execution Time (us)</th>
                        <th>DRAM Utilization (%)</th>
                        <th>Compute Utilization (%)</th>
                        <th>Arithmetic Intensity (FLOP/B)</th>
                        <th>Active Warps (%)</th>
                        <th>Registers / Thread</th>
                        <th>Local Spills (B)</th>
                    </tr>
                </thead>
                <tbody>{ncu_rows or '<tr><td colspan="9">No ncu_metrics.json found</td></tr>'}</tbody>
            </table>
        </div>
    </div>
    """


def generate_dashboard(stages: list[dict], output_path: Path):
    f1 = build_plot1_pareto(stages)
    f2 = build_plot2_prefill(stages)
    f3 = build_plot3_decode(stages)
    f4, p4_controls_html, roofline_json = build_plot4_roofline(stages)

    cfg = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }
    p1_html = f1.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p2_html = f2.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p3_html = f3.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p4_html = f4.to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id="roofline-plot")
    tables_html = build_tables_html(stages)

    stage_list = ", ".join(s["name"] for s in stages)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Inference Kernel Optimization Benchmark</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: {BG_PAGE};
            color: {TEXT_MAIN};
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 28px 36px;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            border-bottom: 2px solid #ff2a2a;
            padding-bottom: 14px;
            margin-bottom: 24px;
        }}
        header h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        header .meta {{
            font-family: monospace;
            font-size: 13px;
            color: {TEXT_MUTED};
        }}
        header .meta span {{ color: #ff2a2a; font-weight: 600; }}
        .grid-2x2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: {BG_CARD};
            border: 1px solid #242424;
            border-top: 2px solid #ff2a2a;
            border-radius: 4px;
            padding: 12px;
        }}
        .plot4-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            padding: 6px 6px 10px 6px;
        }}
        .plot4-title {{
            font-size: 17px;
            color: {TEXT_MAIN};
            font-family: Inter, sans-serif;
        }}

        /* --- Plotly Dropdown dark-grey hover & active state --- */
        .js-plotly-plot .plotly .updatemenu-item-rect {{
            fill: #1a1a1a !important;
            stroke: #333333 !important;
            transition: fill 0.12s ease;
        }}
        .js-plotly-plot .plotly .updatemenu-header-group .updatemenu-item-rect {{
            stroke: #ff2a2a !important;
        }}
        .js-plotly-plot .plotly .updatemenu-button:hover .updatemenu-item-rect,
        .js-plotly-plot .plotly .updatemenu-dropdown-button:hover .updatemenu-item-rect,
        .js-plotly-plot .plotly .updatemenu-item-rect:hover {{
            fill: #2e2e2e !important;
            stroke: #ff2a2a !important;
        }}
        .js-plotly-plot .plotly .updatemenu-item-text {{
            fill: #ffffff !important;
        }}
        .js-plotly-plot .plotly .updatemenu-header-arrow {{
            fill: #ff2a2a !important;
        }}

        /* --- Pin modebar cleanly at top-right above dropdown --- */
        .js-plotly-plot .plotly .modebar {{
            top: 2px !important;
            right: 6px !important;
            background: rgba(18, 18, 18, 0.9) !important;
            border-radius: 4px;
            padding: 2px 4px;
        }}
        .js-plotly-plot .plotly .modebar-btn path {{
            fill: #888888 !important;
        }}
        .js-plotly-plot .plotly .modebar-btn:hover path {{
            fill: #ff2a2a !important;
        }}

        .tables-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .table-card {{ padding: 20px; overflow-x: auto; }}
        .table-card h2 {{
            font-size: 16px;
            color: {TEXT_MAIN};
            letter-spacing: -0.01em;
        }}
        .table-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .filter-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .pill-group {{
            display: inline-flex;
            background: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 4px;
            overflow: hidden;
        }}
        .pill {{
            background: transparent;
            color: {TEXT_MUTED};
            border: none;
            padding: 6px 12px;
            font-family: Inter, sans-serif;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.12s, color 0.12s;
        }}
        .pill:hover {{
            background: #262626;
            color: #ffffff;
        }}
        .pill.active {{
            background: #ff2a2a;
            color: #000000;
        }}
        .dark-select {{
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #ff2a2a;
            border-radius: 4px;
            padding: 5px 10px;
            font-family: Inter, monospace;
            font-size: 11.5px;
            outline: none;
            cursor: pointer;
        }}
        .dark-select:hover {{
            background-color: #262626;
        }}
        .dark-select option {{
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: "JetBrains Mono", Consolas, monospace;
            font-size: 12.5px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #222222;
            white-space: nowrap;
        }}
        th {{
            color: {TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.04em;
            background-color: #171717;
        }}
        tr:hover td {{ background-color: #191919; }}
        tr.group-start td {{ border-top: 2px solid #2e2e2e; }}
        .stage-cell {{ font-weight: 700; color: #ffffff; }}
        .stage-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .highlight-red {{ color: #ff2a2a; font-weight: 700; }}
        .ai-cell {{ color: #ff7b54; font-weight: 600; }}
        .sub {{ color: #777777; font-size: 11px; }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 700;
            margin-right: 6px;
            text-transform: uppercase;
        }}
        .badge-prefill {{ background: #ff2a2a; color: #000000; }}
        .badge-decode {{ background: #ffffff; color: #000000; }}
        @media (max-width: 1100px) {{ .grid-2x2 {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <header>
        <h1>LLM INFERENCE & KERNEL BENCHMARK</h1>
        <div class="meta">Active Stages: <span>{stage_list}</span></div>
    </header>

    <div class="grid-2x2">
        <div class="card">{p1_html}</div>
        <div class="card">{p2_html}</div>
        <div class="card">{p3_html}</div>
        <div class="card">{p4_controls_html}{p4_html}</div>
    </div>

    {tables_html}

    <script>
        const ROOFLINE_STAGE_DATA = {roofline_json};
        let currentRooflinePhase = 'all';
        let currentNcuPhase = 'all';

        function setRooflinePhase(phase) {{
            currentRooflinePhase = phase;
            document.querySelectorAll('#roofline-phase-pills .pill').forEach(btn => {{
                btn.classList.toggle('active', btn.getAttribute('data-phase') === phase);
            }});
            filterRooflinePlot();
        }}

        function filterRooflinePlot() {{
            const plotDiv = document.getElementById('roofline-plot');
            if (!plotDiv || !ROOFLINE_STAGE_DATA.length) return;

            const kernelMode = document.getElementById('roofline-kernel-select').value;
            const stageMode = document.getElementById('roofline-stage-select').value;

            // Step 1: Filter visible kernels per stage
            const filteredByStage = ROOFLINE_STAGE_DATA.map(sObj => {{
                const matchStage = (stageMode === 'all') || (sObj.stage === stageMode);
                if (!matchStage) return [];
                return sObj.kernels.filter(k => {{
                    const matchPhase = (currentRooflinePhase === 'all') || (k.phase_lower === currentRooflinePhase);
                    let matchKernel = true;
                    if (kernelMode === 'core') {{
                        matchKernel = k.is_core;
                    }} else if (kernelMode !== 'all') {{
                        matchKernel = (k.group === kernelMode);
                    }}
                    return matchPhase && matchKernel;
                }});
            }});

            // Step 2: Detect overlapping points across stages in log-space
            // so we don't double-stamp text and can append "(×N)" on the top trace
            const eps = 0.03; // ~7% log10 proximity threshold
            const clusters = [];
            filteredByStage.forEach((ks, sIdx) => {{
                ks.forEach((k, kIdx) => {{
                    const lx = Math.log10(k.x);
                    const ly = Math.log10(k.y);
                    let found = clusters.find(c => Math.abs(c.lx - lx) < eps && Math.abs(c.ly - ly) < eps);
                    if (!found) {{
                        found = {{ lx, ly, members: [] }};
                        clusters.push(found);
                    }}
                    found.members.push({{ sIdx, kIdx, stage: ROOFLINE_STAGE_DATA[sIdx].stage, label: k.label }});
                }});
            }});

            const xs = [];
            const ys = [];
            const texts = [];
            const cdatas = [];
            const symbols = [];
            const traceIndices = [];

            filteredByStage.forEach((ks, sIdx) => {{
                traceIndices.push(sIdx + 1); // Trace 0 is the fixed Hardware Ceiling line
                xs.push(ks.map(k => k.x));
                ys.push(ks.map(k => k.y));
                symbols.push(ks.map(k => k.symbol));

                const stageTexts = [];
                const stageCdatas = [];

                ks.forEach((k, kIdx) => {{
                    const lx = Math.log10(k.x);
                    const ly = Math.log10(k.y);
                    const cluster = clusters.find(c => Math.abs(c.lx - lx) < eps && Math.abs(c.ly - ly) < eps);
                    const count = cluster ? cluster.members.length : 1;
                    const lastMember = cluster ? cluster.members[cluster.members.length - 1] : null;
                    const stageNames = cluster ? cluster.members.map(m => m.stage).join(' & ') : ROOFLINE_STAGE_DATA[sIdx].stage;

                    // Only draw the text label on the top-most overlapping marker to keep font crisp
                    if (count > 1) {{
                        if (lastMember && lastMember.sIdx === sIdx && lastMember.kIdx === kIdx) {{
                            stageTexts.push(`${{k.label}} (×${{count}})`);
                        }} else {{
                            stageTexts.push('');
                        }}
                    }} else {{
                        stageTexts.push(k.label);
                    }}

                    stageCdatas.push([k.phase, k.exec_us, k.dram_gbps, k.label, stageNames]);
                }});

                texts.push(stageTexts);
                cdatas.push(stageCdatas);
            }});

            Plotly.restyle(plotDiv, {{
                x: xs,
                y: ys,
                text: texts,
                customdata: cdatas,
                'marker.symbol': symbols
            }}, traceIndices);
        }}

        function setNcuPhase(phase) {{
            currentNcuPhase = phase;
            document.querySelectorAll('#ncu-phase-pills .pill').forEach(btn => {{
                btn.classList.toggle('active', btn.getAttribute('data-phase') === phase);
            }});
            filterNcuTable();
        }}

        function filterNcuTable() {{
            const kernelMode = document.getElementById('ncu-kernel-select').value;
            const stageMode = document.getElementById('ncu-stage-select').value;
            const rows = document.querySelectorAll('#ncu-table tbody tr.ncu-row');

            let prevVisibleGroup = null;
            rows.forEach(row => {{
                const rPhase = row.getAttribute('data-phase');
                const rGroup = row.getAttribute('data-group');
                const rStage = row.getAttribute('data-stage');
                const rCore = row.getAttribute('data-core') === 'true';

                const matchPhase = (currentNcuPhase === 'all') || (rPhase === currentNcuPhase);
                const matchStage = (stageMode === 'all') || (rStage === stageMode);
                let matchKernel = true;
                if (kernelMode === 'core') {{
                    matchKernel = rCore;
                }} else if (kernelMode !== 'all') {{
                    matchKernel = (rGroup === kernelMode);
                }}

                if (matchPhase && matchStage && matchKernel) {{
                    row.style.display = '';
                    row.classList.toggle('group-start', prevVisibleGroup !== null && prevVisibleGroup !== rGroup);
                    prevVisibleGroup = rGroup;
                }} else {{
                    row.style.display = 'none';
                    row.classList.remove('group-start');
                }}
            }});
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            filterNcuTable();
            filterRooflinePlot();
        }});
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] Dashboard generated: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Plot LLM Inference & Kernel Benchmarks")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results directory")
    parser.add_argument("--stages", nargs="*", help="Specific stage folders to plot (default: all)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HTML file path")
    args = parser.parse_args()

    # Auto-detect benchmark/results or benchmarks/results
    if args.results_dir:
        base_dir = Path(args.results_dir)
    elif Path("benchmark/results").exists():
        base_dir = Path("benchmark/results")
    else:
        base_dir = Path("benchmarks/results")

    if args.stages:
        stage_dirs = [base_dir / s for s in args.stages if (base_dir / s).is_dir()]
    else:
        stage_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()]) if base_dir.exists() else []

    if not stage_dirs:
        raise SystemExit(f"No valid stage directories found in '{base_dir}'.")

    stages_data = [load_stage_data(d) for d in stage_dirs]
    out_file = Path(args.output) if args.output else (base_dir / "dashboard.html")
    generate_dashboard(stages_data, out_file)


if __name__ == "__main__":
    main()