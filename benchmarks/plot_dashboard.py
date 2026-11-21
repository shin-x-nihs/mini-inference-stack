import argparse
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go

# --- Red, White & Black Palette (SemiAnalysis-inspired) ---
BG_PAGE = "#0a0a0a"
BG_CARD = "#121212"
GRID_COLOR = "#222222"
TEXT_MAIN = "#ffffff"
TEXT_MUTED = "#a0a0a0"

# --- Track-Aware Progressive Palettes (Dark Theme) ---
# Custom Track: Warm Crimson -> Coral/Orange -> Amber -> Gold progression
CUSTOM_TRACK_COLORS = [
    "#ff2a2a",  # Stage 0: Pure Crimson Red
    "#ff8c42",  # Stage 1: Vibrant Tangerine / Coral Orange
    "#ffd166",  # Stage 2: Warm Amber Gold
    "#f4a261",  # Stage 3: Sandy Gold
    "#ff5d8f",  # Stage 4: Electric Rose
    "#fff3b0",  # Stage 5: Pale Cream Gold
]

# vLLM Reference Track: Cool Ice White -> Electric Sky Cyan -> Mint Teal -> Violet
VLLM_TRACK_COLORS = [
    "#e2e8f0",  # Stage 0: Crisp Ice White / Silver
    "#38bdf8",  # Stage 1: Electric Sky Cyan
    "#2dd4bf",  # Stage 2: Bright Mint Teal
    "#a78bfa",  # Stage 3: Soft Lavender Violet
    "#818cf8",  # Stage 4: Periwinkle Indigo
]

OTHER_TRACK_COLORS = [
    "#c084fc",
    "#f472b6",
    "#4ade80",
    "#facc15",
]

# Concentric marker radii by stage number so overlapping points form a clean bullseye
STAGE_ROOFLINE_SIZES = [15, 10, 7, 5, 4, 4]


def extract_int(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def parse_stage_meta(stage_name: str) -> tuple[str, int, int, str, str, int]:
    """
    Returns (track, track_order, stage_num, color, marker_symbol, marker_size)
    from a stage name like 'stage-0-custom' or 'stage-2-vllm'.
    """
    low = stage_name.lower()
    stage_num = extract_int(stage_name)
    if "custom" in low:
        track = "custom"
        track_order = 0
        color = CUSTOM_TRACK_COLORS[stage_num % len(CUSTOM_TRACK_COLORS)]
        symbol = "circle"
    elif "vllm" in low:
        track = "vllm"
        track_order = 1
        color = VLLM_TRACK_COLORS[stage_num % len(VLLM_TRACK_COLORS)]
        symbol = "diamond"
    else:
        track = "other"
        track_order = 2
        color = OTHER_TRACK_COLORS[stage_num % len(OTHER_TRACK_COLORS)]
        symbol = "square"

    size = (
        STAGE_ROOFLINE_SIZES[stage_num]
        if stage_num < len(STAGE_ROOFLINE_SIZES)
        else STAGE_ROOFLINE_SIZES[-1]
    )
    return track, track_order, stage_num, color, symbol, size


def stage_sort_key(stage_or_dir) -> tuple[int, int, str]:
    name = stage_or_dir["name"] if isinstance(stage_or_dir, dict) else stage_or_dir.name
    _, track_order, stage_num, _, _, _ = parse_stage_meta(name)
    return (track_order, stage_num, name)


def padded_range(values: list[float], pad: float = 1.15) -> list[float]:
    valid = [v for v in values if v is not None and v > 0]
    return [0.0, max(valid) * pad] if valid else [0.0, 1.0]


def canonical_kernel_info(
    label: str, phase: str, raw_kernel_name: str = ""
) -> tuple[str, str, bool, int]:
    """
    Maps raw stage kernel labels (e.g. 'Decode Linear GEMV' vs 'Decode Linear GEMM')
    to a unified (canonical_group, short_plot_label, is_core, sort_order).
    """
    lbl = label.lower()
    raw = raw_kernel_name.lower()
    p = phase.capitalize() if phase else "Other"
    p_offset = 0 if p == "Prefill" else 100

    if "linear" in lbl or "gemm" in lbl or "gemv" in lbl:
        return (f"{p} Linear (GEMM/GEMV)", f"{p} Linear", True, p_offset + 1)
    if "combine" in lbl or "combine" in raw:
        return (f"{p} Attention Combine", f"{p} Attn Combine", False, p_offset + 3)
    if (
        "flash_attention" in lbl
        or "_flash_attention_fwd" in lbl
        or "flash_attention" in raw
        or "_flash_attention_fwd" in raw
    ):
        return (f"{p} Attention", f"{p} Attn", True, p_offset + 2)
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
    track, track_order, stage_num, color, symbol, size = parse_stage_meta(stage_dir.name)
    data = {
        "name": stage_dir.name,
        "track": track,
        "track_order": track_order,
        "stage_num": stage_num,
        "color": color,
        "symbol": symbol,
        "marker_size": size,
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
                    k.get("label", ""), k.get("phase", ""), k.get("raw_kernel_name", "")
                )
                k["canonical_group"] = group
                k["short_label"] = short_lbl
                k["is_core"] = is_core
                k["sort_order"] = order
                k["effective_tflops"] = effective_tflops(k)
            data["ncu"] = ncu_data

    return data


def base_layout(xaxis_title: str, yaxis_title: str) -> dict:
    return {
        "paper_bgcolor": BG_CARD,
        "plot_bgcolor": BG_CARD,
        "font": {"color": TEXT_MUTED, "family": "Inter, monospace", "size": 12},
        "margin": {"l": 60, "r": 28, "t": 42, "b": 50},
        "height": 385,
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": TEXT_MAIN, "size": 11.5},
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        "xaxis": {
            "title": {"text": xaxis_title, "font": {"color": TEXT_MAIN}},
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "linecolor": "#333333",
        },
        "yaxis": {
            "title": {"text": yaxis_title, "font": {"color": TEXT_MAIN}},
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "linecolor": "#333333",
        },
    }


def default_active_track(stages: list[dict]) -> str:
    return "all"


def build_plot1_pareto(stages: list[dict], active_track: str) -> tuple[go.Figure, str]:
    fig = go.Figure(layout=base_layout(
        "System Throughput (tokens/s)",
        "TPOT Mean (ms/token)"
    ))

    metrics = [
        ("mean_tpot", "TPOT Mean (ms/token)"),
        ("p99_itl", "ITL P99 (ms)"),
        ("p90_itl", "ITL P90 (ms)"),
        ("p50_itl", "ITL P50 (ms)"),
    ]

    # Alternate text positions on dense Pareto curves so C=1..C=16 don't overlap
    pos_cycle = ["top center", "bottom center", "top right", "bottom left", "top left"]

    for s in stages:
        color = s["color"]
        pts = s["pareto"]
        is_vis = (active_track == "all") or (s["track"] == active_track)
        text_positions = [pos_cycle[i % len(pos_cycle)] for i in range(len(pts))]
        fig.add_trace(go.Scatter(
            x=[p["throughput"] for p in pts],
            y=[p["mean_tpot"] for p in pts],
            text=[f"C={p['concurrency']}" for p in pts],
            textposition=text_positions,
            textfont={"size": 10, "color": color},
            mode="lines+markers+text",
            name=s["name"],
            visible=True if is_vis else "legendonly",
            meta={"track": s["track"], "stage": s["name"]},
            line={"color": color, "width": 2.5},
            marker={
                "size": 8,
                "color": color,
                "symbol": s["symbol"],
                "line": {"width": 1.2, "color": BG_CARD},
            },
            hovertemplate="<b>%{fullData.name}</b> (%{text})<br>Throughput: %{x:.1f} tok/s<br>Latency: %{y:.2f} ms<extra></extra>",
        ))

    vis_stages = [s for s in stages if active_track == "all" or s["track"] == active_track] or stages
    vis_x = [p["throughput"] for s in vis_stages for p in s["pareto"]]
    vis_y = [p["mean_tpot"] for s in vis_stages for p in s["pareto"]]
    fig.update_xaxes(range=padded_range(vis_x))
    fig.update_yaxes(range=padded_range(vis_y))

    options_html = "".join(
        f'<option value="{key}"{" selected" if idx == 0 else ""}>{label}</option>'
        for idx, (key, label) in enumerate(metrics)
    )
    header_html = f"""
    <div class="plot-header-bar">
        <div class="plot-title">Throughput vs. Latency (System Capacity)</div>
        <div class="filter-controls">
            <select id="plot1-metric-select" class="dark-select" onchange="updatePlot1Metric()">
                {options_html}
            </select>
        </div>
    </div>
    """
    return fig, header_html


def build_plot2_prefill(stages: list[dict], active_track: str) -> tuple[go.Figure, str]:
    fig = go.Figure(layout=base_layout(
        "Input Prompt Length (tokens)",
        "Time to First Token - TTFT (ms)"
    ))

    for s in stages:
        color = s["color"]
        pts = s["context"]
        is_vis = (active_track == "all") or (s["track"] == active_track)
        fig.add_trace(go.Scatter(
            x=[p["length"] for p in pts],
            y=[p["ttft"] for p in pts],
            mode="lines+markers",
            name=s["name"],
            visible=True if is_vis else "legendonly",
            meta={"track": s["track"], "stage": s["name"]},
            line={"color": color, "width": 2.5},
            marker={"size": 8, "color": color, "symbol": s["symbol"], "line": {"width": 1, "color": BG_CARD}},
            hovertemplate="<b>%{fullData.name}</b><br>Prompt: %{x} tok<br>TTFT: %{y:.2f} ms<extra></extra>",
        ))

    vis_stages = [s for s in stages if active_track == "all" or s["track"] == active_track] or stages
    vis_x = [p["length"] for s in vis_stages for p in s["context"]]
    vis_y = [p["ttft"] for s in vis_stages for p in s["context"]]
    if vis_x:
        fig.update_xaxes(range=[0, max(vis_x) * 1.08])
    fig.update_yaxes(range=padded_range(vis_y))

    header_html = """
    <div class="plot-header-bar">
        <div class="plot-title">Prompt Size vs. Time to First Token</div>
        <div class="filter-controls"></div>
    </div>
    """
    return fig, header_html


def build_plot3_decode(stages: list[dict], active_track: str) -> tuple[go.Figure, str]:
    fig = go.Figure(layout=base_layout(
        "Context Sequence Length (tokens)",
        "TPOT (ms/token)"
    ))

    metrics = [
        ("tpot", "TPOT (ms/token)"),
        ("gen_speed", "Generation Speed (tokens/s/req)"),
    ]

    for s in stages:
        color = s["color"]
        pts = s["context"]
        is_vis = (active_track == "all") or (s["track"] == active_track)
        fig.add_trace(go.Scatter(
            x=[p["length"] for p in pts],
            y=[p["tpot"] for p in pts],
            mode="lines+markers",
            name=s["name"],
            visible=True if is_vis else "legendonly",
            meta={"track": s["track"], "stage": s["name"]},
            line={"color": color, "width": 2.5},
            marker={"size": 8, "color": color, "symbol": s["symbol"], "line": {"width": 1, "color": BG_CARD}},
            hovertemplate="<b>%{fullData.name}</b><br>Context: %{x} tok<br>Value: %{y:.2f}<extra></extra>",
        ))

    vis_stages = [s for s in stages if active_track == "all" or s["track"] == active_track] or stages
    vis_x = [p["length"] for s in vis_stages for p in s["context"]]
    vis_y = [p["tpot"] for s in vis_stages for p in s["context"]]
    if vis_x:
        fig.update_xaxes(range=[0, max(vis_x) * 1.08])
    fig.update_yaxes(range=padded_range(vis_y))

    options_html = "".join(
        f'<option value="{key}"{" selected" if idx == 0 else ""}>{label}</option>'
        for idx, (key, label) in enumerate(metrics)
    )
    header_html = f"""
    <div class="plot-header-bar">
        <div class="plot-title">Context Length vs. Generation Speed</div>
        <div class="filter-controls">
            <select id="plot3-metric-select" class="dark-select" onchange="updatePlot3Metric()">
                {options_html}
            </select>
        </div>
    </div>
    """
    return fig, header_html


def build_plot4_roofline(stages: list[dict], active_track: str) -> tuple[go.Figure, str, str]:
    layout = base_layout(
        "Arithmetic Intensity (FLOP/Byte)",
        "Attained Performance (TFLOP/s)"
    )

    fig = go.Figure(layout=layout)

    peak_bw_gbps = 896.0
    peak_compute_tflops = 40.0
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
        name=f"Ceiling ({peak_bw_gbps:.0f} GB/s | {peak_compute_tflops:.0f} TFLOP/s)",
        meta={"track": "ceiling", "stage": "ceiling"},
        line={"color": "#ff2a2a", "width": 2, "dash": "dash"},
        hoverinfo="skip",
    ))

    ncu_stages = [s for s in stages if s["ncu"]]
    all_tflops = []
    group_orders = {}
    js_stage_kernels = []

    for s in ncu_stages:
        color = s["color"]
        marker_size = s["marker_size"]
        track_symbol = s["symbol"]
        is_vis = (active_track == "all") or (s["track"] == active_track)

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
                "short_label": k.get("short_label", k["label"]),
                "phase": k.get("phase", ""),
                "phase_lower": k.get("phase", "").lower(),
                "group": g,
                "is_core": bool(k.get("is_core")),
                "exec_us": k.get("execution_time_us", 0.0),
                "dram_gbps": k.get("achieved_dram_gbps", 0.0),
                "symbol": track_symbol,
            })

        js_stage_kernels.append({
            "stage": s["name"],
            "track": s["track"],
            "stage_num": s["stage_num"],
            "color": color,
            "symbol": track_symbol,
            "size": marker_size,
            "kernels": stage_k_list,
        })

        # Default initial view: Core Kernels (Attn & Linear)
        core_ks = [k for k in stage_k_list if k["is_core"]]
        fig.add_trace(go.Scatter(
            x=[k["x"] for k in core_ks],
            y=[k["y"] for k in core_ks],
            text=[k["short_label"] for k in core_ks],
            customdata=[[k["phase"], k["exec_us"], k["dram_gbps"], k["label"], s["name"], s["track"].upper()] for k in core_ks],
            mode="markers+text",
            textposition="top center",
            textfont={"size": 10, "color": TEXT_MUTED},
            name=s["name"],
            visible=True if is_vis else "legendonly",
            meta={"track": s["track"], "stage": s["name"]},
            marker={
                "size": marker_size,
                "color": color,
                "symbol": track_symbol,
                "line": {"width": 1.3, "color": "#000000"},
            },
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Stage: <b>%{customdata[4]}</b> [%{customdata[5]} | %{customdata[0]}]<br>"
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
        '<option value="all" selected>All Track Stages</option>'
        + "".join(
            f'<option value="{s["name"]}" data-track="{s["track"]}">{s["name"]}</option>'
            for s in ncu_stages
        )
    )

    controls_html = f"""
    <div class="plot-header-bar">
        <div class="plot-title">
            Hardware Roofline
            <span class="roofline-key" title="Shape = Track | Size = Stage Index">
                (● Custom &nbsp;◆ vLLM &nbsp;|&nbsp; Size: S0 &gt; S1 &gt; S2)
            </span>
        </div>
        <div class="filter-controls">
            <div class="pill-group" id="roofline-phase-pills">
                <button class="pill active" data-phase="all" onclick="setRooflinePhase('all')">All</button>
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
    # Stages are already sorted by (track_order, stage_num, name)
    # Table A: Nsight Systems Timeline Overview
    nsys_rows = ""
    prev_track = None
    for s in stages:
        n = s["nsys"]
        if not n:
            continue

        track_divider = "group-start" if (prev_track is not None and s["track"] != prev_track) else ""
        prev_track = s["track"]

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
        dec_sub = f' <span class="sub">({dec_tot:.1f} ms)</span>' if dec_tot is not None else ""

        kb = n.get("kernel_time_breakdown")
        if kb:
            lin_p = kb.get("linear_pct", 0.0)
            att_p = kb.get("attention_pct", 0.0)
            oth_p = kb.get("norm_act_other_pct", 0.0)
            lin_m = kb.get("linear_ms", 0.0)
            att_m = kb.get("attention_ms", 0.0)
            oth_m = kb.get("norm_act_other_ms", 0.0)
            if s["track"] == "custom":
                ms_str = f"({int(lin_m)} / {int(att_m)} / {int(oth_m)} ms)"
            else:
                ms_str = f"({lin_m:.1f} / {att_m:.1f} / {oth_m:.1f} ms)"
            kb_cell = (
                f"{lin_p:.1f}% / {att_p:.1f}% / {oth_p:.1f}% "
                f'<span class="sub">{ms_str}</span>'
            )
        else:
            kb_cell = "-"

        nsys_rows += f"""
        <tr class="{track_divider}">
            <td class="stage-cell"><span class="stage-dot" style="background:{s['color']};"></span>{s['name']}</td>
            <td>{wall_ms:.2f}{wall_sub}</td>
            <td><span class="highlight-red">{act_pct:.1f}%</span>{act_sub}</td>
            <td>{idle_pct:.1f}%{idle_sub}</td>
            <td>{launch_cell}</td>
            <td>{sync_ms:.2f}</td>
            <td>{pref_ms:.2f} ms{pref_sub}</td>
            <td>{dec_step:.2f} ms/step{dec_sub}</td>
            <td>{kb_cell}</td>
        </tr>"""

    # Table B: Nsight Compute Kernel Details (Grouped by Kernel, then Track, then Stage)
    flat_entries = []
    group_orders = {}

    for s_idx, s in enumerate(stages):
        ncu = s["ncu"]
        if not ncu:
            continue
        ceilings = ncu.get("hardware_ceilings", {})
        peak_dram = ceilings.get("estimated_peak_dram_gbps", 896.0)
        peak_tc = ceilings.get("estimated_peak_tensor_tflops", 40.0)

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
                "track_order": s["track_order"],
                "stage_num": s["stage_num"],
                "color": s["color"],
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

    # Sort by Canonical Kernel first, then Track (custom -> vllm), then Stage number
    flat_entries.sort(key=lambda x: (x["order"], x["group"], x["track_order"], x["stage_num"], x["stage_idx"]))

    ncu_rows = ""
    prev_group = None
    for item in flat_entries:
        group_divider = "group-start" if (prev_group is not None and item["group"] != prev_group) else ""
        prev_group = item["group"]
        phase_badge = f"<span class='badge badge-{item['phase']}'>{item['phase_label']}</span>"
        stage_dot = item["color"]

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
    stages = sorted(stages, key=stage_sort_key)
    active_track = default_active_track(stages)

    f1, p1_header_html = build_plot1_pareto(stages, active_track)
    f2, p2_header_html = build_plot2_prefill(stages, active_track)
    f3, p3_header_html = build_plot3_decode(stages, active_track)
    f4, p4_controls_html, roofline_json = build_plot4_roofline(stages, active_track)

    cfg = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }
    p1_html = f1.to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id="plot1-pareto")
    p2_html = f2.to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id="plot2-prefill")
    p3_html = f3.to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id="plot3-decode")
    p4_html = f4.to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id="roofline-plot")
    tables_html = build_tables_html(stages)

    stage_badges_html = " ".join(
        f'<span class="stage-pill-tag"><span class="stage-dot" style="background:{s["color"]};"></span>{s["name"]}</span>'
        for s in stages
    )

    stages_client_json = json.dumps([
        {
            "name": s["name"],
            "track": s["track"],
            "stage_num": s["stage_num"],
            "color": s["color"],
            "symbol": s["symbol"],
            "pareto": s["pareto"],
            "context": s["context"],
        }
        for s in stages
    ])

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
            padding: 26px 34px;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            border-bottom: 2px solid #ff2a2a;
            padding-bottom: 16px;
            margin-bottom: 22px;
        }}
        .header-left {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        header h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        header .meta {{
            font-family: "JetBrains Mono", Consolas, monospace;
            font-size: 12px;
            color: {TEXT_MUTED};
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .stage-pill-tag {{
            display: inline-flex;
            align-items: center;
            background: #161616;
            border: 1px solid #2a2a2a;
            border-radius: 4px;
            padding: 2px 8px;
            color: #e5e5e5;
            font-size: 11.5px;
            font-weight: 600;
        }}
        .global-track-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: #141414;
            border: 1px solid #333333;
            border-left: 3px solid #ff2a2a;
            border-radius: 6px;
            padding: 8px 14px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.55);
        }}
        .track-bar-label {{
            font-size: 11.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #cccccc;
            white-space: nowrap;
        }}
        .track-pill-group {{
            display: inline-flex;
            background: #0b0b0b;
            border: 1px solid #383838;
            border-radius: 5px;
            overflow: hidden;
        }}
        .track-pill {{
            background: transparent;
            color: #a8a8a8;
            border: none;
            padding: 8px 16px;
            font-family: Inter, sans-serif;
            font-size: 12.5px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.14s ease;
            display: inline-flex;
            align-items: center;
            gap: 7px;
        }}
        .track-pill:hover {{
            background: #222222;
            color: #ffffff;
        }}
        .track-pill.active {{
            background: #ff2a2a;
            color: #000000;
            box-shadow: inset 0 0 8px rgba(0, 0, 0, 0.25);
        }}
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
            padding: 12px 14px;
        }}
        .plot-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            padding: 4px 4px 10px 4px;
            border-bottom: 1px solid #1e1e1e;
            margin-bottom: 4px;
            min-height: 40px;
        }}
        .plot-title {{
            font-size: 16px;
            font-weight: 600;
            color: {TEXT_MAIN};
            font-family: Inter, sans-serif;
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .roofline-key {{
            font-size: 11px;
            font-family: "JetBrains Mono", Consolas, monospace;
            font-weight: 500;
            color: #999999;
        }}

        /* --- Interactive Clickable Legend Hover Enlargement --- */
        .js-plotly-plot .plotly .legend .traces {{
            cursor: pointer !important;
        }}
        .js-plotly-plot .plotly .legend .traces .legendtext {{
            transition: font-size 0.12s ease, fill 0.12s ease, font-weight 0.12s ease;
        }}
        .js-plotly-plot .plotly .legend .traces:hover .legendtext {{
            font-size: 14px !important;
            font-weight: 700 !important;
            fill: #ffffff !important;
        }}
        .js-plotly-plot .plotly .legend .traces:hover .legendtoggle {{
            cursor: pointer !important;
        }}

        /* --- Pin modebar cleanly at top-right --- */
        .js-plotly-plot .plotly .modebar {{
            top: 2px !important;
            right: 4px !important;
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
            margin-bottom: 12px;
        }}
        .table-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .table-header-bar h2 {{
            margin-bottom: 0;
        }}
        .filter-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
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
            padding: 5px 11px;
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
        tr.group-start td {{ border-top: 2px solid #333333; }}
        .stage-cell {{ font-weight: 700; color: #ffffff; }}
        .stage-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 7px;
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
        <div class="header-left">
            <h1>LLM INFERENCE &amp; KERNEL BENCHMARK</h1>
            <div class="meta">Stages (Track-Ordered): {stage_badges_html}</div>
        </div>
        <div class="global-track-bar">
            <span class="track-bar-label">Plots Track View:</span>
            <div class="track-pill-group" id="global-track-pills">
                <button class="track-pill{' active' if active_track == 'custom' else ''}" data-track="custom" onclick="setGlobalTrack('custom')">● Custom Track</button>
                <button class="track-pill{' active' if active_track == 'vllm' else ''}" data-track="vllm" onclick="setGlobalTrack('vllm')">◆ vLLM Reference</button>
                <button class="track-pill{' active' if active_track == 'all' else ''}" data-track="all" onclick="setGlobalTrack('all')">●◆ Both Tracks</button>
            </div>
        </div>
    </header>

    <div class="grid-2x2">
        <div class="card">{p1_header_html}{p1_html}</div>
        <div class="card">{p2_header_html}{p2_html}</div>
        <div class="card">{p3_header_html}{p3_html}</div>
        <div class="card">{p4_controls_html}{p4_html}</div>
    </div>

    {tables_html}

    <script>
        const STAGES_CLIENT_DATA = {stages_client_json};
        const ROOFLINE_STAGE_DATA = {roofline_json};
        let currentGlobalTrack = '{active_track}';
        let currentRooflinePhase = 'all';
        let currentNcuPhase = 'all';

        const PLOT1_LABELS = {{
            'p99_itl': 'ITL P99 (ms)',
            'p90_itl': 'ITL P90 (ms)',
            'p50_itl': 'ITL P50 (ms)',
            'mean_tpot': 'TPOT Mean (ms/token)'
        }};

        const PLOT3_LABELS = {{
            'tpot': 'TPOT (ms/token)',
            'gen_speed': 'Generation Speed (tokens/s/req)'
        }};

        function paddedMax(vals, pad = 1.15) {{
            const valid = vals.filter(v => typeof v === 'number' && !isNaN(v) && v > 0);
            if (!valid.length) return [0, 1];
            return [0, Math.max(...valid) * pad];
        }}

        function getActiveTracesData(plotDiv) {{
            if (!plotDiv || !plotDiv.data) return [];
            return plotDiv.data.filter(t => t.visible === true || t.visible === undefined);
        }}

        function rescalePlot1Axes() {{
            const plotDiv = document.getElementById('plot1-pareto');
            if (!plotDiv || !plotDiv.data) return;
            const active = getActiveTracesData(plotDiv);
            const xs = [];
            const ys = [];
            active.forEach(t => {{
                (t.x || []).forEach(v => xs.push(v));
                (t.y || []).forEach(v => ys.push(v));
            }});
            const metricKey = document.getElementById('plot1-metric-select').value;
            Plotly.relayout(plotDiv, {{
                'xaxis.range': paddedMax(xs, 1.15),
                'yaxis.range': paddedMax(ys, 1.15),
                'yaxis.title.text': PLOT1_LABELS[metricKey] || 'Latency (ms)'
            }});
        }}

        function rescalePlot2Axes() {{
            const plotDiv = document.getElementById('plot2-prefill');
            if (!plotDiv || !plotDiv.data) return;
            const active = getActiveTracesData(plotDiv);
            const xs = [];
            const ys = [];
            active.forEach(t => {{
                (t.x || []).forEach(v => xs.push(v));
                (t.y || []).forEach(v => ys.push(v));
            }});
            Plotly.relayout(plotDiv, {{
                'xaxis.range': paddedMax(xs, 1.08),
                'yaxis.range': paddedMax(ys, 1.15)
            }});
        }}

        function rescalePlot3Axes() {{
            const plotDiv = document.getElementById('plot3-decode');
            if (!plotDiv || !plotDiv.data) return;
            const active = getActiveTracesData(plotDiv);
            const xs = [];
            const ys = [];
            active.forEach(t => {{
                (t.x || []).forEach(v => xs.push(v));
                (t.y || []).forEach(v => ys.push(v));
            }});
            const metricKey = document.getElementById('plot3-metric-select').value;
            Plotly.relayout(plotDiv, {{
                'xaxis.range': paddedMax(xs, 1.08),
                'yaxis.range': paddedMax(ys, 1.15),
                'yaxis.title.text': PLOT3_LABELS[metricKey] || 'TPOT (ms/token)'
            }});
        }}

        function updatePlot1Metric() {{
            const plotDiv = document.getElementById('plot1-pareto');
            if (!plotDiv) return;
            const key = document.getElementById('plot1-metric-select').value;
            const yArrays = STAGES_CLIENT_DATA.map(s => s.pareto.map(p => p[key]));
            Plotly.restyle(plotDiv, {{ y: yArrays }}).then(() => rescalePlot1Axes());
        }}

        function updatePlot3Metric() {{
            const plotDiv = document.getElementById('plot3-decode');
            if (!plotDiv) return;
            const key = document.getElementById('plot3-metric-select').value;
            const yArrays = STAGES_CLIENT_DATA.map(s => s.context.map(p => p[key]));
            Plotly.restyle(plotDiv, {{ y: yArrays }}).then(() => rescalePlot3Axes());
        }}

        function setGlobalTrack(track) {{
            currentGlobalTrack = track;
            document.querySelectorAll('#global-track-pills .track-pill').forEach(btn => {{
                btn.classList.toggle('active', btn.getAttribute('data-track') === track);
            }});

            const visFlags = STAGES_CLIENT_DATA.map(s =>
                (track === 'all' || s.track === track) ? true : 'legendonly'
            );

            ['plot1-pareto', 'plot2-prefill', 'plot3-decode'].forEach(id => {{
                const div = document.getElementById(id);
                if (div) Plotly.restyle(div, {{ visible: visFlags }});
            }});

            rescalePlot1Axes();
            rescalePlot2Axes();
            rescalePlot3Axes();

            // Sync Plot 4 Roofline stage dropdown options & trace visibility
            const stageSelect = document.getElementById('roofline-stage-select');
            if (stageSelect) {{
                stageSelect.value = 'all';
                Array.from(stageSelect.options).forEach(opt => {{
                    const optTrack = opt.getAttribute('data-track');
                    if (!optTrack) return;
                    opt.style.display = (track === 'all' || optTrack === track) ? '' : 'none';
                }});
            }}

            const roofDiv = document.getElementById('roofline-plot');
            if (roofDiv && ROOFLINE_STAGE_DATA.length) {{
                const roofVis = ROOFLINE_STAGE_DATA.map(sObj =>
                    (track === 'all' || sObj.track === track) ? true : 'legendonly'
                );
                const traceIndices = ROOFLINE_STAGE_DATA.map((_, idx) => idx + 1);
                Plotly.restyle(roofDiv, {{ visible: roofVis }}, traceIndices).then(() => {{
                    filterRooflinePlot();
                }});
            }}
        }}

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

            // Step 1: Filter kernels per stage, respecting current trace visibility and dropdown filters
            const filteredByStage = ROOFLINE_STAGE_DATA.map((sObj, sIdx) => {{
                const traceObj = plotDiv.data ? plotDiv.data[sIdx + 1] : null;
                const isTraceVisible = !traceObj || traceObj.visible === true || traceObj.visible === undefined;
                if (!isTraceVisible) return [];

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

            // Step 2: Detect overlapping points across visible stages in log-space
            // so we don't double-stamp text and can append "(×N)" on the top trace
            const eps = 0.035; // ~8% log10 proximity threshold
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
                    found.members.push({{
                        sIdx,
                        kIdx,
                        stage: ROOFLINE_STAGE_DATA[sIdx].stage,
                        short_label: k.short_label || k.label
                    }});
                }});
            }});

            // Alternate text positions across distinct clusters to prevent nearby labels from colliding
            const posCycle = ['top center', 'bottom center', 'top right', 'bottom left', 'top left'];
            clusters.sort((a, b) => a.lx - b.lx || a.ly - b.ly);
            clusters.forEach((c, cIdx) => {{
                c.textPos = posCycle[cIdx % posCycle.length];
            }});

            const xs = [];
            const ys = [];
            const texts = [];
            const textPositions = [];
            const cdatas = [];
            const symbols = [];
            const traceIndices = [];

            filteredByStage.forEach((ks, sIdx) => {{
                const sObj = ROOFLINE_STAGE_DATA[sIdx];
                traceIndices.push(sIdx + 1); // Trace 0 is the fixed Hardware Ceiling line
                xs.push(ks.map(k => k.x));
                ys.push(ks.map(k => k.y));
                symbols.push(ks.map(k => k.symbol));

                const stageTexts = [];
                const stageTextPos = [];
                const stageCdatas = [];

                ks.forEach((k, kIdx) => {{
                    const lx = Math.log10(k.x);
                    const ly = Math.log10(k.y);
                    const cluster = clusters.find(c => Math.abs(c.lx - lx) < eps && Math.abs(c.ly - ly) < eps);
                    const count = cluster ? cluster.members.length : 1;
                    const lastMember = cluster ? cluster.members[cluster.members.length - 1] : null;
                    const stageNames = cluster ? cluster.members.map(m => m.stage).join(' & ') : sObj.stage;
                    const dispLabel = k.short_label || k.label;

                    stageTextPos.push(cluster ? cluster.textPos : 'top center');

                    // Only draw the text label on the top-most overlapping marker to keep font crisp
                    if (count > 1) {{
                        if (lastMember && lastMember.sIdx === sIdx && lastMember.kIdx === kIdx) {{
                            stageTexts.push(`${{dispLabel}} (×${{count}})`);
                        }} else {{
                            stageTexts.push('');
                        }}
                    }} else {{
                        stageTexts.push(dispLabel);
                    }}

                    stageCdatas.push([k.phase, k.exec_us, k.dram_gbps, k.label, stageNames, sObj.track.toUpperCase()]);
                }});

                texts.push(stageTexts);
                textPositions.push(stageTextPos);
                cdatas.push(stageCdatas);
            }});

            Plotly.restyle(plotDiv, {{
                x: xs,
                y: ys,
                text: texts,
                textposition: textPositions,
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
            setGlobalTrack(currentGlobalTrack);

            // Re-scale axes or re-filter Roofline labels whenever user clicks a stage in a Plotly legend
            const p1 = document.getElementById('plot1-pareto');
            const p2 = document.getElementById('plot2-prefill');
            const p3 = document.getElementById('plot3-decode');
            const p4 = document.getElementById('roofline-plot');

            if (p1 && p1.on) p1.on('plotly_restyle', () => setTimeout(rescalePlot1Axes, 20));
            if (p2 && p2.on) p2.on('plotly_restyle', () => setTimeout(rescalePlot2Axes, 20));
            if (p3 && p3.on) p3.on('plotly_restyle', () => setTimeout(rescalePlot3Axes, 20));
            if (p4 && p4.on) {{
                let filteringRoof = false;
                p4.on('plotly_restyle', (evtData) => {{
                    if (filteringRoof) return;
                    if (evtData && evtData[0] && ('visible' in evtData[0])) {{
                        filteringRoof = true;
                        setTimeout(() => {{
                            filterRooflinePlot();
                            filteringRoof = false;
                        }}, 20);
                    }}
                }});
            }}
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
        stage_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir()],
            key=stage_sort_key,
        ) if base_dir.exists() else []

    if not stage_dirs:
        raise SystemExit(f"No valid stage directories found in '{base_dir}'.")

    stages_data = [load_stage_data(d) for d in stage_dirs]
    # Filter out empty directories that contain no benchmark or profiling data
    stages_data = [
        s for s in stages_data
        if s["pareto"] or s["context"] or s["nsys"] or s["ncu"]
    ]
    stages_data.sort(key=stage_sort_key)

    if not stages_data:
        raise SystemExit(f"No benchmark data found in stage directories of '{base_dir}'.")

    out_file = Path(args.output) if args.output else (base_dir / "dashboard.html")
    generate_dashboard(stages_data, out_file)


if __name__ == "__main__":
    main()