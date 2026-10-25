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
            data["ncu"] = json.load(fp)

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


def build_plot4_roofline(stages: list[dict]) -> go.Figure:
    fig = go.Figure(layout=base_layout(
        "Hardware Roofline (Compute vs. Memory)",
        "Arithmetic Intensity (FLOP/Byte)",
        "Attained Performance (TFLOP/s)"
    ))

    # Determine hardware ceilings from first available ncu_metrics.json
    peak_bw_gbps = 512.0
    peak_compute_tflops = 100.0
    for s in stages:
        if s["ncu"] and "hardware_ceilings" in s["ncu"]:
            c = s["ncu"]["hardware_ceilings"]
            peak_bw_gbps = c.get("estimated_peak_dram_gbps", peak_bw_gbps)
            peak_compute_tflops = c.get("estimated_peak_tensor_tflops", peak_compute_tflops)
            break

    # Ridge point where Memory Roof meets Compute Roof: AI = TFLOPs * 1000 / GBps
    ridge_ai = (peak_compute_tflops * 1000.0) / peak_bw_gbps
    ai_min, ai_max = 0.02, 4096.0

    # Roofline Ceiling Line
    fig.add_trace(go.Scatter(
        x=[ai_min, ridge_ai, ai_max],
        y=[(peak_bw_gbps * ai_min) / 1000.0, peak_compute_tflops, peak_compute_tflops],
        mode="lines",
        name=f"Hardware Ceiling ({peak_bw_gbps:.0f} GB/s | {peak_compute_tflops:.0f} TFLOP/s)",
        line=dict(color="#ff2a2a", width=2, dash="dash"),
        hoverinfo="skip",
    ))

    all_ai, all_tflops = [], []
    for idx, s in enumerate(stages):
        if not s["ncu"]:
            continue
        color = STAGE_COLORS[idx % len(STAGE_COLORS)]
        kernels = [
            k for k in s["ncu"].get("kernels", [])
            if k.get("arithmetic_intensity_flop_byte", 0) > 0 and k.get("attained_tflops", 0) > 0
        ]
        all_ai.extend([k["arithmetic_intensity_flop_byte"] for k in kernels])
        all_tflops.extend([k["attained_tflops"] for k in kernels])

        fig.add_trace(go.Scatter(
            x=[k["arithmetic_intensity_flop_byte"] for k in kernels],
            y=[k["attained_tflops"] for k in kernels],
            text=[k["label"] for k in kernels],
            customdata=[[k["phase"], k["execution_time_us"], k["achieved_dram_gbps"]] for k in kernels],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=10, color=TEXT_MUTED),
            name=s["name"],
            marker=dict(
                size=10,
                color=color,
                symbol=["diamond" if k["phase"] == "Prefill" else "circle" for k in kernels],
                line=dict(width=1.5, color="#000000"),
            ),
            hovertemplate=(
                "<b>%{text}</b> (%{fullData.name} - %{customdata[0]})<br>"
                "AI: %{x:.2f} FLOP/B<br>"
                "Perf: %{y:.3f} TFLOP/s<br>"
                "DRAM: %{customdata[2]:.1f} GB/s<br>"
                "Duration: %{customdata[1]:.2f} us<extra></extra>"
            ),
        ))

    fig.update_xaxes(type="log", range=[math.log10(ai_min), math.log10(ai_max)])
    min_y = min(all_tflops) * 0.4 if all_tflops else 0.0005
    fig.update_yaxes(type="log", range=[math.log10(min_y), math.log10(peak_compute_tflops * 1.8)])
    return fig


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
        dec_sub = f' <span class="sub">({dec_tot:.1f} ms total)</span>' if dec_tot is not None else ""

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

    # Table B: Nsight Compute Kernel Details
    ncu_rows = ""
    for s in stages:
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

            ai = k.get("arithmetic_intensity_flop_byte", 0.0)
            phase_badge = f"<span class='badge badge-{k.get('phase', '').lower()}'>{k.get('phase', '')}</span>"

            ncu_rows += f"""
            <tr>
                <td class="stage-cell">{s['name']}</td>
                <td>{phase_badge} {k.get('label', '')}</td>
                <td>{k.get('execution_time_us', 0):.2f}</td>
                <td>{dram_pct:.1f}% <span class="sub">({k.get('achieved_dram_gbps', 0):.1f} GB/s)</span></td>
                <td>{comp_pct:.1f}% <span class="sub">({k.get('attained_tflops', 0):.2f} TFLOP/s)</span></td>
                <td class="ai-cell">{ai:.2f}</td>
                <td>{k.get('active_warps_pct', 0):.1f}%</td>
                <td>{k.get('registers_per_thread', 0)}</td>
                <td>{k.get('local_spills_bytes', 0)}</td>
            </tr>"""

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
                        <th>CUDA Sync Wait (ms)</th>
                        <th>Prefill Duration</th>
                        <th>Decode Duration</th>
                        <th>Kernel Time (Lin / Attn / Oth)</th>
                    </tr>
                </thead>
                <tbody>{nsys_rows or '<tr><td colspan="9">No nsys_metrics.json found</td></tr>'}</tbody>
            </table>
        </div>

        <div class="card table-card">
            <h2>Table B: Kernel Details (Nsight Compute)</h2>
            <table>
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
    f4 = build_plot4_roofline(stages)

    cfg = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }
    p1_html = f1.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p2_html = f2.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p3_html = f3.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    p4_html = f4.to_html(full_html=False, include_plotlyjs=False, config=cfg)
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

        /* --- Fix 1: Dropdown menu dark-grey hover & active state --- */
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

        /* --- Fix 2: Pin modebar cleanly at top-right above dropdown --- */
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
            margin-bottom: 14px;
            color: {TEXT_MAIN};
            letter-spacing: -0.01em;
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
        .stage-cell {{ font-weight: 700; color: #ffffff; }}
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
        <div class="card">{p4_html}</div>
    </div>

    {tables_html}
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