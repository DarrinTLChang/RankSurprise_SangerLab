# plotting.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import rankdata

from config import *
from .utils import *


def _moving_average(y: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins is None or window_bins <= 1:
        return y
    kernel = np.ones(int(window_bins), dtype=float) / float(int(window_bins))
    return np.convolve(y.astype(float), kernel, mode="same")


def add_raster_row(
    fig: go.Figure,
    x_ms: np.ndarray,
    y_row: float,
    elec: str,
    cl: int,
    bursts_for_unit: list[dict[str, Any]],
    row: int,
    col: int,
    *,
    show_burst_overlay: bool = True,
    burst_color: str = "rgb(220,0,0)",
    ):
    spk = x_ms
    fig.add_trace(go.Scattergl(
        x=spk / 1000.0,
        y=np.full_like(spk, y_row, dtype=float),
        mode="markers",
        text=np.full(spk.size, f"{elec} C{cl}"),
        hovertemplate="%{text}<br>t=%{x:.3f}s<extra></extra>",
        marker=dict(
            symbol="line-ns-open",
            size=RASTER_DOT_SIZE,
            line=dict(width=1),
            color=color_for_nonburst(elec),
        ),
        opacity=RASTER_OPACITY_BASE,
        showlegend=False
    ), row=row, col=col)

    if bursts_for_unit and show_burst_overlay:
        mask = np.zeros_like(spk, dtype=bool)
        for b in bursts_for_unit:
            mask |= (spk >= b["Start_ms"]) & (spk <= b["End_ms"])
        if mask.any():
            fig.add_trace(go.Scattergl(
                x=spk[mask] / 1000.0,
                y=np.full(mask.sum(), y_row, dtype=float),
                mode="markers",
                text=np.full(mask.sum(), f"{elec} C{cl}"),
                hovertemplate="%{text}<br>t=%{x:.3f}s<extra></extra>",
                marker=dict(symbol="line-ns-open", size=RASTER_DOT_SIZE,
                            line=dict(width=1), color=burst_color),
                opacity=RASTER_OPACITY_BURST,
                showlegend=False
            ), row=row, col=col)


def add_coactivity_panel(fig: go.Figure, co_df: pd.DataFrame,
                         record_len_s: float, row: int, col: int,
                         bin_s: float, stride_s: float,
                         connect: bool = True,
                         smooth_sec: float = 0.0):
    _df = co_df[pd.to_numeric(co_df["Window_start_s"], errors="coerce").notna()].copy()
    _df["Window_start_s"] = _df["Window_start_s"].astype(float)
    _df["Active_channels"] = _df["Active_channels"].astype(float)

    x = _df["Window_start_s"].to_numpy() + (bin_s / 2.0)
    y = _df["Active_channels"].to_numpy()

    if smooth_sec and smooth_sec > 0:
        smooth_bins = max(1, int(round(smooth_sec / stride_s)))
        y = _moving_average(y, smooth_bins)

    fig.add_trace(go.Scattergl(
        x=x, y=y,
        mode="lines+markers" if connect else "markers",
        marker=dict(size=COACTIVITY_DOT_SIZE, color=BURST_CHANNELS_COLOR),
        line=dict(color=BURST_CHANNELS_COLOR),
        hovertemplate="t=%{x:.3f}s<br>Active=%{y}<extra></extra>",
        name="Burst channels",
        showlegend=False
    ), row=row, col=col)

    fig.update_xaxes(range=[0, record_len_s], row=row, col=col)


def add_region_coactivity_panel(fig: go.Figure, df_reg: pd.DataFrame,
                                record_len_s: float, row: int, col: int,
                                bin_s: float, stride_s: float,
                                connect: bool = True, smooth_sec: float = 0.0,
                                show_legend: bool = True):
    cols = [c for c in df_reg.columns if c not in ("Window_start_s", "Window_end_s")]
    for region in sorted(cols):
        colr = REGION_COLORS.get(region, "rgb(30,30,200)")

        _df = df_reg[pd.to_numeric(df_reg["Window_start_s"], errors="coerce").notna()].copy()
        _df["Window_start_s"] = _df["Window_start_s"].astype(float)
        _df[region] = pd.to_numeric(_df[region], errors="coerce")

        x = _df["Window_start_s"].to_numpy() + (bin_s / 2.0)
        y = _df[region].to_numpy()

        if smooth_sec and smooth_sec > 0:
            smooth_bins = max(1, int(round(smooth_sec / stride_s)))
            y = _moving_average(y, smooth_bins)

        if REGIONAL_OUTLINE_WIDTH and REGIONAL_OUTLINE_WIDTH > 0:
            fig.add_trace(go.Scattergl(
                x=x, y=y,
                mode="lines",
                line=dict(color=REGIONAL_OUTLINE_COLOR, width=REGIONAL_OUTLINE_WIDTH),
                marker=dict(size=0),
                opacity=1.0,
                hoverinfo="skip",
                showlegend=False
            ), row=row, col=col)

        fig.add_trace(go.Scattergl(
            x=x, y=y,
            mode="lines",
            line=dict(color=colr, width=REGIONAL_LINE_WIDTH),
            marker=dict(size=0),
            opacity=REGIONAL_OPACITY,
            name=region,
            legendgroup=region,
            showlegend=show_legend,
            hovertemplate="%{y}<extra></extra>"
        ), row=row, col=col)

    fig.update_xaxes(range=[0, record_len_s], row=row, col=col)


def save_fig_interactive(fig: go.Figure, base_name):
    from pathlib import Path
    base_name = Path(base_name)
    base_name.parent.mkdir(parents=True, exist_ok=True)
    html_path = base_name if base_name.suffix == ".html" else base_name.with_suffix(".html")
    fig.write_html(
        str(html_path),
        include_plotlyjs="embed",
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False}
    )
    print(f"• Saved interactive {html_path}")


def add_region_legend_burst(fig: go.Figure, *, row: int = 1, col: int = 1):
    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgb(220,0,0)", width=24),
            name="Network Burst (L)",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="line-ns", size=18,
                color="rgba(0,0,0,0)",
                line=dict(width=4, color="rgb(220,0,0)")
            ),
            name="Regional Bursts (L)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgb(153,51,255)", width=24),
            name="Network Burst (R)",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="line-ns", size=8,
                color="rgba(0,0,0,0)",
                line=dict(width=4, color="rgb(153,51,255)")
            ),
            name="Regional Bursts (R)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[0], y=[0],
            marker=dict(color="rgb(249, 153, 153)"),
            name="Network Burst (L)",
            legendgroup="FR/EMG",
            legendgrouptitle_text="FR/EMG",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[0], y=[0],
            marker=dict(color="rgb(214, 173, 255)"),
            name="Network Burst (R)",
            legendgroup="FR/EMG",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def add_emg_panel(fig: go.Figure,
                  t_s: np.ndarray,
                  emg_traces: list[tuple[str, np.ndarray]],
                  row: int,
                  col: int = 1,
                  downsample: int = 20,
                  side_label: str = "",
                  line_color: str = "black",
                  show_y_ticks: bool = False):
    if t_s is None or len(t_s) == 0 or not emg_traces:
        return

    t = np.asarray(t_s, dtype=float)
    keep = np.isfinite(t)
    t = t[keep]

    for name, y in emg_traces:
        yy = np.asarray(y, dtype=float)
        yy = yy[keep]
        m = np.isfinite(yy)
        tt = t[m]
        yy = yy[m]

        if downsample and downsample > 1:
            tt = tt[::downsample]
            yy = yy[::downsample]

        fig.add_trace(go.Scattergl(
            x=tt, y=yy,
            mode="lines",
            line=dict(color=line_color, width=1),
            name=name,
            showlegend=False,
        ), row=row, col=col)

    fig.update_yaxes(
        title_text=side_label,
        row=row, col=col,
        showticklabels=show_y_ticks,
        ticks="" if not show_y_ticks else None,
        ticklen=4 if show_y_ticks else 0,
        ticksuffix="p" if show_y_ticks else None,
    )


def plot_isi_rank(total_isi) -> go.Figure:
    ranks = rankdata(total_isi, method="ordinal")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=ranks, nbinsx=50, name="ISI ranks")
    )
    fig.update_layout(
        title="ISI Rank Distribution",
        xaxis_title="Rank",
        yaxis_title="Count"
    )
    return fig


def _compute_network_bars(network_windows, all_bursts):
    """Convert onset windows + burst list into merged bar intervals (seconds)."""
    if not network_windows:
        return []
    bars = bars_from_onset_windows_using_all_bursts(
        network_windows, all_bursts,
        time_unit="ms", min_unique_channels=1, channel_mode="elec_cluster",
        use_onset_plus_length=NETWORK_SPAN_ONSET_PLUS_LENGTH,
    )
    return _merge_windows_s(bars)


def _fr_y_range(fr_df: pd.DataFrame, fixed_max: float | None = None):
    """Return (y0, y1) for FR panel. If fixed_max set, use [0, fixed_max] for consistent scale."""
    if fixed_max is not None and fixed_max > 0:
        return 0.0, float(fixed_max)
    if fr_df is None or len(fr_df) == 0 or "FR_Hz" not in fr_df.columns:
        return 0.0, 1.0
    vals = pd.to_numeric(fr_df["FR_Hz"], errors="coerce").dropna()
    if len(vals) == 0:
        return 0.0, 1.0
    y_max = float(vals.max())
    return 0.0, y_max * 1.02 if y_max > 0 else 1.0


def _emg_y_range(traces):
    vals = np.concatenate([np.asarray(y, float) for _, y in traces])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -12.0, 2.0
    y0, y1 = float(np.min(vals)), float(np.max(vals))
    if y1 <= y0:
        y1 = y0 + max(abs(y0) * 0.01, 1e-12) if y0 != 0 else 1.0
    return y0, y1



SIDE_COLORS = {
    "L": {"burst": "rgb(210,0,0)", "network": "red", "emg_shader": "rgba(240,0,0,0.4)"},
    "R": {"burst": "rgb(110,0,170)", "network": "rgb(110,0,170)", "emg_shader": "rgba(153,51,255,0.4)"},
}


def make_sangerlab_presentation_figure(
    spike_struct_L,
    spike_struct_R,
    STATS,
    all_bursts_L,
    all_bursts_R,
    record_len_s: float,
    patient: str,
    period: str,
    bin_s: float,
    stride_s: float,
    run_params: dict,
    *,
    presentation_mode: bool = False,
    allowed_clusters: set[tuple[str, int]] | None = None,
    emg_t_s: np.ndarray | None = None,
    emg_traces_L: list[tuple[str, np.ndarray]] | None = None,
    emg_traces_R: list[tuple[str, np.ndarray]] | None = None,
    emg_downsample: int = 100,
    emg_panel_labels: tuple[str, str] | None = None,
    emg_y_range: tuple[float, float] | None = None,
    network_windows_L=None,
    network_windows_R=None,
    network_bars_L=None,
    network_bars_R=None,
    region_windows_by_region_L=None,
    region_windows_by_region_R=None,
    region_bars_by_region_L=None,
    region_bars_by_region_R=None,
    disable_bursts: bool = False,
    fr_df_L: pd.DataFrame | None = None,
    fr_df_R: pd.DataFrame | None = None,
    show_firing_rate: bool = True,
) -> go.Figure:

    show_emg = (
        emg_t_s is not None
        and emg_traces_L is not None and len(emg_traces_L) > 0
        and emg_traces_R is not None and len(emg_traces_R) > 0
    )
    show_fr = (
        show_firing_rate
        and fr_df_L is not None and len(fr_df_L) > 0
        and fr_df_R is not None and len(fr_df_R) > 0
    )

    # ---- subplot layout: raster, then neo (EMG/proxy), then FR ----
    if show_emg and show_fr:
        n_rows, row_heights = 6, [0.38, 0.38, 0.27, 0.27, 0.20, 0.20]
    elif show_emg:
        n_rows, row_heights = 4, [0.22, 0.22, 0.28, 0.28]
    elif show_fr:
        n_rows, row_heights = 4, [0.3, 0.3, 0.24, 0.24]
    else:
        n_rows, row_heights = 2, [0.60, 0.40]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # ---- per-side data ----
    sides = [
        {"tag": "L", "row": 1, "label": "Left",
         "struct": spike_struct_L, "bursts": all_bursts_L,
         "net_wins": network_windows_L, "net_bars": network_bars_L if network_bars_L is not None else [],
         "reg_wins": region_windows_by_region_L, "reg_bars": region_bars_by_region_L if region_bars_by_region_L is not None else {},
         "fr_df": fr_df_L, "emg_traces": emg_traces_L},
        {"tag": "R", "row": 2, "label": "Right",
         "struct": spike_struct_R, "bursts": all_bursts_R,
         "net_wins": network_windows_R, "net_bars": network_bars_R if network_bars_R is not None else [],
         "reg_wins": region_windows_by_region_R, "reg_bars": region_bars_by_region_R if region_bars_by_region_R is not None else {},
         "fr_df": fr_df_R, "emg_traces": emg_traces_R},
    ]

    # ---- raster panels ----
    region_orders: dict[str, list[str]] = {}

    for side in sides:
        row_all = 0
        last_region = None
        tickvals: list[float] = []
        ticktext: list[str] = []
        region_seen: set[str] = set()
        region_first_seen: list[str] = []
        colors = SIDE_COLORS[side["tag"]]

        for ch in np.ravel(side["struct"]):
            elec = str(ch.electrode)
            reg = infer_region(elec)
            for cl, arr in enumerate(np.ravel(ch.time)):
                if (elec, int(cl)) not in STATS.index:
                    continue
                if allowed_clusters is not None and (elec, int(cl)) not in allowed_clusters:
                    continue
                if reg not in region_seen:
                    region_seen.add(reg)
                    region_first_seen.append(reg)

                spk = np.asarray(arr).ravel().astype(float)
                spk = spk[np.isfinite(spk)]
                if spk.size < 2:
                    continue
                spk = np.sort(spk)

                bursts_for_unit = [
                    b for b in side["bursts"]
                    if b.get("Electrode") == elec and int(b.get("Cluster", -1)) == int(cl)
                ]

                y_plot = row_all * RASTER_ROW_SPACING
                row_all += 1
                tickvals.append(y_plot)
                ticktext.append(reg if reg != last_region else "")
                last_region = reg

                add_raster_row(
                    fig, spk, y_plot, elec, int(cl), bursts_for_unit,
                    row=side["row"], col=1,
                    show_burst_overlay=(not disable_bursts),
                    burst_color=colors["burst"],
                )

        y_max = row_all * RASTER_ROW_SPACING
        y_min = -(8.0 * RASTER_ROW_SPACING)

        fig.update_yaxes(
            title_text=side["label"], tickvals=tickvals, ticktext=ticktext,
            tickfont=dict(size=14), ticks="", ticklen=0,
            range=[y_min, y_max], autorange="reversed",
            row=side["row"], col=1, showline=False, zeroline=False,
        )
        fig.add_shape(
            type="line", x0=0.0, x1=0.0, y0=0.0, y1=float(y_max),
            line=dict(color="black", width=5), layer="above",
            row=side["row"], col=1,
        )
        fig.update_xaxes(showticklabels=False, row=side["row"], col=1)
        region_orders[side["tag"]] = region_first_seen

    # ---- network bars (for FR and EMG overlays) ----
    # Use pre-computed bars from detection (already filtered and merged)
    net_bars_L = network_bars_L if network_bars_L is not None else []
    net_bars_R = network_bars_R if network_bars_R is not None else []

    # ---- firing-rate panels (rows 5,6 when EMG also shown; else 3,4) ----
    if show_fr:
        for i, side in enumerate(sides):
            fr_row = (5 if show_emg else 3) + i
            add_firing_rate_panel(
                fig, side["fr_df"], record_len_s,
                row=fr_row, col=1,
                bin_s=bin_s, stride_s=stride_s,
                connect=True, smooth_sec=SMOOTH_PANEL_SEC,
                line_color=SIDE_COLORS[side["tag"]]["burst"],
            )
            fig.update_xaxes(showticklabels=False, row=fr_row, col=1)

            # Network burst overlay on FR panel: both L and R bars on each panel
            y0, y1 = _fr_y_range(side["fr_df"], fixed_max=FR_Y_MAX)
            for bars, color_key in [(net_bars_L, "L"), (net_bars_R, "R")]:
                if bars:
                    add_emg_shaders_from_windows(
                        fig, bars, row=fr_row, col=1,
                        y0=y0, y1=y1,
                        color=SIDE_COLORS[color_key]["emg_shader"],
                        opacity=1, layer="below", time_unit="s",
                    )

    # ---- network & region burst bars on raster ----
    y_top = -5 * RASTER_ROW_SPACING
    bar_height = 3.5 * RASTER_ROW_SPACING

    for side in sides:
        colors = SIDE_COLORS[side["tag"]]

        if (side["net_bars"] or side["net_wins"]) and plot_network_bar:
            # Use pre-computed bars from detection (net_bars alone is enough to draw)
            net_bars = side["net_bars"]
            if net_bars:
                add_burst_bars_top_SIMPLE(
                    fig, net_bars, row=side["row"], col=1,
                    y_top=y_top, bar_height=bar_height,
                    color=colors["network"], opacity=0.9, line_width=1,
                )

        if side["reg_wins"] and plot_region_bar:
            # Use pre-computed bars from detection
            plot_region_bars_from_precomputed(
                fig, row=side["row"], col=1,
                region_bars_by_region=side["reg_bars"],
                region_colors=REGION_COLORS,
                region_order=list(reversed(region_orders[side["tag"]])),
                y_top_start=(y_top - 2.5 * bar_height),
                bar_height=bar_height / 2,
                y_step=1.2 * bar_height,
                opacity=0.9, line_width=1.0,
            )

    # ---- EMG/proxy panels (rows 3,4; before FR when both shown) ----
    emg_labels = emg_panel_labels if emg_panel_labels is not None else ("Summed EMG (L)", "Summed EMG (R)")
    if show_emg:
        for i, side in enumerate(sides):
            emg_row = 3 + i

            add_emg_panel(
                fig, emg_t_s, side["emg_traces"], row=emg_row, col=1,
                downsample=emg_downsample,
                side_label=emg_labels[i],
                line_color=SIDE_COLORS[side["tag"]]["burst"],
                show_y_ticks=(emg_panel_labels is not None),
            )

            y0, y1 = (emg_y_range if emg_y_range is not None else _emg_y_range(side["emg_traces"]))
            for bars, color_key in [(net_bars_L, "L"), (net_bars_R, "R")]:
                if bars:
                    add_emg_shaders_from_windows(
                        fig, bars, row=emg_row, col=1,
                        y0=y0, y1=y1,
                        color=SIDE_COLORS[color_key]["emg_shader"],
                        opacity=1, layer="below", time_unit="s",
                    )

            # Y-axis range: fixed when provided (e.g. [0,1] for proxy), else tight to data
            fig.update_yaxes(range=[y0, y1], row=emg_row, col=1)
            fig.update_xaxes(showticklabels=False, row=emg_row, col=1)

    # ---- x-axis and title (time scale visible on bottom row) ----
    fig.update_xaxes(
        title_text="Time (s)", range=[0, record_len_s],
        showticklabels=True,
        row=n_rows, col=1,
    )
    for r in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    if presentation_mode:
        fig_title = f"{patient} \u2022 {period} \u2014 {run_params['method']}"
    else:
        fig_title = figure_title_from_params(run_params, patient, period)

    add_region_legend_burst(fig, row=1, col=1)
    fig.update_layout(
        title=fig_title,
        template="simple_white",
        margin=COMMON_MARGINS,
        hovermode="closest",
        height=1200,
        showlegend=True,
        legend_title_text="Legend",
        legend=dict(itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
    )

    return fig


def bars_from_onset_windows_using_all_bursts(
    network_windows_ms,
    all_bursts,
    *,
    time_unit: str = "ms",
    channel_mode: str = "elec_cluster",
    min_unique_channels: int = 1,
    pad_ms: float = 0.0,
    use_onset_plus_length: bool = False,
) -> list[tuple[float, float]]:
    if not network_windows_ms:
        return []

    if time_unit.lower() == "s":
        windows_ms = [(float(a) * 1000.0, float(b) * 1000.0) for a, b in network_windows_ms]
    else:
        windows_ms = [(float(a), float(b)) for a, b in network_windows_ms]

    def ch_id(b):
        elec = str(b.get("Electrode", ""))
        if channel_mode == "elec":
            return elec
        cl = int(b.get("Cluster", -1))
        return f"{elec}|{cl}"

    bars_s: list[tuple[float, float]] = []

    for w0, w1 in windows_ms:
        if not (np.isfinite(w0) and np.isfinite(w1)) or w1 <= w0:
            continue

        overlapping = []
        chans = set()

        for b in all_bursts:
            s = float(b.get("Start_ms", np.nan))
            e = float(b.get("End_ms", np.nan))
            if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
                continue
            if (s >= w0) and (s <= w1):
                overlapping.append((s, e))
                chans.add(ch_id(b))

        if not overlapping:
            continue
        if len(chans) < int(min_unique_channels):
            continue

        if use_onset_plus_length:
            lengths = [e - s for (s, e) in overlapping]
            median_length_ms = float(np.median(lengths))
            b0 = w0 / 1000.0
            b1 = (w0 + median_length_ms + pad_ms) / 1000.0
        else:
            starts = [s for s, _ in overlapping]
            ends = [e for _, e in overlapping]
            b0 = float(np.median(starts)) - pad_ms
            b1 = float(np.median(ends)) + pad_ms
            b0, b1 = b0 / 1000.0, b1 / 1000.0
        if b1 <= b0:
            continue

        bars_s.append((b0, b1))

    return bars_s


def add_burst_bars_top_SIMPLE(
    fig, windows_s, *,
    row: int, col: int = 1,
    y_top: float, bar_height: float,
    color: str = "red", opacity: float = 1.0,
    layer: str = "above",
    line_color: str | None = None,
    line_width: float = 0.0,
):
    if not windows_s:
        return
    if line_color is None:
        line_color = color

    for (t0, t1) in windows_s:
        if t1 <= t0:
            continue
        fig.add_shape(
            type="rect",
            x0=float(t0), x1=float(t1),
            y0=float(y_top - bar_height), y1=float(y_top),
            fillcolor=color, opacity=opacity,
            line=dict(color=line_color, width=float(line_width)),
            layer=layer,
            row=row, col=col,
        )


def plot_region_bars_from_precomputed(
    fig, *, row: int, col: int,
    region_bars_by_region: dict[str, list[tuple[float, float]]],
    region_colors: dict,
    region_order: list[str],
    y_top_start: float,
    bar_height: float,
    y_step: float,
    opacity: float = 0.9,
    line_width: float = 1.0,
):
    """Plot region bars using pre-computed bars from detection."""
    y_top = float(y_top_start)

    for reg in region_order:
        bars_s = region_bars_by_region.get(reg, [])
        if not bars_s:
            continue

        add_burst_bars_top_SIMPLE(
            fig, bars_s,
            row=row, col=col,
            y_top=y_top, bar_height=bar_height,
            color=region_colors.get(reg, "blue"),
            opacity=opacity, line_width=line_width,
        )

        y_top -= float(y_step)


def plot_region_bars_simple(
    fig, *, row: int, col: int,
    region_windows_by_region: dict,
    all_bursts: list[dict],
    region_colors: dict,
    region_order: list[str],
    y_top_start: float,
    bar_height: float,
    y_step: float,
    min_unique_channels: int = 1,
    channel_mode: str = "elec_cluster",
    opacity: float = 0.9,
    line_width: float = 1.0,
):
    y_top = float(y_top_start)

    for reg in region_order:
        wins = region_windows_by_region.get(reg, [])
        if not wins:
            continue

        reg_bursts = [b for b in all_bursts if str(b.get("_Region", infer_region(str(b["Electrode"])))) == reg]

        windows_s = bars_from_onset_windows_using_all_bursts(
            wins, reg_bursts,
            time_unit="ms",
            min_unique_channels=min_unique_channels,
            channel_mode=channel_mode,
        )

        add_burst_bars_top_SIMPLE(
            fig, windows_s,
            row=row, col=col,
            y_top=y_top, bar_height=bar_height,
            color=region_colors.get(reg, "blue"),
            opacity=0.95, line_width=line_width,
        )

        y_top -= float(y_step)


def add_emg_shaders_from_windows(
    fig: go.Figure,
    windows_ms: list[tuple[float, float]] | None,
    *, row: int, col: int = 1,
    y0: float, y1: float,
    color: str = "rgb(220,0,0)",
    opacity: float = 0.18,
    layer: str = "below",
    time_unit: str = "ms",
):
    if not windows_ms:
        return

    for (t0, t1) in windows_ms:
        if t1 <= t0:
            continue

        if time_unit.lower() == "ms":
            x0 = float(t0) / 1000.0
            x1 = float(t1) / 1000.0
        elif time_unit.lower() == "s":
            x0 = float(t0)
            x1 = float(t1)
        else:
            raise ValueError("time_unit must be 'ms' or 's'")

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=float(y0), y1=float(y1),
            fillcolor=color,
            opacity=float(opacity),
            line_width=0,
            layer=layer,
            row=row, col=col,
        )


def _merge_windows_s(wins):
    wins = sorted(wins, key=lambda t: t[0])
    merged = []
    for t0, t1 in wins:
        if not merged:
            merged.append([t0, t1])
        elif t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [(a, b) for a, b in merged]


def population_firing_rate_binned(
    spike_struct, STATS: pd.DataFrame, record_len_s: float, *,
    bin_s: float, stride_s: float,
    allowed_clusters: set[tuple[str, int]] | None = None,
    normalize_by_units: bool = True,
) -> pd.DataFrame:
    bin_s = float(bin_s)
    stride_s = float(stride_s)
    if bin_s <= 0 or stride_s <= 0:
        raise ValueError("bin_s and stride_s must be > 0")

    all_spk_s = []
    n_units = 0

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        for cl, arr in enumerate(np.ravel(ch.time)):
            cl = int(cl)
            if (elec, cl) not in STATS.index:
                continue
            if allowed_clusters is not None and (elec, cl) not in allowed_clusters:
                continue

            spk = np.asarray(arr).ravel().astype(float)
            spk = spk[np.isfinite(spk)]
            if spk.size == 0:
                continue

            n_units += 1
            all_spk_s.append(np.sort(spk) / 1000.0)

    if n_units == 0:
        return pd.DataFrame(columns=["Window_start_s", "Window_end_s", "FR_Hz", "SpikeCount", "N_units"])

    spikes_s = np.sort(np.concatenate(all_spk_s))

    last_start = max(0.0, record_len_s - bin_s)
    starts = np.arange(0.0, last_start + 1e-12, stride_s)
    ends = starts + bin_s

    left_idx = np.searchsorted(spikes_s, starts, side="left")
    right_idx = np.searchsorted(spikes_s, ends, side="left")
    counts = (right_idx - left_idx).astype(float)

    fr = counts / bin_s
    if normalize_by_units:
        fr = fr / float(n_units)

    return pd.DataFrame({
        "Window_start_s": starts,
        "Window_end_s": ends,
        "FR_Hz": fr,
        "SpikeCount": counts,
        "N_units": float(n_units),
    })


def add_firing_rate_panel(
    fig: go.Figure, fr_df: pd.DataFrame, record_len_s: float,
    row: int, col: int, *,
    bin_s: float, stride_s: float,
    connect: bool = True, smooth_sec: float = 0.0,
    name: str = "Population firing rate",
    line_color: str | None = None,
):
    if fr_df is None or len(fr_df) == 0:
        return

    _df = fr_df.copy()
    _df["Window_start_s"] = pd.to_numeric(_df["Window_start_s"], errors="coerce")
    _df["FR_Hz"] = pd.to_numeric(_df["FR_Hz"], errors="coerce")
    _df = _df[_df["Window_start_s"].notna() & _df["FR_Hz"].notna()]

    x = _df["Window_start_s"].to_numpy(dtype=float) + (float(bin_s) / 2.0)
    y = _df["FR_Hz"].to_numpy(dtype=float)

    if smooth_sec and smooth_sec > 0:
        smooth_bins = max(1, int(round(float(smooth_sec) / float(stride_s))))
        y = _moving_average(y, smooth_bins)

    line_kw = dict(width=1)
    if line_color is not None:
        line_kw["color"] = line_color

    fig.add_trace(go.Scattergl(
        x=x, y=y,
        mode="lines" if connect else "markers",
        line=line_kw,
        marker=dict(size=COACTIVITY_DOT_SIZE),
        hovertemplate="t=%{x:.3f}s<br>FR=%{y:.2f} Hz<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.update_xaxes(range=[0, record_len_s], row=row, col=col)


# =====================================================================
# ISI plotting helpers (formerly maxisi_isi_plots.py)
# =====================================================================

def make_isi_hist_figure(isis_in_ms: np.ndarray, isis_out_ms: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=isis_out_ms, name="Non-burst ISIs", opacity=0.6))
    fig.add_trace(go.Histogram(x=isis_in_ms, name="Burst ISIs", opacity=0.6))
    fig.update_layout(
        title=title,
        template="simple_white",
        barmode="overlay",
        xaxis_title="Interspike interval (ms)",
        yaxis_title="Count",
    )
    if ISI_HIST_MAX_MS is not None:
        fig.update_xaxes(range=[0, ISI_HIST_MAX_MS])
    return fig


def make_isi_bar_figure(isis_in_ms: np.ndarray, isis_out_ms: np.ndarray, title: str) -> go.Figure:
    def _stats(a: np.ndarray) -> tuple[float, float, int]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.mean(a)), float(np.std(a, ddof=0)), int(a.size)

    m_in, sd_in, n_in = _stats(isis_in_ms)
    m_out, sd_out, n_out = _stats(isis_out_ms)

    x = ["Burst", "Non-burst"]
    y = [m_in, m_out]

    plus_in = sd_in if np.isfinite(sd_in) else 0.0
    plus_out = sd_out if np.isfinite(sd_out) else 0.0
    minus_in = min(plus_in, m_in) if np.isfinite(m_in) else 0.0
    minus_out = min(plus_out, m_out) if np.isfinite(m_out) else 0.0

    err = dict(
        type="data",
        symmetric=False,
        array=[plus_in, plus_out],
        arrayminus=[minus_in, minus_out],
        visible=bool(ISI_BAR_SHOW_STD),
    )

    txt = [f"n={n_in}", f"n={n_out}"]
    fig = go.Figure()
    if ISI_BAR_SHOW_STD:
        fig.add_trace(go.Bar(x=x, y=y, error_y=err, text=txt, textposition="outside", name="Mean ISI"))
    else:
        fig.add_trace(go.Bar(x=x, y=y, text=txt, textposition="outside", name="Mean ISI"))

    fig.update_layout(
        title=title,
        template="simple_white",
        xaxis_title="Class",
        yaxis_title="Mean ISI (ms)",
        bargap=0.4,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig
