# plotting.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from config import *
from .numeric import moving_average
from .utils import *

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
            left = int(np.searchsorted(spk, float(b["Start_ms"]), side="left"))
            right = int(np.searchsorted(spk, float(b["End_ms"]), side="right"))
            mask[left:right] = True
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


def add_region_legend_burst(
    fig: go.Figure,
    *,
    row: int = 1,
    col: int = 1,
    include_fr_emg: bool = True,
    side_tags: tuple[str, ...] = ("L", "R"),
):
    # Raster legend items: keep only the sides that are actually plotted.
    if "L" in side_tags:
        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="rgb(220,0,0)", width=24),
                name="Network Burst (L)",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    symbol="line-ns", size=18,
                    color="rgba(0,0,0,0)",
                    line=dict(width=4, color="rgb(220,0,0)"),
                ),
                name="Regional Bursts (L)",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

    if "R" in side_tags:
        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="rgb(153,51,255)", width=24),
                name="Network Burst (R)",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    symbol="line-ns", size=8,
                    color="rgba(0,0,0,0)",
                    line=dict(width=4, color="rgb(153,51,255)"),
                ),
                name="Regional Bursts (R)",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

    # FR/EMG legend group: optionally hide it (kilosort/kilosortset request).
    if include_fr_emg:
        if "L" in side_tags:
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

        if "R" in side_tags:
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

# Multi-panel presentation figures (raster + EMG/FR) use a fixed height. A single raster-only
# subplot used to inherit the full 1200px, which adds empty vertical space and browser scroll;
# use ~20% above the nominal ~600px single-row share instead.
PRESENTATION_FIG_HEIGHT_DEFAULT = 1200
PRESENTATION_FIG_HEIGHT_RASTER_ONLY_ONE_SIDE = int(round(600 * 1.2))


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
    show_proxy_shaders: bool | None = None,
    compact_kilosort_title: bool = False,
    spike_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> go.Figure:

    left_has = spike_struct_L is not None and np.ravel(spike_struct_L).size > 0
    right_has = spike_struct_R is not None and np.ravel(spike_struct_R).size > 0

    side_tags: list[str] = []
    if left_has:
        side_tags.append("L")
    if right_has:
        side_tags.append("R")
    if not side_tags:
        # Fallback: keep a single raster axis rather than failing.
        side_tags = ["L"]

    def _emg_ok(tag: str) -> bool:
        traces = emg_traces_L if tag == "L" else emg_traces_R
        return emg_t_s is not None and traces is not None and len(traces) > 0

    def _fr_ok(tag: str) -> bool:
        df = fr_df_L if tag == "L" else fr_df_R
        return df is not None and len(df) > 0

    show_emg = all(_emg_ok(tag) for tag in side_tags)
    show_fr = show_firing_rate and all(_fr_ok(tag) for tag in side_tags)

    n_sides = len(side_tags)
    # ---- subplot layout: raster, then neo (EMG/proxy), then FR ----
    if show_emg and show_fr:
        n_rows = 3 * n_sides
        row_heights = [0.38] * n_sides + [0.27] * n_sides + [0.20] * n_sides
    elif show_emg:
        n_rows = 2 * n_sides
        row_heights = [0.22] * n_sides + [0.28] * n_sides
    elif show_fr:
        n_rows = 2 * n_sides
        row_heights = [0.30] * n_sides + [0.24] * n_sides
    else:
        n_rows = n_sides
        row_heights = [1.0] if n_sides == 1 else [0.60, 0.40]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # ---- per-side data ----
    label_for_tag = {"L": "Left", "R": "Right"}
    sides = []
    for i, tag in enumerate(side_tags):
        label = "" if compact_kilosort_title else label_for_tag.get(tag, "")
        row = i + 1
        sides.append(
            {
                "tag": tag,
                "row": row,
                "label": label,
                "struct": spike_struct_L if tag == "L" else spike_struct_R,
                "bursts": all_bursts_L if tag == "L" else all_bursts_R,
                "net_wins": network_windows_L if tag == "L" else network_windows_R,
                "net_bars": (network_bars_L if tag == "L" else network_bars_R)
                if (network_bars_L if tag == "L" else network_bars_R) is not None
                else [],
                "reg_wins": region_windows_by_region_L if tag == "L" else region_windows_by_region_R,
                "reg_bars": (region_bars_by_region_L if tag == "L" else region_bars_by_region_R)
                if (region_bars_by_region_L if tag == "L" else region_bars_by_region_R) is not None
                else {},
                "fr_df": fr_df_L if tag == "L" else fr_df_R,
                "emg_traces": emg_traces_L if tag == "L" else emg_traces_R,
            }
        )

    # ---- raster panels ----
    region_orders: dict[str, list[str]] = {}

    for side in sides:
        bursts_by_unit: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for burst in side["bursts"]:
            key = (str(burst.get("Electrode", "")), int(burst.get("Cluster", -1)))
            bursts_by_unit.setdefault(key, []).append(burst)

        row_all = 0
        tickvals: list[float] = []
        ticktext: list[str] = []
        depth_pairs: list[tuple[float, float]] = []  # (y_plot, depth_um)
        region_seen: set[str] = set()
        region_first_seen: list[str] = []
        colors = SIDE_COLORS[side["tag"]]

        ch_list = list(np.ravel(side["struct"]))
        # Sort units for raster display.
        # - Default: globally by depth (mingled across shanks).
        # - Optional: group by shank blocks, each sorted by depth (legacy look).
        last_reg_for_labels = None
        try:
            if bool(KILOSORT_RASTER_GROUP_BY_SHANK):
                ch_list.sort(
                    key=lambda c: (
                        infer_region(str(getattr(c, "electrode", ""))),
                        infer_depth_um(str(getattr(c, "electrode", ""))) is None,
                        infer_depth_um(str(getattr(c, "electrode", ""))) or 0.0,
                    )
                )
            else:
                ch_list.sort(
                    key=lambda c: (
                        infer_depth_um(str(getattr(c, "electrode", ""))) is None,
                        infer_depth_um(str(getattr(c, "electrode", ""))) or 0.0,
                    )
                )
        except Exception:
            pass

        for ch in ch_list:
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

                key = (elec, int(cl))
                if spike_cache is not None and key in spike_cache:
                    spk = spike_cache[key]
                else:
                    spk = np.sort(np.asarray(arr).ravel().astype(float))
                spk = spk[np.isfinite(spk)]
                if spk.size < 2:
                    continue
                bursts_for_unit = bursts_by_unit.get(key, [])

                y_plot = row_all * RASTER_ROW_SPACING
                row_all += 1
                tickvals.append(y_plot)
                d = infer_depth_um(elec)
                if str(elec).startswith("rat_") and d is not None and np.isfinite(d):
                    if bool(KILOSORT_RASTER_GROUP_BY_SHANK):
                        # Shank-separated mode: label block boundaries only (Shank 0, Shank 1, ...)
                        if reg != last_reg_for_labels:
                            ticktext.append(display_region_name(reg))
                            last_reg_for_labels = reg
                        else:
                            ticktext.append("")
                    else:
                        # Mingled mode: show depth scale ticks (not every row)
                        depth_pairs.append((float(y_plot), float(d)))
                        ticktext.append("")
                else:
                    ticktext.append(display_region_name(reg) if (bool(KILOSORT_RASTER_GROUP_BY_SHANK) and reg != last_reg_for_labels) else "")
                    last_reg_for_labels = reg if bool(KILOSORT_RASTER_GROUP_BY_SHANK) else last_reg_for_labels

                add_raster_row(
                    fig, spk, y_plot, elec, int(cl), bursts_for_unit,
                    row=side["row"], col=1,
                    show_burst_overlay=(not disable_bursts),
                    burst_color=colors["burst"],
                )

        y_max = row_all * RASTER_ROW_SPACING
        # Reserve headroom for the top-of-raster bar stack (network + region bars).
        # This keeps the raster panel visually consistent across stages (e.g. Stage 1 vs Stage 2),
        # even when later stages draw additional bars.
        y_min = -(8.0 * RASTER_ROW_SPACING)
        y_top_reserved = -5.0 * RASTER_ROW_SPACING
        bar_height_reserved = 3.5 * RASTER_ROW_SPACING
        if bool(plot_network_bar):
            y_min = min(y_min, y_top_reserved - bar_height_reserved)
        if bool(plot_region_bar):
            n_regions = max(0, len(region_first_seen))
            if n_regions > 0:
                reg_bar_h = bar_height_reserved / 2.0
                reg_step = 1.2 * bar_height_reserved
                reg_top = y_top_reserved - 2.5 * bar_height_reserved
                reg_bottom = reg_top - reg_bar_h - (n_regions - 1) * reg_step
                y_min = min(y_min, reg_bottom)

        # Build readable y ticks. For mingled Kilosort plots we show a depth scale (µm) at regular intervals.
        if depth_pairs and (not bool(KILOSORT_RASTER_GROUP_BY_SHANK)):
            depth_pairs.sort(key=lambda t: t[1])
            depths = np.array([d for (_, d) in depth_pairs], dtype=float)
            yrows = np.array([y for (y, _) in depth_pairs], dtype=float)
            lo = float(np.nanmin(depths))
            hi = float(np.nanmax(depths))
            step = float(getattr(__import__("config"), "KILOSORT_DEPTH_TICK_UM", 200.0))
            step = 200.0 if not np.isfinite(step) or step <= 0 else step
            tick_depths = np.arange(np.floor(lo / step) * step, hi + step, step)
            tickvals = []
            ticktext = []
            for td in tick_depths:
                idx = int(np.argmin(np.abs(depths - td)))
                tickvals.append(float(yrows[idx]))
                ticktext.append(f"{td:.0f} µm")

        fig.update_yaxes(
            title_text=side["label"], tickvals=tickvals, ticktext=ticktext,
            tickfont=dict(size=14), ticks="", ticklen=0,
            range=[y_max, y_min], autorange=False,
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
        fr_row_start = (2 * n_sides + 1) if show_emg else (n_sides + 1)
        for i, side in enumerate(sides):
            fr_row = fr_row_start + i
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
                region_order=list(reversed(region_orders[side["tag"]])),
                y_top_start=(y_top - 2.5 * bar_height),
                bar_height=bar_height / 2,
                y_step=1.2 * bar_height,
                opacity=0.9, line_width=1.0,
            )

    # ---- EMG/proxy panels (rows 3,4; before FR when both shown) ----
    if show_emg:
        if show_proxy_shaders is None:
            # Default to config toggle for backward compatibility.
            show_proxy_shaders = bool(SHOW_PROXY_SHADERS)
        if emg_panel_labels is not None:
            label_L, label_R = emg_panel_labels
        else:
            label_L, label_R = ("Summed EMG (L)", "Summed EMG (R)")
        emg_row_start = n_sides + 1
        for i, side in enumerate(sides):
            emg_row = emg_row_start + i
            side_label = label_L if side["tag"] == "L" else label_R

            add_emg_panel(
                fig, emg_t_s, side["emg_traces"], row=emg_row, col=1,
                downsample=emg_downsample,
                side_label=side_label,
                line_color=SIDE_COLORS[side["tag"]]["burst"],
                show_y_ticks=(emg_panel_labels is not None),
            )

            y0, y1 = (emg_y_range if emg_y_range is not None else _emg_y_range(side["emg_traces"]))
            if show_proxy_shaders:
                # Shade only the matching hemisphere's network bars on each panel.
                bars = net_bars_L if side["tag"] == "L" else net_bars_R
                if bars:
                    add_emg_shaders_from_windows(
                        fig, bars, row=emg_row, col=1,
                        y0=y0, y1=y1,
                        color=SIDE_COLORS[side["tag"]]["emg_shader"],
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
        if compact_kilosort_title:
            fig_title = f"{patient} \u2014 {run_params['method']}"
        else:
            fig_title = f"{patient} \u2022 {period} \u2014 {run_params['method']}"
    else:
        fig_title = figure_title_from_params(
            run_params, patient, period, compact_kilosort=compact_kilosort_title,
        )

    add_region_legend_burst(
        fig,
        row=1,
        col=1,
        include_fr_emg=(not compact_kilosort_title),
        side_tags=tuple(side_tags),
    )
    raster_only = not show_emg and not show_fr
    fig_height = (
        PRESENTATION_FIG_HEIGHT_RASTER_ONLY_ONE_SIDE
        if (raster_only and n_sides == 1)
        else PRESENTATION_FIG_HEIGHT_DEFAULT
    )
    fig.update_layout(
        title=fig_title,
        template="simple_white",
        margin=COMMON_MARGINS,
        hovermode="closest",
        height=fig_height,
        showlegend=True,
        legend_title_text="" if compact_kilosort_title else "Legend",
        legend=dict(itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
    )

    return fig


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
    region_order: list[str],
    y_top_start: float,
    bar_height: float,
    y_step: float,
    opacity: float = 0.9,
    line_width: float = 1.0,
):
    """Plot region (or shank) bars using pre-computed windows; colors from ``color_for_plot_group``."""
    y_top = float(y_top_start)

    for reg in region_order:
        bars_s = region_bars_by_region.get(reg, [])
        if not bars_s:
            continue

        add_burst_bars_top_SIMPLE(
            fig, bars_s,
            row=row, col=col,
            y_top=y_top, bar_height=bar_height,
            color=color_for_plot_group(str(reg)),
            opacity=opacity, line_width=line_width,
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


def population_firing_rate_binned(
    spike_struct, STATS: pd.DataFrame, record_len_s: float, *,
    bin_s: float, stride_s: float,
    allowed_clusters: set[tuple[str, int]] | None = None,
    normalize_by_units: bool = True,
    spike_cache: dict[tuple[str, int], np.ndarray] | None = None,
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

            key = (elec, cl)
            if spike_cache is not None and key in spike_cache:
                spk = spike_cache[key]
            else:
                spk = np.sort(np.asarray(arr).ravel().astype(float))
            spk = spk[np.isfinite(spk)]
            if spk.size == 0:
                continue

            n_units += 1
            all_spk_s.append(spk / 1000.0)

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
        y = moving_average(y, smooth_bins)

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
