#!/usr/bin/env python3
"""
Proxy lag analysis relative to network burst onsets.

Two analyses:
  1. Peak lag: For each burst, find when the proxy peaks relative to burst_start_ms.
     → "the proxy peaks X ms after network burst onset"
  2. Slope onset: Find when the proxy starts rising (derivative crosses threshold)
     relative to burst_start_ms.
     → "the proxy starts rising X ms before/after the spikes"

Search window for each burst: [burst_start - pre_pad, burst_end + post_pad].
Uses proxy from rasters_all_with_fast_proxies Excel (20 ms resolution).

Usage:
  python proxy_lag_analysis.py
  python proxy_lag_analysis.py --pre-pad-ms 300 --post-pad-ms 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d

from config import PROXY_ROOT, BURST_ROOT, PROXY_ANALYSIS_ROOT, PROXY_ANALYSIS_RUN_TAG, SpikeTime_Mat_File
from pipeline.burst_paths import burst_network_csv_path, burst_region_csv_path
from pipeline.utils import dataset_labels
from compare_proxy_vs_network import (
    load_hemi_proxy_from_excel,
    load_proxy_excel_with_region_columns,
    load_region_bursts_by_region,
    derive_proxy_xlsx_path,
    DEFAULT_PROXY_ROOT,
)

DEFAULT_OUT_DIR = PROXY_ANALYSIS_ROOT / "proxy_lag_plots"
PRE_PAD_MS = 300   # how far before burst_start to begin search
POST_PAD_MS = 500   # how far after burst_end to end search


# ============================================================
# Discovery & loading (reused from onset_triggered_proxy)
# ============================================================
def _allowed_patient_periods() -> set[tuple[str, str]]:
    """(patient, period) from uncommented SpikeTime_Mat_File (config). Period normalized to no space."""
    out = set()
    for mat_file in SpikeTime_Mat_File:
        patient, period = dataset_labels(mat_file)
        out.add((patient, period.replace(" ", "")))
    return out


def _discover_patient_periods(burst_root: Path, run_tag: str,
                              allowed: set[tuple[str, str]] | None = None):
    """If allowed is not None, only include (patient, period) in that set (from SpikeTime_Mat_File)."""
    results = []
    for patient_dir in sorted(burst_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        for period_dir in sorted(patient_dir.iterdir()):
            if not period_dir.is_dir():
                continue
            period = period_dir.name
            if allowed is not None and (patient, period.replace(" ", "")) not in allowed:
                continue
            # New layout: burst_root already points at the method root (e.g. F:\SangerLabBursts_RS),
            # so runs live at patient/PeriodN/run_tag (no extra /rankSurprise/ layer).
            tag_dir = period_dir / run_tag
            if not tag_dir.is_dir():
                continue
            csv_L = burst_network_csv_path(tag_dir, "left")
            csv_R = burst_network_csv_path(tag_dir, "right")
            if csv_L.exists() or csv_R.exists():
                results.append((patient, period, csv_L, csv_R))
    return results


def _load_bursts(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    required = {"burst_start_ms", "burst_end_ms"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    if "burst_duration_ms" not in df.columns:
        df = df.assign(burst_duration_ms=df["burst_end_ms"] - df["burst_start_ms"])
    return df.dropna(subset=list(required) + ["burst_duration_ms"])


def _windows_to_bursts_df(windows: list[tuple[float, float]]) -> pd.DataFrame:
    """Convert list of (start_ms, end_ms) to DataFrame with burst_start_ms, burst_end_ms, burst_duration_ms."""
    if not windows:
        return pd.DataFrame()
    rows = [
        {"burst_start_ms": s, "burst_end_ms": e, "burst_duration_ms": e - s}
        for s, e in windows
    ]
    return pd.DataFrame(rows)


# ============================================================
# Analysis 1: Peak lag
# ============================================================
def _find_peak_lags(t_ms: np.ndarray, proxy: np.ndarray,
                    bursts_df: pd.DataFrame,
                    pre_pad: float, post_pad: float) -> np.ndarray:
    """
    For each burst, find the proxy peak in [start - pre_pad, end + post_pad].
    Returns array of (peak_time_ms - burst_start_ms) for each burst.
    Positive = peak comes after burst start.
    """
    lags = np.full(len(bursts_df), np.nan)
    for i, row in bursts_df.iterrows():
        t0 = row["burst_start_ms"] - pre_pad
        t1 = row["burst_end_ms"] + post_pad
        mask = (t_ms >= t0) & (t_ms <= t1)
        if mask.sum() < 3:
            continue
        seg_t = t_ms[mask]
        seg_proxy = proxy[mask]
        finite = np.isfinite(seg_proxy)
        if finite.sum() < 3:
            continue
        peak_idx = np.nanargmax(seg_proxy[finite])
        peak_time = seg_t[finite][peak_idx]
        lags[i] = peak_time - row["burst_start_ms"]
    return lags


# ============================================================
# Analysis 2: Slope onset (derivative-based)
# ============================================================
def _find_slope_onsets(t_ms: np.ndarray, proxy: np.ndarray,
                       bursts_df: pd.DataFrame,
                       pre_pad: float, post_pad: float,
                       smooth_n: int = 3) -> np.ndarray:
    """
    For each burst, find when the proxy first starts rising (derivative > 0
    sustained for at least 2 consecutive bins) in the search window.
    Returns (slope_onset_time - burst_start_ms). Negative = proxy starts
    rising before the burst.
    """
    dt = float(np.median(np.diff(t_ms)))
    lags = np.full(len(bursts_df), np.nan)

    for i, row in bursts_df.iterrows():
        t0 = row["burst_start_ms"] - pre_pad
        t1 = row["burst_end_ms"] + post_pad
        mask = (t_ms >= t0) & (t_ms <= t1)
        if mask.sum() < 5:
            continue
        seg_t = t_ms[mask]
        seg_proxy = proxy[mask]

        # Smooth to reduce noise, then take derivative
        smoothed = uniform_filter1d(seg_proxy, size=smooth_n)
        deriv = np.diff(smoothed) / dt

        # Find first point where derivative is positive for ≥2 consecutive bins
        pos = deriv > 0
        for j in range(len(pos) - 1):
            if pos[j] and pos[j + 1]:
                onset_time = seg_t[j]
                lags[i] = onset_time - row["burst_start_ms"]
                break

    return lags


# ============================================================
# Plotting
# ============================================================
def _plot_lag_histogram(lags: np.ndarray, title: str, xlabel: str, out_path: Path,
                        color: str = "tab:blue"):
    finite = lags[np.isfinite(lags)]
    if len(finite) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=30, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    med = np.median(finite)
    ax.axvline(med, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {med:.0f} ms")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_grouped_boxplot(records_left: list[dict], records_right: list[dict],
                          val_col: str, ylabel: str, title: str, out_path: Path):
    from matplotlib.patches import Patch

    df_L = pd.DataFrame(records_left) if records_left else pd.DataFrame()
    df_R = pd.DataFrame(records_right) if records_right else pd.DataFrame()

    patients_L = set(df_L["patient"].unique()) if not df_L.empty else set()
    patients_R = set(df_R["patient"].unique()) if not df_R.empty else set()
    patients = sorted(patients_L | patients_R)
    if not patients:
        return
    groups = patients + ["ALL"]

    box_width = 0.35
    positions_L, positions_R = [], []
    data_L, data_R = [], []

    for i, grp in enumerate(groups):
        center = i * 1.0
        positions_L.append(center - box_width / 2 - 0.02)
        positions_R.append(center + box_width / 2 + 0.02)
        if grp == "ALL":
            data_L.append(df_L[val_col].dropna().values if not df_L.empty else np.array([]))
            data_R.append(df_R[val_col].dropna().values if not df_R.empty else np.array([]))
        else:
            data_L.append(df_L.loc[df_L["patient"] == grp, val_col].dropna().values
                          if not df_L.empty else np.array([]))
            data_R.append(df_R.loc[df_R["patient"] == grp, val_col].dropna().values
                          if not df_R.empty else np.array([]))

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.6), 8))

    if any(len(d) > 0 for d in data_L):
        bp_L = ax.boxplot(data_L, positions=positions_L, widths=box_width,
                          patch_artist=True, showfliers=True, manage_ticks=False)
        for patch in bp_L["boxes"]:
            patch.set_facecolor("#4878CF")
            patch.set_alpha(0.7)
        for med in bp_L["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

    if any(len(d) > 0 for d in data_R):
        bp_R = ax.boxplot(data_R, positions=positions_R, widths=box_width,
                          patch_artist=True, showfliers=True, manage_ticks=False)
        for patch in bp_R["boxes"]:
            patch.set_facecolor("#E8A02F")
            patch.set_alpha(0.7)
        for med in bp_R["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

    ax.axhline(0, color="red", linestyle="--", linewidth=1,
               label="burst_start (t=0)")

    sep_x = (len(patients) - 0.5) * 1.0
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.6)

    ax.yaxis.set_major_locator(plt.MultipleLocator(250))
    ax.grid(axis="y", which="major", alpha=0.3)

    ax.set_xticks([i * 1.0 for i in range(len(groups))])
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor="#4878CF", alpha=0.7, label="Left"),
                       Patch(facecolor="#E8A02F", alpha=0.7, label="Right"),
                       plt.Line2D([0], [0], color="red", linestyle="--", label="burst_start (t=0)")],
              loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {title}: {out_path}")


# ============================================================
# Main
# ============================================================
MODES = ("hemi_to_hemi", "region_to_hemi", "hemi_to_region")


def _run_hemi_to_hemi(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                      peak_records, slope_records, summary_rows):
    """Hemi proxy vs hemisphere (network) bursts — original behavior."""
    for patient, period, csv_L, csv_R in discoveries:
        for side, csv_path in [("left", csv_L), ("right", csv_R)]:
            bursts_df = _load_bursts(csv_path)
            if bursts_df.empty:
                continue

            px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
            if px_path is None or not px_path.exists():
                continue

            t_ms, hemi_proxy, _ = load_hemi_proxy_from_excel(px_path)
            t_ms = t_ms.astype(float)

            peak_lags = _find_peak_lags(t_ms, hemi_proxy, bursts_df, pre_pad, post_pad)
            finite_peaks = peak_lags[np.isfinite(peak_lags)]
            for v in finite_peaks:
                peak_records[side].append({"patient": patient, "period": period, "lag_ms": v})

            slope_lags = _find_slope_onsets(t_ms, hemi_proxy, bursts_df, pre_pad, post_pad)
            finite_slopes = slope_lags[np.isfinite(slope_lags)]
            for v in finite_slopes:
                slope_records[side].append({"patient": patient, "period": period, "lag_ms": v})

            pat_dir = out_dir / "per_patient" / patient
            pat_dir.mkdir(parents=True, exist_ok=True)
            _plot_lag_histogram(peak_lags, f"{patient} • {period} • {side.capitalize()} — Peak lag",
                                "Peak lag (ms, relative to burst_start)",
                                pat_dir / f"peak_lag_{period}_{side}.png")
            _plot_lag_histogram(slope_lags, f"{patient} • {period} • {side.capitalize()} — Slope onset lag",
                                "Slope onset (ms, relative to burst_start)",
                                pat_dir / f"slope_onset_{period}_{side}.png", color="tab:green")

            med_peak = float(np.median(finite_peaks)) if len(finite_peaks) else np.nan
            med_slope = float(np.median(finite_slopes)) if len(finite_slopes) else np.nan
            summary_rows.append({
                "patient": patient, "period": period, "side": side, "region": "",
                "n_bursts": len(bursts_df),
                "median_peak_lag_ms": med_peak, "median_slope_onset_ms": med_slope,
            })
            print(f"  {patient} {period} {side}: {len(bursts_df)} bursts, "
                  f"peak lag median={med_peak:.0f} ms, slope onset median={med_slope:.0f} ms")


def _run_region_to_hemi(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                        peak_records, slope_records, summary_rows):
    """Region proxy vs hemisphere (network) bursts — one analysis per region."""
    for patient, period, csv_L, csv_R in discoveries:
        for side, csv_path in [("left", csv_L), ("right", csv_R)]:
            bursts_df = _load_bursts(csv_path)
            if bursts_df.empty:
                continue

            px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
            if px_path is None or not px_path.exists():
                continue

            t_ms, _, region_proxies = load_proxy_excel_with_region_columns(px_path)
            t_ms = t_ms.astype(float)
            if not region_proxies:
                continue

            for region, proxy_vals in region_proxies.items():
                if proxy_vals.shape[0] != t_ms.shape[0]:
                    continue
                peak_lags = _find_peak_lags(t_ms, proxy_vals, bursts_df, pre_pad, post_pad)
                finite_peaks = peak_lags[np.isfinite(peak_lags)]
                for v in finite_peaks:
                    peak_records[side].append({"patient": patient, "period": period, "side": side, "region": region, "lag_ms": v})

                slope_lags = _find_slope_onsets(t_ms, proxy_vals, bursts_df, pre_pad, post_pad)
                finite_slopes = slope_lags[np.isfinite(slope_lags)]
                for v in finite_slopes:
                    slope_records[side].append({"patient": patient, "period": period, "side": side, "region": region, "lag_ms": v})

                med_peak = float(np.median(finite_peaks)) if len(finite_peaks) else np.nan
                med_slope = float(np.median(finite_slopes)) if len(finite_slopes) else np.nan
                summary_rows.append({
                    "patient": patient, "period": period, "side": side, "region": region,
                    "n_bursts": len(bursts_df),
                    "median_peak_lag_ms": med_peak, "median_slope_onset_ms": med_slope,
                })
            print(f"  {patient} {period} {side}: region proxies vs hemi bursts "
                  f"({len(region_proxies)} regions)")


def _run_hemi_to_region(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                        peak_records, slope_records, summary_rows):
    """Hemisphere proxy vs region bursts — one analysis per region."""
    for patient, period, csv_L, csv_R in discoveries:
        tag_dir = csv_L.parent
        region_csv_L = burst_region_csv_path(tag_dir, "left")
        region_csv_R = burst_region_csv_path(tag_dir, "right")

        for side, region_csv in [("left", region_csv_L), ("right", region_csv_R)]:
            by_region = load_region_bursts_by_region(region_csv)
            if not by_region:
                continue

            px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
            if px_path is None or not px_path.exists():
                continue

            t_ms, hemi_proxy, _ = load_hemi_proxy_from_excel(px_path)
            t_ms = t_ms.astype(float)

            for region, windows in by_region.items():
                bursts_df = _windows_to_bursts_df(windows)
                if bursts_df.empty:
                    continue

                peak_lags = _find_peak_lags(t_ms, hemi_proxy, bursts_df, pre_pad, post_pad)
                finite_peaks = peak_lags[np.isfinite(peak_lags)]
                for v in finite_peaks:
                    peak_records[side].append({"patient": patient, "period": period, "side": side, "region": region, "lag_ms": v})

                slope_lags = _find_slope_onsets(t_ms, hemi_proxy, bursts_df, pre_pad, post_pad)
                finite_slopes = slope_lags[np.isfinite(slope_lags)]
                for v in finite_slopes:
                    slope_records[side].append({"patient": patient, "period": period, "side": side, "region": region, "lag_ms": v})

                med_peak = float(np.median(finite_peaks)) if len(finite_peaks) else np.nan
                med_slope = float(np.median(finite_slopes)) if len(finite_slopes) else np.nan
                summary_rows.append({
                    "patient": patient, "period": period, "side": side, "region": region,
                    "n_bursts": len(bursts_df),
                    "median_peak_lag_ms": med_peak, "median_slope_onset_ms": med_slope,
                })
            print(f"  {patient} {period} {side}: hemi proxy vs region bursts "
                  f"({len(by_region)} regions)")


def _plot_grouped_boxplot_by_region(peak_records, slope_records, out_dir: Path):
    """Boxplots when records have 'region': one box per region (all patients/sides pooled per region)."""
    df_peak = pd.DataFrame(peak_records["left"] + peak_records["right"])
    df_slope = pd.DataFrame(slope_records["left"] + slope_records["right"])
    if df_peak.empty and df_slope.empty:
        return
    reg_peak = df_peak["region"].dropna().unique().tolist() if "region" in df_peak.columns else []
    reg_slope = df_slope["region"].dropna().unique().tolist() if "region" in df_slope.columns else []
    regions = sorted(set(reg_peak) | set(reg_slope))
    regions = [r for r in regions if r]
    if not regions:
        return

    for val_col, ylabel, title, fname in [
        ("lag_ms", "Peak lag (ms)", "Peak lag (region proxy/hemi proxy vs bursts) by region", "boxplot_peak_lag_by_region.png"),
        ("lag_ms", "Slope onset (ms)", "Slope onset by region", "boxplot_slope_onset_by_region.png"),
    ]:
        df = df_peak if "peak" in fname else df_slope
        if df.empty or val_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(regions) * 1.2), 6))
        data_left = [df.loc[(df["region"] == r) & (df["side"] == "left"), val_col].dropna().values for r in regions]
        data_right = [df.loc[(df["region"] == r) & (df["side"] == "right"), val_col].dropna().values for r in regions]
        # Simplify: one box per region (pool L and R or separate)
        positions = np.arange(len(regions))
        bp = ax.boxplot([df.loc[df["region"] == r, val_col].dropna().values for r in regions],
                        positions=positions, widths=0.5, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#6baed6")
            patch.set_alpha(0.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(regions, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  {title}: {out_dir / fname}")


def process_all(burst_root: Path, run_tag: str, proxy_root: Path,
                pre_pad: float, post_pad: float, out_dir: Path, mode: str):
    allowed = _allowed_patient_periods() if SpikeTime_Mat_File else None
    discoveries = _discover_patient_periods(burst_root, run_tag, allowed=allowed)
    if not discoveries:
        print(f"No data found under {burst_root} with run_tag:\n  {run_tag}")
        return

    print(f"Found {len(discoveries)} patient-period(s)  [mode={mode}]")
    proxy_root = proxy_root.resolve()

    peak_records: dict[str, list[dict]] = {"left": [], "right": []}
    slope_records: dict[str, list[dict]] = {"left": [], "right": []}
    summary_rows: list[dict] = []

    if mode == "hemi_to_hemi":
        _run_hemi_to_hemi(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                          peak_records, slope_records, summary_rows)
    elif mode == "region_to_hemi":
        _run_region_to_hemi(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                           peak_records, slope_records, summary_rows)
    elif mode == "hemi_to_region":
        _run_hemi_to_region(discoveries, proxy_root, pre_pad, post_pad, out_dir,
                            peak_records, slope_records, summary_rows)
    else:
        raise ValueError(f"mode must be one of {MODES}")

    # --- Grouped boxplots ---
    print()
    if mode == "hemi_to_hemi":
        if peak_records["left"] or peak_records["right"]:
            _plot_grouped_boxplot(peak_records["left"], peak_records["right"], val_col="lag_ms",
                                  ylabel="Peak lag (ms)", title="Proxy peak lag relative to burst onset (by subject)",
                                  out_path=out_dir / "boxplot_peak_lag.png")
        if slope_records["left"] or slope_records["right"]:
            _plot_grouped_boxplot(slope_records["left"], slope_records["right"], val_col="lag_ms",
                                  ylabel="Slope onset (ms)", title="Proxy slope onset relative to burst onset (by subject)",
                                  out_path=out_dir / "boxplot_slope_onset.png")
    else:
        _plot_grouped_boxplot_by_region(peak_records, slope_records, out_dir)

    # --- Grand histograms (hemi_to_hemi only) ---
    if mode == "hemi_to_hemi":
        for side in ("left", "right"):
            peak_vals = np.array([r["lag_ms"] for r in peak_records[side]])
            slope_vals = np.array([r["lag_ms"] for r in slope_records[side]])
            if len(peak_vals) >= 2:
                _plot_lag_histogram(peak_vals, f"All subjects — {side.capitalize()} — Peak lag",
                                    "Peak lag (ms, relative to burst_start)", out_dir / f"grand_peak_lag_{side}.png")
            if len(slope_vals) >= 2:
                _plot_lag_histogram(slope_vals, f"All subjects — {side.capitalize()} — Slope onset",
                                    "Slope onset (ms, relative to burst_start)",
                                    out_dir / f"grand_slope_onset_{side}.png", color="tab:green")

    # --- Summary CSV ---
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "lag_analysis_summary.csv", index=False)
        print(f"\nSummary: {out_dir / 'lag_analysis_summary.csv'}")

    for side in ("left", "right"):
        if peak_records[side]:
            pd.DataFrame(peak_records[side]).to_csv(out_dir / f"peak_lags_{side}.csv", index=False)
        if slope_records[side]:
            pd.DataFrame(slope_records[side]).to_csv(out_dir / f"slope_onsets_{side}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Proxy lag analysis: peak lag + slope onset relative to burst_start_ms."
    )
    parser.add_argument("--burst-root", type=Path, default=BURST_ROOT)
    parser.add_argument("--run-tag", type=str, default=PROXY_ANALYSIS_RUN_TAG)
    parser.add_argument("--pre-pad-ms", type=float, default=PRE_PAD_MS,
                        help=f"Search window start before burst_start (default: {PRE_PAD_MS})")
    parser.add_argument("--post-pad-ms", type=float, default=POST_PAD_MS,
                        help=f"Search window end after burst_end (default: {POST_PAD_MS})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", type=str, default="hemi_to_hemi", choices=MODES,
                        help="hemi_to_hemi (default), region_to_hemi, hemi_to_region")
    args = parser.parse_args()

    out_base = args.out_dir.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = out_base / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy_root = Path(PROXY_ROOT or DEFAULT_PROXY_ROOT).resolve()
    print(f"Burst root  : {args.burst_root}")
    print(f"Proxy root  : {proxy_root}")
    print(f"Run tag     : {args.run_tag}")
    print(f"Mode       : {args.mode}")
    print(f"Search window: burst_start - {args.pre_pad_ms:.0f} ms  →  burst_end + {args.post_pad_ms:.0f} ms")
    print(f"Output     : {out_dir}\n")

    process_all(args.burst_root, args.run_tag, proxy_root,
                args.pre_pad_ms, args.post_pad_ms, out_dir, args.mode)


if __name__ == "__main__":
    main()
