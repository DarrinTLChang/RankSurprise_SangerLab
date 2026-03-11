#!/usr/bin/env python3
"""
Pre-onset proxy analysis: does the proxy rise *before* a network burst starts?

Since stimulation fires at burst onset and corrupts post-onset recording, we
only look at the pre-onset window (default: -500 ms to 0).

For every network burst onset, extracts the pre-onset proxy window and computes:
  - % change: (late half - early half) / |early half| × 100
  - Correlation: Pearson r between proxy and time (positive = rising toward onset)

Outputs:
  1. Per-patient/side pre-onset time-series plots (mean ± SEM)
  2. Grouped boxplot of % change per patient (Left + Right + ALL)
  3. Grouped boxplot of correlation per patient
  4. Per-patient heatmaps of individual onset windows
  5. Grand-average pre-onset traces
  6. Summary CSV with statistics + Wilcoxon p-values

Usage:
  python onset_triggered_proxy.py
  python onset_triggered_proxy.py --window-ms 500
  python onset_triggered_proxy.py --run-tag "separateGPi__SNR=1.2__..."
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

from config import PROXY_ROOT, BURST_ROOT, PROXY_ANALYSIS_ROOT, PROXY_ANALYSIS_RUN_TAG, SpikeTime_Mat_File
from pipeline.utils import dataset_labels
from compare_proxy_vs_network import (
    load_hemi_proxy_from_excel,
    load_proxy_excel_with_region_columns,
    load_region_bursts_by_region,
    derive_proxy_xlsx_path,
    DEFAULT_PROXY_ROOT,
)

# ============================================================
# Defaults
# ============================================================
DEFAULT_OUT_DIR = PROXY_ANALYSIS_ROOT / "onset_triggered_plots"
WINDOW_MS = 200  # look-back window in ms (0 to -WINDOW_MS before onset)
MODES = ("hemi_to_hemi", "region_to_hemi", "hemi_to_region")


# ============================================================
# Discovery & loading
# ============================================================
def _allowed_patient_periods() -> set[tuple[str, str]]:
    """(patient, period) from uncommented SpikeTime_Mat_File (config). Period normalized to no space."""
    out = set()
    for mat_file in SpikeTime_Mat_File:
        patient, period = dataset_labels(mat_file)
        out.add((patient, period.replace(" ", "")))
    return out


def _discover_patient_periods(burst_root: Path, run_tag: str,
                              allowed: set[tuple[str, str]] | None = None) -> list[tuple[str, str, Path, Path]]:
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
            tag_dir = period_dir / "rankSurprise" / run_tag
            if not tag_dir.is_dir():
                continue
            csv_L = tag_dir / "network_bursts_RS_left.csv"
            csv_R = tag_dir / "network_bursts_RS_right.csv"
            if csv_L.exists() or csv_R.exists():
                results.append((patient, period, csv_L, csv_R))
    return results


def _onsets_from_region_windows(windows: list[tuple[float, float]]) -> np.ndarray:
    """Extract onset times (start of each window) from region burst windows."""
    if not windows:
        return np.array([], dtype=float)
    return np.array([w[0] for w in windows], dtype=float)


def _load_onsets_ms(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return np.array([])
    df = pd.read_csv(csv_path)
    if "burst_start_ms" not in df.columns:
        return np.array([])
    vals = pd.to_numeric(df["burst_start_ms"], errors="coerce").dropna().to_numpy(float)
    return np.sort(vals)


# ============================================================
# Window extraction (pre-onset only: -window_ms to 0)
# ============================================================
def _extract_pre_windows(t_ms: np.ndarray, proxy: np.ndarray,
                         onsets_ms: np.ndarray, window_ms: float):
    """
    For each onset, extract proxy in [onset - window_ms, onset].
    Returns (windows: 2D array, rbins: 1D array of negative time offsets).
    """
    dt = float(np.median(np.diff(t_ms)))
    n_bins = int(window_ms / dt) + 1
    rbins = np.linspace(-window_ms, 0, n_bins)

    windows = np.full((len(onsets_ms), n_bins), np.nan)
    for i, onset in enumerate(onsets_ms):
        target_times = onset + rbins
        indices = np.searchsorted(t_ms, target_times, side="left")
        indices = np.clip(indices, 0, len(t_ms) - 1)
        actual_times = t_ms[indices]
        close_enough = np.abs(actual_times - target_times) <= dt
        windows[i, close_enough] = proxy[indices[close_enough]]

    return windows, rbins


# ============================================================
# Per-onset metrics
# ============================================================
def _per_onset_pct_change(windows: np.ndarray, rbins: np.ndarray) -> np.ndarray:
    """
    % change = (late_half - early_half) / |early_half| × 100.
    Early half = first half of window (far from onset).
    Late half  = second half (close to onset).
    """
    mid = len(rbins) // 2
    early_means = np.nanmean(windows[:, :mid], axis=1)
    late_means = np.nanmean(windows[:, mid:], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (late_means - early_means) / np.abs(early_means) * 100.0
    pct[~np.isfinite(pct)] = np.nan
    return pct


def _per_onset_correlations(windows: np.ndarray, rbins: np.ndarray) -> np.ndarray:
    """
    Pearson r between proxy and time within the pre-onset window.
    Positive r = proxy rising toward onset.
    """
    n_onsets = windows.shape[0]
    corrs = np.full(n_onsets, np.nan)
    for i in range(n_onsets):
        row = windows[i]
        valid = np.isfinite(row)
        if valid.sum() >= 4:
            r, _ = sp_stats.pearsonr(rbins[valid], row[valid])
            corrs[i] = r
    return corrs


# ============================================================
# Plotting helpers
# ============================================================
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

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.6), 5.5))

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

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    for positions, data_list in [(positions_L, data_L), (positions_R, data_R)]:
        for pos, vals in zip(positions, data_list):
            if len(vals) >= 5:
                _, pval = sp_stats.wilcoxon(vals, alternative="greater")
                star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                if star:
                    y_top = np.percentile(vals, 95)
                    ax.text(pos, y_top, star, ha="center", va="bottom",
                            fontsize=7, fontweight="bold")

    sep_x = (len(patients) - 0.5) * 1.0
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.6)

    ax.set_xticks([i * 1.0 for i in range(len(groups))])
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor="#4878CF", alpha=0.7, label="Left"),
                       Patch(facecolor="#E8A02F", alpha=0.7, label="Right")],
              loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {title}: {out_path}")


def _plot_heatmap(windows: np.ndarray, rbins: np.ndarray, title: str, out_path: Path):
    """Rows = onsets sorted by correlation, columns = time bins."""
    corrs = _per_onset_correlations(windows, rbins)
    order = np.argsort(np.nan_to_num(corrs, nan=-999))
    sorted_win = windows[order]

    fig, ax = plt.subplots(figsize=(8, max(3, windows.shape[0] * 0.06)))
    t_s = rbins / 1000.0
    im = ax.imshow(sorted_win, aspect="auto",
                   extent=[t_s[0], t_s[-1], windows.shape[0], 0],
                   cmap="RdBu_r", interpolation="nearest")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time before onset (s)")
    ax.set_ylabel("Onset (sorted by r)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Proxy value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main processing
# ============================================================
def process_all(burst_root: Path, run_tag: str, proxy_root: Path,
                window_ms: float, out_dir: Path, mode: str = "hemi_to_hemi"):
    allowed = _allowed_patient_periods() if SpikeTime_Mat_File else None
    discoveries = _discover_patient_periods(burst_root, run_tag, allowed=allowed)
    if not discoveries:
        print(f"No patient/period data found under {burst_root} with run_tag:\n  {run_tag}")
        return

    print(f"Found {len(discoveries)} patient-period(s)  [mode={mode}]")
    proxy_root = proxy_root.resolve()

    grand: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    rel_bins: dict[str, np.ndarray] = {}
    pct_records: dict[str, list[dict]] = {"left": [], "right": []}
    corr_records: dict[str, list[dict]] = {"left": [], "right": []}
    summary_rows: list[dict] = []

    def _iter_ota_pairs():
        for patient, period, csv_L, csv_R in discoveries:
            tag_dir = csv_L.parent
            for side, csv_path in [("left", csv_L), ("right", csv_R)]:
                if mode == "hemi_to_hemi":
                    onsets_ms = _load_onsets_ms(csv_path)
                    if onsets_ms.size == 0:
                        continue
                    px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                    if px_path is None or not px_path.exists():
                        continue
                    t_ms, hemi_proxy, _ = load_hemi_proxy_from_excel(px_path)
                    t_ms = t_ms.astype(float)
                    yield patient, period, side, "", t_ms, hemi_proxy, onsets_ms
                elif mode == "region_to_hemi":
                    onsets_ms = _load_onsets_ms(csv_path)
                    if onsets_ms.size == 0:
                        continue
                    px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                    if px_path is None or not px_path.exists():
                        continue
                    t_ms, _, region_proxies = load_proxy_excel_with_region_columns(px_path)
                    t_ms = t_ms.astype(float)
                    for region, proxy_vals in region_proxies.items():
                        if proxy_vals.shape[0] != t_ms.shape[0]:
                            continue
                        yield patient, period, side, region, t_ms, proxy_vals, onsets_ms
                else:  # hemi_to_region
                    region_csv = tag_dir / f"region_bursts_RS_{side}.csv"
                    by_region = load_region_bursts_by_region(region_csv)
                    if not by_region:
                        continue
                    px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                    if px_path is None or not px_path.exists():
                        continue
                    t_ms, hemi_proxy, _ = load_hemi_proxy_from_excel(px_path)
                    t_ms = t_ms.astype(float)
                    for region, windows in by_region.items():
                        onsets_ms = _onsets_from_region_windows(windows)
                        if onsets_ms.size == 0:
                            continue
                        yield patient, period, side, region, t_ms, hemi_proxy, onsets_ms

    for patient, period, side, region_label, t_ms, proxy, onsets_ms in _iter_ota_pairs():
            windows, rbins = _extract_pre_windows(t_ms, proxy, onsets_ms, window_ms)
            if windows.shape[0] == 0:
                continue

            mean_trace = np.nanmean(windows, axis=0)
            sem_trace = np.nanstd(windows, axis=0) / np.sqrt(
                np.sum(np.isfinite(windows), axis=0).clip(1)
            )

            if not region_label:
                grand[side].append(mean_trace)
                if side not in rel_bins:
                    rel_bins[side] = rbins

            pcts = _per_onset_pct_change(windows, rbins)
            corrs = _per_onset_correlations(windows, rbins)

            for p in pcts:
                if np.isfinite(p):
                    pct_records[side].append({"patient": patient, "period": period, "pct": p, "region": region_label})
            for c in corrs:
                if np.isfinite(c):
                    corr_records[side].append({"patient": patient, "period": period, "corr": c, "region": region_label})

            finite_pcts = pcts[np.isfinite(pcts)]
            finite_corrs = corrs[np.isfinite(corrs)]
            pval_pct = (sp_stats.wilcoxon(finite_pcts, alternative="greater")[1]
                        if len(finite_pcts) >= 5 else np.nan)
            pval_corr = (sp_stats.wilcoxon(finite_corrs, alternative="greater")[1]
                         if len(finite_corrs) >= 5 else np.nan)

            mid = len(rbins) // 2
            early_val = float(np.nanmean(mean_trace[:mid]))
            late_val = float(np.nanmean(mean_trace[mid:]))

            summary_rows.append({
                "patient": patient, "period": period, "side": side, "region": region_label,
                "n_onsets": len(onsets_ms),
                "proxy_early": early_val,
                "proxy_late": late_val,
                "mean_pct_change": float(np.nanmean(finite_pcts)) if len(finite_pcts) else np.nan,
                "median_pct_change": float(np.nanmedian(finite_pcts)) if len(finite_pcts) else np.nan,
                "wilcoxon_p_pct": pval_pct,
                "mean_corr": float(np.nanmean(finite_corrs)) if len(finite_corrs) else np.nan,
                "median_corr": float(np.nanmedian(finite_corrs)) if len(finite_corrs) else np.nan,
                "wilcoxon_p_corr": pval_corr,
            })

            pat_dir = out_dir / "per_patient" / patient
            if region_label:
                pat_dir = pat_dir / region_label
            pat_dir.mkdir(parents=True, exist_ok=True)
            file_suffix = f"{period}_{side}" + (f"_{region_label}" if region_label else "")
            fig, ax = plt.subplots(figsize=(7, 4))
            t_s = rbins / 1000.0
            ax.plot(t_s, mean_trace, color="tab:blue", linewidth=1.2)
            ax.fill_between(t_s, mean_trace - sem_trace, mean_trace + sem_trace,
                            alpha=0.25, color="tab:blue")
            ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="Burst onset")
            ax.set_xlabel("Time before onset (s)")
            ax.set_ylabel("Hemi proxy" if not region_label else "Proxy")
            ax.set_title(f"{patient} • {period} • {side.capitalize()}" + (f" • {region_label}" if region_label else "") +
                         f" (n={len(onsets_ms)}, r={np.nanmean(finite_corrs):.2f}, p={pval_corr:.3g})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(pat_dir / f"ota_{file_suffix}.png", dpi=150)
            plt.close(fig)

            _plot_heatmap(windows, rbins,
                          f"{patient} • {period} • {side.capitalize()}" + (f" • {region_label}" if region_label else "") + f" (n={len(onsets_ms)})",
                          pat_dir / f"heatmap_{file_suffix}.png")

            tag = f"{patient} {period} {side}" + (f" {region_label}" if region_label else "")
            print(f"  {tag}: {len(onsets_ms)} onsets, "
                  f"mean %Δ={np.nanmean(finite_pcts):.1f}%, "
                  f"mean r={np.nanmean(finite_corrs):.3f}, "
                  f"p_corr={pval_corr:.3g}")

    # --- Grouped boxplots ---
    print()
    if pct_records["left"] or pct_records["right"]:
        _plot_grouped_boxplot(
            pct_records["left"], pct_records["right"],
            val_col="pct",
            ylabel="Pre-onset proxy % change (late vs early)",
            title="Pre-onset proxy % change (by subject)",
            out_path=out_dir / "boxplot_pct_change.png",
        )
    if corr_records["left"] or corr_records["right"]:
        _plot_grouped_boxplot(
            corr_records["left"], corr_records["right"],
            val_col="corr",
            ylabel="Correlation (proxy vs time, pre-onset)",
            title="Pre-onset proxy correlation (by subject)",
            out_path=out_dir / "boxplot_correlation.png",
        )

    # --- Grand-average pre-onset traces ---
    for side in ("left", "right"):
        traces = grand[side]
        if not traces or side not in rel_bins:
            continue
        rbins = rel_bins[side]
        stacked = np.array(traces)
        grand_mean = np.nanmean(stacked, axis=0)
        grand_sem = np.nanstd(stacked, axis=0) / np.sqrt(stacked.shape[0])

        t_s = rbins / 1000.0
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_s, grand_mean, color="tab:blue", linewidth=1.5)
        ax.fill_between(t_s, grand_mean - grand_sem, grand_mean + grand_sem,
                        alpha=0.25, color="tab:blue")
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Burst onset")
        ax.set_xlabel("Time before onset (s)")
        ax.set_ylabel("Hemi proxy (grand average)")
        ax.set_title(f"Pre-onset proxy — {side.capitalize()} "
                     f"(n={len(traces)} patient-periods)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"grand_average_{side}.png", dpi=150)
        plt.close(fig)
        print(f"  Grand average ({side}): {len(traces)} patient-periods → "
              f"{out_dir / f'grand_average_{side}.png'}")

    # --- Summary CSV ---
    if summary_rows:
        csv_path = out_dir / "onset_triggered_summary.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"\nSummary: {csv_path}")

    # --- Per-onset records CSVs ---
    for side in ("left", "right"):
        if pct_records[side]:
            pd.DataFrame(pct_records[side]).to_csv(
                out_dir / f"per_onset_pct_{side}.csv", index=False)
        if corr_records[side]:
            pd.DataFrame(corr_records[side]).to_csv(
                out_dir / f"per_onset_corr_{side}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-onset proxy analysis around network burst onsets."
    )
    parser.add_argument("--burst-root", type=Path, default=BURST_ROOT,
                        help=f"Root of outputs_RS_burst (default: {BURST_ROOT})")
    parser.add_argument("--run-tag", type=str, default=PROXY_ANALYSIS_RUN_TAG,
                        help="Run tag folder name under rankSurprise/ (default from config)")
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS,
                        help=f"Pre-onset look-back window in ms (default: {WINDOW_MS})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--mode", type=str, default="hemi_to_hemi", choices=MODES,
                        help="hemi_to_hemi (default), region_to_hemi, hemi_to_region")
    args = parser.parse_args()

    out_base = args.out_dir.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = out_base / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy_root = Path(PROXY_ROOT or DEFAULT_PROXY_ROOT).resolve()
    print(f"Burst root : {args.burst_root}")
    print(f"Proxy root : {proxy_root}")
    print(f"Run tag    : {args.run_tag}")
    print(f"Mode      : {args.mode}")
    print(f"Window     : -{args.window_ms:.0f} ms to 0 (pre-onset only)")
    print(f"Output     : {out_dir}\n")

    process_all(args.burst_root, args.run_tag, proxy_root, args.window_ms, out_dir, args.mode)


if __name__ == "__main__":
    main()
