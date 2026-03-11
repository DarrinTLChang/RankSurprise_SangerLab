#!/usr/bin/env python3
"""
Slope-based proxy detection analysis.

For each burst onset, measure the max slope (d/dt) of the proxy signal in a
window around burst_start_ms.  Compare to random baseline slopes.  Show:

  1. Per-patient boxplot: onset slopes vs baseline slopes
  2. Threshold convergence: best slope threshold per patient/side
  3. Slope-based performance curve (sensitivity, precision, FA/min)

Usage:
  python proxy_slope_detection.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d

from config import PROXY_ROOT, BURST_ROOT, PROXY_ANALYSIS_ROOT, PROXY_ANALYSIS_RUN_TAG, SpikeTime_Mat_File
from pipeline.utils import dataset_labels
from compare_proxy_vs_network import (
    load_hemi_proxy_from_excel,
    load_proxy_excel_with_region_columns,
    load_region_bursts_by_region,
    derive_proxy_xlsx_path,
    DEFAULT_PROXY_ROOT,
)

DEFAULT_OUT_DIR = PROXY_ANALYSIS_ROOT / "proxy_slope_detection"
TOLERANCE_MS = 300
SCALE_TO_P = 1e12
SMOOTH_MS = 5.0       # smoothing window for derivative
ONSET_PRE_MS = 200    # look 200ms before onset
ONSET_POST_MS = 100   # look 100ms after onset
N_BASELINE_SAMPLES = 500  # random baseline windows per recording
MODES = ("hemi_to_hemi", "region_to_hemi", "hemi_to_region")


def _windows_to_bursts_df(windows: list[tuple[float, float]]) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame()
    return pd.DataFrame([{"burst_start_ms": s, "burst_end_ms": e} for s, e in windows])


# ============================================================
# Discovery / loading (shared with proxy_detection_demo)
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
            tag_dir = period_dir / "rankSurprise" / run_tag
            if not tag_dir.is_dir():
                continue
            csv_L = tag_dir / "network_bursts_RS_left.csv"
            csv_R = tag_dir / "network_bursts_RS_right.csv"
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
    return df.dropna(subset=list(required))


# ============================================================
# Slope computation
# ============================================================
def compute_slope(t_ms, proxy_p, smooth_ms=SMOOTH_MS):
    """Smooth proxy, compute derivative (p/ms)."""
    dt = float(np.median(np.diff(t_ms)))
    smooth_n = max(1, int(smooth_ms / dt))
    smoothed = uniform_filter1d(proxy_p, size=smooth_n)
    slope = np.diff(smoothed) / dt
    t_slope = t_ms[:-1] + dt / 2
    return t_slope, slope


def _onset_slopes(t_slope, slope, bursts_df, pre_ms=ONSET_PRE_MS, post_ms=ONSET_POST_MS):
    """Max slope in [onset - pre_ms, onset + post_ms] for each burst."""
    max_slopes = []
    for _, row in bursts_df.iterrows():
        onset = row["burst_start_ms"]
        mask = (t_slope >= onset - pre_ms) & (t_slope <= onset + post_ms)
        if mask.sum() < 3:
            continue
        max_slopes.append(float(np.max(slope[mask])))
    return np.array(max_slopes)


def _baseline_slopes(t_slope, slope, bursts_df, n_samples=N_BASELINE_SAMPLES,
                     window_ms=300, margin_ms=500):
    """Max slope in random non-burst windows for baseline comparison."""
    burst_starts = bursts_df["burst_start_ms"].values
    burst_ends = bursts_df["burst_end_ms"].values
    t_min, t_max = t_slope[0], t_slope[-1]

    rng = np.random.default_rng(42)
    max_slopes = []
    attempts = 0
    while len(max_slopes) < n_samples and attempts < n_samples * 10:
        attempts += 1
        center = rng.uniform(t_min + window_ms, t_max - window_ms)
        near_burst = np.any(
            (center >= burst_starts - margin_ms) & (center <= burst_ends + margin_ms)
        )
        if near_burst:
            continue
        mask = (t_slope >= center - window_ms / 2) & (t_slope <= center + window_ms / 2)
        if mask.sum() < 3:
            continue
        max_slopes.append(float(np.max(slope[mask])))
    return np.array(max_slopes)


# ============================================================
# Plot 1: Onset slopes vs baseline slopes (per patient)
# ============================================================
def plot_onset_vs_baseline(all_records, out_path: Path):
    """
    Side-by-side boxplot: onset max-slopes vs baseline max-slopes per patient.
    """
    df = pd.DataFrame(all_records)
    if df.empty:
        return
    patients = sorted(df["patient"].unique())
    groups = patients + ["ALL"]

    box_width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 2.0), 6))

    for i, grp in enumerate(groups):
        center = i * 1.0
        if grp == "ALL":
            onset_vals = df.loc[df["type"] == "onset", "slope"].dropna().values
            base_vals = df.loc[df["type"] == "baseline", "slope"].dropna().values
        else:
            onset_vals = df.loc[(df["patient"] == grp) & (df["type"] == "onset"), "slope"].dropna().values
            base_vals = df.loc[(df["patient"] == grp) & (df["type"] == "baseline"), "slope"].dropna().values

        pos_onset = center - box_width / 2 - 0.02
        pos_base = center + box_width / 2 + 0.02

        if len(onset_vals) > 0:
            bp = ax.boxplot([onset_vals], positions=[pos_onset], widths=box_width,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            bp["boxes"][0].set_facecolor("#D44B4B")
            bp["boxes"][0].set_alpha(0.7)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.5)

        if len(base_vals) > 0:
            bp = ax.boxplot([base_vals], positions=[pos_base], widths=box_width,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            bp["boxes"][0].set_facecolor("#AAAAAA")
            bp["boxes"][0].set_alpha(0.7)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.5)

        if len(onset_vals) >= 5 and len(base_vals) >= 5:
            _, pval = sp_stats.mannwhitneyu(onset_vals, base_vals, alternative="greater")
            star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            if star:
                y_top = max(np.percentile(onset_vals, 95),
                            np.percentile(base_vals, 95))
                ax.text(center, y_top, star, ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    sep_x = (len(patients) - 0.5) * 1.0
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.6)

    ax.set_xticks([i * 1.0 for i in range(len(groups))])
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Max slope (p/ms)")
    ax.set_title("Proxy slope at burst onset vs baseline")
    ax.legend(handles=[Patch(facecolor="#D44B4B", alpha=0.7, label="Burst onset"),
                       Patch(facecolor="#AAAAAA", alpha=0.7, label="Baseline")],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Onset vs baseline: {out_path}")


# ============================================================
# Plot 2: Threshold convergence across patients
# ============================================================
def _find_best_slope_threshold(t_slope, slope, bursts_df, tolerance_ms):
    """Sweep slope thresholds, return best F1 threshold + metrics."""
    finite_slope = slope[np.isfinite(slope)]
    percentiles = np.arange(80, 100, 0.5)
    thresholds = np.percentile(finite_slope, percentiles)

    best_f1, best_thr, best_pct = 0.0, 0.0, 0.0
    best_sens, best_prec, best_fa = 0.0, 0.0, 0.0

    recording_dur_min = (t_slope[-1] - t_slope[0]) / 1000 / 60

    for pct, thr in zip(percentiles, thresholds):
        above = slope >= thr
        crossings_idx = np.where(np.diff(above.astype(int)) == 1)[0] + 1
        if len(crossings_idx) == 0:
            continue
        crossing_times = t_slope[crossings_idx]

        burst_starts = bursts_df["burst_start_ms"].values
        burst_ends = bursts_df["burst_end_ms"].values
        n_bursts = len(bursts_df)
        burst_detected = np.zeros(n_bursts, dtype=bool)
        n_tp = 0
        n_fp = 0
        for ct in crossing_times:
            matched = False
            for j in range(n_bursts):
                if burst_starts[j] - tolerance_ms <= ct <= burst_ends[j] + tolerance_ms:
                    burst_detected[j] = True
                    matched = True
                    break
            if matched:
                n_tp += 1
            else:
                n_fp += 1

        n_det = int(burst_detected.sum())
        sens = n_det / n_bursts if n_bursts > 0 else 0
        prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0
        fa = n_fp / recording_dur_min if recording_dur_min > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_pct = pct
            best_sens = sens
            best_prec = prec
            best_fa = fa

    return best_thr, best_pct, best_f1, best_sens, best_prec, best_fa


def plot_threshold_convergence(threshold_records, out_path: Path):
    """
    Show best slope threshold per patient/side as a strip plot.
    If thresholds converge, a horizontal band will emerge.
    """
    df = pd.DataFrame(threshold_records)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))

    labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#4878CF" if row["side"] == "left" else "#E8A02F"
        ax.scatter(i, row["best_threshold"], color=color, s=80, zorder=5, edgecolors="black")
        labels.append(f"{row['patient']}\n{row['period']}\n{row['side'][0].upper()}")

        ax.annotate(f"F1={row['best_f1']:.2f}", (i, row["best_threshold"]),
                    textcoords="offset points", xytext=(0, 10), fontsize=7,
                    ha="center", va="bottom")

    median_thr = df["best_threshold"].median()
    ax.axhline(median_thr, color="red", linestyle="--", linewidth=1.5,
               label=f"Median threshold = {median_thr:.4f} p/ms")
    q25 = df["best_threshold"].quantile(0.25)
    q75 = df["best_threshold"].quantile(0.75)
    ax.axhspan(q25, q75, alpha=0.15, color="red", label=f"IQR [{q25:.4f}, {q75:.4f}]")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Best slope threshold (p/ms)")
    ax.set_title("Slope threshold convergence across patients\n"
                 "(each dot = best F1 threshold for one recording)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Threshold convergence: {out_path}")


# ============================================================
# Plot 3: Slope-based performance curve (one per recording)
# ============================================================
def plot_slope_performance_curve(t_slope, slope, bursts_df, tolerance_ms,
                                 tag: str, out_path: Path):
    finite_slope = slope[np.isfinite(slope)]
    percentiles = np.arange(50, 100, 1.0)
    thresholds = np.percentile(finite_slope, percentiles)
    recording_dur_min = (t_slope[-1] - t_slope[0]) / 1000 / 60

    sens_list, prec_list, fa_list = [], [], []
    burst_starts = bursts_df["burst_start_ms"].values
    burst_ends = bursts_df["burst_end_ms"].values
    n_bursts = len(bursts_df)

    for thr in thresholds:
        above = slope >= thr
        crossings_idx = np.where(np.diff(above.astype(int)) == 1)[0] + 1
        if len(crossings_idx) == 0:
            sens_list.append(0.0)
            prec_list.append(0.0)
            fa_list.append(0.0)
            continue
        crossing_times = t_slope[crossings_idx]
        burst_detected = np.zeros(n_bursts, dtype=bool)
        n_tp, n_fp = 0, 0
        for ct in crossing_times:
            matched = False
            for j in range(n_bursts):
                if burst_starts[j] - tolerance_ms <= ct <= burst_ends[j] + tolerance_ms:
                    burst_detected[j] = True
                    matched = True
                    break
            n_tp += int(matched)
            n_fp += int(not matched)

        n_det = int(burst_detected.sum())
        sens_list.append(n_det / n_bursts if n_bursts > 0 else 0)
        prec_list.append(n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0)
        fa_list.append(n_fp / recording_dur_min if recording_dur_min > 0 else 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(percentiles, sens_list, color="blue", linewidth=2.5, label="Sensitivity")
    ax1.plot(percentiles, prec_list, color="green", linewidth=2.5, label="Precision")
    ax1.set_xlabel("Slope threshold (percentile)", fontsize=11)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(percentiles, fa_list, color="red", linewidth=1.5, linestyle=":",
             alpha=0.7, label="False alarms / min")
    ax2.set_ylabel("False alarms per minute", color="red", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title(f"Slope-based detection performance — {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def _iter_proxy_burst_pairs(discoveries, proxy_root: Path, mode: str):
    proxy_root = proxy_root.resolve()
    for patient, period, csv_L, csv_R in discoveries:
        tag_dir = csv_L.parent
        for side, csv_path in [("left", csv_L), ("right", csv_R)]:
            if mode == "hemi_to_hemi":
                bursts_df = _load_bursts(csv_path)
                if bursts_df.empty:
                    continue
                px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                if px_path is None or not px_path.exists():
                    continue
                t_ms, hemi_proxy, _ = load_hemi_proxy_from_excel(px_path)
                t_ms = t_ms.astype(float)
                proxy_p = hemi_proxy.astype(float) * SCALE_TO_P
                yield patient, period, side, "", t_ms, proxy_p, bursts_df
            elif mode == "region_to_hemi":
                bursts_df = _load_bursts(csv_path)
                if bursts_df.empty:
                    continue
                px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                if px_path is None or not px_path.exists():
                    continue
                t_ms, _, region_proxies = load_proxy_excel_with_region_columns(px_path)
                t_ms = t_ms.astype(float)
                for region, proxy_vals in region_proxies.items():
                    if proxy_vals.shape[0] != t_ms.shape[0]:
                        continue
                    proxy_p = proxy_vals.astype(float) * SCALE_TO_P
                    yield patient, period, side, region, t_ms, proxy_p, bursts_df
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
                proxy_p = hemi_proxy.astype(float) * SCALE_TO_P
                for region, windows in by_region.items():
                    bursts_df = _windows_to_bursts_df(windows)
                    if bursts_df.empty:
                        continue
                    yield patient, period, side, region, t_ms, proxy_p, bursts_df


def process_all(burst_root: Path, run_tag: str, proxy_root: Path,
                tolerance_ms: float, out_dir: Path, mode: str = "hemi_to_hemi"):
    allowed = _allowed_patient_periods() if SpikeTime_Mat_File else None
    discoveries = _discover_patient_periods(burst_root, run_tag, allowed=allowed)
    if not discoveries:
        print(f"No data found under {burst_root} with run_tag:\n  {run_tag}")
        return

    print(f"Found {len(discoveries)} patient-period(s)  [mode={mode}]")
    proxy_root = proxy_root.resolve()

    all_slope_records = []
    threshold_records = []

    for patient, period, side, region_label, t_ms, proxy_p, bursts_df in _iter_proxy_burst_pairs(
            discoveries, proxy_root, mode):
        tag = f"{patient} • {period} • {side.capitalize()}" + (f" • {region_label}" if region_label else "")
        pat_dir = out_dir / patient
        if region_label:
            pat_dir = pat_dir / region_label
        pat_dir.mkdir(parents=True, exist_ok=True)

        t_slope, slope = compute_slope(t_ms, proxy_p)

        onset_sl = _onset_slopes(t_slope, slope, bursts_df)
        for v in onset_sl:
            all_slope_records.append({"patient": patient, "side": side, "type": "onset", "slope": v})
        base_sl = _baseline_slopes(t_slope, slope, bursts_df)
        for v in base_sl:
            all_slope_records.append({"patient": patient, "side": side, "type": "baseline", "slope": v})

        best_thr, best_pct, best_f1, best_sens, best_prec, best_fa = \
            _find_best_slope_threshold(t_slope, slope, bursts_df, tolerance_ms)

        threshold_records.append({
            "patient": patient, "period": period, "side": side, "region": region_label,
            "best_threshold": best_thr, "best_pct": best_pct,
            "best_f1": best_f1, "sensitivity": best_sens,
            "precision": best_prec, "fa_per_min": best_fa,
            "n_bursts": len(bursts_df),
        })

        file_suffix = f"{period}_{side}" + (f"_{region_label}" if region_label else "")
        plot_slope_performance_curve(
            t_slope, slope, bursts_df, tolerance_ms, tag,
            pat_dir / f"slope_performance_{file_suffix}.png",
        )

        print(f"  {tag}: {len(bursts_df)} bursts, "
              f"onset slope median={np.median(onset_sl):.4f} p/ms, "
              f"baseline median={np.median(base_sl):.4f} p/ms, "
              f"best F1={best_f1:.2f} @ {best_pct:.0f}th pct "
              f"(sens={best_sens:.0%}, prec={best_prec:.0%}, FA={best_fa:.1f}/min)")

    # --- Cross-patient plots ---
    if all_slope_records:
        plot_onset_vs_baseline(all_slope_records,
                               out_dir / "boxplot_onset_vs_baseline_slope.png")

    if threshold_records:
        plot_threshold_convergence(threshold_records,
                                   out_dir / "threshold_convergence.png")
        csv_path = out_dir / "slope_detection_summary.csv"
        pd.DataFrame(threshold_records).to_csv(csv_path, index=False)
        print(f"\nSummary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Slope-based proxy detection analysis."
    )
    parser.add_argument("--burst-root", type=Path, default=BURST_ROOT)
    parser.add_argument("--run-tag", type=str, default=PROXY_ANALYSIS_RUN_TAG)
    parser.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
    print(f"Output     : {out_dir}\n")

    process_all(args.burst_root, args.run_tag, proxy_root,
                args.tolerance_ms, out_dir, args.mode)


if __name__ == "__main__":
    main()
