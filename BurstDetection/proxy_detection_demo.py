#!/usr/bin/env python3
"""
Real-time proxy vs offline burst detection demonstration.

Compares:
  OFFLINE (gold standard): Full spike sorting → burst detection → network bursts
  REAL-TIME (fast proxy):  Bandpass filter + NEO → simple threshold

Produces:
  1. Two-panel overlay: proxy signal (p-units) + threshold vs offline burst events
  2. Detection performance curve: sensitivity & precision vs threshold
  3. Burst vs non-burst proxy amplitude separation
  4. Summary CSV with detection stats + false alarm rate

Usage:
  python proxy_detection_demo.py
  python proxy_detection_demo.py --run-tag "separateGPi__..."
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

DEFAULT_OUT_DIR = PROXY_ANALYSIS_ROOT / "proxy_detection_demo"
TOLERANCE_MS = 300
SCALE_TO_P = 1e12  # raw proxy → pico-units
MODES = ("hemi_to_hemi", "region_to_hemi", "hemi_to_region", "region_to_region")


def _windows_to_bursts_df(windows: list[tuple[float, float]]) -> pd.DataFrame:
    """Convert list of (start_ms, end_ms) to DataFrame with burst_start_ms, burst_end_ms."""
    if not windows:
        return pd.DataFrame()
    return pd.DataFrame([{"burst_start_ms": s, "burst_end_ms": e} for s, e in windows])


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
    return df.dropna(subset=list(required))


def _classify_crossings(crossing_times_ms: np.ndarray,
                        bursts_df: pd.DataFrame,
                        tolerance_ms: float):
    burst_starts = bursts_df["burst_start_ms"].values
    burst_ends = bursts_df["burst_end_ms"].values
    n_bursts = len(bursts_df)

    is_tp = np.zeros(len(crossing_times_ms), dtype=bool)
    burst_detected = np.zeros(n_bursts, dtype=bool)

    for i, ct in enumerate(crossing_times_ms):
        for j in range(n_bursts):
            if burst_starts[j] - tolerance_ms <= ct <= burst_ends[j] + tolerance_ms:
                is_tp[i] = True
                burst_detected[j] = True
                break

    return crossing_times_ms[is_tp], crossing_times_ms[~is_tp], burst_detected


def _find_upward_crossings(t_ms: np.ndarray, proxy: np.ndarray,
                           threshold: float) -> np.ndarray:
    above = proxy >= threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    return t_ms[crossings]


def _evaluate_threshold(t_ms, proxy, bursts_df, threshold, tolerance_ms):
    crossings = _find_upward_crossings(t_ms, proxy, threshold)
    if len(crossings) == 0:
        return 0.0, 0.0, 0, 0, 0
    tp_times, fp_times, burst_detected = _classify_crossings(
        crossings, bursts_df, tolerance_ms
    )
    n_tp = len(tp_times)
    n_fp = len(fp_times)
    n_detected = int(burst_detected.sum())
    n_bursts = len(bursts_df)

    sensitivity = n_detected / n_bursts if n_bursts > 0 else 0.0
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    return sensitivity, precision, n_detected, n_tp, n_fp


# ============================================================
# Plot 1: Two-panel overlay (hero figure)
# ============================================================
def plot_two_panel_overlay(t_ms, proxy_p, bursts_df, threshold_p, tolerance_ms,
                           tag: str, out_path: Path,
                           t_range_s: tuple | None = None):
    """
    Top panel:  Real-time proxy (p-units) with threshold + TP/FP markers
    Bottom panel: Offline-detected network bursts (ground truth bars)
    """
    crossings = _find_upward_crossings(t_ms, proxy_p, threshold_p)
    tp_times, fp_times, burst_detected = _classify_crossings(
        crossings, bursts_df, tolerance_ms
    )
    sens, prec, n_det, n_tp, n_fp = _evaluate_threshold(
        t_ms, proxy_p, bursts_df, threshold_p, tolerance_ms
    )

    t_s = t_ms / 1000.0
    recording_dur_s = (t_ms[-1] - t_ms[0]) / 1000
    fa_per_min = n_fp / (recording_dur_s / 60) if recording_dur_s > 0 else 0

    fig, (ax_proxy, ax_bursts) = plt.subplots(
        2, 1, figsize=(16, 6), height_ratios=[3, 1], sharex=True
    )

    # --- Top: proxy signal ---
    # Clip y-axis to avoid outlier compression
    finite_p = proxy_p[np.isfinite(proxy_p)]
    y_lo = float(np.percentile(finite_p, 1))
    y_hi = float(np.percentile(finite_p, 99.5))
    y_pad = (y_hi - y_lo) * 0.1
    y_lo -= y_pad
    y_hi += y_pad

    ax_proxy.plot(t_s, proxy_p, color="#4878CF", linewidth=0.5, alpha=0.8)
    ax_proxy.axhline(threshold_p, color="#E8A02F", linestyle="--", linewidth=1.5,
                     label=f"Threshold ({threshold_p:.1f} p)")

    if len(tp_times) > 0:
        tp_y = np.interp(tp_times, t_ms, proxy_p).clip(y_lo, y_hi)
        ax_proxy.scatter(tp_times / 1000, tp_y, color="green", marker="v", s=40,
                         zorder=5, label=f"Hit ({n_tp})")
    if len(fp_times) > 0:
        fp_y = np.interp(fp_times, t_ms, proxy_p).clip(y_lo, y_hi)
        ax_proxy.scatter(fp_times / 1000, fp_y, color="red", marker="x", s=30,
                         zorder=5, label=f"False alarm ({n_fp})")

    ax_proxy.set_ylim(y_lo, y_hi)
    ax_proxy.set_ylabel("Fast proxy (p)")
    ax_proxy.set_title(
        f"Real-time proxy detection vs offline burst detection — {tag}\n"
        f"Sensitivity = {sens:.0%} ({n_det}/{len(bursts_df)} bursts)   "
        f"Precision = {prec:.0%}   "
        f"False alarms = {fa_per_min:.1f}/min",
        fontsize=11,
    )
    ax_proxy.legend(loc="upper right", fontsize=8)
    ax_proxy.text(0.005, 0.95, "REAL-TIME\n(filter + NEO + threshold)",
                  transform=ax_proxy.transAxes, fontsize=8, va="top",
                  fontweight="bold", color="#4878CF",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Bottom: offline burst events ---
    for _, row in bursts_df.iterrows():
        ax_bursts.axvspan(row["burst_start_ms"] / 1000, row["burst_end_ms"] / 1000,
                          alpha=0.7, color="#D44B4B", linewidth=0)
    ax_bursts.set_ylim(0, 1)
    ax_bursts.set_yticks([])
    ax_bursts.set_ylabel("Offline\nbursts", fontsize=9, rotation=0, labelpad=45, va="center")
    ax_bursts.set_xlabel("Time (s)")
    ax_bursts.text(0.005, 0.8, "OFFLINE (gold standard)\n(spike sorting + burst detection)",
                   transform=ax_bursts.transAxes, fontsize=8, va="top",
                   fontweight="bold", color="#D44B4B",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if t_range_s is not None:
        ax_proxy.set_xlim(t_range_s)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Plot 2: Performance curve
# ============================================================
def plot_performance_curve(t_ms, proxy_p, bursts_df, tolerance_ms,
                           tag: str, out_path: Path):
    percentiles = np.arange(50, 100, 1)
    thresholds = np.percentile(proxy_p[np.isfinite(proxy_p)], percentiles)

    recording_dur_min = (t_ms[-1] - t_ms[0]) / 1000 / 60

    sensitivities, precisions, fa_rates = [], [], []
    for thr in thresholds:
        sens, prec, _, _, n_fp = _evaluate_threshold(
            t_ms, proxy_p, bursts_df, thr, tolerance_ms
        )
        sensitivities.append(sens)
        precisions.append(prec)
        fa_rates.append(n_fp / recording_dur_min if recording_dur_min > 0 else 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(percentiles, sensitivities, color="#4878CF", linewidth=2.5,
             label="Sensitivity (% of offline bursts detected)")
    ax1.plot(percentiles, precisions, color="green", linewidth=2.5,
             label="Precision (% of proxy detections that match)")
    ax1.set_xlabel("Threshold (percentile of proxy signal)", fontsize=11)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(percentiles, fa_rates, color="red", linewidth=1.5, linestyle=":",
             alpha=0.7, label="False alarms / min")
    ax2.set_ylabel("False alarms per minute", color="red", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title(
        f"How well can the real-time proxy replicate offline burst detection?\n{tag}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return percentiles, thresholds, sensitivities, precisions, fa_rates


# ============================================================
# Plot 2b: ROC curve (TPR vs FPR)
# ============================================================
NEG_WINDOW_MS = 300  # length of "negative" windows for FPR

def _negative_windows(t_ms: np.ndarray, bursts_df: pd.DataFrame,
                      tolerance_ms: float, window_ms: float = NEG_WINDOW_MS):
    """Non-overlapping windows that do not overlap any burst (±tolerance)."""
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    burst_intervals = []
    for _, row in bursts_df.iterrows():
        burst_intervals.append((
            row["burst_start_ms"] - tolerance_ms,
            row["burst_end_ms"] + tolerance_ms,
        ))
    windows = []
    t = t_min
    while t + window_ms <= t_max:
        overlap = any(
            (t < e and t + window_ms > s)
            for s, e in burst_intervals
        )
        if not overlap:
            windows.append((t, t + window_ms))
        t += window_ms
    return windows


def _tpr_fpr_at_threshold(t_ms, proxy_p, bursts_df, threshold, tolerance_ms,
                          neg_windows: list[tuple[float, float]]):
    """TPR = fraction of bursts with ≥1 crossing; FPR = fraction of neg windows with ≥1 crossing."""
    crossings = _find_upward_crossings(t_ms, proxy_p, threshold)
    n_bursts = len(bursts_df)
    burst_starts = bursts_df["burst_start_ms"].values
    burst_ends = bursts_df["burst_end_ms"].values

    tp = 0
    for j in range(n_bursts):
        for ct in crossings:
            if burst_starts[j] - tolerance_ms <= ct <= burst_ends[j] + tolerance_ms:
                tp += 1
                break

    n_neg = len(neg_windows)
    fp_neg = 0
    for (s, e) in neg_windows:
        for ct in crossings:
            if s <= ct <= e:
                fp_neg += 1
                break

    tpr = tp / n_bursts if n_bursts > 0 else 0.0
    fpr = fp_neg / n_neg if n_neg > 0 else 0.0
    return tpr, fpr


ROC_PERCENTILES = np.arange(50, 100, 0.5)


def _roc_tpr_fpr_one_recording(t_ms, proxy_p, bursts_df, tolerance_ms
                               ) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Return (tprs, fprs, n_bursts, n_neg) at ROC_PERCENTILES for one recording, or None if no neg windows."""
    neg_windows = _negative_windows(t_ms, bursts_df, tolerance_ms)
    if len(neg_windows) == 0:
        return None
    n_bursts = len(bursts_df)
    n_neg = len(neg_windows)
    thresholds = np.percentile(proxy_p[np.isfinite(proxy_p)], ROC_PERCENTILES)
    tprs, fprs = [], []
    for thr in thresholds:
        tpr, fpr = _tpr_fpr_at_threshold(
            t_ms, proxy_p, bursts_df, thr, tolerance_ms, neg_windows
        )
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(tprs), np.array(fprs), n_bursts, n_neg


def plot_roc_curve(t_ms, proxy_p, bursts_df, tolerance_ms, tag: str, out_path: Path):
    """Plot ROC curve (TPR vs FPR) and AUC as threshold is swept."""
    data = _roc_tpr_fpr_one_recording(t_ms, proxy_p, bursts_df, tolerance_ms)
    if data is None:
        return
    tprs, fprs, _, _ = data

    order = np.argsort(fprs)
    fpr_s = fprs[order]
    tpr_s = tprs[order]
    auc = float(np.trapz(tpr_s, fpr_s))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, color="#4878CF", linewidth=2, label=f"Proxy (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (neg windows with detection)")
    ax.set_ylabel("True positive rate (bursts detected)")
    ax.set_title(f"ROC — {tag}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pooled_roc_from_records(roc_records: list[tuple[np.ndarray, np.ndarray, int, int]]
                             ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (pooled_fpr, pooled_tpr, auc) from per-recording ROC records."""
    if not roc_records:
        return None
    n_p = len(ROC_PERCENTILES)
    total_tp = np.zeros(n_p)
    total_fp_neg = np.zeros(n_p)
    total_bursts = 0
    total_neg = 0
    for tprs, fprs, n_bursts, n_neg in roc_records:
        total_tp += tprs * n_bursts
        total_fp_neg += fprs * n_neg
        total_bursts += n_bursts
        total_neg += n_neg
    if total_bursts == 0 or total_neg == 0:
        return None
    pooled_tpr = total_tp / total_bursts
    pooled_fpr = total_fp_neg / total_neg

    order = np.argsort(pooled_fpr)
    fpr_s = pooled_fpr[order]
    tpr_s = pooled_tpr[order]
    auc = float(np.trapz(tpr_s, fpr_s))

    return pooled_fpr, pooled_tpr, auc


def plot_pooled_roc(roc_records: list[tuple[np.ndarray, np.ndarray, int, int]], out_path: Path):
    """Pool TPR/FPR across recordings (percentile-based threshold): one master ROC and AUC."""
    pooled = _pooled_roc_from_records(roc_records)
    if pooled is None:
        return
    pooled_fpr, pooled_tpr, auc = pooled

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pooled_fpr, pooled_tpr, color="#4878CF", linewidth=2,
            label=f"Pooled (AUC = {auc:.3f}, n={len(roc_records)} rec)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (neg windows with detection)")
    ax.set_ylabel("True positive rate (bursts detected)")
    ax.set_title("ROC — pooled across all recordings (percentile threshold)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pooled ROC (n={len(roc_records)}): {out_path}")


def plot_pooled_roc_by_group(
    roc_records_meta: list[dict],
    group_key: str,
    out_path: Path,
    title_prefix: str,
):
    """Pooled ROC per group (e.g., per region or per patient) in one overview figure."""
    if not roc_records_meta:
        return
    groups = sorted({rec.get(group_key, "") for rec in roc_records_meta if rec.get(group_key, "")})
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    n_plotted = 0
    for grp in groups:
        group_recs = [
            (rec["tprs"], rec["fprs"], rec["n_bursts"], rec["n_neg"])
            for rec in roc_records_meta
            if rec.get(group_key, "") == grp
        ]
        pooled = _pooled_roc_from_records(group_recs)
        if pooled is None:
            continue
        pooled_fpr, pooled_tpr, auc = pooled
        ax.plot(
            pooled_fpr,
            pooled_tpr,
            linewidth=2,
            label=f"{grp} (AUC={auc:.3f}, n={len(group_recs)} rec)",
        )
        n_plotted += 1

    if n_plotted == 0:
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (neg windows with detection)")
    ax.set_ylabel("True positive rate (bursts detected)")
    ax.set_title(title_prefix)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pooled ROC by {group_key}: {out_path}")


def plot_f1_histogram(summary_rows: list[dict], out_path: Path):
    """Boxplot of best F1 per recording, grouped by subject and hemisphere (like other grouped plots)."""
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    if "f1" not in df.columns or "side" not in df.columns or "patient" not in df.columns:
        return

    records_left: list[dict] = []
    records_right: list[dict] = []
    for _, row in df.iterrows():
        f1_val = row["f1"]
        if pd.isna(f1_val):
            continue
        rec = {"patient": row["patient"], "f1": float(f1_val)}
        if row["side"] == "left":
            records_left.append(rec)
        elif row["side"] == "right":
            records_right.append(rec)

    if not records_left and not records_right:
        return

    _plot_grouped_boxplot(
        records_left,
        records_right,
        val_col="f1",
        ylabel="F1 score (best threshold per recording)",
        title="Proxy detection F1 (by subject)",
        out_path=out_path,
    )


def plot_f1_by_region_boxplot(summary_rows: list[dict], out_path: Path, mode: str):
    """Boxplot of best F1 per recording, grouped by region label across patients."""
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    if "f1" not in df.columns or "side" not in df.columns or "region" not in df.columns:
        return

    records_left: list[dict] = []
    records_right: list[dict] = []
    for _, row in df.iterrows():
        f1_val = row["f1"]
        region = row["region"]
        if pd.isna(f1_val) or not region:
            continue
        rec = {"patient": region, "f1": float(f1_val)}
        if row["side"] == "left":
            records_left.append(rec)
        elif row["side"] == "right":
            records_right.append(rec)

    if not records_left and not records_right:
        return

    title = f"Proxy detection F1 by region ({mode})"
    _plot_grouped_boxplot(
        records_left,
        records_right,
        val_col="f1",
        ylabel="F1 score (best threshold per recording)",
        title=title,
        out_path=out_path,
        x_label="Region",
    )


def plot_auc_by_mode(base_dir: Path, out_path: Path | None = None):
    """Read detection_summary.csv from base_dir/{hemi_to_hemi,region_to_hemi,hemi_to_region,region_to_region} and plot AUC by mode (no rerun)."""
    base_dir = base_dir.resolve()
    mode_dirs = ["hemi_to_hemi", "region_to_hemi", "hemi_to_region", "region_to_region"]
    rows: list[dict] = []
    for mode in mode_dirs:
        csv_path = base_dir / mode / "detection_summary.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "auc" not in df.columns:
            continue
        for _, r in df.iterrows():
            auc_val = r.get("auc")
            if pd.isna(auc_val):
                continue
            rows.append({"mode": mode, "auc": float(auc_val), "patient": r.get("patient"), "side": r.get("side")})
    if not rows:
        print("No AUC data found (need detection_summary.csv with 'auc' in each mode dir under base).")
        return
    out_path = out_path or (base_dir / "auc_by_mode.png")
    df = pd.DataFrame(rows)
    modes_ordered = [m for m in mode_dirs if m in df["mode"].unique()]
    data_by_mode = [df.loc[df["mode"] == m, "auc"].values for m in modes_ordered]
    fig, ax = plt.subplots(figsize=(max(6, len(modes_ordered) * 2), 5))
    bp = ax.boxplot(data_by_mode, labels=modes_ordered, patch_artist=True, showfliers=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#4878CF")
        box.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_ylabel("AUC (per recording)")
    ax.set_xlabel("Mode")
    ax.set_title("ROC AUC by analysis mode (from existing detection_summary.csv)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  AUC by mode: {out_path}")


# ============================================================
# Plot 3: Amplitude separation
# ============================================================
def plot_amplitude_separation(t_ms, proxy_p, bursts_df, tag: str, out_path: Path):
    in_burst = np.zeros(len(t_ms), dtype=bool)
    for _, row in bursts_df.iterrows():
        mask = (t_ms >= row["burst_start_ms"]) & (t_ms <= row["burst_end_ms"])
        in_burst |= mask

    burst_vals = proxy_p[in_burst & np.isfinite(proxy_p)]
    non_burst_vals = proxy_p[~in_burst & np.isfinite(proxy_p)]

    fig, ax = plt.subplots(figsize=(6, 7))
    bp = ax.boxplot([non_burst_vals, burst_vals],
                    tick_labels=["Outside burst\n(background)", "During burst\n(offline-detected)"],
                    patch_artist=True, widths=0.5, showfliers=False)
    bp["boxes"][0].set_facecolor("#4878CF")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#E8A02F")
    bp["boxes"][1].set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)

    if len(burst_vals) > 0 and len(non_burst_vals) > 0:
        d = (np.mean(burst_vals) - np.mean(non_burst_vals)) / np.std(non_burst_vals)
        ax.text(0.95, 0.95, f"Cohen's d = {d:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_ylabel("Fast proxy amplitude (p)")
    ax.set_title(f"Proxy amplitude separation — {tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Per-burst correlation + grouped boxplot
# ============================================================
PAD_MS = 300  # look before/after burst for correlation window

def _per_burst_correlations(t_ms, proxy_p, bursts_df, pad_ms=PAD_MS):
    """
    For each burst, extract proxy in [start - pad, end + pad] and correlate
    with a binary indicator (1 during burst, 0 outside). Positive r = proxy
    is higher during the burst.
    """
    corrs = []
    for _, row in bursts_df.iterrows():
        t0 = row["burst_start_ms"] - pad_ms
        t1 = row["burst_end_ms"] + pad_ms
        mask = (t_ms >= t0) & (t_ms <= t1)
        if mask.sum() < 6:
            continue
        seg_t = t_ms[mask]
        seg_p = proxy_p[mask]
        indicator = ((seg_t >= row["burst_start_ms"]) &
                     (seg_t <= row["burst_end_ms"])).astype(float)
        valid = np.isfinite(seg_p)
        if valid.sum() < 6 or indicator[valid].std() == 0:
            continue
        r, _ = sp_stats.pearsonr(indicator[valid], seg_p[valid])
        corrs.append(r)
    return np.array(corrs)


def _per_burst_pct_change(t_ms, proxy_p, bursts_df, pad_ms=PAD_MS):
    """
    For each burst, compute % change = (mean_during - mean_outside) / |mean_outside| × 100.
    'during' = [burst_start, burst_end]; 'outside' = the pad regions on either side.
    """
    pcts = []
    for _, row in bursts_df.iterrows():
        t0 = row["burst_start_ms"] - pad_ms
        t1 = row["burst_end_ms"] + pad_ms
        mask = (t_ms >= t0) & (t_ms <= t1)
        if mask.sum() < 6:
            continue
        seg_t = t_ms[mask]
        seg_p = proxy_p[mask]
        valid = np.isfinite(seg_p)
        if valid.sum() < 6:
            continue
        in_burst = ((seg_t >= row["burst_start_ms"]) &
                    (seg_t <= row["burst_end_ms"])) & valid
        outside = (~((seg_t >= row["burst_start_ms"]) &
                     (seg_t <= row["burst_end_ms"]))) & valid
        if in_burst.sum() < 2 or outside.sum() < 2:
            continue
        mean_during = np.mean(seg_p[in_burst])
        mean_outside = np.mean(seg_p[outside])
        if mean_outside == 0:
            continue
        pct = (mean_during - mean_outside) / abs(mean_outside) * 100.0
        pcts.append(pct)
    return np.array(pcts)


def _plot_grouped_boxplot(records_left, records_right, val_col: str,
                          ylabel: str, title: str, out_path: Path,
                          x_label: str = "Subject"):
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

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.6), 6))

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
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor="#4878CF", alpha=0.7, label="Left"),
                       Patch(facecolor="#E8A02F", alpha=0.7, label="Right")],
              loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {title}: {out_path}")


# ============================================================
# Main
# ============================================================
def _iter_proxy_burst_pairs(discoveries, proxy_root: Path, mode: str):
    """Yield (patient, period, side, region_label, t_ms, proxy_p, bursts_df) for each proxy-vs-burst pair."""
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

            elif mode == "hemi_to_region":
                region_csv = burst_region_csv_path(tag_dir, side)
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

            else:  # region_to_region
                region_csv = burst_region_csv_path(tag_dir, side)
                by_region = load_region_bursts_by_region(region_csv)
                if not by_region:
                    continue
                px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
                if px_path is None or not px_path.exists():
                    continue
                t_ms, _, region_proxies = load_proxy_excel_with_region_columns(px_path)
                t_ms = t_ms.astype(float)
                for region in by_region:
                    if region not in region_proxies:
                        continue
                    proxy_vals = region_proxies[region]
                    if proxy_vals.shape[0] != t_ms.shape[0]:
                        continue
                    proxy_p = proxy_vals.astype(float) * SCALE_TO_P
                    bursts_df = _windows_to_bursts_df(by_region[region])
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
    summary_rows = []
    roc_records: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    roc_records_meta: list[dict] = []
    corr_records_left: list[dict] = []
    corr_records_right: list[dict] = []
    pct_records_left: list[dict] = []
    pct_records_right: list[dict] = []

    for patient, period, side, region_label, t_ms, proxy_p, bursts_df in _iter_proxy_burst_pairs(
            discoveries, proxy_root, mode):
        tag = f"{patient} • {period} • {side.capitalize()}"
        if region_label:
            tag += f" • {region_label}"
        pat_dir = out_dir / patient
        if region_label:
            pat_dir = pat_dir / region_label
        pat_dir.mkdir(parents=True, exist_ok=True)

        recording_dur_s = (t_ms[-1] - t_ms[0]) / 1000
        recording_dur_min = recording_dur_s / 60

        file_suffix = f"{period}_{side}" + (f"_{region_label}" if region_label else "")

        pcts, thrs, sens_list, prec_list, fa_list = plot_performance_curve(
            t_ms, proxy_p, bursts_df, tolerance_ms, tag,
            pat_dir / f"performance_{file_suffix}.png",
        )
        plot_roc_curve(
            t_ms, proxy_p, bursts_df, tolerance_ms, tag,
            pat_dir / f"roc_{file_suffix}.png",
        )
        one_roc = _roc_tpr_fpr_one_recording(t_ms, proxy_p, bursts_df, tolerance_ms)
        if one_roc is not None:
            tprs, fprs, n_bursts, n_neg = one_roc
            order = np.argsort(fprs)
            auc = float(np.trapz(tprs[order], fprs[order]))
            roc_records.append((tprs, fprs, n_bursts, n_neg))
            roc_records_meta.append(
                {
                    "patient": patient,
                    "region": region_label or "",
                    "side": side,
                    "tprs": tprs,
                    "fprs": fprs,
                    "n_bursts": n_bursts,
                    "n_neg": n_neg,
                }
            )
        else:
            auc = np.nan
        f1_scores = [2 * s * p / (s + p) if (s + p) > 0 else 0 for s, p in zip(sens_list, prec_list)]
        best_idx = int(np.argmax(f1_scores))
        best_pct, best_thr = pcts[best_idx], thrs[best_idx]
        best_sens, best_prec = sens_list[best_idx], prec_list[best_idx]
        best_f1, best_fa = f1_scores[best_idx], fa_list[best_idx]

        plot_two_panel_overlay(
            t_ms, proxy_p, bursts_df, best_thr, tolerance_ms, tag,
            pat_dir / f"overlay_{file_suffix}.png",
        )
        if not bursts_df.empty:
            first_burst_s = bursts_df["burst_start_ms"].min() / 1000
            zoom_start = max(t_ms[0] / 1000, first_burst_s - 10)
            zoom_end = zoom_start + 60
        else:
            zoom_start, zoom_end = t_ms[0] / 1000, t_ms[0] / 1000 + 60
        plot_two_panel_overlay(
            t_ms, proxy_p, bursts_df, best_thr, tolerance_ms, f"{tag} (60s window)",
            pat_dir / f"overlay_zoomed_{file_suffix}.png", t_range_s=(zoom_start, zoom_end),
        )
        plot_amplitude_separation(t_ms, proxy_p, bursts_df, tag, pat_dir / f"amplitude_{file_suffix}.png")

        burst_corrs = _per_burst_correlations(t_ms, proxy_p, bursts_df)
        burst_pcts = _per_burst_pct_change(t_ms, proxy_p, bursts_df)
        corr_target = corr_records_left if side == "left" else corr_records_right
        pct_target = pct_records_left if side == "left" else pct_records_right
        for r_val in burst_corrs:
            corr_target.append({"patient": patient, "corr": r_val, "region": region_label})
        for p_val in burst_pcts:
            pct_target.append({"patient": patient, "pct": p_val, "region": region_label})

        summary_rows.append({
            "patient": patient, "period": period, "side": side, "region": region_label,
            "n_bursts": len(bursts_df), "recording_min": recording_dur_min,
            "best_threshold_pct": best_pct, "best_threshold_p": best_thr,
            "sensitivity": best_sens, "precision": best_prec, "f1": best_f1,
            "auc": auc,
            "false_alarms_per_min": best_fa,
            "median_burst_corr": float(np.nanmedian(burst_corrs)) if len(burst_corrs) > 0 else np.nan,
        })
        print(f"  {tag}: {len(bursts_df)} bursts in {recording_dur_min:.1f} min, "
              f"F1={best_f1:.2f} @ {best_pct:.0f}th pct "
              f"(sens={best_sens:.0%}, prec={best_prec:.0%}, FA={best_fa:.1f}/min)")

    if corr_records_left or corr_records_right:
        _plot_grouped_boxplot(
            corr_records_left, corr_records_right, val_col="corr",
            ylabel="Pearson r (proxy vs burst indicator)",
            title="Proxy–burst correlation (by subject)",
            out_path=out_dir / "boxplot_burst_correlation.png",
        )
    if pct_records_left or pct_records_right:
        _plot_grouped_boxplot(
            pct_records_left, pct_records_right, val_col="pct",
            ylabel="% change (during burst vs outside)",
            title="Proxy % change during burst (by subject)",
            out_path=out_dir / "boxplot_burst_pct_change.png",
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "detection_summary.csv", index=False)
        print(f"\nSummary: {out_dir / 'detection_summary.csv'}")
    if roc_records:
        plot_pooled_roc(roc_records, out_dir / "roc_pooled.png")
        if mode == "region_to_region" and roc_records_meta:
            plot_pooled_roc_by_group(
                roc_records_meta,
                group_key="region",
                out_path=out_dir / "roc_pooled_by_region.png",
                title_prefix="ROC — pooled per region (region_to_region)",
            )
            plot_pooled_roc_by_group(
                roc_records_meta,
                group_key="patient",
                out_path=out_dir / "roc_pooled_by_patient.png",
                title_prefix="ROC — pooled per patient (region_to_region)",
            )
    if summary_rows:
        plot_f1_histogram(summary_rows, out_dir / "boxplot_f1_by_subject.png")
        if mode in {"region_to_hemi", "hemi_to_region", "region_to_region"}:
            plot_f1_by_region_boxplot(summary_rows, out_dir / f"boxplot_f1_by_region_{mode}.png", mode)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time proxy vs offline burst detection demonstration."
    )
    parser.add_argument("--burst-root", type=Path, default=BURST_ROOT)
    parser.add_argument("--run-tag", type=str, default=PROXY_ANALYSIS_RUN_TAG)
    parser.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS,
                        help=f"TP tolerance window in ms (default: {TOLERANCE_MS})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", type=str, default="hemi_to_hemi", choices=MODES,
                        help="Single run: hemi_to_hemi (default), region_to_hemi, hemi_to_region, region_to_region")
    parser.add_argument("--modes", type=str, default=None,
                        help="Run multiple modes in one go: comma-separated, e.g. region_to_hemi,hemi_to_region,region_to_region")
    parser.add_argument("--plot-auc-by-mode", action="store_true",
                        help="Only plot AUC by mode from existing detection_summary.csv (no patient rerun)")
    parser.add_argument("--auc-plot-base", type=Path, default=None,
                        help="Base dir for --plot-auc-by-mode (default: same as --out-dir)")
    args = parser.parse_args()

    if args.plot_auc_by_mode:
        base = (args.auc_plot_base or args.out_dir).resolve()
        plot_auc_by_mode(base)
        return

    if args.modes:
        mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
        bad = [m for m in mode_list if m not in MODES]
        if bad:
            parser.error(f"--modes: invalid mode(s) {bad}. Choose from: {list(MODES)}")
        modes_to_run = mode_list
    else:
        modes_to_run = [args.mode]

    out_base = args.out_dir.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    proxy_root = Path(PROXY_ROOT or DEFAULT_PROXY_ROOT).resolve()

    print(f"Burst root   : {args.burst_root}")
    print(f"Proxy root   : {proxy_root}")
    print(f"Run tag      : {args.run_tag}")
    print(f"Modes       : {modes_to_run}")
    print(f"TP tolerance : ±{args.tolerance_ms:.0f} ms")
    print("Comparing: OFFLINE (spike sort + burst detection)")
    print("       vs: REAL-TIME (filter + NEO + threshold)\n")

    for mode in modes_to_run:
        out_dir = out_base / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output       : {out_dir}\n")
        process_all(args.burst_root, args.run_tag, proxy_root,
                    args.tolerance_ms, out_dir, mode)
        print()


if __name__ == "__main__":
    main()
