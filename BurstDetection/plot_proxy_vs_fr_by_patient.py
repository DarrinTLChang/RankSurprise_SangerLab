#!/usr/bin/env python3
"""
Standalone script: correlate proxy (from rasters_all_with_fast_proxies) with
firing rate (computed from spike .mat files) at both hemisphere and region level.

Produces one bar-chart PNG for LEFT and one for RIGHT, each with two bars
per patient: Region↔Region (blue) and Hemi↔Hemi (orange).

NO burst detection is run — only spike loading, FR binning, proxy loading,
and Pearson correlation, so this is much faster than running main.py.

Usage:
  python plot_proxy_vs_fr_by_patient.py
  python plot_proxy_vs_fr_by_patient.py --out-dir ./my_plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio

# Import pipeline helpers (config values, utils, stats, FR computation)
from config import (
    SpikeTime_Mat_File,
    PROXY_ROOT,
    pooling_toggle,
    alpha_meanisi_toggle,
    FR_MIN_HZ,
    SNR_MIN,
    coactivity_bins_s,
    OVERLAP_FRACTION,
)
from pipeline.utils import (
    dataset_labels,
    compute_recording_duration_s,
    split_spike_struct_by_side,
    infer_region,
)
from pipeline.stats import build_cache
from pipeline.plotting import population_firing_rate_binned
from compare_proxy_vs_network import (
    load_proxy_excel_with_region_columns,
    load_hemi_proxy_from_excel,
    derive_proxy_xlsx_path,
    DEFAULT_PROXY_ROOT,
)

# ============================================================
# Defaults
# ============================================================
DEFAULT_OUT_DIR = Path("proxy_fr_plots")


def _resample_to_bins(t_proxy_s: np.ndarray, proxy_vals: np.ndarray,
                      bin_starts: np.ndarray, bin_width: float) -> np.ndarray:
    """Average proxy values within each FR bin."""
    edges = np.append(bin_starts, bin_starts[-1] + bin_width)
    idx = np.digitize(t_proxy_s, edges) - 1
    resampled = np.full(len(bin_starts), np.nan)
    for i in range(len(bin_starts)):
        m = idx == i
        if m.any():
            v = proxy_vals[m]
            v = v[np.isfinite(v)]
            if v.size > 0:
                resampled[i] = float(np.mean(v))
    return resampled


def process_one_dataset(mat_file: str, bin_s: float) -> list[dict]:
    """Load spikes + proxy, compute FR, correlate. Returns list of result dicts."""
    patient, period = dataset_labels(mat_file)
    print(f"  {patient} • {period} …", end=" ", flush=True)

    mat = spio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    spike_struct = mat["spikeTime"]
    record_len_s = compute_recording_duration_s(spike_struct)

    # Build cluster stats (reads cached CSV if available, otherwise computes)
    # Use a temp directory that won't collide; we only need STATS for filtering.
    tmp_dir = Path("_proxy_fr_tmp") / patient / period.replace(" ", "")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    STATS_indiv, STATS_POOLED = build_cache(
        mat_file=mat_file,
        cluster_stats_csv=tmp_dir / "cluster_stats.csv",
        pooled_cluster_stats_csv=tmp_dir / "pooled_cluster_stats.csv",
        base_thr=0,
        record_len_s=record_len_s,
        adaptie_thr_toggle=alpha_meanisi_toggle,
        pooling_toggle=pooling_toggle,
    )
    STATS = STATS_POOLED if pooling_toggle else STATS_indiv
    allowed_clusters = set(STATS.index)

    spike_struct_L, spike_struct_R = split_spike_struct_by_side(spike_struct)

    stride_s = max(bin_s * (1.0 - OVERLAP_FRACTION), 1e-9)

    # Proxy paths
    proxy_root = Path(PROXY_ROOT or DEFAULT_PROXY_ROOT).resolve()

    results: list[dict] = []

    for side_tag, struct_side in [("left", spike_struct_L), ("right", spike_struct_R)]:
        px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side_tag)
        if px_path is None or not Path(px_path).exists():
            continue

        # Population FR for this side
        fr_df = population_firing_rate_binned(
            struct_side, STATS, record_len_s,
            bin_s=bin_s, stride_s=stride_s,
            allowed_clusters=allowed_clusters,
            normalize_by_units=True,
        )
        if fr_df is None or len(fr_df) == 0:
            continue
        fr_t = fr_df["Window_start_s"].to_numpy(float)
        fr_y = fr_df["FR_Hz"].to_numpy(float)

        # Load proxy with region columns
        t_ms_px, hemi_px, region_px = load_proxy_excel_with_region_columns(px_path)
        t_s_px = t_ms_px.astype(float) / 1000.0

        # --- Hemi-level correlation ---
        hemi_resampled = _resample_to_bins(t_s_px, hemi_px, fr_t, bin_s)
        valid = np.isfinite(hemi_resampled) & np.isfinite(fr_y)
        hemi_corr = float(np.corrcoef(fr_y[valid], hemi_resampled[valid])[0, 1]) if valid.sum() > 2 else np.nan
        results.append({"patient": patient, "period": period, "side": side_tag,
                        "level": "hemi", "region": "all", "correlation": hemi_corr})

        # --- Region-level correlations ---
        for reg_name, reg_px_vals in region_px.items():
            reg_resampled = _resample_to_bins(t_s_px, reg_px_vals, fr_t, bin_s)
            reg_clusters = {(e, c) for (e, c) in allowed_clusters if infer_region(e) == reg_name}
            if not reg_clusters:
                continue
            reg_fr = population_firing_rate_binned(
                struct_side, STATS, record_len_s,
                bin_s=bin_s, stride_s=stride_s,
                allowed_clusters=reg_clusters,
                normalize_by_units=True,
            )
            if reg_fr is None or len(reg_fr) == 0:
                continue
            reg_fr_y = reg_fr["FR_Hz"].to_numpy(float)
            valid = np.isfinite(reg_resampled) & np.isfinite(reg_fr_y)
            reg_corr = float(np.corrcoef(reg_fr_y[valid], reg_resampled[valid])[0, 1]) if valid.sum() > 2 else np.nan
            results.append({"patient": patient, "period": period, "side": side_tag,
                            "level": "region", "region": reg_name, "correlation": reg_corr})

    n = len(results)
    print(f"{n} rows")
    return results


def plot_side(df_all: pd.DataFrame, side: str, out_path: Path, use_sem: bool = True):
    """Bar chart for one side: Region↔Region and Hemi↔Hemi per patient."""
    df = df_all[df_all["side"] == side].copy()
    if df.empty:
        print(f"  No data for side={side}, skipping.")
        return

    hemi = df[df["level"] == "hemi"].groupby("patient")["correlation"]
    hemi_mean = hemi.mean()
    hemi_err = hemi.sem() if use_sem else hemi.std()

    region = df[df["level"] == "region"].groupby("patient")["correlation"]
    region_mean = region.mean()
    region_err = region.sem() if use_sem else region.std()

    patients = sorted(
        set(hemi_mean.index) | set(region_mean.index),
        key=lambda p: (p.replace("s", "").zfill(4), p),
    )
    if not patients:
        return

    x = np.arange(len(patients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(patients) * 0.9), 5))

    r_y = [float(region_mean.get(p, 0)) for p in patients]
    r_e = [float(region_err.get(p, 0)) if pd.notna(region_err.get(p)) else 0 for p in patients]
    ax.bar(x - width / 2, r_y, width, yerr=r_e, capsize=3,
           label="Region↔Region", color="tab:blue")

    h_y = [float(hemi_mean.get(p, 0)) for p in patients]
    h_e = [float(hemi_err.get(p, 0)) if pd.notna(hemi_err.get(p)) else 0 for p in patients]
    ax.bar(x + width / 2, h_y, width, yerr=h_e, capsize=3,
           label="Hemi↔Hemi", color="tab:orange")

    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Proxy vs firing rate correlation (by subject) — {side.capitalize()}")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone: correlate proxy with firing rate (no burst detection needed)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Directory to save plots and CSV (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--use-std", action="store_true",
                        help="Use standard deviation instead of SEM for error bars.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SpikeTime_Mat_File:
        print("No datasets in SpikeTime_Mat_File (config.py). Nothing to do.")
        sys.exit(0)

    bin_s_list = coactivity_bins_s if isinstance(coactivity_bins_s, (list, tuple)) else [coactivity_bins_s]
    bin_s = bin_s_list[0]

    print(f"Processing {len(SpikeTime_Mat_File)} dataset(s), bin={bin_s}s …\n")

    all_rows: list[dict] = []
    for mat_file in SpikeTime_Mat_File:
        try:
            rows = process_one_dataset(mat_file, bin_s)
            all_rows.extend(rows)
        except Exception as exc:
            patient, period = dataset_labels(mat_file)
            print(f"  {patient} • {period}: ERROR — {exc}")

    if not all_rows:
        print("\nNo results collected. Check that proxy Excel files exist.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "proxy_vs_fr_all.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows → {csv_path}")

    for side in ("left", "right"):
        out_path = out_dir / f"proxy_vs_fr_{side}.png"
        plot_side(df, side, out_path, use_sem=not args.use_std)


if __name__ == "__main__":
    main()
