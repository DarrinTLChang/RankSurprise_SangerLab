#!/usr/bin/env python3
"""
Per-channel ROC comparison: for each proxy "channel" (hemi + each region column),
compute ROC vs network burst indicator and report AUC. Use this to identify
channels that track network burst well (high AUC) vs poorly (low AUC) for
channel reduction / real-time proxy selection.

Uses the same ROC definition as proxy_detection_demo: percentile thresholds,
TPR = fraction of bursts with ≥1 crossing, FPR = fraction of negative windows
with ≥1 crossing.

Usage:
  python channel_burst_roc.py
  python channel_burst_roc.py --burst-root /path --run-tag "separateGPi__..."
  python channel_burst_roc.py --out-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import BURST_ROOT, PROXY_ANALYSIS_ROOT, PROXY_ANALYSIS_RUN_TAG, SpikeTime_Mat_File
from pipeline.utils import dataset_labels
from compare_proxy_vs_network import load_all_proxy_columns, derive_proxy_xlsx_path, DEFAULT_PROXY_ROOT
from proxy_detection_demo import (
    _discover_patient_periods,
    _allowed_patient_periods,
    _load_bursts,
    _roc_tpr_fpr_one_recording,
    TOLERANCE_MS,
    SCALE_TO_P,
)

DEFAULT_OUT_DIR = PROXY_ANALYSIS_ROOT / "channel_burst_roc"


def _auc_from_roc(tprs: np.ndarray, fprs: np.ndarray) -> float:
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def run_all(
    burst_root: Path,
    run_tag: str,
    proxy_root: Path,
    tolerance_ms: float,
    out_dir: Path,
) -> None:
    allowed = _allowed_patient_periods() if SpikeTime_Mat_File else None
    discoveries = _discover_patient_periods(burst_root, run_tag, allowed=allowed)
    if not discoveries:
        print(f"No data found under {burst_root} with run_tag:\n  {run_tag}")
        return

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proxy_root = proxy_root.resolve()

    rows: list[dict] = []
    for patient, period, csv_L, csv_R in discoveries:
        for side, csv_path in [("left", csv_L), ("right", csv_R)]:
            bursts_df = _load_bursts(csv_path)
            if bursts_df.empty:
                continue
            px_path = derive_proxy_xlsx_path(proxy_root, patient, period, side=side)
            if px_path is None or not px_path.exists():
                continue
            try:
                t_ms, channels = load_all_proxy_columns(px_path)
            except Exception as e:
                print(f"  Skip {patient} {period} {side}: {e}")
                continue
            t_ms = t_ms.astype(float)
            if t_ms.size == 0 or not channels:
                continue

            for ch_name, ch_vals in channels.items():
                if ch_vals.shape[0] != t_ms.shape[0]:
                    continue
                proxy_p = ch_vals.astype(float) * SCALE_TO_P
                roc = _roc_tpr_fpr_one_recording(t_ms, proxy_p, bursts_df, tolerance_ms)
                if roc is None:
                    auc = np.nan
                else:
                    tprs, fprs, _, _ = roc
                    auc = _auc_from_roc(tprs, fprs)
                rows.append({
                    "patient": patient,
                    "period": period,
                    "side": side,
                    "channel": ch_name,
                    "auc": auc,
                    "n_bursts": len(bursts_df),
                })

    if not rows:
        print("No per-channel ROC results.")
        return

    df = pd.DataFrame(rows)
    csv_path = out_dir / "channel_burst_auc.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot: AUC by channel (boxplot across recordings)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    channels_ordered = sorted(df["channel"].unique())
    data_by_ch = [df.loc[df["channel"] == ch, "auc"].dropna().values for ch in channels_ordered]
    if not any(len(d) > 0 for d in data_by_ch):
        return

    fig, ax = plt.subplots(figsize=(max(6, len(channels_ordered) * 0.6), 5))
    bp = ax.boxplot(
        data_by_ch,
        labels=channels_ordered,
        patch_artist=True,
        showfliers=True,
    )
    for box in bp["boxes"]:
        box.set_facecolor("#4878CF")
        box.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_ylabel("AUC (vs network burst)")
    ax.set_xlabel("Channel")
    ax.set_title("Per-channel ROC AUC (higher = better alignment with network burst)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "channel_burst_auc_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_dir / 'channel_burst_auc_boxplot.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-channel ROC AUC vs network burst (for channel reduction / good vs bad channel selection)."
    )
    parser.add_argument("--burst-root", type=Path, default=BURST_ROOT)
    parser.add_argument("--run-tag", type=str, default=PROXY_ANALYSIS_RUN_TAG)
    parser.add_argument("--proxy-root", type=Path, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    proxy_root = Path(args.proxy_root or DEFAULT_PROXY_ROOT).resolve()
    print(f"Burst root : {args.burst_root}")
    print(f"Proxy root: {proxy_root}")
    print(f"Run tag   : {args.run_tag}")
    print(f"Output    : {args.out_dir}\n")

    run_all(
        args.burst_root,
        args.run_tag,
        proxy_root,
        args.tolerance_ms,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
