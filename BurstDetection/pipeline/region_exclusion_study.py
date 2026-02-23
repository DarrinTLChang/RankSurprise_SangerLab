# region_exclusion_study.py
"""
Region exclusion study: compare network burst spans when excluding K regions
(leave-K-out, LKO) to ground-truth spans (all regions).

Reads existing burst CSVs from a run directory; does not re-run full detection.
Uses overlap (IoU) between predicted and GT spans as the comparison metric.
"""

from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MIN_SPIKES_IN_BURST,
    MIN_UNIQUE_CHANNELS_NETWORK,
    RS_Limit_network,
    RS_Percentile_Limit_network,
    RS_alpha_network,
)
from pipeline.detection import RS_detect_burst, compute_spans_and_bars
from pipeline.utils import infer_region


def _load_csv_safe(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    return pd.read_csv(path)


def _burst_in_region_windows(onset_ms: float, region_windows: list[tuple[float, float]]) -> bool:
    for w0, w1 in region_windows:
        if w0 <= onset_ms <= w1:
            return True
    return False


def load_gt_and_bursts(
    run_dir: Path,
    *,
    infer_region_fn=None,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """
    Load ground-truth network spans and the burst lists used for network detection.

    run_dir: directory containing unit_bursts_RS_*.csv, region_bursts_RS_*.csv,
             network_bursts_RS_*.csv (same path passed to rs_burst_detection).

    Returns:
        gt_spans_left, gt_spans_right: list of (span_start_ms, span_end_ms)
        bursts_left, bursts_right: list of burst dicts with Start_ms, End_ms, Electrode, Cluster, Region
        regions: sorted list of unique region names present in the data
    """
    if infer_region_fn is None:
        infer_region_fn = infer_region

    def _load_gt(side: str) -> list[tuple[float, float]]:
        path = run_dir / f"network_bursts_RS_{side}.csv"
        df = _load_csv_safe(path, required=True)
        if df is None or len(df) == 0:
            return []
        spans = []
        for _, row in df.iterrows():
            s = pd.to_numeric(row.get("span_start_ms"), errors="coerce")
            e = pd.to_numeric(row.get("span_end_ms"), errors="coerce")
            if np.isfinite(s) and np.isfinite(e) and e > s:
                spans.append((float(s), float(e)))
        return spans

    def _load_unit_bursts(side: str) -> pd.DataFrame:
        path = run_dir / f"unit_bursts_RS_{side}.csv"
        df = _load_csv_safe(path, required=True)
        return df

    def _load_region_windows(side: str) -> dict[str, list[tuple[float, float]]]:
        path = run_dir / f"region_bursts_RS_{side}.csv"
        df = _load_csv_safe(path, required=False)
        if df is None or len(df) == 0:
            return {}
        out: dict[str, list[tuple[float, float]]] = {}
        for _, row in df.iterrows():
            reg = str(row.get("Region", ""))
            if reg not in out:
                out[reg] = []
            t0 = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            t1 = pd.to_numeric(row.get("onset_end_ms"), errors="coerce")
            if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
                out[reg].append((float(t0), float(t1)))
        return out

    def _build_burst_list(side: str) -> tuple[list[dict[str, Any]], set[str]]:
        units = _load_unit_bursts(side)
        region_windows = _load_region_windows(side)
        regions_in_side: set[str] = set()

        burst_dicts: list[dict[str, Any]] = []
        for _, row in units.iterrows():
            elec = str(row.get("Electrode", ""))
            cl = int(pd.to_numeric(row.get("Cluster", -1), errors="coerce"))
            onset_ms = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            onset_end_ms = pd.to_numeric(row.get("onset_end_ms"), errors="coerce")
            if not np.isfinite(onset_ms) or not np.isfinite(onset_end_ms):
                continue
            reg = infer_region_fn(elec)
            regions_in_side.add(reg)

            # If we have region-level detection, only include bursts that fall in a region window
            if region_windows:
                if reg not in region_windows or not _burst_in_region_windows(onset_ms, region_windows[reg]):
                    continue

            burst_dicts.append({
                "Electrode": elec,
                "Cluster": cl,
                "Start_ms": float(onset_ms),
                "End_ms": float(onset_end_ms),
            })
            burst_dicts[-1]["Region"] = reg

        return burst_dicts, regions_in_side

    gt_left = _load_gt("left")
    gt_right = _load_gt("right")
    bursts_left, regs_l = _build_burst_list("left")
    bursts_right, regs_r = _build_burst_list("right")
    regions = sorted(regs_l | regs_r)

    return gt_left, gt_right, bursts_left, bursts_right, regions


def load_gt_spans(run_dir: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Load ground-truth network spans (span_start_ms, span_end_ms) per side from network_bursts CSVs."""
    run_dir = Path(run_dir)
    gt_left, gt_right = [], []

    for side, out in [("left", gt_left), ("right", gt_right)]:
        path = run_dir / f"network_bursts_RS_{side}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            s = pd.to_numeric(row.get("span_start_ms"), errors="coerce")
            e = pd.to_numeric(row.get("span_end_ms"), errors="coerce")
            if np.isfinite(s) and np.isfinite(e) and e > s:
                out.append((float(s), float(e)))

    return gt_left, gt_right


def run_network_with_excluded_regions(
    bursts: list[dict[str, Any]],
    excluded_regions: set[str],
    *,
    limit=RS_Limit_network,
    RSalpha=RS_alpha_network,
    min_spikes=MIN_SPIKES_IN_BURST,
    RS_Percentile_Limit=RS_Percentile_Limit_network,
    min_unique_channels=MIN_UNIQUE_CHANNELS_NETWORK,
) -> tuple[list[dict[str, Any]], list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Run network-level RS detection and span computation using only bursts whose
    Region is not in excluded_regions.

    Returns:
        network_bursts_rows: list of dicts (same format as detection pipeline CSV)
        pred_spans: list of (span_start_ms, span_end_ms) for IoU
        bars_s: merged (start_s, end_s) bars for plotting
    """
    filtered = [b for b in bursts if b.get("Region") not in excluded_regions]
    if len(filtered) == 0:
        return [], [], []

    onsets_ms = np.sort(np.asarray([float(b["Start_ms"]) for b in filtered], dtype=float))
    if onsets_ms.size < min_spikes:
        return [], [], []

    start_idx, length, RSvals = RS_detect_burst(
        onsets_ms,
        limit=limit,
        RSalpha=RSalpha,
        min_spikes=min_spikes,
        RS_Percentile_Limit=RS_Percentile_Limit,
    )

    windows: list[tuple[float, float]] = []
    rs_vals_for_windows: list[float] = []
    num_events_for_windows: list[int] = []
    for s, L, rs in zip(start_idx, length, RSvals):
        s, L = int(s), int(L)
        e = s + L - 1
        if e < s or e >= onsets_ms.size:
            continue
        t0 = float(onsets_ms[s])
        t1 = float(onsets_ms[e])
        if t1 <= t0:
            continue
        windows.append((t0, t1))
        rs_vals_for_windows.append(float(rs))
        num_events_for_windows.append(int(L))

    if not windows:
        return [], [], []

    spans_ms, bars_s = compute_spans_and_bars(
        windows,
        filtered,
        min_unique_channels=min_unique_channels,
        channel_mode="elec_cluster",
    )

    network_bursts_rows: list[dict[str, Any]] = []
    pred_spans: list[tuple[float, float]] = []
    for w, span, rs_val, num_ev in zip(windows, spans_ms, rs_vals_for_windows, num_events_for_windows):
        if not (np.isfinite(span[0]) and np.isfinite(span[1])) or span[1] <= span[0]:
            continue
        span_dur = span[1] - span[0]
        network_bursts_rows.append({
            "onset_start_ms": w[0],
            "onset_end_ms": w[1],
            "onset_duration_ms": float(w[1] - w[0]),
            "span_start_ms": span[0],
            "span_end_ms": span[1],
            "span_duration_ms": span_dur,
            "num_onsets": num_ev,
            "RS": rs_val,
        })
        pred_spans.append((span[0], span[1]))

    return network_bursts_rows, pred_spans, bars_s


def span_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two intervals (ms)."""
    a0, a1 = a
    b0, b1 = b
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def mean_iou_vs_gt(
    gt_spans: list[tuple[float, float]],
    pred_spans: list[tuple[float, float]],
) -> tuple[float, int]:
    """
    For each GT span, take the max IoU with any predicted span; mean IoU over GT.
    N_gt_matched: number of GT spans that have IoU > 0 with at least one pred.
    (One pred can overlap multiple GTs, so N_gt_matched can exceed N_pred_spans.)
    """
    if not gt_spans:
        return float("nan"), 0
    ious = []
    for g in gt_spans:
        best = 0.0
        for p in pred_spans:
            best = max(best, span_iou(g, p))
        ious.append(best)
    n_matched = sum(1 for x in ious if x > 0)
    return float(np.mean(ious)), n_matched


def iter_permutation_dirs(
    run_dir: Path,
    regions: list[str],
    *,
    max_excluded: int | None = None,
    out_subdir: str = "region_exclusion_study",
):
    """
    Yield (included_regions, perm_dir) in the same order as run_region_exclusion_study creates folders.
    included_regions is a sorted list of region names; perm_dir is the path to that permutation's folder.
    """
    run_dir = Path(run_dir)
    N = len(regions)
    if N == 0:
        return
    K_max = min(max_excluded if max_excluded is not None else N - 1, N - 1)
    if K_max < 1:
        K_max = 1
    out_dir = run_dir / out_subdir
    for K in range(1, K_max + 1):
        for excluded in combinations(regions, K):
            included = sorted(set(regions) - set(excluded))
            perm_dir_name = "_".join(included).replace(" ", "_").replace("/", "-")
            perm_dir = out_dir / perm_dir_name
            yield included, perm_dir


def get_permutation_data_for_plotting(
    all_bursts_L: list[dict[str, Any]],
    all_bursts_R: list[dict[str, Any]],
    region_bars_by_region_L: dict[str, list[tuple[float, float]]],
    region_bars_by_region_R: dict[str, list[tuple[float, float]]],
    included_regions: set[str] | list[str],
    *,
    infer_region_fn=None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[tuple[float, float]]],
    dict[str, list[tuple[float, float]]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """
    Filter bursts and region bars to included regions only; run network detection on filtered bursts.
    Returns (filtered_bursts_L, filtered_bursts_R, region_bars_L, region_bars_R, network_bars_L, network_bars_R)
    for use with make_sangerlab_presentation_figure.
    """
    if infer_region_fn is None:
        infer_region_fn = infer_region
    inc = set(included_regions)

    def with_region(b: dict) -> dict:
        return dict(b, Region=infer_region_fn(str(b.get("Electrode", ""))))

    filtered_L = [with_region(b) for b in all_bursts_L if infer_region_fn(str(b.get("Electrode", ""))) in inc]
    filtered_R = [with_region(b) for b in all_bursts_R if infer_region_fn(str(b.get("Electrode", ""))) in inc]

    _, _, bars_L = run_network_with_excluded_regions(filtered_L, set())
    _, _, bars_R = run_network_with_excluded_regions(filtered_R, set())

    reg_bars_L = {k: v for k, v in (region_bars_by_region_L or {}).items() if k in inc}
    reg_bars_R = {k: v for k, v in (region_bars_by_region_R or {}).items() if k in inc}

    return filtered_L, filtered_R, reg_bars_L, reg_bars_R, bars_L, bars_R


def run_region_exclusion_study(
    run_dir: Path,
    *,
    max_excluded: int | None = None,
    infer_region_fn=None,
    out_subdir: str = "region_exclusion_study",
) -> pd.DataFrame:
    """
    Run leave-K-out (LKO) for K = 1, 2, ..., min(max_excluded, N-1) where N = number of regions.
    Compare predicted spans to GT using mean IoU. Write results to run_dir / out_subdir / results.csv.

    run_dir: directory containing the burst CSVs (same as output_RS_burst_dir in main).
    max_excluded: max number of regions to exclude at once (default: N-1 so at least one region remains).
    """
    if infer_region_fn is None:
        infer_region_fn = infer_region

    run_dir = Path(run_dir)
    gt_left, gt_right, bursts_left, bursts_right, regions = load_gt_and_bursts(
        run_dir, infer_region_fn=infer_region_fn
    )

    N = len(regions)
    if N == 0:
        raise ValueError("No regions found in burst data")
    K_max = min(max_excluded if max_excluded is not None else N - 1, N - 1)
    if K_max < 1:
        K_max = 1

    rows: list[dict[str, Any]] = []
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unit burst CSVs are copied; region_bursts are filtered per permutation
    unit_copy_filenames = [
        "unit_bursts_RS_left.csv",
        "unit_bursts_RS_right.csv",
    ]

    for K in range(1, K_max + 1):
        for excluded in combinations(regions, K):
            excluded_set = set(excluded)
            included = sorted(set(regions) - excluded_set)
            included_set = set(included)
            included_str = "|".join(included)
            # Folder name: regions included (path-safe)
            perm_dir_name = "_".join(included).replace(" ", "_").replace("/", "-")
            perm_dir = out_dir / perm_dir_name
            perm_dir.mkdir(parents=True, exist_ok=True)

            # Copy unit burst CSVs (unchanged across permutations)
            for fname in unit_copy_filenames:
                src = run_dir / fname
                if src.exists():
                    shutil.copy2(src, perm_dir / fname)

            # Write region_bursts filtered to included regions only (per side)
            for side in ("left", "right"):
                path = run_dir / f"region_bursts_RS_{side}.csv"
                if path.exists():
                    df_reg = pd.read_csv(path)
                    if "Region" in df_reg.columns:
                        df_reg = df_reg[df_reg["Region"].astype(str).isin(included_set)]
                    df_reg.to_csv(perm_dir / f"region_bursts_RS_{side}.csv", index=False)

            for side, gt_spans, bursts in [
                ("left", gt_left, bursts_left),
                ("right", gt_right, bursts_right),
            ]:
                network_rows, pred_spans, _ = run_network_with_excluded_regions(bursts, excluded_set)
                pd.DataFrame(network_rows).to_csv(
                    perm_dir / f"network_bursts_RS_{side}.csv", index=False
                )
                mean_iou, n_matched = mean_iou_vs_gt(gt_spans, pred_spans)
                rows.append({
                    "Side": side,
                    "K_excluded": K,
                    "Regions_included": included_str,
                    "N_gt_spans": len(gt_spans),
                    "N_pred_spans": len(pred_spans),
                    "Mean_IoU": mean_iou,
                    "N_gt_matched": n_matched,
                })

    df = pd.DataFrame(rows)
    out_path = out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"• Region exclusion study: wrote {len(df)} rows → {out_path}")
    print(f"• Per-permutation output folders under {out_dir} (unit/region copied, network CSVs generated)")
    return df
