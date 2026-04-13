#!/usr/bin/env python3
"""
Compare fast hemispheric proxy activity to offline RS network bursts.

HOW TO CALL
-----------
  # Loop over all patients (uses paths and RUN_TAGS at top of this file):
  python compare_proxy_vs_network.py --loop

  # outputs_RS_burst_2 (min burst dur = 0ms), separateGPi run_tag only: set USE_RS_BURST_2 = True in file, then:
  python compare_proxy_vs_network.py --loop
  # Or without changing the file:
  python compare_proxy_vs_network.py --loop --rs-burst-root "/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst_2" --run-tag "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__region__network" --out-dir "/Volumes/D_Drive/SangerLabBursts/outputs_fast_proxy_RS_burst_2"

  # Single run (one proxy Excel + one network CSV):
  python compare_proxy_vs_network.py --proxy-xlsx <path> --network-csv <path>

  # Override paths or run_tags from CLI:
  python compare_proxy_vs_network.py --loop --proxy-root /path/to/proxies --rs-burst-root /path/to/RS_burst
  python compare_proxy_vs_network.py --loop --run-tag "separateGPi__..." "another_tag"

Paths and run_tags are defined at the top of this file (edit like config.py).
Output (--loop) under .../outputs_fast_proxy/patient/PeriodN/run_tag/:
  - hemi_proxy to hemi_burst: proxy_vs_network_results_left.csv, proxy_vs_network_results_right.csv
  - region_proxy_to_hemi_burst/results_left.csv, results_right.csv (one row per region)
  - region_proxy_to_region_burst/results_left.csv, results_right.csv (one row per region)

Two correlation metrics when --fixed-onset-window-ms > 0:
  - Duration-based: written to --out-dir (e.g. outputs_fast_proxy); mask = [burst_start, burst_end].
  - Onset-fixed: written to a separate folder with duration in the name, e.g. outputs_fast_proxy_onset_fixed_300ms
    (same patient/PeriodN/run_tag layout; mask = [onset, onset+fixed_window_ms]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# New output layout config
from config import BURST_ROOT, PROXY_ANALYSIS_ROOT
from pipeline.burst_paths import BURST_TIMINGS_SUBDIR, burst_network_csv_path, burst_region_csv_path

# ============================================================
# PATHS  (edit these like the dataset list in config.py)
# ============================================================
DEFAULT_PROXY_ROOT = Path("/Volumes/D_Drive/rasters_all_with_fast_proxies(neo)")
DEFAULT_RS_BURST_ROOT = Path(BURST_ROOT)
DEFAULT_OUT_FAST_PROXY = Path(PROXY_ANALYSIS_ROOT) / "outputs_fast_proxy"

# outputs_RS_burst_2: same layout but min burst dur = 0ms; rankSurprise has separateGPi__... and SNR=1.2__... (we use only separateGPi)
RS_BURST_2_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst_2")
OUT_FAST_PROXY_RS_BURST_2 = Path("/Volumes/D_Drive/SangerLabBursts/outputs_fast_proxy_RS_burst_2")
RUN_TAG_RS_BURST_2 = "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__region__network"
# Set to True to use RS_burst_2 root and separateGPi-only run_tag (and write to outputs_fast_proxy_RS_burst_2)
USE_RS_BURST_2 = False

# Onset-only metric: ignore burst end; use fixed window after each onset (matches "detect start, then fixed-duration stimulation").
# Correlation = proxy vs mask where mask=1 in [onset, onset+FIXED_ONSET_WINDOW_MS] for each onset.
FIXED_ONSET_WINDOW_MS = 400.0  # ms after each burst onset to count as "in burst"

# ============================================================
# RUN TAGS  (comment / uncomment to select which run_tags to process)
# ============================================================
RUN_TAGS = [
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=0ms__minCh=1__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=50ms__minCh=1__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=3%__limReg=75__aNet=2%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=8%__limClust=75__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=4%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=5__region__network",

    # "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=3%__limReg=75__aNet=2%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
]


def load_bilateral_proxy_csv(
    path: Path,
    *,
    time_col: str,
    left_col: str,
    right_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single CSV with time (seconds) and separate columns for left- and right-hemisphere proxy.
    Returns (t_s, y_left, y_right) as aligned float arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path}")

    df = pd.read_csv(path)
    for c in (time_col, left_col, right_col):
        if c not in df.columns:
            raise ValueError(
                f"Missing column {c!r} in {path}. Available columns: {list(df.columns)}"
            )

    t_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    y_l = pd.to_numeric(df[left_col], errors="coerce").to_numpy(dtype=float)
    y_r = pd.to_numeric(df[right_col], errors="coerce").to_numpy(dtype=float)

    n = t_s.shape[0]
    if y_l.shape[0] != n or y_r.shape[0] != n:
        raise ValueError(
            f"Length mismatch in {path}: time={n}, {left_col}={y_l.shape[0]}, {right_col}={y_r.shape[0]}"
        )

    return t_s, y_l, y_r


def load_hemi_proxy_from_excel(path: Path, sheet_name: str = "proxies") -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load time axis (ms) and a hemispheric proxy column from the given Excel file.
    Returns (t_ms, hemi_proxy, hemi_col_name).
    """
    return load_hemi_proxy_from_table(
        path,
        sheet_name=sheet_name,
        time_col="t_center_s",
        hemi_col="hemi_proxy",
    )


def _read_proxy_table(path: Path, sheet_name: str = "proxies") -> pd.DataFrame:
    """
    Read a proxy table from .xlsx (sheet) or .csv.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported proxy file type: {path} (expected .csv or .xlsx)")


def load_hemi_proxy_from_table(
    path: Path,
    *,
    sheet_name: str = "proxies",
    time_col: str = "t_center_s",
    hemi_col: str = "hemi_proxy",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load (t_ms, hemi_proxy, hemi_col_name) from either:
    - Excel (.xlsx/.xls): reads `sheet_name`
    - CSV (.csv): reads the whole file

    `time_col` is in seconds; it is converted to ms.
    """
    path = Path(path)
    df = _read_proxy_table(path, sheet_name=sheet_name)

    # Friendly fallback for CSVs that use time_s instead of t_center_s
    if time_col not in df.columns and time_col == "t_center_s" and "time_s" in df.columns:
        time_col = "time_s"

    if time_col not in df.columns:
        raise ValueError(
            f"Could not find time column '{time_col}' in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    t_ms = (pd.to_numeric(df[time_col], errors="coerce").astype(float) * 1000.0).to_numpy()

    if hemi_col not in df.columns:
        raise ValueError(
            f"Could not find proxy column '{hemi_col}' in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    hemi_proxy = pd.to_numeric(df[hemi_col], errors="coerce").to_numpy(dtype=float)

    if t_ms.shape[0] != hemi_proxy.shape[0]:
        raise ValueError(
            f"Time axis and proxy length mismatch in {path}: "
            f"{t_ms.shape[0]} vs {hemi_proxy.shape[0]}"
        )

    return t_ms, hemi_proxy, hemi_col


def load_proxy_excel_with_region_columns(
    path: Path, sheet_name: str = "proxies"
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Load time axis, hemi_proxy, and all region proxy columns (proxy_GPi1, proxy_VOSTN, etc.).
    Returns (t_ms, hemi_proxy, region_proxies) where region_proxies keys are region names
    without the "proxy_" prefix (e.g. "GPi1", "VOSTN").
    """
    path = Path(path)
    df = _read_proxy_table(path, sheet_name=sheet_name)
    time_col = "t_center_s"
    if time_col not in df.columns and "time_s" in df.columns:
        time_col = "time_s"
    if time_col not in df.columns:
        raise ValueError(f"Missing {time_col} in {path}. Columns: {list(df.columns)}")
    t_ms = (pd.to_numeric(df[time_col], errors="coerce").astype(float) * 1000.0).to_numpy()
    hemi_proxy = (
        pd.to_numeric(df["hemi_proxy"], errors="coerce").to_numpy(dtype=float)
        if "hemi_proxy" in df.columns
        else np.full_like(t_ms, np.nan)
    )
    region_proxies: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col.startswith("proxy_") and col != "proxy_":
            region_name = col[len("proxy_"):].strip()
            region_proxies[region_name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return t_ms, hemi_proxy, region_proxies


def load_all_proxy_columns(
    path: Path, sheet_name: str = "proxies"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load time axis and every proxy column as a separate "channel": hemi_proxy (key "hemi")
    plus all proxy_* columns (key = region name without prefix).
    Returns (t_ms, channels) where channels[channel_name] is the time series.
    """
    path = Path(path)
    df = _read_proxy_table(path, sheet_name=sheet_name)
    time_col = "t_center_s"
    if time_col not in df.columns and "time_s" in df.columns:
        time_col = "time_s"
    if time_col not in df.columns:
        raise ValueError(f"Missing {time_col} in {path}. Columns: {list(df.columns)}")
    t_ms = (pd.to_numeric(df[time_col], errors="coerce").astype(float) * 1000.0).to_numpy()
    channels: dict[str, np.ndarray] = {}
    if "hemi_proxy" in df.columns:
        channels["hemi"] = pd.to_numeric(df["hemi_proxy"], errors="coerce").to_numpy(dtype=float)
    for col in df.columns:
        if col.startswith("proxy_") and col != "proxy_":
            name = col[len("proxy_"):].strip()
            channels[name] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return t_ms, channels


def load_region_bursts_by_region(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Load ``burst_timings/region_bursts_L.csv`` (or R); return dict region -> list of (start_ms, end_ms).
    Uses burst_start_ms, burst_end_ms when present; else onset/span columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return {}
    if df.empty or "Region" not in df.columns:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        reg = str(row["Region"]).strip()
        if "burst_start_ms" in df.columns and "burst_end_ms" in df.columns:
            t0 = pd.to_numeric(row.get("burst_start_ms"), errors="coerce")
            t1 = pd.to_numeric(row.get("burst_end_ms"), errors="coerce")
        else:
            t0 = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            if not np.isfinite(t0):
                continue
            t0 = float(t0)
            span_dur = pd.to_numeric(row.get("span_duration_ms"), errors="coerce")
            t1 = t0 + float(span_dur) if np.isfinite(span_dur) and span_dur > 0 else pd.to_numeric(row.get("onset_end_ms", row.get("span_end_ms", t0)), errors="coerce")
        if not (np.isfinite(t0) and np.isfinite(t1) and float(t1) > float(t0)):
            continue
        pair = (float(t0), float(t1))
        lst = out.setdefault(reg, [])
        if pair not in lst:
            lst.append(pair)
    for reg in out:
        out[reg] = sorted(out[reg], key=lambda p: p[0])
    return out


def load_network_bursts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load network burst start/end times (ms) from ``burst_timings/network_bursts_L.csv`` (or R, or combined ``network_LR.csv``).
    Supports both schemas:
    - burst_start_ms, burst_end_ms (new)
    - onset_start_ms with onset_end_ms or span_duration_ms (legacy / copied runs)
    Returns (starts_ms, ends_ms). For onset-fixed metrics we only need starts; ends can be dummy if missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    starts_list: list[float] = []
    ends_list: list[float] = []

    if "burst_start_ms" in df.columns and "burst_end_ms" in df.columns:
        for _, row in df.iterrows():
            s = pd.to_numeric(row.get("burst_start_ms"), errors="coerce")
            e = pd.to_numeric(row.get("burst_end_ms"), errors="coerce")
            if np.isfinite(s):
                starts_list.append(float(s))
                ends_list.append(float(e) if np.isfinite(e) and e > s else float(s) + 1.0)
    elif "onset_start_ms" in df.columns:
        for _, row in df.iterrows():
            s = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            if not np.isfinite(s):
                continue
            s = float(s)
            e = pd.to_numeric(row.get("onset_end_ms"), errors="coerce")
            if np.isfinite(e) and e > s:
                ends_list.append(float(e))
            else:
                span = pd.to_numeric(row.get("span_duration_ms"), errors="coerce")
                ends_list.append(s + float(span) if np.isfinite(span) and span > 0 else s + 1.0)
            starts_list.append(s)
    else:
        raise ValueError(
            f"{path} must have burst_start_ms and burst_end_ms, or onset_start_ms. "
            f"Available columns: {list(df.columns)}"
        )

    if not starts_list:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.array(starts_list, dtype=float), np.array(ends_list, dtype=float)


def build_network_mask(t_ms: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask over t_ms indicating time bins that fall inside
    any network burst interval [start, end].
    """
    mask = np.zeros_like(t_ms, dtype=bool)
    for s, e in zip(starts, ends):
        if not np.isfinite(s) or not np.isfinite(e) or e <= s:
            continue
        mask |= (t_ms >= s) & (t_ms <= e)
    return mask


def build_burst_mask_from_windows(t_ms: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    """Build boolean mask over t_ms for intervals in windows [(start_ms, end_ms), ...]."""
    mask = np.zeros_like(t_ms, dtype=bool)
    for s, e in windows:
        if not np.isfinite(s) or not np.isfinite(e) or e <= s:
            continue
        mask |= (t_ms >= s) & (t_ms <= e)
    return mask


def build_mask_from_onsets_fixed_duration(
    t_ms: np.ndarray, starts_ms: np.ndarray, fixed_window_ms: float
) -> np.ndarray:
    """
    Build mask from onset times only: for each onset, mask=1 in [onset, onset+fixed_window_ms].
    Ignores burst end; use when only onset detection matters (e.g. fixed-duration stimulation).
    """
    mask = np.zeros_like(t_ms, dtype=bool)
    for s in starts_ms:
        if not np.isfinite(s):
            continue
        end = float(s) + float(fixed_window_ms)
        mask |= (t_ms >= s) & (t_ms <= end)
    return mask


def run_comparison(
    proxy_xlsx: Path,
    network_csv: Path,
    sheet_name: str = "proxies",
    fixed_onset_window_ms: float | None = None,
) -> dict | None:
    """
    Run one proxy-vs-network comparison. Returns a dict of metrics, or None if
    comparison could not be done (no bursts, or mask all in/out).

    Two metrics:
    - Duration-based: mask = 1 during [burst_start, burst_end] (pipeline duration).
    - Onset-fixed: mask = 1 during [onset, onset+fixed_onset_window_ms] (onset-only; no burst end).
    """
    t_ms, hemi_proxy, hemi_col = load_hemi_proxy_from_excel(proxy_xlsx, sheet_name=sheet_name)
    starts, ends = load_network_bursts(network_csv)
    if len(starts) == 0:
        return None

    out: dict = {"n_network_bursts": len(starts)}

    # 1) Duration-based mask (original)
    mask = build_network_mask(t_ms, starts, ends)
    n_in = int(mask.sum())
    n_out = int((~mask).sum())
    if n_in > 0 and n_out > 0:
        network_indicator = mask.astype(float)
        corr = float(np.corrcoef(network_indicator, hemi_proxy)[0, 1])
        inside = hemi_proxy[mask]
        outside = hemi_proxy[~mask]
        out["correlation"] = corr
        out["mean_hemi_proxy_inside"] = float(np.nanmean(inside))
        out["mean_hemi_proxy_outside"] = float(np.nanmean(outside))
        out["median_hemi_proxy_inside"] = float(np.nanmedian(inside))
        out["median_hemi_proxy_outside"] = float(np.nanmedian(outside))
        out["n_time_bins_inside"] = n_in
        out["n_time_bins_outside"] = n_out
    else:
        out["correlation"] = np.nan
        out["mean_hemi_proxy_inside"] = out["mean_hemi_proxy_outside"] = np.nan
        out["median_hemi_proxy_inside"] = out["median_hemi_proxy_outside"] = np.nan
        out["n_time_bins_inside"] = n_in
        out["n_time_bins_outside"] = n_out

    # 2) Onset-fixed mask (no burst end; fixed window after each onset)
    if fixed_onset_window_ms is not None and fixed_onset_window_ms > 0:
        mask_onset = build_mask_from_onsets_fixed_duration(t_ms, starts, fixed_onset_window_ms)
        n_in_o = int(mask_onset.sum())
        n_out_o = int((~mask_onset).sum())
        if n_in_o > 0 and n_out_o > 0:
            ind_onset = mask_onset.astype(float)
            corr_onset = float(np.corrcoef(ind_onset, hemi_proxy)[0, 1])
            inside_o = hemi_proxy[mask_onset]
            outside_o = hemi_proxy[~mask_onset]
            out["correlation_onset_fixed"] = corr_onset
            out["mean_hemi_proxy_inside_onset_fixed"] = float(np.nanmean(inside_o))
            out["mean_hemi_proxy_outside_onset_fixed"] = float(np.nanmean(outside_o))
            out["median_hemi_proxy_inside_onset_fixed"] = float(np.nanmedian(inside_o))
            out["median_hemi_proxy_outside_onset_fixed"] = float(np.nanmedian(outside_o))
            out["n_time_bins_inside_onset_fixed"] = n_in_o
            out["n_time_bins_outside_onset_fixed"] = n_out_o
            out["fixed_onset_window_ms"] = fixed_onset_window_ms
        else:
            out["correlation_onset_fixed"] = np.nan
            out["mean_hemi_proxy_inside_onset_fixed"] = out["mean_hemi_proxy_outside_onset_fixed"] = np.nan
            out["median_hemi_proxy_inside_onset_fixed"] = out["median_hemi_proxy_outside_onset_fixed"] = np.nan
            out["n_time_bins_inside_onset_fixed"] = n_in_o
            out["n_time_bins_outside_onset_fixed"] = n_out_o
            out["fixed_onset_window_ms"] = fixed_onset_window_ms

    return out


def run_comparison_region_proxy_vs_hemi_burst(
    proxy_xlsx: Path,
    network_csv: Path,
    sheet_name: str = "proxies",
    fixed_onset_window_ms: float | None = None,
) -> list[dict]:
    """
    For each region proxy column (proxy_GPi1, etc.), correlate with hemisphere network burst mask.
    Also computes onset-fixed correlation when fixed_onset_window_ms is set.
    """
    t_ms, _, region_proxies = load_proxy_excel_with_region_columns(proxy_xlsx, sheet_name=sheet_name)
    starts, ends = load_network_bursts(network_csv)
    if len(starts) == 0:
        return []
    mask = build_network_mask(t_ms, starts, ends)
    n_in = int(mask.sum())
    n_out = int((~mask).sum())
    if n_in == 0 or n_out == 0:
        return []
    network_indicator = mask.astype(float)
    mask_onset = None
    if fixed_onset_window_ms is not None and fixed_onset_window_ms > 0:
        mask_onset = build_mask_from_onsets_fixed_duration(t_ms, starts, fixed_onset_window_ms)
        if mask_onset.sum() == 0 or (~mask_onset).sum() == 0:
            mask_onset = None
    rows: list[dict] = []
    for region, proxy_vals in region_proxies.items():
        if proxy_vals.shape[0] != t_ms.shape[0]:
            continue
        r: dict = {
            "region": region,
            "correlation": float(np.corrcoef(network_indicator, proxy_vals)[0, 1]),
            "mean_proxy_inside": float(np.nanmean(proxy_vals[mask])),
            "mean_proxy_outside": float(np.nanmean(proxy_vals[~mask])),
            "median_proxy_inside": float(np.nanmedian(proxy_vals[mask])),
            "median_proxy_outside": float(np.nanmedian(proxy_vals[~mask])),
            "n_time_bins_inside": n_in,
            "n_time_bins_outside": n_out,
            "n_network_bursts": len(starts),
        }
        if mask_onset is not None:
            r["correlation_onset_fixed"] = float(np.corrcoef(mask_onset.astype(float), proxy_vals)[0, 1])
            r["mean_proxy_inside_onset_fixed"] = float(np.nanmean(proxy_vals[mask_onset]))
            r["mean_proxy_outside_onset_fixed"] = float(np.nanmean(proxy_vals[~mask_onset]))
            r["fixed_onset_window_ms"] = fixed_onset_window_ms
        rows.append(r)
    return rows


def run_comparison_region_proxy_vs_region_burst(
    proxy_xlsx: Path,
    region_bursts_csv: Path,
    sheet_name: str = "proxies",
    fixed_onset_window_ms: float | None = None,
) -> list[dict]:
    """
    For each region that has both a proxy column and region bursts, correlate proxy with that region's burst mask.
    Onset-fixed: use [start, start+fixed_onset_window_ms] per burst instead of [start, end].
    """
    t_ms, _, region_proxies = load_proxy_excel_with_region_columns(proxy_xlsx, sheet_name=sheet_name)
    by_region = load_region_bursts_by_region(region_bursts_csv)
    rows: list[dict] = []
    for region, proxy_vals in region_proxies.items():
        if proxy_vals.shape[0] != t_ms.shape[0]:
            continue
        windows = by_region.get(region, [])
        if not windows:
            continue
        mask = build_burst_mask_from_windows(t_ms, windows)
        n_in = int(mask.sum())
        n_out = int((~mask).sum())
        if n_in == 0 or n_out == 0:
            continue
        burst_indicator = mask.astype(float)
        r: dict = {
            "region": region,
            "correlation": float(np.corrcoef(burst_indicator, proxy_vals)[0, 1]),
            "mean_proxy_inside": float(np.nanmean(proxy_vals[mask])),
            "mean_proxy_outside": float(np.nanmean(proxy_vals[~mask])),
            "median_proxy_inside": float(np.nanmedian(proxy_vals[mask])),
            "median_proxy_outside": float(np.nanmedian(proxy_vals[~mask])),
            "n_time_bins_inside": n_in,
            "n_time_bins_outside": n_out,
            "n_region_bursts": len(windows),
        }
        if fixed_onset_window_ms is not None and fixed_onset_window_ms > 0:
            starts = np.array([w[0] for w in windows])
            mask_onset = build_mask_from_onsets_fixed_duration(t_ms, starts, fixed_onset_window_ms)
            if mask_onset.sum() > 0 and (~mask_onset).sum() > 0:
                r["correlation_onset_fixed"] = float(np.corrcoef(mask_onset.astype(float), proxy_vals)[0, 1])
                r["mean_proxy_inside_onset_fixed"] = float(np.nanmean(proxy_vals[mask_onset]))
                r["mean_proxy_outside_onset_fixed"] = float(np.nanmean(proxy_vals[~mask_onset]))
                r["fixed_onset_window_ms"] = fixed_onset_window_ms
        rows.append(r)
    return rows


def find_run_dirs_with_network_bursts(rs_burst_root: Path) -> list[tuple[Path, str, str, str]]:
    """
    Discover (run_dir, patient, period_label, run_tag) under rs_burst_root where
    ``burst_timings/network_bursts_L.csv`` or ``..._R.csv`` exists.
    period_label is e.g. "Period2"; patient is e.g. "s508"; run_tag is the run_dir name.
    """
    rs_burst_root = Path(rs_burst_root).resolve()
    if not rs_burst_root.is_dir():
        return []
    seen: set[tuple[str, str, str]] = set()
    result: list[tuple[Path, str, str, str]] = []
    for csv_name in ("network_bursts_L.csv", "network_bursts_R.csv"):
        for f in rs_burst_root.rglob(csv_name):
            if f.parent.name != BURST_TIMINGS_SUBDIR:
                continue
            run_dir = f.parent.parent
            # New layout: run_dir = .../patient/PeriodN/run_tag (need 3 levels under root)
            try:
                patient = run_dir.parent.parent.name
                period_label = run_dir.parent.name  # Period1, Period2, ...
            except IndexError:
                continue
            run_tag = run_dir.name
            key = (patient, period_label, run_tag)
            if key not in seen:
                seen.add(key)
                result.append((run_dir, patient, period_label, run_tag))
    return sorted(result, key=lambda x: (x[1], x[2], x[3]))


def derive_proxy_xlsx_path(
    proxy_root: Path,
    patient: str,
    period_label: str,
    side: str,
) -> Path | None:
    """
    Derive proxy Excel path for (patient, period_label, side).
    proxy_root contains folders sXXX_YYYYMM; period folder is period1, period2 (lowercase).
    Filename: "subject s508 • period 2_L_bursts_rates_proxies.xlsx" (space before period number).
    Returns path if the patient folder exists (first sXXX_* match); caller checks file exists.
    """
    proxy_root = Path(proxy_root).resolve()
    # Period2 -> 2, Period10 -> 10
    period_num = period_label.replace("Period", "").strip()
    if not period_num.isdigit():
        return None
    period_num = int(period_num)
    # Patient folder: s508_202201 (any s508_*)
    patient_folders = sorted(proxy_root.glob(f"{patient}_*"))
    if not patient_folders:
        return None
    patient_dir = patient_folders[0]
    period_dir = patient_dir / f"period{period_num}"
    # "subject s508 • period 2_L_bursts_rates_proxies.xlsx"
    side_char = "L" if side == "left" else "R"
    xlsx_name = f"subject {patient} • period {period_num}_{side_char}_bursts_rates_proxies.xlsx"
    return period_dir / xlsx_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fast hemispheric proxy activity to RS network bursts."
    )
    parser.add_argument(
        "--proxy-xlsx",
        type=Path,
        default=None,
        help="Path to bursts_rates_proxies.xlsx (single-run mode; required if not --loop).",
    )
    parser.add_argument(
        "--network-csv",
        type=Path,
        default=None,
        help="Path to burst_timings/network_bursts_L.csv (or R) (single-run mode; required if not --loop).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over all patients: discover from rs-burst-root, match proxy by patient/period.",
    )
    parser.add_argument(
        "--proxy-root",
        type=Path,
        default=DEFAULT_PROXY_ROOT,
        help=f"Root for fast proxy folders sXXX_YYYYMM/periodN/... (default: {DEFAULT_PROXY_ROOT}).",
    )
    parser.add_argument(
        "--rs-burst-root",
        type=Path,
        default=DEFAULT_RS_BURST_ROOT,
        help=f"Root for burst runs under new layout: patient/PeriodN/run_tag (default: {DEFAULT_RS_BURST_ROOT}).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        nargs="*",
        default=None,
        help="In --loop mode, only process these run_tag(s) (exact match). "
             "Defaults to the RUN_TAGS list defined at the top of this file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_FAST_PROXY,
        help=f"Root for outputs_fast_proxy/patient/PeriodN/run_tag/ (default: {DEFAULT_OUT_FAST_PROXY}).",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="proxies",
        help="Sheet name in the Excel file containing proxy time series (default: proxies).",
    )
    parser.add_argument(
        "--fixed-onset-window-ms",
        type=float,
        default=None,
        help="Onset-only metric: mask = [onset, onset+this] ms (no burst end). Use 0 to disable. (default: from FIXED_ONSET_WINDOW_MS in file)",
    )
    args = parser.parse_args()

    # Optional: use outputs_RS_burst_2 (minDur 0ms) and separateGPi-only run_tag
    rs_burst_root = Path(args.rs_burst_root).resolve()
    out_dir_arg = Path(args.out_dir).resolve() if args.out_dir else None
    run_tag_arg = args.run_tag
    if USE_RS_BURST_2:
        rs_burst_root = RS_BURST_2_ROOT.resolve()
        if out_dir_arg is None:
            out_dir_arg = OUT_FAST_PROXY_RS_BURST_2.resolve()
        if run_tag_arg is None:
            run_tag_arg = [RUN_TAG_RS_BURST_2]
        print(f"Using RS_burst_2 (minDur 0ms): root={rs_burst_root}, out={out_dir_arg}, run_tag={run_tag_arg}")

    fixed_onset_ms = args.fixed_onset_window_ms if args.fixed_onset_window_ms is not None else FIXED_ONSET_WINDOW_MS
    if fixed_onset_ms is not None and fixed_onset_ms <= 0:
        fixed_onset_ms = None

    if args.loop:
        # Discover run dirs from RS burst root; filter by --run-tag or file-level RUN_TAGS
        run_dirs = find_run_dirs_with_network_bursts(rs_burst_root)
        active_tags = run_tag_arg if run_tag_arg is not None else RUN_TAGS
        if active_tags:
            tag_set = set(active_tags)
            run_dirs = [(rd, p, pl, rt) for rd, p, pl, rt in run_dirs if rt in tag_set]
            print(f"Filtered to {len(run_dirs)} run dir(s) matching {len(tag_set)} run_tag(s)")
        else:
            print(f"Found {len(run_dirs)} run dir(s) with network burst CSVs under {rs_burst_root}")

        out_dir = out_dir_arg if out_dir_arg is not None else Path(DEFAULT_OUT_FAST_PROXY).resolve()
        # Onset-fixed results go to a separate root with duration in the name, e.g. outputs_fast_proxy_onset_fixed_300ms
        out_dir_onset = None
        if fixed_onset_ms is not None and fixed_onset_ms > 0:
            out_dir_onset = out_dir.parent / f"{out_dir.name}_onset_fixed_{int(fixed_onset_ms)}ms"
            print(f"Onset-fixed output directory: {out_dir_onset}")
        print(f"Output directory: {out_dir}")
        n_written = 0
        for run_dir, patient, period_label, run_tag in run_dirs:
            base_out = out_dir / patient / period_label / run_tag
            base_out_onset = (out_dir_onset / patient / period_label / run_tag) if out_dir_onset is not None else None
            any_written_this_run = False
            for side in ("left", "right"):
                network_csv = burst_network_csv_path(run_dir, side)
                region_bursts_csv = burst_region_csv_path(run_dir, side)
                if not network_csv.exists():
                    continue
                proxy_xlsx = derive_proxy_xlsx_path(args.proxy_root, patient, period_label, side)
                if proxy_xlsx is None:
                    print(f"  Skip {patient} {period_label} {side}: no proxy folder for {patient} under proxy-root")
                    continue
                if not proxy_xlsx.exists():
                    print(f"  Skip {patient} {period_label} {side}: proxy file not found: {proxy_xlsx}")
                    continue

                # 1) hemi_proxy to hemi_burst (same path as before: base_out / proxy_vs_network_results_{side}.csv)
                try:
                    metrics = run_comparison(
                        proxy_xlsx, network_csv, sheet_name=args.sheet, fixed_onset_window_ms=fixed_onset_ms
                    )
                except Exception as e:
                    print(f"  Skip {patient} {period_label} {run_tag} {side}: {e}")
                    continue
                if metrics is not None:
                    base_out.mkdir(parents=True, exist_ok=True)
                    row = {"patient": patient, "period": period_label, "run_tag": run_tag, **metrics}
                    out_path = base_out / f"proxy_vs_network_results_{side}.csv"
                    pd.DataFrame([row]).to_csv(out_path, index=False)
                    n_written += 1
                    any_written_this_run = True
                    print(f"  {patient} {period_label} {side} hemi→hemi: corr={metrics['correlation']:.4f} → {out_path}")
                    # Onset-fixed only: write to separate folder (e.g. ..._onset_fixed_300ms)
                    if base_out_onset is not None and "correlation_onset_fixed" in metrics:
                        row_onset = {
                            "patient": patient,
                            "period": period_label,
                            "run_tag": run_tag,
                            "correlation_onset_fixed": metrics["correlation_onset_fixed"],
                            "mean_hemi_proxy_inside_onset_fixed": metrics.get("mean_hemi_proxy_inside_onset_fixed"),
                            "mean_hemi_proxy_outside_onset_fixed": metrics.get("mean_hemi_proxy_outside_onset_fixed"),
                            "median_hemi_proxy_inside_onset_fixed": metrics.get("median_hemi_proxy_inside_onset_fixed"),
                            "median_hemi_proxy_outside_onset_fixed": metrics.get("median_hemi_proxy_outside_onset_fixed"),
                            "n_time_bins_inside_onset_fixed": metrics.get("n_time_bins_inside_onset_fixed"),
                            "n_time_bins_outside_onset_fixed": metrics.get("n_time_bins_outside_onset_fixed"),
                            "n_network_bursts": metrics["n_network_bursts"],
                            "fixed_onset_window_ms": metrics.get("fixed_onset_window_ms"),
                        }
                        base_out_onset.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame([row_onset]).to_csv(base_out_onset / f"proxy_vs_network_results_{side}.csv", index=False)
                        n_written += 1
                        print(f"  {patient} {period_label} {side} hemi→hemi (onset-fixed): corr={metrics['correlation_onset_fixed']:.4f} → {base_out_onset}")

                # 2) region_proxy to hemi_burst → base_out / region_proxy_to_hemi_burst / results_{side}.csv
                try:
                    region_hemi_rows = run_comparison_region_proxy_vs_hemi_burst(
                        proxy_xlsx, network_csv, sheet_name=args.sheet, fixed_onset_window_ms=fixed_onset_ms
                    )
                except Exception:
                    region_hemi_rows = []
                if region_hemi_rows:
                    base_out.mkdir(parents=True, exist_ok=True)
                    sub = base_out / "region_proxy_to_hemi_burst"
                    sub.mkdir(parents=True, exist_ok=True)
                    for r in region_hemi_rows:
                        r["patient"] = patient
                        r["period"] = period_label
                        r["run_tag"] = run_tag
                        r["side"] = side
                    out_path = sub / f"results_{side}.csv"
                    pd.DataFrame(region_hemi_rows).to_csv(out_path, index=False)
                    n_written += 1
                    any_written_this_run = True
                    print(f"  {patient} {period_label} {side} region→hemi: {len(region_hemi_rows)} regions → {out_path}")
                    if base_out_onset is not None and region_hemi_rows and "correlation_onset_fixed" in region_hemi_rows[0]:
                        sub_onset = base_out_onset / "region_proxy_to_hemi_burst"
                        sub_onset.mkdir(parents=True, exist_ok=True)
                        onset_only = []
                        for r in region_hemi_rows:
                            onset_only.append({
                                "patient": patient, "period": period_label, "run_tag": run_tag, "side": side,
                                "region": r["region"],
                                "correlation_onset_fixed": r["correlation_onset_fixed"],
                                "mean_proxy_inside_onset_fixed": r.get("mean_proxy_inside_onset_fixed"),
                                "mean_proxy_outside_onset_fixed": r.get("mean_proxy_outside_onset_fixed"),
                                "n_network_bursts": r["n_network_bursts"],
                                "fixed_onset_window_ms": r.get("fixed_onset_window_ms"),
                            })
                        pd.DataFrame(onset_only).to_csv(sub_onset / f"results_{side}.csv", index=False)
                        n_written += 1
                        print(f"  {patient} {period_label} {side} region→hemi (onset-fixed): {len(onset_only)} regions → {sub_onset}")

                # 3) region_proxy to region_burst → base_out / region_proxy_to_region_burst / results_{side}.csv
                if not region_bursts_csv.exists():
                    continue
                try:
                    region_region_rows = run_comparison_region_proxy_vs_region_burst(
                        proxy_xlsx, region_bursts_csv, sheet_name=args.sheet, fixed_onset_window_ms=fixed_onset_ms
                    )
                except Exception:
                    region_region_rows = []
                if region_region_rows:
                    base_out.mkdir(parents=True, exist_ok=True)
                    sub = base_out / "region_proxy_to_region_burst"
                    sub.mkdir(parents=True, exist_ok=True)
                    for r in region_region_rows:
                        r["patient"] = patient
                        r["period"] = period_label
                        r["run_tag"] = run_tag
                        r["side"] = side
                    out_path = sub / f"results_{side}.csv"
                    pd.DataFrame(region_region_rows).to_csv(out_path, index=False)
                    n_written += 1
                    any_written_this_run = True
                    print(f"  {patient} {period_label} {side} region→region: {len(region_region_rows)} regions → {out_path}")
                    if base_out_onset is not None and region_region_rows and "correlation_onset_fixed" in region_region_rows[0]:
                        sub_onset = base_out_onset / "region_proxy_to_region_burst"
                        sub_onset.mkdir(parents=True, exist_ok=True)
                        onset_only = []
                        for r in region_region_rows:
                            onset_only.append({
                                "patient": patient, "period": period_label, "run_tag": run_tag, "side": side,
                                "region": r["region"],
                                "correlation_onset_fixed": r["correlation_onset_fixed"],
                                "mean_proxy_inside_onset_fixed": r.get("mean_proxy_inside_onset_fixed"),
                                "mean_proxy_outside_onset_fixed": r.get("mean_proxy_outside_onset_fixed"),
                                "n_region_bursts": r["n_region_bursts"],
                                "fixed_onset_window_ms": r.get("fixed_onset_window_ms"),
                            })
                        pd.DataFrame(onset_only).to_csv(sub_onset / f"results_{side}.csv", index=False)
                        n_written += 1
                        print(f"  {patient} {period_label} {side} region→region (onset-fixed): {len(onset_only)} regions → {sub_onset}")

        if n_written == 0:
            print("No (patient, period, side) had both proxy and network CSV; nothing written.")
        else:
            print(f"Wrote {n_written} result file(s) under {out_dir}")
        return

    # Single-run mode
    if args.proxy_xlsx is None or args.network_csv is None:
        parser.error("Provide both --proxy-xlsx and --network-csv, or use --loop.")
    t_ms, hemi_proxy, hemi_col = load_hemi_proxy_from_excel(args.proxy_xlsx, sheet_name=args.sheet)
    print(f"Loaded proxy: {args.proxy_xlsx}")
    print(f"  time bins: {t_ms.shape[0]}")
    print(f"  hemi proxy column: {hemi_col}")

    starts, ends = load_network_bursts(args.network_csv)
    print(f"Loaded network bursts: {args.network_csv}")
    print(f"  N bursts: {len(starts)}")

    if len(starts) == 0:
        print("No network bursts found; nothing to compare.")
        return

    mask = build_network_mask(t_ms, starts, ends)
    n_in = int(mask.sum())
    n_out = int((~mask).sum())
    print(f"Time bins inside bursts:  {n_in}")
    print(f"Time bins outside bursts: {n_out}")

    if n_in == 0 or n_out == 0:
        print("Network mask is all inside or all outside; correlation is not meaningful.")
        return

    network_indicator = mask.astype(float)
    corr = np.corrcoef(network_indicator, hemi_proxy)[0, 1]
    inside = hemi_proxy[mask]
    outside = hemi_proxy[~mask]

    print("\n=== Proxy vs Network (duration-based mask) ===")
    print(f"Point-biserial correlation (network vs {hemi_col}): {corr:.4f}")
    print(f"Mean {hemi_col} inside bursts:  {np.nanmean(inside):.4f}")
    print(f"Mean {hemi_col} outside bursts: {np.nanmean(outside):.4f}")
    print(f"Median {hemi_col} inside bursts:  {np.nanmedian(inside):.4f}")
    print(f"Median {hemi_col} outside bursts: {np.nanmedian(outside):.4f}")

    if fixed_onset_ms is not None and fixed_onset_ms > 0:
        mask_onset = build_mask_from_onsets_fixed_duration(t_ms, starts, fixed_onset_ms)
        n_in_o = int(mask_onset.sum())
        n_out_o = int((~mask_onset).sum())
        if n_in_o > 0 and n_out_o > 0:
            corr_onset = np.corrcoef(mask_onset.astype(float), hemi_proxy)[0, 1]
            inside_o = hemi_proxy[mask_onset]
            outside_o = hemi_proxy[~mask_onset]
            print(f"\n=== Onset-fixed ({fixed_onset_ms:.0f} ms after each onset; no burst end) ===")
            print(f"Point-biserial correlation: {corr_onset:.4f}")
            print(f"Mean {hemi_col} inside:  {np.nanmean(inside_o):.4f}")
            print(f"Mean {hemi_col} outside: {np.nanmean(outside_o):.4f}")


if __name__ == "__main__":
    main()

