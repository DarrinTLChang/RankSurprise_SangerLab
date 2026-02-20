# coactivity.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .utils import *
from config import *


def burst_coactivity_by_channel(
    burst_list: list[dict[str, Any]],
    record_len_s: float,
    win_s: float = 1.0,
    stride_s: float | None = None
) -> pd.DataFrame:
    if stride_s is None or stride_s <= 0:
        stride_s = win_s
    n_win = int(np.floor((record_len_s - win_s) / stride_s)) + 1
    if n_win < 1:
        n_win = 1
        stride_s = win_s

    channel_windows: dict[str, np.ndarray] = {}

    for b in burst_list:
        ch_name = b["Electrode"]
        if ch_name not in channel_windows:
            channel_windows[ch_name] = np.zeros(n_win, dtype=bool)

        s = float(b["Start_ms"]) / 1000.0
        e = float(b["End_ms"])   / 1000.0

        k_low  = int(np.ceil((s - win_s) / stride_s))
        k_high = int(np.floor(e / stride_s))
        if k_high < 0 or k_low > n_win - 1:
            continue
        k_low  = max(0, k_low)
        k_high = min(n_win - 1, k_high)
        channel_windows[ch_name][k_low:k_high+1] = True

    counts = np.zeros(n_win, dtype=int)
    for mask in channel_windows.values():
        counts += mask.astype(int)

    starts = np.arange(n_win) * stride_s
    df_co = pd.DataFrame({
        "Window_start_s": starts,
        "Window_end_s"  : starts + win_s,
        "Active_channels": counts
    })

    mean_val = df_co["Active_channels"].mean()
    std_val  = df_co["Active_channels"].std(ddof=0)
    df_co = pd.concat([
        df_co,
        pd.DataFrame({
            "Window_start_s": ["mean", "std"],
            "Window_end_s"  : ["mean", "std"],
            "Active_channels": [mean_val, std_val]
        })
    ], ignore_index=True)
    return df_co


def burst_coactivity_by_cluster(
    burst_list: list[dict[str, Any]],
    record_len_s: float,
    win_s: float = 1.0,
    stride_s: float | None = None
) -> pd.DataFrame:
    if stride_s is None or stride_s <= 0:
        stride_s = win_s
    n_win = int(np.floor((record_len_s - win_s) / stride_s)) + 1
    if n_win < 1:
        n_win = 1
        stride_s = win_s

    cluster_windows: dict[tuple[str, int], np.ndarray] = {}

    for b in burst_list:
        key = (b["Electrode"], int(b["Cluster"]))
        if key not in cluster_windows:
            cluster_windows[key] = np.zeros(n_win, dtype=bool)

        s = float(b["Start_ms"]) / 1000.0
        e = float(b["End_ms"])   / 1000.0

        k_low  = int(np.ceil((s - win_s) / stride_s))
        k_high = int(np.floor(e / stride_s))
        if k_high < 0 or k_low > n_win - 1:
            continue
        k_low  = max(0, k_low)
        k_high = min(n_win - 1, k_high)
        cluster_windows[key][k_low:k_high+1] = True

    counts = np.zeros(n_win, dtype=int)
    for mask in cluster_windows.values():
        counts += mask.astype(int)

    starts = np.arange(n_win) * stride_s
    df_co = pd.DataFrame({
        "Window_start_s": starts,
        "Window_end_s"  : starts + win_s,
        "Active_channels": counts
    })

    mean_val = df_co["Active_channels"].mean()
    std_val  = df_co["Active_channels"].std(ddof=0)
    df_co = pd.concat([
        df_co,
        pd.DataFrame({
            "Window_start_s": ["mean", "std"],
            "Window_end_s"  : ["mean", "std"],
            "Active_channels": [mean_val, std_val]
        })
    ], ignore_index=True)
    return df_co


def burst_coactivity_by_region(
    burst_list: list[dict[str, Any]],
    record_len_s: float,
    win_s: float = 1.0,
    stride_s: float | None = None
) -> pd.DataFrame:
    if stride_s is None or stride_s <= 0:
        stride_s = win_s

    n_win = int(np.floor((record_len_s - win_s) / stride_s)) + 1
    if n_win < 1:
        n_win = 1
        stride_s = win_s

    region_channel_windows: dict[str, dict[str, np.ndarray]] = {}

    for b in burst_list:
        ch_name = b["Electrode"]
        region  = infer_region(ch_name)
        if region not in region_channel_windows:
            region_channel_windows[region] = {}
        if ch_name not in region_channel_windows[region]:
            region_channel_windows[region][ch_name] = np.zeros(n_win, dtype=bool)

        s = float(b["Start_ms"]) / 1000.0
        e = float(b["End_ms"])   / 1000.0

        k_low  = int(np.ceil((s - win_s) / stride_s))
        k_high = int(np.floor(e / stride_s))
        if k_high < 0 or k_low > n_win - 1:
            continue
        k_low  = max(0, k_low)
        k_high = min(n_win - 1, k_high)
        region_channel_windows[region][ch_name][k_low:k_high+1] = True

    starts = np.arange(n_win) * stride_s
    data: dict[str, Any] = {
        "Window_start_s": starts,
        "Window_end_s": starts + win_s,
    }

    for region, ch_map in region_channel_windows.items():
        counts = np.zeros(n_win, dtype=int)
        for mask in ch_map.values():
            counts += mask.astype(int)
        data[region] = counts

    df = pd.DataFrame(data)

    summary = {"Window_start_s": ["mean", "std"], "Window_end_s": ["mean", "std"]}
    for region in [c for c in df.columns if c not in ("Window_start_s", "Window_end_s")]:
        summary[region] = [df[region].mean(), df[region].std(ddof=0)]
    return pd.concat([df, pd.DataFrame(summary)], ignore_index=True)
