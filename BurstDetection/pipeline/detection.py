# detection.py
#
# All burst-detection algorithms:
#   - detect_bursts()     : maxISI / alpha-meanISI threshold method
#   - RS_detect_burst()   : Rank Surprise method
#
# Rank-surprise algorithm adapted from Subhasis Ray's Matlab -> Python port.
# Reference: Gourevitch, B. & Eggermont, J. J. A nonparametric approach for
# detection of bursts in spike trains. J Neurosci Methods 160, 349-358 (2007).
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3, or (at your option) any
# later version.  When redistributing the code preserve the reference
# and the credits.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.stats import norm
from config import *

from .burst_paths import burst_network_csv_path, burst_region_csv_path, burst_unit_csv_path


# =====================================================================
# maxISI / alpha-meanISI burst detection
# =====================================================================

def detect_bursts(
    spikes_ms: np.ndarray,
    elec: str,
    cl: int,
    STATS,
    override_thr: float | None = None,
    override_min_spikes: int | None = None,
    ibi_merge_factor=1.5,
):
    if spikes_ms.size < 2:
        return []
    row = STATS.loc[(elec, cl)]
    thr = float(override_thr) if override_thr is not None else float(row["thr"])
    fr_hz = float(row["FR_Hz"])

    isis = np.diff(spikes_ms)
    split = np.where(isis > thr)[0] + 1
    segs = np.split(spikes_ms, split)

    bursts_indiv = []
    for seg in segs:
        min_spikes = (
            int(override_min_spikes)
            if override_min_spikes is not None
            else MIN_SPIKES_IN_BURST
        )

        if seg.size < min_spikes:
            continue
        dur = seg[-1] - seg[0]
        # MIN_BURST_DURATION is applied only at final network burst level, not at unit level
        bursts_indiv.append(
            dict(
                Start_ms=float(seg[0]),
                End_ms=float(seg[-1]),
                Duration_ms=float(dur),
                Num_Spikes=int(seg.size),
                thr=thr,
                FR_Hz=fr_hz,
            )
        )
    if len(bursts_indiv) < 2:
        return bursts_indiv

    bursts_merged = [bursts_indiv[0].copy()]

    for b in bursts_indiv[1:]:
        prev = bursts_merged[-1]
        ibi = float(b["Start_ms"] - prev["End_ms"])

        if ibi <= ibi_merge_factor * thr:
            prev["End_ms"] = max(prev["End_ms"], float(b["End_ms"]))
            prev["Num_Spikes"] = int(prev["Num_Spikes"]) + int(b["Num_Spikes"])
            prev["Duration_ms"] = float(prev["End_ms"] - prev["Start_ms"])
        else:
            bursts_merged.append(b.copy())

    return bursts_merged


# =====================================================================
# Rank Surprise burst detection
# =====================================================================

def RS_detect_burst(spiketimes, limit=None, RSalpha=-np.log(0.01), min_spikes=3, RS_Percentile_Limit=75):
    """Detect bursts in spiketimes using Rank-Surprise method.

    Return a tuple (start, length, RS) where
      start:  spike number of burst start for each burst detected
      length: burst length for each burst detected (in spikes)
      RS:     rank surprise value for each burst detected.
    """
    q_lim = 30
    l_min = min_spikes

    alternate = np.ones(400)
    alternate[1::2] = -1

    log_fac = np.cumsum(np.log(np.r_[1:q_lim + 1]))

    ISI = np.diff(spiketimes)
    N = len(ISI)

    if limit is None:
        limit = np.percentile(ISI, RS_Percentile_Limit)

    R = _val2rank(ISI)

    ISI_limit = np.diff(np.where(ISI < limit, 1, 0))
    begin_int = np.nonzero(ISI_limit == 1)[0] + 1
    if ISI[0] < limit:
        begin_int = np.r_[0, begin_int]
    end_int = np.nonzero(ISI_limit == -1)[0]
    if len(end_int) < len(begin_int):
        end_int = np.r_[end_int, N - 1]
    length_int = end_int - begin_int + 1

    archive_burst_RS = []
    archive_burst_length = []
    archive_burst_start = []

    for n_j, p_j in zip(begin_int, length_int):
        subseq_RS = []
        for i in range(p_j - (l_min - 1) + 1):
            for q in range(l_min - 1, p_j - i + 1):
                u = np.sum(R[n_j + i:n_j + i + q])
                u = int(np.floor(u))
                if q < q_lim:
                    kmax = int(np.floor((u - q) / N))
                    if kmax < 0:
                        continue
                    k = np.arange(kmax + 1, dtype=int)
                    length_k = len(k)
                    t1 = np.tile(k, (q, 1))
                    t2 = np.tile(np.r_[:q], (length_k, 1)).transpose()
                    arg = u - t1 * N - t2
                    if np.any(arg <= 0):
                        arg = np.maximum(arg, 1e-300)
                    l1 = np.log(arg)
                    ss = np.sum(l1, axis=0)
                    l2 = log_fac[np.r_[0, k[1:] - 1]]
                    l3 = log_fac[q - k - 1]
                    fac1 = np.exp(ss - l2 - l3 - q * np.log(N))
                    fac2 = alternate[:length_k]
                    prob = np.dot(fac1, fac2)
                    prob = float(np.clip(prob, np.finfo(float).tiny, 1.0))
                else:
                    prob = norm.cdf((u - q * (N + 1) / 2) / np.sqrt(q * (N ** 2 - 1) / 12))
                    prob = float(np.clip(prob, np.finfo(float).tiny, 1.0))
                RS = -np.log(prob)
                if RS > RSalpha:
                    subseq_RS.append((np.r_[RS, i, q]))

        if len(subseq_RS) > 0:
            subseq_RS = sorted(subseq_RS, key=itemgetter(0), reverse=True)
            while len(subseq_RS) > 0:
                current_burst = subseq_RS[0]
                archive_burst_RS.append(current_burst[0])
                archive_burst_length.append(current_burst[2] + 1)
                archive_burst_start.append(n_j + current_burst[1])
                subseq_RS = [row for row in subseq_RS[1:]
                             if ((row[1] + row[2] - 1) < current_burst[1]) or
                             (row[1] > (current_burst[1] + current_burst[2] - 1))]

    ind_sort = np.argsort(archive_burst_start)
    archive_burst_start = np.take(archive_burst_start, ind_sort).astype(int)
    archive_burst_RS = np.take(archive_burst_RS, ind_sort)
    archive_burst_length = np.take(archive_burst_length, ind_sort).astype(int)
    return (archive_burst_start, archive_burst_length, archive_burst_RS)


def _val2rank(values):
    """Convert values to ranks, with mean of ranks for tied values."""
    lp = len(values)
    cl = np.argsort(values)
    rk = np.ones(lp)
    rk[cl] = np.r_[0:lp]
    cl2 = np.argsort(-values)
    rk2 = np.ones(lp)
    rk2[cl2] = np.r_[:lp]
    ranks = (lp + 1 - rk2 + rk) / 2
    return ranks


# =====================================================================
# Unified span computation (used for CSV and plotting)
# =====================================================================

def _ch_id(b: dict, channel_mode: str = "elec_cluster") -> str:
    """Get channel ID from burst dict."""
    elec = str(b.get("Electrode", ""))
    if channel_mode == "elec":
        return elec
    cl = int(b.get("Cluster", -1))
    return f"{elec}|{cl}"


def compute_spans_and_bars(
    windows_ms: list[tuple[float, float]],
    bursts: list[dict],
    *,
    min_unique_channels: int = 1,
    channel_mode: str = "elec_cluster",
    use_onset_plus_length: bool = False,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute spans for each window and merged bars for plotting.

    If use_onset_plus_length is False: span = (median(starts), median(ends)).
    If use_onset_plus_length is True: span = (w0, w0 + median(burst lengths)).

    Returns:
        spans_ms: List of (span_start_ms, span_end_ms) per window (for CSV)
        bars_s: List of merged (start_s, end_s) bars (for plotting)
    """
    spans_ms: list[tuple[float, float]] = []
    bars_s: list[tuple[float, float]] = []

    for w0, w1 in windows_ms:
        if not (np.isfinite(w0) and np.isfinite(w1)) or w1 <= w0:
            spans_ms.append((float("nan"), float("nan")))
            continue

        overlapping = []
        chans = set()

        for b in bursts:
            s = float(b.get("Start_ms", np.nan))
            e = float(b.get("End_ms", np.nan))
            if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
                continue
            if (s >= w0) and (s <= w1):  # onset within window
                overlapping.append((s, e))
                chans.add(_ch_id(b, channel_mode))

        if not overlapping or len(chans) < min_unique_channels:
            spans_ms.append((float("nan"), float("nan")))
            continue

        if use_onset_plus_length:
            lengths = [e - s for (s, e) in overlapping]
            median_length_ms = float(np.median(lengths))
            span_start = float(w0)
            span_end = w0 + median_length_ms
        else:
            starts = [x[0] for x in overlapping]
            ends = [x[1] for x in overlapping]
            span_start = float(np.median(starts))
            span_end = float(np.median(ends))

        if span_end <= span_start:
            spans_ms.append((float("nan"), float("nan")))
            continue

        spans_ms.append((span_start, span_end))
        bars_s.append((span_start / 1000.0, span_end / 1000.0))
    
    # Merge overlapping bars
    bars_s = _merge_windows_s(bars_s)
    
    return spans_ms, bars_s


def _merge_windows_s(wins: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping windows."""
    if not wins:
        return []
    wins = sorted(wins, key=lambda t: t[0])
    merged = []
    for t0, t1 in wins:
        if not merged:
            merged.append([t0, t1])
        elif t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [(a, b) for a, b in merged]


# =====================================================================
# Multi-stage burst detection pipeline
# =====================================================================

@dataclass
class BurstResults:
    all_bursts: list[dict[str, Any]] = field(default_factory=list)
    all_bursts_L: list[dict[str, Any]] = field(default_factory=list)
    all_bursts_R: list[dict[str, Any]] = field(default_factory=list)
    region_windows: dict = field(default_factory=dict)
    region_windows_L: dict = field(default_factory=dict)
    region_windows_R: dict = field(default_factory=dict)
    network_windows: list = field(default_factory=list)
    network_windows_L: list = field(default_factory=list)
    network_windows_R: list = field(default_factory=list)
    network_bars_L: list[tuple[float, float]] = field(default_factory=list)
    network_bars_R: list[tuple[float, float]] = field(default_factory=list)
    region_bars_by_region_L: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    region_bars_by_region_R: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def rs_burst_detection(
    spike_struct,
    spike_struct_L,
    spike_struct_R,
    STATS: pd.DataFrame,
    run_dir: Path,
    iter_units_fn,
    infer_region_fn,
    burst_in_window_fn,
) -> BurstResults:
    """Run the full burst-detection pipeline (unit -> region -> network).

    Burst CSVs are written under ``run_dir / burst_timings /`` with short names
    (e.g. ``network_LR.csv``, ``network_bursts_L.csv``).

    Parameters
    ----------
    run_dir : Path
        Run output directory (e.g. main ``run_dir``).
    iter_units_fn : callable
        ``iter_units_from_stats(spike_struct, STATS)`` — yields (elec, cl, spk).
    infer_region_fn : callable
        ``infer_region(electrode_name)`` — returns region string.
    burst_in_window_fn : callable
        ``burst_in_window(t_ms, windows)`` — returns bool.
    """
    s_list = [
        (spike_struct, "combined"),
        (spike_struct_L, "left"),
        (spike_struct_R, "right"),
    ]

    def _write_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    res = BurstResults()

    if maxisi_toggle or alpha_meanisi_toggle:
        for elec, cl, spk in iter_units_fn(spike_struct, STATS):
            detect_bursts(spk, elec, cl, STATS, ibi_merge_factor=ibi_merge_factor)
        return res

    if not rankSurprise_toggle:
        return res

    all_bursts_list: list[list[dict]] = []
    region_window_list: list[dict] = []
    network_window_list: list[list] = []
    network_bars_temp: dict[str, list[tuple[float, float]]] = {}
    region_bars_temp: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for spike_struct_side, side_tag in s_list:
        # Stage 1: unit-level bursts
        unit_bursts: list[dict] = []
        for elec, cl, spk in iter_units_fn(spike_struct_side, STATS):
            start_idx, length_spikes, RSvals = RS_detect_burst(
                spk,
                limit=RS_Limit_stage1,
                RSalpha=RS_alpha_stage1,
                min_spikes=MIN_SPIKES_IN_BURST,
                RS_Percentile_Limit=RS_Percentile_Limit_stage1,
            )
            for s, L, rs in zip(start_idx, length_spikes, RSvals):
                s = int(s); e = int(s + L - 1)
                if 0 <= s < spk.size and 0 <= e < spk.size:
                    dur_ms = float(spk[e] - spk[s])
                    # By default MIN_BURST_DURATION is applied only at the final
                    # network level. When APPLY_MIN_BURST_DURATION_ALL_STAGES is
                    # True, also drop short bursts at the unit level here.
                    if APPLY_MIN_BURST_DURATION_ALL_STAGES and dur_ms < float(MIN_BURST_DURATION):
                        continue
                    unit_bursts.append({
                        "Electrode": elec,
                        "Cluster": int(cl),
                        "Start_ms": float(spk[s]),
                        "End_ms": float(spk[e]),
                        "Duration_ms": dur_ms,
                        "Num_Spikes": int(L),
                        "RS": float(rs),
                        "Method": "RankSurprise",
                    })

        all_bursts = unit_bursts
        # RS unit-level CSVs: unified burst fields
        unit_csv_rows = [
            {
                "Electrode": b["Electrode"],
                "Cluster": b["Cluster"],
                "burst_start_ms": b["Start_ms"],
                "burst_end_ms": b["End_ms"],
                "burst_duration_ms": b["Duration_ms"],
                "num_onsets": b["Num_Spikes"],
                "RS": b["RS"],
                "Method": b["Method"],
            }
            for b in unit_bursts
        ]
        _write_csv(pd.DataFrame(unit_csv_rows), burst_unit_csv_path(run_dir, side_tag))

        # Stage 1.5: region-level bursts
        region_windows_by_region: dict[str, list[tuple[float, float]]] = {}

        if RS_region_burst_toggle:
            region_onsets: dict[str, list[float]] = {}
            for b in unit_bursts:
                reg = infer_region_fn(str(b["Electrode"]))
                b["_Region"] = reg
                region_onsets.setdefault(reg, []).append(float(b["Start_ms"]))

            region_bursts: list[dict] = []
            for reg, onsets in region_onsets.items():
                onsets_ms = np.asarray(onsets, dtype=float)
                onsets_ms = onsets_ms[np.isfinite(onsets_ms)]
                onsets_ms = np.sort(onsets_ms)

                if onsets_ms.size < MIN_SPIKES_IN_BURST:
                    continue

                unit_bursts_in_reg = [b for b in unit_bursts if b.get("_Region") == reg]

                start_idx_r, length_r, RSvals_r = RS_detect_burst(
                    onsets_ms,
                    limit=RS_Limit_region,
                    RSalpha=RS_alpha_region,
                    min_spikes=MIN_SPIKES_IN_BURST,
                    RS_Percentile_Limit=RS_Percentile_Limit_region,
                )

                windows: list[tuple[float, float]] = []
                rs_vals_for_windows: list[float] = []
                num_events_for_windows: list[int] = []
                for s, L, rs in zip(start_idx_r, length_r, RSvals_r):
                    s = int(s); e = int(s + L - 1)
                    t0 = float(onsets_ms[s]); t1 = float(onsets_ms[e])
                    if t1 <= t0:
                        continue
                    dur_ms = t1 - t0
                    # When APPLY_MIN_BURST_DURATION_ALL_STAGES is True, also
                    # drop short region-level bursts here. Otherwise, MIN_BURST_DURATION
                    # is enforced only at the final network level below.
                    if APPLY_MIN_BURST_DURATION_ALL_STAGES and dur_ms < float(MIN_BURST_DURATION):
                        continue
                    windows.append((t0, t1))
                    rs_vals_for_windows.append(float(rs))
                    num_events_for_windows.append(int(L))

                if windows:
                    spans_ms, bars_s = compute_spans_and_bars(
                        windows, unit_bursts_in_reg,
                        min_unique_channels=MIN_UNIQUE_CHANNELS_REGION,
                        channel_mode="elec_cluster",
                        use_onset_plus_length=REGION_SPAN_ONSET_PLUS_LENGTH,
                    )
                    
                    # Only keep windows that passed channel filter (have valid spans)
                    filtered_windows = []
                    for i, (w, span, rs_val, num_ev) in enumerate(zip(windows, spans_ms, rs_vals_for_windows, num_events_for_windows)):
                        if np.isfinite(span[0]) and np.isfinite(span[1]):
                            filtered_windows.append(w)
                            span_dur = span[1] - span[0]
                            region_bursts.append({
                                "Region": reg,
                                "burst_start_ms": w[0],
                                "burst_end_ms": w[0] + span_dur,
                                "burst_duration_ms": span_dur,
                                "num_onsets": num_ev,
                                "RS": rs_val,
                                "Method": "RS_region_onsets",
                            })
                    
                    if filtered_windows:
                        region_windows_by_region[reg] = filtered_windows
                        # Store merged bars for this region (will be added to BurstResults later)
                        if side_tag in ("left", "right"):
                            if side_tag not in region_bars_temp:
                                region_bars_temp[side_tag] = {}
                            region_bars_temp[side_tag][reg] = bars_s

            all_bursts_region: list[dict] = []
            for b in unit_bursts:
                reg = b.get("_Region", infer_region_fn(str(b["Electrode"])))
                wins = region_windows_by_region.get(reg, [])
                flag = burst_in_window_fn(float(b["Start_ms"]), wins)
                b["IsRegionBurst"] = bool(flag)
                if flag:
                    all_bursts_region.append(b)

            _write_csv(pd.DataFrame(region_bursts), burst_region_csv_path(run_dir, side_tag))

            region_windows_rows = []
            for reg, wins in region_windows_by_region.items():
                for (t0, t1) in wins:
                    region_windows_rows.append({
                        "Region": reg,
                        "Start_ms": float(t0),
                        "End_ms": float(t1),
                        "Duration_ms": float(t1 - t0),
                    })
            # pd.DataFrame(region_windows_rows).to_csv(run_dir / f"region_windows_RS_{side_tag}.csv", index=False)

            all_bursts = all_bursts_region

        region_window_list.append(region_windows_by_region)

        # Stage 2: network-level bursts
        network_windows: list[tuple[float, float]] = []

        if RS_NETWORK_ONSETS_TOGGLE:
            onset_source = all_bursts
            onsets_ms = np.sort(np.asarray([b["Start_ms"] for b in onset_source], float))

            network_bursts: list[dict] = []
            if onsets_ms.size >= MIN_SPIKES_IN_BURST:
                start_idx2, length2, RSvals2 = RS_detect_burst(
                    onsets_ms,
                    limit=RS_Limit_network,
                    RSalpha=RS_alpha_network,
                    min_spikes=MIN_SPIKES_IN_BURST,
                    RS_Percentile_Limit=RS_Percentile_Limit_network,
                )

                windows: list[tuple[float, float]] = []
                rs_vals_for_windows: list[float] = []
                num_events_for_windows: list[int] = []
                for s2, L2, rs2 in zip(start_idx2, length2, RSvals2):
                    s2 = int(s2); e2 = int(s2 + L2 - 1)
                    t0 = float(onsets_ms[s2]); t1 = float(onsets_ms[e2])
                    if t1 <= t0:
                        continue
                    # MIN_BURST_DURATION applied only at final network burst level (here)
                    if (t1 - t0) < float(MIN_BURST_DURATION):
                        continue
                    windows.append((t0, t1))
                    rs_vals_for_windows.append(float(rs2))
                    num_events_for_windows.append(int(L2))

                if windows:
                    spans_ms, bars_s = compute_spans_and_bars(
                        windows, all_bursts,
                        min_unique_channels=MIN_UNIQUE_CHANNELS_NETWORK,
                        channel_mode="elec_cluster",
                        use_onset_plus_length=NETWORK_SPAN_ONSET_PLUS_LENGTH,
                    )
                    
                    # Only keep windows that passed channel filter (have valid spans)
                    for w, span, rs_val, num_ev in zip(windows, spans_ms, rs_vals_for_windows, num_events_for_windows):
                        if np.isfinite(span[0]) and np.isfinite(span[1]):
                            network_windows.append(w)
                            span_dur = span[1] - span[0]
                            network_bursts.append({
                                "burst_start_ms": w[0],
                                "burst_end_ms": w[0] + span_dur,
                                "burst_duration_ms": span_dur,
                                "num_onsets": num_ev,
                                "RS": rs_val,
                            })
                    
                    # Store merged bars for this side
                    network_bars_temp[side_tag] = bars_s

            for b in all_bursts:
                b["IsNetworkBurst"] = burst_in_window_fn(float(b["Start_ms"]), network_windows)

            _write_csv(pd.DataFrame(network_bursts), burst_network_csv_path(run_dir, side_tag))

            network_windows_rows = [{
                "Start_ms": float(t0),
                "End_ms": float(t1),
                "Duration_ms": float(t1 - t0),
            } for (t0, t1) in network_windows]
            # pd.DataFrame(network_windows_rows).to_csv(run_dir / f"network_windows_RS_{side_tag}.csv", index=False)

            if plot_network_bursts:
                all_bursts = [b for b in all_bursts if b.get("IsNetworkBurst")]

        network_window_list.append(network_windows)
        all_bursts_list.append(all_bursts)

    res.all_bursts = all_bursts_list[0]
    res.all_bursts_L = all_bursts_list[1]
    res.all_bursts_R = all_bursts_list[2]
    res.region_windows = region_window_list[0]
    res.region_windows_L = region_window_list[1]
    res.region_windows_R = region_window_list[2]
    res.network_windows = network_window_list[0]
    res.network_windows_L = network_window_list[1]
    res.network_windows_R = network_window_list[2]
    
    # Assign pre-computed bars
    res.network_bars_L = network_bars_temp.get("left", [])
    res.network_bars_R = network_bars_temp.get("right", [])
    res.region_bars_by_region_L = region_bars_temp.get("left", {})
    res.region_bars_by_region_R = region_bars_temp.get("right", {})

    print(f"Burst counts: combined={len(res.all_bursts)}, L={len(res.all_bursts_L)}, R={len(res.all_bursts_R)}")

    return res
