# stats.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import scipy.io as spio
from pathlib import Path
from config import *


def snr_fails_filter(snr_val: float) -> bool:
    """True if cluster should be excluded: NaN, below SNR_MIN, or above SNR_MAX (if set)."""
    if np.isnan(snr_val) or snr_val < SNR_MIN:
        return True
    if SNR_MAX is not None and snr_val > SNR_MAX:
        return True
    return False


def build_cache(
    mat_file: str,
    cluster_stats_csv: Path,
    pooled_cluster_stats_csv:Path,
    base_thr: float,
    record_len_s: float,
    adaptie_thr_toggle: bool,
    pooling_toggle,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Scan the .mat file and compute per-cluster statistics.

    Writes:
      - isi_stats.csv
      - cluster_stats.csv

    Returns:
      DataFrame indexed by (Electrode, Cluster)
    """
    mat = spio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

    print(f"[DEBUG build_cache] SNR_MIN={SNR_MIN}, SNR_MAX={SNR_MAX}")
    rows: list[dict[str, Any]] = []
    _skipped_snr = 0
    _kept = 0

    for ch in np.ravel(mat["spikeTime"]):
        elec = str(ch.electrode)
        snr_arr = np.asarray(ch.snr).flatten()
        
        for cl, arr in enumerate(np.ravel(ch.time)):
            snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
            spk = np.asarray(arr).flatten()
            if spk.size < 2:
                continue

            fr_hz = spk.size / record_len_s
            if fr_hz < FR_MIN_HZ:
                continue
            if snr_fails_filter(snr_val):
                _skipped_snr += 1
                print(f"  [SKIP] {elec} cl={cl} SNR={snr_val:.3f} (fails filter)")
                continue
            _kept += 1

            isis = np.diff(spk.astype(float))

            isi_mean = float(np.mean(isis))
            isi_median = float(np.median(isis))

            isi_std = float(np.std(isis, ddof=0))
            start_ms = float(spk[0])
            end_ms   = float(spk[-1])
            duration_ms = end_ms - start_ms

            if adaptie_thr_toggle:
                if fr_hz > 0:
                    thr = ALPHA_MAXISI * isi_mean
                else:
                    thr = MAX_MAXISI_MS 

                thr = float(np.clip(thr, MIN_MAXISI_MS, MAX_MAXISI_MS))
            else:
                thr = float(base_thr)

            rows.append(
                dict(
                    Electrode=elec,
                    Cluster=cl,
                    FR_Hz=fr_hz,
                    ISI_mean_ms=isi_mean,
                    ISI_median_ms= isi_median,
                    ISI_std_ms=isi_std,
                    Start_ms=start_ms,
                    End_ms=end_ms,
                    Duration_ms=duration_ms,
                    thr=thr,
                    SNR=snr_val,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(cluster_stats_csv, index=False)

    print(f"[DEBUG build_cache] Kept {_kept} clusters, skipped {_skipped_snr} by SNR filter")
    if _kept > 0:
        snr_vals = [r["SNR"] for r in rows if not np.isnan(r["SNR"])]
        print(f"[DEBUG build_cache] SNR range of kept: {min(snr_vals):.2f} – {max(snr_vals):.2f}")
    print(f"• Wrote {len(df)} rows → {cluster_stats_csv}")

    if pooling_toggle:
        pooled_rows: list[dict[str, Any]] = []
        for ch in np.ravel(mat["spikeTime"]):
            elec = str(ch.electrode)
            snr_arr = np.asarray(ch.snr).flatten()
            pooled = []
            for cl, arr in enumerate(np.ravel(ch.time)):
                snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
                spk = np.asarray(arr).flatten()
                if spk.size < 2:
                    continue
                fr_hz_cl = spk.size / record_len_s
                if fr_hz_cl < FR_MIN_HZ:
                    continue
                if snr_fails_filter(snr_val):
                    continue
                pooled.append(spk)
            if not pooled:
                continue
            pooled_spk = np.sort(np.concatenate(pooled))
            fr_hz = pooled_spk.size / record_len_s
            isis = np.diff(pooled_spk.astype(float))

            pooled_isi_mean = float(np.mean(isis))
            pooled_isi_median = float(np.median(isis))
            pooled_isi_std = float(np.std(isis, ddof=0))
            start_ms = float(pooled_spk[0])
            end_ms   = float(pooled_spk[-1])
            duration_ms = end_ms - start_ms

            if adaptie_thr_toggle:
                if fr_hz > 0:
                    thr = ALPHA_MAXISI * pooled_isi_mean
                else:
                    thr = MAX_MAXISI_MS 
                thr = float(np.clip(thr, MIN_MAXISI_MS, MAX_MAXISI_MS))
            else:
                thr = float(base_thr)

            pooled_rows.append(
                dict(
                    Electrode=elec,
                    Cluster= 0,
                    FR_Hz=fr_hz,
                    ISI_mean_ms=pooled_isi_mean,
                    ISI_median_ms= pooled_isi_median,
                    ISI_std_ms= pooled_isi_std,
                    Start_ms=start_ms,
                    End_ms=end_ms,
                    Duration_ms=duration_ms,
                    thr=thr,
                    SNR=np.nan,
                )
            )
        df_pooled = pd.DataFrame(pooled_rows)
        df_pooled.to_csv(pooled_cluster_stats_csv, index=False)
        print(f"• Wrote {len(df_pooled)} rows → {pooled_cluster_stats_csv}")

    df_idx = df.set_index(["Electrode", "Cluster"])

    if pooling_toggle:
        df_pooled_idx = df_pooled.set_index(["Electrode", "Cluster"])
    else:
        df_pooled_idx = None

    return df_idx, df_pooled_idx


def compute_network_burst_isi_stats(
    spike_struct,
    network_windows_ms,
    STATS,
    *,
    allowed_clusters=None,
):
    """
    Returns:
        per_window_df  -> ISI stats per network burst window
        pooled_stats   -> overall pooled ISI stats (across all windows/units)
    Notes:
        - Mean ISI is defined if N_ISI >= 1 (>=2 spikes within a unit-window)
        - Sample STD (ddof=1) is defined if N_ISI >= 2 (>=3 spikes within a unit-window)
    """

    per_window_rows = []
    pooled_isis = []

    if not network_windows_ms:
        return pd.DataFrame(columns=["Start_ms","End_ms","Mean_ISI_ms","STD_ISI_ms","N_ISI","N_spikes"]), {
            "mean_ms": np.nan, "std_ms": np.nan, "n_isi": 0
        }

    for w0_ms, w1_ms in network_windows_ms:
        w0_ms = float(w0_ms)
        w1_ms = float(w1_ms)
        if not (np.isfinite(w0_ms) and np.isfinite(w1_ms)) or (w1_ms <= w0_ms):
            continue

        window_isis = []
        window_spike_count = 0

        for ch in np.ravel(spike_struct):
            elec = str(ch.electrode)

            for cl, arr in enumerate(np.ravel(ch.time)):
                cl = int(cl)

                if (elec, cl) not in STATS.index:
                    continue
                if allowed_clusters is not None and (elec, cl) not in allowed_clusters:
                    continue

                spk = np.asarray(arr, dtype=float).ravel()
                spk = spk[np.isfinite(spk)]
                if spk.size < 2:
                    continue

                spk_win = spk[(spk >= w0_ms) & (spk <= w1_ms)]
                if spk_win.size < 2:
                    continue

                spk_win = np.sort(spk_win)
                isi = np.diff(spk_win)

                if isi.size:
                    window_isis.append(isi)
                    pooled_isis.append(isi)
                    window_spike_count += int(spk_win.size)

        if window_isis:
            window_isis = np.concatenate(window_isis).astype(float)
            n_isi = int(window_isis.size)

            mean_isi = float(np.mean(window_isis)) if n_isi >= 1 else np.nan
            std_isi  = float(np.std(window_isis, ddof=0)) if n_isi >= 2 else np.nan

            per_window_rows.append({
                "Start_ms": w0_ms,
                "End_ms": w1_ms,
                "Mean_ISI_ms": mean_isi,
                "STD_ISI_ms": std_isi,
                "N_ISI": n_isi,
                "N_spikes": int(window_spike_count),
            })
        else:
            per_window_rows.append({
                "Start_ms": w0_ms,
                "End_ms": w1_ms,
                "Mean_ISI_ms": np.nan,
                "STD_ISI_ms": np.nan,
                "N_ISI": 0,
                "N_spikes": 0,
            })

    per_window_df = pd.DataFrame(per_window_rows)

    if pooled_isis:
        pooled_isis = np.concatenate(pooled_isis).astype(float)
        n_isi = int(pooled_isis.size)

        pooled_mean = float(np.mean(pooled_isis)) if n_isi >= 1 else np.nan
        pooled_std  = float(np.std(pooled_isis, ddof=0)) if n_isi >= 2 else np.nan

        pooled_stats = {"mean_ms": pooled_mean, "std_ms": pooled_std, "n_isi": n_isi}
    else:
        pooled_stats = {"mean_ms": np.nan, "std_ms": np.nan, "n_isi": 0}

    return per_window_df, pooled_stats
