# main.py
from __future__ import annotations
import argparse
import time
import re
from pathlib import Path
import numpy as np
import scipy.io as spio
import pandas as pd

from config import *
import config as cfg
import pipeline.detection as detection
import pipeline.utils as utils
from pipeline.emg import *
from pipeline.detection import *
from pipeline.utils import *
from pipeline.stats import *
from pipeline.coactivity import *
from pipeline.plotting import *
from pipeline.isi import *
from pipeline.kilosort_loader import parse_dataset_spec

REPO_ROOT = Path(__file__).resolve().parent.parent


def _datasets_from_indices(spec: str | None, entries: list) -> list:
    """
    If spec is set, treat it as comma-separated 0-based indices into ``entries``.
    Otherwise return a copy of the full list.
    """
    if not spec or not str(spec).strip():
        return list(entries)
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        n = len(entries)
        if n == 0:
            raise ValueError("SpikeTime_Mat_File is empty")
        if i < 0 or i >= n:
            raise ValueError(f"Dataset index {i} out of range (valid: 0–{n - 1})")
        out.append(entries[i])
    return out


def _datasets_from_contains(fragments: list[str] | None, entries: list) -> list:
    """
    Keep entries whose string form contains every fragment (AND). Stable when you add rows
    to config — use path snippets like ``m361`` + ``imec2`` instead of numeric indices.
    """
    if not fragments:
        return list(entries)
    out = []
    for e in entries:
        s = str(e)
        if all(f in s for f in fragments):
            out.append(e)
    if not out:
        raise ValueError(
            f"No SpikeTime_Mat_File entry contains all of {fragments!r} (substring match, AND)."
        )
    return out


def _filter_entries_by_kind(entries: list, kind: str | None) -> list:
    """Keep only entries whose dataset spec kind matches (kilosort vs kilosortset vs mat)."""
    if not kind:
        return list(entries)
    out = []
    for e in entries:
        k, _ = parse_dataset_spec(str(e))
        if k == kind:
            out.append(e)
    if not out:
        raise ValueError(
            f"No SpikeTime_Mat_File entries match --only-dataset-kind {kind!r} after other filters."
        )
    return out


def _resolve_proxy_csv(p: str | None) -> Path | None:
    """Resolve PROXY_CSV relative to repo root when needed."""
    if not p:
        return None
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return pp
    if pp.exists():
        return pp.resolve()
    cand = (REPO_ROOT / pp).resolve()
    return cand if cand.exists() else pp


def _load_bilateral_proxy_csv(
    csv_path: Path,
    *,
    time_col: str,
    left_col: str,
    right_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load bilateral proxy CSV columns used by the main raster/proxy workflow.
    Returns (time_s, left_vals, right_vals) after dropping rows with missing values.
    """
    df = pd.read_csv(csv_path)
    required = [time_col, left_col, right_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required proxy columns: {missing}")
    sub = df[required].dropna()
    if sub.empty:
        raise ValueError("Proxy CSV has no valid rows after dropping NaNs.")
    t_s = np.asarray(sub[time_col].to_numpy(), dtype=float)
    hemi_l = np.asarray(sub[left_col].to_numpy(), dtype=float)
    hemi_r = np.asarray(sub[right_col].to_numpy(), dtype=float)
    order = np.argsort(t_s)
    return t_s[order], hemi_l[order], hemi_r[order]


def _resolve_dataset_entry(spec: str) -> str:
    """Resolve mat paths relative to repo root; preserve Kilosort prefixes."""
    kind, path = parse_dataset_spec(spec)
    if kind == "kilosortset":
        parts = [p.strip() for p in path.split(";") if p.strip()]
        resolved_parts: list[str] = []
        for part in parts:
            p = Path(part)
            if p.is_absolute() and p.exists():
                resolved = p
            elif p.exists():
                resolved = p.resolve()
            else:
                cand = (REPO_ROOT / p).resolve()
                resolved = cand if cand.exists() else p
            resolved_parts.append(str(resolved))
        return "kilosortset:" + ";".join(resolved_parts)
    p = Path(path)
    if p.is_absolute() and p.exists():
        resolved = p
    elif p.exists():
        resolved = p.resolve()
    else:
        cand = (REPO_ROOT / p).resolve()
        resolved = cand if cand.exists() else p
    if kind == "kilosort":
        return f"kilosort:{resolved}"
    return str(resolved)


def _resolve_win_shuff_params_ms(record_len_s: float) -> tuple[float, float]:
    """
    Effective Stage-1 WIN-SHUFF params for one recording.

    - Fixed mode: use RS_WIN_SHUFF_WINDOW_MS / RS_WIN_SHUFF_BIN_MS.
    - Auto mode: derive from recording length (fractions), clamp to min/max bounds,
      then snap so window/bin is an integer count.
    """
    ws_ms = float(cfg.RS_WIN_SHUFF_WINDOW_MS)
    bin_ms = float(cfg.RS_WIN_SHUFF_BIN_MS)

    if bool(cfg.RS_WIN_SHUFF_STAGE1_ENABLE) and bool(getattr(cfg, "RS_WIN_SHUFF_AUTO_FROM_RECORDING", False)):
        rec_ms = max(float(record_len_s) * 1000.0, 1e-9)
        ws_ms = rec_ms * float(getattr(cfg, "RS_WIN_SHUFF_AUTO_WINDOW_FRACTION_RECORDING", 0.10))
        bin_ms = ws_ms * float(getattr(cfg, "RS_WIN_SHUFF_AUTO_BIN_FRACTION_OF_WINDOW", 0.10))

        ws_ms = float(np.clip(
            ws_ms,
            float(getattr(cfg, "RS_WIN_SHUFF_AUTO_WINDOW_MIN_MS", 50.0)),
            float(getattr(cfg, "RS_WIN_SHUFF_AUTO_WINDOW_MAX_MS", 1000.0)),
        ))
        bin_ms = float(np.clip(
            bin_ms,
            float(getattr(cfg, "RS_WIN_SHUFF_AUTO_BIN_MIN_MS", 1.0)),
            float(getattr(cfg, "RS_WIN_SHUFF_AUTO_BIN_MAX_MS", 50.0)),
        ))

    ws_ms = max(ws_ms, 1e-6)
    bin_ms = max(min(bin_ms, ws_ms), 1e-6)
    n_bins = max(1, int(round(ws_ms / bin_ms)))
    bin_ms = ws_ms / float(n_bins)
    return ws_ms, bin_ms


def run_single_dataset(
    mat_file: str,
    base_thr: float,
    coactivity_bins_s: float,
    EMG_MAT: str,
    NOTES_TXT: str,
):
    t0 = time.perf_counter()
    patient, period = dataset_labels_any(mat_file)

    from pipeline.kilosort_loader import load_kilosort_dir, save_spiketime_mat

    kind, raw_path = parse_dataset_spec(mat_file)
    if kind == "kilosort":
        spike_struct, _ks_meta = load_kilosort_dir(
            raw_path,
            good_only=KILOSORT_GOOD_ONLY,
            max_duration_s=KILOSORT_MAX_DURATION_S,

        )
        mat_file_for_cache = None
    elif kind == "kilosortset":
        from pipeline.kilosort_loader import load_kilosort_dirs
        ks_dirs = [p.strip() for p in raw_path.split(";") if p.strip()]
        spike_struct, _ks_meta = load_kilosort_dirs(
            ks_dirs,
            good_only=KILOSORT_GOOD_ONLY,
            max_duration_s=KILOSORT_MAX_DURATION_S,
        )
        mat_file_for_cache = None
    else:
        mat = spio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
        spike_struct = mat["spikeTime"]
        mat_file_for_cache = raw_path

    thr_method_name, thr_method_id = get_thresholding_method_name()
    record_len_s = compute_recording_duration_s(spike_struct)
    fs = float((np.ravel(spike_struct)[0]).dataSegmentLength)

    # Per-recording WIN-SHUFF parameter resolution (fixed vs auto-from-recording).
    if thr_method_name == "rankSurprise":
        ws_ms_eff, bin_ms_eff = _resolve_win_shuff_params_ms(record_len_s)
        for mod in (cfg, detection, utils):
            mod.RS_WIN_SHUFF_WINDOW_MS = float(ws_ms_eff)
            mod.RS_WIN_SHUFF_BIN_MS = float(bin_ms_eff)
        if bool(cfg.RS_WIN_SHUFF_STAGE1_ENABLE):
            if bool(getattr(cfg, "RS_WIN_SHUFF_AUTO_FROM_RECORDING", False)):
                print(
                    f"• WIN-SHUFF(auto): window={ws_ms_eff:g}ms, bin={bin_ms_eff:g}ms "
                    f"(record={record_len_s:.3f}s)"
                )
            else:
                print(f"• WIN-SHUFF(fixed): window={ws_ms_eff:g}ms, bin={bin_ms_eff:g}ms")

    # New output layout: one root per method.
    if thr_method_name == "rankSurprise":
        if kind in ("kilosort", "kilosortset"):
            method_root = Path(OUTPUT_ROOT_RS_MOUSE)
        else:
            method_root = Path(OUTPUT_ROOT_RS)
    elif thr_method_name == "maxISI":
        method_root = Path(OUTPUT_ROOT_MAXISI)
    elif thr_method_name == "alpha_meanISI":
        method_root = Path(OUTPUT_ROOT_ALPHA_MEANISI)
    else:
        method_root = Path(OUTPUT_ROOT_RS)
    # Windows paths cannot contain characters like '?'.
    period_dir = re.sub(r"[<>:\"/\\\\|?*]", "", period.replace(" ", ""))
    patient_dir = patient

    # Kilosort output layout override:
    # Legacy: .../<session>/<shankX>/<imecY>/kilosort4  (rat)
    # Flat:   .../<line>/<gate_folder>/<imec_folder>/      (mouse; KS files in imec_folder)
    #   <METHOD_ROOT>\<patient_dir>\<period_dir>\<run_tag>\...
    if kind in ("kilosort", "kilosortset"):
        def _sanitize_path_token(s: str) -> str:
            return re.sub(r"[<>:\"/\\\\|?*]", "", str(s))

        try:
            if kind == "kilosort":
                p0 = Path(raw_path)
            else:
                p0 = Path([p.strip() for p in raw_path.split(";") if p.strip()][0])
            parts = p0.parts
            if p0.name.lower() == "kilosort4" and len(parts) >= 4:
                session = parts[-4]
                shank = parts[-3]
                imec = parts[-2]
                patient_dir = _sanitize_path_token(session)
                if kind == "kilosort":
                    period_dir = f"{_sanitize_path_token(imec)}_{_sanitize_path_token(shank)}"
                else:
                    period_dir = f"{_sanitize_path_token(imec)}_allshanks"
            else:
                # Flat KS root (no trailing kilosort4 directory)
                if len(parts) >= 4:
                    patient_dir = _sanitize_path_token(parts[-3])
                elif len(parts) >= 2:
                    patient_dir = _sanitize_path_token(parts[-2])
                else:
                    patient_dir = "kilosort"
                gate = parts[-2] if len(parts) >= 2 else "gate"
                leaf = parts[-1] if len(parts) >= 1 else "imec"
                if kind == "kilosort":
                    period_dir = _sanitize_path_token(f"{gate}_{leaf}")
                else:
                    period_dir = f"{_sanitize_path_token(leaf)}_allshanks"
        except Exception:
            patient_dir = "kilosort"
            if kind == "kilosort":
                period_dir = "imec_shank"
            else:
                period_dir = "imec_allshanks"

    def _dataset_tag_for_gallery(kind: str, raw_path: str, patient: str) -> str:
        # Prefer existing patient label when it already looks like m360_shank2_imec0.
        if patient and "shank" in patient and "imec" in patient:
            return patient
        if kind == "kilosort":
            return patient
        if kind == "kilosortset":
            # Match outputs folder name from dataset_labels_any (incl. _separated when grouped).
            return patient
        # Fallback: sanitize patient string
        return re.sub(r"[<>:\"/\\\\|?*]", "_", patient or "dataset")

    run_params = build_run_params(thr_method_name, base_thr)
    if kind in ("kilosort", "kilosortset"):
        # Persist any Kilosort time clipping into naming/metadata.
        run_params["max_duration_s"] = KILOSORT_MAX_DURATION_S
        run_tag = run_tag_from_params_kilosort(
            run_params,
            good_only=bool(KILOSORT_GOOD_ONLY),
            sep_shank=bool(KILOSORT_RASTER_GROUP_BY_SHANK),
        )
    else:
        run_tag = run_tag_from_params(run_params)
    max_dur_s = KILOSORT_MAX_DURATION_S
    if max_dur_s is not None:
        run_tag = f"{run_tag}_[0-{max_dur_s}s]"
    run_dir = method_root / patient_dir / period_dir / run_tag
    isi_dir = run_dir / "isi_summary"
    raster_dir = run_dir / "raster_plots"
    for d in (run_dir, isi_dir, raster_dir):
        d.mkdir(parents=True, exist_ok=True)
    save_run_params(run_params, run_dir)

    # Optional: export Kilosort loads to MATLAB spikeTime .mat for inspection/reuse.
    if kind in ("kilosort", "kilosortset") and bool(KILOSORT_WRITE_SPIKETIME_MAT):
        out_mat = run_dir / "spikeTime_from_kilosort.mat"
        try:
            save_spiketime_mat(out_mat, spike_struct)
            print(f"• Wrote spikeTime .mat → {out_mat}")
        except Exception as e:
            print(f"• spikeTime .mat export failed ({out_mat}): {e}")

    print(f"\n• Loading {patient} • {period}")
    print(f"• Recording duration: {record_len_s:.3f} s")

    # Build per-cluster stats
    STATS_indiv, STATS_POOLED = build_cache(
        mat_file=mat_file_for_cache,
        cluster_stats_csv=run_dir / "cluster_stats.csv",
        pooled_cluster_stats_csv=run_dir / "pooled_cluster_stats.csv",
        base_thr=base_thr,
        record_len_s=record_len_s,
        adaptie_thr_toggle=alpha_meanisi_toggle,
        pooling_toggle=pooling_toggle,
        spike_struct=spike_struct if kind in ("kilosort", "kilosortset") else None,
        skip_snr_filter=(kind in ("kilosort", "kilosortset") and KILOSORT_SKIP_PIPELINE_SNR_FILTER),
    )

    if pooling_toggle:
        spike_struct = pool_spike_struct_per_channel(spike_struct, record_len_s)
        STATS = STATS_POOLED
    else:
        STATS = STATS_indiv
    allowed_clusters = set(STATS.index)

    spike_struct_L, spike_struct_R = split_spike_struct_by_side(spike_struct)

    df_spikes = build_spike_labels_df(spike_struct, STATS, allowed_clusters, ibi_merge_factor)
    df_spikes.to_csv(run_dir / "indiv_spike_labels.csv", index=False)

    # --------------------------------------------------
    # Burst detection
    # --------------------------------------------------
    bd = rs_burst_detection(
        spike_struct, spike_struct_L, spike_struct_R,
        STATS, run_dir,
        iter_units_fn=iter_units_from_stats,
        infer_region_fn=infer_region,
        burst_in_window_fn=burst_in_window,
    )
    all_bursts = bd.all_bursts
    all_bursts_L = bd.all_bursts_L
    all_bursts_R = bd.all_bursts_R
    stage1_unit_bursts_L = bd.stage1_unit_bursts_L
    stage1_unit_bursts_R = bd.stage1_unit_bursts_R
    stage2_region_bursts_L = bd.stage2_region_bursts_L
    stage2_region_bursts_R = bd.stage2_region_bursts_R
    stage3_network_bursts_L = bd.stage3_network_bursts_L
    stage3_network_bursts_R = bd.stage3_network_bursts_R
    network_windows_L = bd.network_windows_L
    network_windows_R = bd.network_windows_R
    network_bars_L = bd.network_bars_L
    network_bars_R = bd.network_bars_R
    region_windows_L = bd.region_windows_L
    region_windows_R = bd.region_windows_R
    region_bars_by_region_L = bd.region_bars_by_region_L
    region_bars_by_region_R = bd.region_bars_by_region_R

    # --------------------------------------------------
    # EMG (for Sanger presentation figure)
    # --------------------------------------------------
    t_emg = None
    emg_traces_L = None
    emg_traces_R = None
    emg_traces_LR = None
    event_times = None
    event_names = None

    if PLOT_EMG and EMG_MAT is not None:
        IDX_R = [0, 1, 2, 3]
        IDX_L = [4, 5, 6, 7]
        IDX_LR = IDX_R + IDX_L

        emg_fs, emg_data, emg_segment_length = load_emg_raw(EMG_MAT)
        emg_processed, selected_channel_names = preprocess_emg_exact(emg_data, fs_for_filter=fs)
        if NOTES_TXT is not None:
            event_times, event_names = parse_notes_events(NOTES_TXT)
        t_emg = make_emg_timebase(emg_segment_length, emg_processed.shape[1])

        mask = t_emg <= record_len_s
        t_emg = t_emg[mask]

        emg_traces_L = build_emg_traces(
            IDX_L, "L",
            emg_processed=emg_processed,
            selected_channel_names=selected_channel_names,
            mask=mask,
        )
        emg_traces_R = build_emg_traces(
            IDX_R, "R",
            emg_processed=emg_processed,
            selected_channel_names=selected_channel_names,
            mask=mask,
        )
        emg_traces_LR = build_emg_traces(
            IDX_LR, "LR",
            emg_processed=emg_processed,
            selected_channel_names=selected_channel_names,
            mask=mask,
        )

        if emg_traces_L is not None:
            emg_traces_L = collapse_emg_traces(emg_traces_L, mode="sum", label="EMG summed L")
        if emg_traces_R is not None:
            emg_traces_R = collapse_emg_traces(emg_traces_R, mode="sum", label="EMG summed R")
        if emg_traces_LR is not None:
            emg_traces_LR = collapse_emg_traces(emg_traces_LR, mode="sum", label="EMG summed")

    # --------------------------------------------------
    # Proxy (single CSV: time_s + left + right columns; see config PROXY_*)
    # --------------------------------------------------
    t_proxy = None
    proxy_traces_L = None
    proxy_traces_R = None
    proxy_y_range = None
    proxy_t_s_fr = None
    proxy_L_fr = None
    proxy_R_fr = None

    if kind != "kilosort" and (PLOT_PROXY_PANEL or COMPUTE_PROXY_VS_FR):
        proxy_csv_resolved = _resolve_proxy_csv(PROXY_CSV)
        traces_L: list[tuple[str, np.ndarray]] = []
        traces_R: list[tuple[str, np.ndarray]] = []
        t_proxy_s = None

        if PROXY_CSV and proxy_csv_resolved is not None and Path(proxy_csv_resolved).is_file():
            try:
                t_s, hemi_L, hemi_R = _load_bilateral_proxy_csv(
                    Path(proxy_csv_resolved),
                    time_col=PROXY_TIME_COL,
                    left_col=PROXY_LEFT_COL,
                    right_col=PROXY_RIGHT_COL,
                )
                t_proxy_s = np.asarray(t_s, dtype=float)
                traces_L.append(("Proxy hemi L", hemi_L))
                traces_R.append(("Proxy hemi R", hemi_R))
            except Exception as e:
                print(f"• Proxy CSV load failed ({proxy_csv_resolved}): {e}")
        elif PROXY_CSV:
            print(f"• Proxy CSV not found: {proxy_csv_resolved or PROXY_CSV}, skipping proxy")

        if t_proxy_s is not None and (traces_L or traces_R):
            t_proxy_s = np.asarray(t_proxy_s, dtype=float)
            # Clip to recording duration so proxy aligns with raster
            mask = (t_proxy_s >= 0) & (t_proxy_s <= record_len_s)
            t_proxy = t_proxy_s[mask]

            def _clip_traces(traces, mask):
                out: list[tuple[str, np.ndarray]] = []
                for name, y in traces:
                    yy = np.asarray(y, dtype=float)
                    if yy.size == mask.size:
                        out.append((name, yy[mask]))
                return out

            clipped_L = _clip_traces(traces_L, mask)
            clipped_R = _clip_traces(traces_R, mask)

            proxy_t_s_fr = np.asarray(t_proxy, dtype=float)
            if clipped_L:
                proxy_L_fr = np.asarray(clipped_L[0][1], dtype=float)
            if clipped_R:
                proxy_R_fr = np.asarray(clipped_R[0][1], dtype=float)

            # Convert to p-units (values are very small, e.g. ~1e-11)
            # Scale by 1e12 so axis is in pico-units (p).
            SCALE_TO_P = 1e12

            def _scale_to_p(traces):
                out: list[tuple[str, np.ndarray]] = []
                for name, y in traces:
                    yy = np.asarray(y, dtype=float) * SCALE_TO_P
                    out.append((name, yy))
                return out

            if clipped_L:
                proxy_traces_L = collapse_emg_traces(
                    _scale_to_p(clipped_L), mode="mean", label="Proxy hemi L [p]"
                )
            if clipped_R:
                proxy_traces_R = collapse_emg_traces(
                    _scale_to_p(clipped_R), mode="mean", label="Proxy hemi R [p]"
                )

            # Y-axis range: min → max of the DOWNSAMPLED data (matches what's actually plotted)
            PROXY_DOWNSAMPLE = 22
            min_val = None
            max_val = None
            for _, y in (proxy_traces_L or []) + (proxy_traces_R or []):
                v = np.asarray(y, dtype=float)[::PROXY_DOWNSAMPLE]
                v = v[np.isfinite(v)]
                if v.size > 0:
                    lo = float(v.min())
                    hi = float(v.max())
                    min_val = lo if min_val is None else min(min_val, lo)
                    max_val = hi if max_val is None else max(max_val, hi)
            if max_val is not None and min_val is not None and max_val > min_val:
                proxy_y_range = (min_val, max_val)
        elif PLOT_PROXY_PANEL and PROXY_CSV:
            print(
                "• Proxy panel: expected PROXY_CSV data but none in recording window or load failed "
                "(check path, PROXY_TIME_COL / LEFT / RIGHT, and time_s within [0, recording length])."
            )

    elif kind == "kilosort" and (PLOT_PROXY_PANEL or COMPUTE_PROXY_VS_FR):
        print("• Kilosort: skipping hemispheric proxy CSV (PLOT_PROXY_PANEL / COMPUTE_PROXY_VS_FR not used).")

    # --------------------------------------------------
    # Co-activity & firing rate
    # --------------------------------------------------
    stride_s = max(coactivity_bins_s * (1.0 - OVERLAP_FRACTION), 1e-9)
    if PLOT_RASTER:
        co_channels = burst_coactivity_by_channel(all_bursts, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)
        co_channels_L = burst_coactivity_by_channel(all_bursts_L, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)
        co_channels_R = burst_coactivity_by_channel(all_bursts_R, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)

        co_clusters = burst_coactivity_by_cluster(all_bursts, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)
        
        co_regions = burst_coactivity_by_region(all_bursts, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)
        co_regions_L = burst_coactivity_by_region(all_bursts_L, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)
        co_regions_R = burst_coactivity_by_region(all_bursts_R, record_len_s, win_s=coactivity_bins_s, stride_s=stride_s)

    fr_df_L = population_firing_rate_binned(
        spike_struct_L, STATS, record_len_s,
        bin_s=coactivity_bins_s, stride_s=stride_s,
        allowed_clusters=allowed_clusters,
        normalize_by_units=True,
    )
    fr_df_R = population_firing_rate_binned(
        spike_struct_R, STATS, record_len_s,
        bin_s=coactivity_bins_s, stride_s=stride_s,
        allowed_clusters=allowed_clusters,
        normalize_by_units=True,
    )

    # --------------------------------------------------
    # Proxy vs firing-rate correlation study (hemi only; single bilateral CSV)
    # --------------------------------------------------
    if COMPUTE_PROXY_VS_FR and proxy_t_s_fr is not None:
        def _resample_to_bins(t_proxy_s, proxy_vals, bin_starts, bin_width):
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

        t_s_px = np.asarray(proxy_t_s_fr, dtype=float)

        for side_tag, fr_df_side, hemi_px in [
            ("left", fr_df_L, proxy_L_fr),
            ("right", fr_df_R, proxy_R_fr),
        ]:
            if hemi_px is None or hemi_px.size == 0:
                continue
            if fr_df_side is None or len(fr_df_side) == 0:
                continue

            fr_t = fr_df_side["Window_start_s"].to_numpy(float)
            fr_y = fr_df_side["FR_Hz"].to_numpy(float)

            hemi_resampled = _resample_to_bins(t_s_px, hemi_px, fr_t, coactivity_bins_s)
            valid = np.isfinite(hemi_resampled) & np.isfinite(fr_y)
            hemi_corr = float(np.corrcoef(fr_y[valid], hemi_resampled[valid])[0, 1]) if valid.sum() > 2 else np.nan

            rows = [{"patient": patient, "period": period, "side": side_tag,
                     "level": "hemi", "region": "all", "correlation": hemi_corr}]

            csv_path = run_dir / f"proxy_vs_fr_{side_tag}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  Proxy vs FR ({side_tag}): {len(rows)} rows → {csv_path}")

    # --------------------------------------------------
    # Main presentation figures
    #   1) SangerLab presentation (raster + EMG + FR)
    #   2) Correlation graph (raster + proxy [p] + FR)
    # --------------------------------------------------

    # 1) Classic SangerLab presentation figure
    # Bottom strip: EMG if PLOT_EMG and loaded; else proxy if PLOT_PROXY_PANEL and CSV loaded (same as config).
    if PLOT_SANGER_PRESENTATION_FIG:
        emg_downsample = 100  # EMG is high-rate
        proxy_downsample = 22  # proxy CSV is ~20 ms bins

        use_emg_bottom = bool(
            PLOT_EMG
            and t_emg is not None
            and len(t_emg) > 0
            and emg_traces_L
            and emg_traces_R
        )
        use_proxy_bottom = bool(
            not use_emg_bottom
            and PLOT_PROXY_PANEL
            and t_proxy is not None
            and len(t_proxy) > 0
            and proxy_traces_L
            and proxy_traces_R
        )

        if use_emg_bottom:
            bottom_t = t_emg
            bottom_L, bottom_R = emg_traces_L, emg_traces_R
            bottom_ds = emg_downsample
            bottom_labels = None
            bottom_y_range = None
        elif use_proxy_bottom:
            bottom_t = t_proxy
            bottom_L, bottom_R = proxy_traces_L, proxy_traces_R
            bottom_ds = proxy_downsample
            bottom_labels = ("Hemi proxy (L)", "Hemi proxy (R)")
            bottom_y_range = proxy_y_range
        else:
            bottom_t = None
            bottom_L, bottom_R = None, None
            bottom_ds = emg_downsample
            bottom_labels = None
            bottom_y_range = None

        def _make_and_save_stage_figure(
            *,
            name: str,
            bursts_L,
            bursts_R,
            net_bars_L,
            net_bars_R,
            reg_bars_L,
            reg_bars_R,
            duplicate_for_gallery: bool = False,
            suppress_burst_visuals: bool = False,
        ):
            if suppress_burst_visuals:
                fig_nw_L = None
                fig_nw_R = None
                fig_rw_L = None
                fig_rw_R = None
                fig_disable_bursts = True
            else:
                fig_nw_L = network_windows_L
                fig_nw_R = network_windows_R
                fig_rw_L = region_windows_L
                fig_rw_R = region_windows_R
                fig_disable_bursts = disable_bursts
            fig = make_sangerlab_presentation_figure(
                spike_struct_L=spike_struct_L,
                spike_struct_R=spike_struct_R,
                STATS=STATS,
                all_bursts_L=bursts_L,
                all_bursts_R=bursts_R,
                record_len_s=record_len_s,
                patient=patient,
                period=period,
                bin_s=coactivity_bins_s,
                stride_s=stride_s,
                run_params=run_params,
                presentation_mode=presentation_mode,
                allowed_clusters=allowed_clusters,
                emg_t_s=bottom_t,
                emg_traces_L=bottom_L,
                emg_traces_R=bottom_R,
                emg_downsample=bottom_ds,
                emg_panel_labels=bottom_labels,
                emg_y_range=bottom_y_range,
                network_windows_L=fig_nw_L,
                network_windows_R=fig_nw_R,
                network_bars_L=net_bars_L,
                network_bars_R=net_bars_R,
                region_windows_by_region_L=fig_rw_L,
                region_windows_by_region_R=fig_rw_R,
                region_bars_by_region_L=reg_bars_L,
                region_bars_by_region_R=reg_bars_R,
                disable_bursts=fig_disable_bursts,
                fr_df_L=fr_df_L,
                fr_df_R=fr_df_R,
                show_firing_rate=PLOT_FR,
                compact_kilosort_title=(kind in ("kilosort", "kilosortset")),
            )
            save_fig_interactive(fig, raster_dir / name)
            if duplicate_for_gallery and DUPLICATE_SANGER_HTML and DUPLICATE_SANGER_HTML_DIR:
                tag = _dataset_tag_for_gallery(kind, raw_path, patient)
                out_base = Path(DUPLICATE_SANGER_HTML_DIR) / tag
                save_fig_interactive(fig, out_base)

        _make_and_save_stage_figure(
            name="stage0_raw",
            bursts_L=[],
            bursts_R=[],
            net_bars_L=[],
            net_bars_R=[],
            reg_bars_L={},
            reg_bars_R={},
            duplicate_for_gallery=False,
            suppress_burst_visuals=True,
        )

        # Stage 1: unit-level RS bursts only (no region/network bars).
        _make_and_save_stage_figure(
            name="stage1_unit_burst",
            bursts_L=stage1_unit_bursts_L,
            bursts_R=stage1_unit_bursts_R,
            net_bars_L=[],
            net_bars_R=[],
            reg_bars_L={},
            reg_bars_R={},
            duplicate_for_gallery=False,
        )

        # Stage 2: region-filtered bursts with region bars only.
        _make_and_save_stage_figure(
            name="stage2_region_burst",
            bursts_L=stage2_region_bursts_L,
            bursts_R=stage2_region_bursts_R,
            net_bars_L=[],
            net_bars_R=[],
            reg_bars_L=region_bars_by_region_L,
            reg_bars_R=region_bars_by_region_R,
            duplicate_for_gallery=False,
        )

        # Stage 3: network-filtered bursts with network bars only.
        _make_and_save_stage_figure(
            name="stage3_network_burst",
            bursts_L=stage3_network_bursts_L,
            bursts_R=stage3_network_bursts_R,
            net_bars_L=network_bars_L,
            net_bars_R=network_bars_R,
            reg_bars_L={},
            reg_bars_R={},
            duplicate_for_gallery=False,
        )

        # Overview: restored full-context plot with region + network bars.
        _make_and_save_stage_figure(
            name="overview_region_burst",
            bursts_L=all_bursts_L,
            bursts_R=all_bursts_R,
            net_bars_L=network_bars_L,
            net_bars_R=network_bars_R,
            reg_bars_L=region_bars_by_region_L,
            reg_bars_R=region_bars_by_region_R,
            duplicate_for_gallery=True,
        )

    # 2) Correlation graph: raster + hemi proxy (p-units) + FR
    if PLOT_CORRELATION_GRAPH and t_proxy is not None and proxy_traces_L is not None and proxy_traces_R is not None:
        proxy_downsample = 22  # Proxy is 20 ms bins (~50 pts/s)
        fig_corr = make_sangerlab_presentation_figure(
            spike_struct_L=spike_struct_L,
            spike_struct_R=spike_struct_R,
            STATS=STATS,
            all_bursts_L=all_bursts_L,
            all_bursts_R=all_bursts_R,
            record_len_s=record_len_s,
            patient=patient,
            period=period,
            bin_s=coactivity_bins_s,
            stride_s=stride_s,
            run_params=run_params,
            presentation_mode=presentation_mode,
            allowed_clusters=allowed_clusters,
            emg_t_s=t_proxy,
            emg_traces_L=proxy_traces_L,
            emg_traces_R=proxy_traces_R,
            emg_downsample=proxy_downsample,
            emg_panel_labels=("Hemi proxy (L)", "Hemi proxy (R)"),
            emg_y_range=proxy_y_range,
            network_windows_L=network_windows_L,
            network_windows_R=network_windows_R,
            network_bars_L=network_bars_L,
            network_bars_R=network_bars_R,
            region_windows_by_region_L=region_windows_L,
            region_windows_by_region_R=region_windows_R,
            region_bars_by_region_L=region_bars_by_region_L,
            region_bars_by_region_R=region_bars_by_region_R,
            disable_bursts=disable_bursts,
            fr_df_L=fr_df_L,
            fr_df_R=fr_df_R,
            show_firing_rate=PLOT_FR,
            compact_kilosort_title=(kind in ("kilosort", "kilosortset")),
        )
        save_fig_interactive(fig_corr, raster_dir / "correlation_graph")

    # # --------------------------------------------------
    # # Network burst ISI stats
    # # --------------------------------------------------
    # isi_df_L, isi_stats_L = compute_network_burst_isi_stats(
    #     spike_struct_L, network_windows_L, STATS, allowed_clusters=allowed_clusters
    # )
    # isi_df_R, isi_stats_R = compute_network_burst_isi_stats(
    #     spike_struct_R, network_windows_R, STATS, allowed_clusters=allowed_clusters
    # )

    # isi_df_L.to_csv(f"{patient}_{period}_ISI_windows_LEFT.csv", index=False)
    # isi_df_R.to_csv(f"{patient}_{period}_ISI_windows_RIGHT.csv", index=False)

    # pd.DataFrame([isi_stats_L]).to_csv(f"{patient}_{period}_ISI_summary_LEFT.csv", index=False)
    # pd.DataFrame([isi_stats_R]).to_csv(f"{patient}_{period}_ISI_summary_RIGHT.csv", index=False)

    elapsed_s = time.perf_counter() - t0
    print(f"• Finished {patient} • {period} in {elapsed_s/60.0:.2f} min ({elapsed_s:.1f} s)")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Burst detection pipeline. Select a subset with --only-datasets (indices) "
        "or --only-dataset-contains (path substrings). Optionally restrict by dataset kind."
    )
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument(
        "--only-datasets",
        type=str,
        default=None,
        metavar="INDICES",
        help="Comma-separated 0-based indices into SpikeTime_Mat_File in config.py (e.g. 0 or 0,2).",
    )
    sel.add_argument(
        "--only-dataset-contains",
        action="append",
        metavar="FRAG",
        dest="only_dataset_contains",
        help="Substring that must appear in the dataset entry string (repeat for AND). "
        "Example: --only-dataset-contains m361 --only-dataset-contains imec2",
    )
    parser.add_argument(
        "--only-dataset-kind",
        type=str,
        default=None,
        choices=("mat", "kilosort", "kilosortset"),
        metavar="KIND",
        help="After index/contains selection, keep only entries of this kind "
        "(mat path, kilosort:, or kilosortset:).",
    )
    args = parser.parse_args()

    if sum([maxisi_toggle, alpha_meanisi_toggle, rankSurprise_toggle]) != 1:
        raise ValueError("Select a SINGLE threshold")

    if args.only_dataset_contains:
        dataset_entries = _datasets_from_contains(args.only_dataset_contains, SpikeTime_Mat_File)
        print(
            f"• --only-dataset-contains (AND) {args.only_dataset_contains!r} "
            f"→ {len(dataset_entries)} run(s)",
            flush=True,
        )
    else:
        dataset_entries = _datasets_from_indices(args.only_datasets, SpikeTime_Mat_File)
        if args.only_datasets:
            print(
                f"• --only-datasets {args.only_datasets!r} → {len(dataset_entries)} run(s) "
                f"(indices in config SpikeTime_Mat_File)",
                flush=True,
            )

    if args.only_dataset_kind:
        dataset_entries = _filter_entries_by_kind(dataset_entries, args.only_dataset_kind)
        print(
            f"• --only-dataset-kind {args.only_dataset_kind!r} "
            f"→ {len(dataset_entries)} run(s)",
            flush=True,
        )

    t_all0 = time.perf_counter()

    if bool(getattr(cfg, "RS_WIN_SHUFF_AUTO_FROM_RECORDING", False)):
        # Auto mode resolves per recording in run_single_dataset; keep a single pass here.
        win_shuff_param_sets = [(float(RS_WIN_SHUFF_WINDOW_MS), float(RS_WIN_SHUFF_BIN_MS))]
    else:
        win_shuff_param_sets = list(getattr(cfg, "WIN_SHUFF_PARAM_SETS", []))
        if not win_shuff_param_sets:
            win_shuff_param_sets = [(float(RS_WIN_SHUFF_WINDOW_MS), float(RS_WIN_SHUFF_BIN_MS))]

    for a_stage1, a_region, a_network in RS_ALPHA_SETS:
        print(
            f"\n=== RS alphas: stage1={a_stage1}, region={a_region}, network={a_network} ===",
            flush=True,
        )
        for ws_ms, bin_ms in win_shuff_param_sets:
            ws_ms = float(ws_ms)
            bin_ms = float(bin_ms)
            if RS_WIN_SHUFF_STAGE1_ENABLE:
                if bool(getattr(cfg, "RS_WIN_SHUFF_AUTO_FROM_RECORDING", False)):
                    print("=== WIN-SHUFF: auto from recording length ===", flush=True)
                else:
                    print(
                        f"=== WIN-SHUFF: window={ws_ms:g}ms, bin={bin_ms:g}ms ===",
                        flush=True,
                    )

            for mod in (cfg, detection, utils):
                mod.RS_alpha_percentage_stage1 = a_stage1
                mod.RS_alpha_stage1 = -np.log(a_stage1)
                mod.RS_alpha_percentage_region = a_region
                mod.RS_alpha_region = -np.log(a_region)
                mod.RS_alpha_percentage_network = a_network
                mod.RS_alpha_network = -np.log(a_network)
                mod.RS_WIN_SHUFF_WINDOW_MS = ws_ms
                mod.RS_WIN_SHUFF_BIN_MS = bin_ms

            for mat_file_raw in dataset_entries:
                mat_file = _resolve_dataset_entry(mat_file_raw)
                patient, period = dataset_labels_any(mat_file)

                print(f"\nRunning {patient}, {period}", flush=True)
                print("Spike :", mat_file, flush=True)

                EMG_MAT = None
                NOTES_TXT = None
                if PLOT_EMG and parse_dataset_spec(mat_file)[0] != "kilosort":
                    emg_candidate, notes_candidate = derive_emg_notes_from_spiketime(
                        parse_dataset_spec(mat_file)[1]
                    )
                    if Path(emg_candidate).exists():
                        EMG_MAT = emg_candidate
                        print("EMG   :", EMG_MAT)
                    else:
                        print("EMG   : not found, skipping")
                    if Path(notes_candidate).exists():
                        NOTES_TXT = notes_candidate
                        print("Notes :", NOTES_TXT)
                    else:
                        print("Notes : not found, skipping")

                for base_thr in maxISI_thresholds:
                    for bin_s in coactivity_bins_s:
                        run_single_dataset(
                            mat_file=mat_file,
                            base_thr=base_thr,
                            coactivity_bins_s=bin_s,
                            EMG_MAT=EMG_MAT,
                            NOTES_TXT=NOTES_TXT,
                        )

    elapsed_all_s = time.perf_counter() - t_all0
    print(
        f"\n=== Total runtime: {elapsed_all_s/60.0:.2f} min ({elapsed_all_s:.1f} s) ===",
        flush=True,
    )
