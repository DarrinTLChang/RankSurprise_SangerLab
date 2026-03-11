# main.py
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import scipy.io as spio
import pandas as pd

from config import *
import config as cfg
import pipeline.detection as detection
import pipeline.region_exclusion as region_exclusion
import pipeline.utils as utils
from pipeline.emg import *
from pipeline.detection import *
from pipeline.utils import *
from pipeline.stats import *
from pipeline.coactivity import *
from pipeline.plotting import *
from pipeline.isi import *
from pipeline.region_exclusion import (
    load_gt_spans,
    load_gt_onset_starts,
    iter_permutation_dirs,
    run_network_with_excluded_regions,
    get_permutation_data_for_plotting,
    mean_iou_vs_gt,
    mean_onset_start_error_ms,
    mean_onset_start_error_pred_to_gt_ms,
)


def run_single_dataset(
    mat_file: str,
    base_thr: float,
    coactivity_bins_s: float,
    EMG_MAT: str,
    NOTES_TXT: str,
):
    patient, period = dataset_labels(mat_file)

    mat = spio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    spike_struct = mat["spikeTime"]
    thr_method_name, thr_method_id = get_thresholding_method_name()
    record_len_s = compute_recording_duration_s(spike_struct)
    fs = float((np.ravel(spike_struct)[0]).dataSegmentLength)

    if NETWORK_SPAN_ONSET_PLUS_LENGTH:
        # New outputs (onset+length span logic) written to external drive
        base_out = Path("/Volumes/D_Drive/SangerLabBursts")
        outputs_root = base_out / "outputs"
        output_RS_burst_root = base_out / "outputs_RS_burst"
        output_region_exclusion_root = base_out / "outputs_region_exclusion"
    else:
        outputs_root = Path("outputs")
        output_RS_burst_root = Path("outputs_RS_burst")
        output_region_exclusion_root = Path("outputs_region_exclusion")
    period_dir = period.replace(" ", "")
    patient_dir = patient

    out_root = outputs_root / patient_dir / period_dir
    meth_dir = out_root / thr_method_name
    
    run_params = build_run_params(thr_method_name, base_thr)
    run_tag = run_tag_from_params(run_params)
    run_dir = meth_dir / run_tag
    output_RS_burst_dir = output_RS_burst_root / patient_dir / period_dir/ thr_method_name / run_tag
    isi_dir = run_dir / "isi_summary"
    raster_dir = run_dir / "raster_plots"
    for d in (run_dir, isi_dir, raster_dir, output_RS_burst_dir):
        d.mkdir(parents=True, exist_ok=True)
    save_run_params(run_params, run_dir)

    print(f"\n• Loading {patient} • {period}")
    print(f"• Recording duration: {record_len_s:.3f} s")

    # Build per-cluster stats
    STATS_indiv, STATS_POOLED = build_cache(
        mat_file=mat_file,
        cluster_stats_csv=run_dir / "cluster_stats.csv",
        pooled_cluster_stats_csv=run_dir / "pooled_cluster_stats.csv",
        base_thr=base_thr,
        record_len_s=record_len_s,
        adaptie_thr_toggle=alpha_meanisi_toggle,
        pooling_toggle=pooling_toggle
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
        STATS, output_RS_burst_dir,
        iter_units_fn=iter_units_from_stats,
        infer_region_fn=infer_region,
        burst_in_window_fn=burst_in_window,
    )
    all_bursts = bd.all_bursts
    all_bursts_L = bd.all_bursts_L
    all_bursts_R = bd.all_bursts_R
    network_windows_L = bd.network_windows_L
    network_windows_R = bd.network_windows_R
    network_bars_L = bd.network_bars_L
    network_bars_R = bd.network_bars_R
    region_windows_L = bd.region_windows_L
    region_windows_R = bd.region_windows_R
    region_bars_by_region_L = bd.region_bars_by_region_L
    region_bars_by_region_R = bd.region_bars_by_region_R

    regions = None  # set by region exclusion study when enabled; used for permutation rasters

    # --------------------------------------------------
    # Region exclusion study (leave-K-out: CSVs per permutation + results)
    # --------------------------------------------------
    if RUN_REGION_EXCLUSION_STUDY and RS_NETWORK_ONSETS_TOGGLE:
        regions = sorted(
            set(infer_region(b["Electrode"]) for b in all_bursts_L)
            | set(infer_region(b["Electrode"]) for b in all_bursts_R)
        )
        if not regions:
            print("• Region exclusion study: no regions found, skipping")
        else:
            bursts_left = [dict(b, Region=infer_region(b["Electrode"])) for b in all_bursts_L]
            bursts_right = [dict(b, Region=infer_region(b["Electrode"])) for b in all_bursts_R]
            gt_left, gt_right = load_gt_spans(output_RS_burst_dir)
            gt_onset_left, gt_onset_right = load_gt_onset_starts(output_RS_burst_dir)
            # Own root mirroring outputs_RS_burst: outputs_region_exclusion / patient / period / method / run_tag
            study_base = output_region_exclusion_root / patient_dir / period_dir / thr_method_name / run_tag
            study_base.mkdir(parents=True, exist_ok=True)
            rows = []
            for included_list, perm_dir in iter_permutation_dirs(study_base, regions, out_subdir=""):
                excluded_set = set(regions) - set(included_list)
                included_set = set(included_list)
                included_str = "|".join(included_list)
                perm_dir.mkdir(parents=True, exist_ok=True)
                # Copy unit burst CSVs
                for fname in ("unit_bursts_RS_left.csv", "unit_bursts_RS_right.csv"):
                    src = output_RS_burst_dir / fname
                    if src.exists():
                        shutil.copy2(src, perm_dir / fname)
                # Write region_bursts filtered to included regions
                for side in ("left", "right"):
                    path = output_RS_burst_dir / f"region_bursts_RS_{side}.csv"
                    if path.exists():
                        df_reg = pd.read_csv(path)
                        if "Region" in df_reg.columns:
                            df_reg = df_reg[df_reg["Region"].astype(str).isin(included_set)]
                        df_reg.to_csv(perm_dir / f"region_bursts_RS_{side}.csv", index=False)
                # Network detection and CSVs per side; collect IoU (append by side for reorder later)
                for side, gt_spans, gt_onset_starts, bursts in [
                    ("left", gt_left, gt_onset_left, bursts_left),
                    ("right", gt_right, gt_onset_right, bursts_right),
                ]:
                    network_rows, pred_spans, _ = run_network_with_excluded_regions(
                        bursts, excluded_set
                    )
                    pd.DataFrame(network_rows).to_csv(
                        perm_dir / f"network_bursts_RS_{side}.csv", index=False
                    )
                    mean_iou, n_gt_matched, n_pred_matched = mean_iou_vs_gt(gt_spans, pred_spans)
                    pred_onset_starts = [r["onset_start_ms"] for r in network_rows]
                    mean_onset_err_ms = mean_onset_start_error_ms(gt_onset_starts, pred_onset_starts)
                    mean_onset_err_pred_to_gt_ms = mean_onset_start_error_pred_to_gt_ms(gt_onset_starts, pred_onset_starts)
                    K_excluded = len(excluded_set)
                    n_gt = len(gt_spans)
                    n_pred = len(pred_spans)
                    pct_gt_matched = (100.0 * n_gt_matched / n_gt) if n_gt else float("nan")
                    pct_pred_matched = (100.0 * n_pred_matched / n_pred) if n_pred else float("nan")
                    rows.append({
                        "Side": side,
                        "K_excluded": K_excluded,
                        "Regions_included": included_str,
                        "N_gt_spans": n_gt,
                        "N_pred_spans": n_pred,
                        "Mean_IoU": mean_iou,
                        "N_gt_matched": n_gt_matched,
                        "Pct_GT_matched": pct_gt_matched,
                        "N_pred_matched": n_pred_matched,
                        "Pct_pred_matched": pct_pred_matched,
                        "Mean_onset_start_error_ms": mean_onset_err_ms,
                        "Mean_onset_start_error_pred_to_gt_ms": mean_onset_err_pred_to_gt_ms,
                    })
            if rows:
                df_out = pd.DataFrame(rows)
                # All Left rows first, then all Right (sort by Side: left < right)
                df_out = df_out.sort_values("Side", kind="stable")
                df_out.to_csv(study_base / "results.csv", index=False)
                print(f"• Region exclusion study: wrote {len(rows)} rows → {study_base / 'results.csv'}")

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
    # Proxy (for correlation graph; neo/p-units)
    # --------------------------------------------------
    t_proxy = None
    proxy_traces_L = None
    proxy_traces_R = None
    proxy_y_range = None
    proxy_path_L = None
    proxy_path_R = None

    if PLOT_PROXY_PANEL or COMPUTE_PROXY_VS_FR:
        from compare_proxy_vs_network import load_hemi_proxy_from_excel, derive_proxy_xlsx_path, DEFAULT_PROXY_ROOT

        proxy_root = Path(PROXY_ROOT or DEFAULT_PROXY_ROOT).resolve()
        proxy_L = derive_proxy_xlsx_path(proxy_root, patient, period, side="left")
        proxy_R = derive_proxy_xlsx_path(proxy_root, patient, period, side="right")
        proxy_path_L = proxy_L
        proxy_path_R = proxy_R

        traces_L: list[tuple[str, np.ndarray]] = []
        traces_R: list[tuple[str, np.ndarray]] = []
        t_proxy_s = None

        if proxy_L is not None and Path(proxy_L).exists():
            t_ms_L, hemi_L, _ = load_hemi_proxy_from_excel(proxy_L)
            t_proxy_s = t_ms_L.astype(float) / 1000.0
            traces_L.append(("Proxy hemi L", hemi_L))
        if proxy_R is not None and Path(proxy_R).exists():
            t_ms_R, hemi_R, _ = load_hemi_proxy_from_excel(proxy_R)
            if t_proxy_s is None:
                t_proxy_s = t_ms_R.astype(float) / 1000.0
            traces_R.append(("Proxy hemi R", hemi_R))

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

    # --------------------------------------------------
    # Co-activity & firing rate
    # --------------------------------------------------
    if PLOT_RASTER:
        stride_s = max(coactivity_bins_s * (1.0 - OVERLAP_FRACTION), 1e-9)

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
    # Proxy vs firing-rate correlation study
    # --------------------------------------------------
    if COMPUTE_PROXY_VS_FR and (proxy_path_L is not None or proxy_path_R is not None):
        from compare_proxy_vs_network import load_proxy_excel_with_region_columns

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

        for side_tag, fr_df_side, struct_side, px_path in [
            ("left", fr_df_L, spike_struct_L, proxy_path_L),
            ("right", fr_df_R, spike_struct_R, proxy_path_R),
        ]:
            if px_path is None or not Path(px_path).exists():
                continue
            if fr_df_side is None or len(fr_df_side) == 0:
                continue

            t_ms_px, hemi_px, region_px = load_proxy_excel_with_region_columns(px_path)
            t_s_px = t_ms_px.astype(float) / 1000.0

            fr_t = fr_df_side["Window_start_s"].to_numpy(float)
            fr_y = fr_df_side["FR_Hz"].to_numpy(float)

            # --- Hemi-level: overall FR vs hemi_proxy ---
            hemi_resampled = _resample_to_bins(t_s_px, hemi_px, fr_t, coactivity_bins_s)
            valid = np.isfinite(hemi_resampled) & np.isfinite(fr_y)
            hemi_corr = float(np.corrcoef(fr_y[valid], hemi_resampled[valid])[0, 1]) if valid.sum() > 2 else np.nan

            rows = [{"patient": patient, "period": period, "side": side_tag,
                     "level": "hemi", "region": "all", "correlation": hemi_corr}]

            # --- Region-level: per-region FR vs per-region proxy ---
            for reg_name, reg_px_vals in region_px.items():
                reg_resampled = _resample_to_bins(t_s_px, reg_px_vals, fr_t, coactivity_bins_s)
                reg_clusters = {(e, c) for (e, c) in allowed_clusters if infer_region(e) == reg_name}
                if not reg_clusters:
                    continue
                reg_fr = population_firing_rate_binned(
                    struct_side, STATS, record_len_s,
                    bin_s=coactivity_bins_s, stride_s=stride_s,
                    allowed_clusters=reg_clusters,
                    normalize_by_units=True,
                )
                if reg_fr is None or len(reg_fr) == 0:
                    continue
                reg_fr_y = reg_fr["FR_Hz"].to_numpy(float)
                valid = np.isfinite(reg_resampled) & np.isfinite(reg_fr_y)
                reg_corr = float(np.corrcoef(reg_fr_y[valid], reg_resampled[valid])[0, 1]) if valid.sum() > 2 else np.nan
                rows.append({"patient": patient, "period": period, "side": side_tag,
                             "level": "region", "region": reg_name, "correlation": reg_corr})

            csv_path = run_dir / f"proxy_vs_fr_{side_tag}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  Proxy vs FR ({side_tag}): {len(rows)} rows → {csv_path}")

    # --------------------------------------------------
    # Main presentation figures
    #   1) SangerLab presentation (raster + EMG + FR)
    #   2) Correlation graph (raster + proxy [p] + FR)
    # --------------------------------------------------

    # 1) Classic SangerLab presentation figure
    if PLOT_SANGER_PRESENTATION_FIG:
        emg_downsample = 100  # EMG is high-rate
        fig = make_sangerlab_presentation_figure(
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
            emg_t_s=t_emg,
            emg_traces_L=emg_traces_L,
            emg_traces_R=emg_traces_R,
            emg_downsample=emg_downsample,
            emg_panel_labels=None,
            emg_y_range=None,
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
        )
        save_fig_interactive(fig, raster_dir / "sangerlab_presentation_LR_coact_EMG")

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
        )
        save_fig_interactive(fig_corr, raster_dir / "correlation_graph")

    # Region exclusion study: Sanger raster per permutation (same figure, filtered data)
    if RUN_REGION_EXCLUSION_STUDY and RS_NETWORK_ONSETS_TOGGLE and regions is not None:
        study_base = output_region_exclusion_root / patient_dir / period_dir / thr_method_name / run_tag
        for included_list, perm_dir in iter_permutation_dirs(study_base, regions, out_subdir=""):
            if not perm_dir.exists():
                continue
            (filtered_L, filtered_R, reg_bars_L, reg_bars_R, net_bars_L, net_bars_R) = (
                get_permutation_data_for_plotting(
                    all_bursts_L, all_bursts_R,
                    region_bars_by_region_L, region_bars_by_region_R,
                    set(included_list),
                )
            )
            inc_set = set(included_list)
            filtered_clusters = {(e, c) for (e, c) in allowed_clusters if infer_region(e) in inc_set}
            fr_perm_L = population_firing_rate_binned(
                spike_struct_L, STATS, record_len_s,
                bin_s=coactivity_bins_s, stride_s=stride_s,
                allowed_clusters=filtered_clusters,
                normalize_by_units=True,
            )
            fr_perm_R = population_firing_rate_binned(
                spike_struct_R, STATS, record_len_s,
                bin_s=coactivity_bins_s, stride_s=stride_s,
                allowed_clusters=filtered_clusters,
                normalize_by_units=True,
            )
            fig_perm = make_sangerlab_presentation_figure(
                spike_struct_L=spike_struct_L,
                spike_struct_R=spike_struct_R,
                STATS=STATS,
                all_bursts_L=filtered_L,
                all_bursts_R=filtered_R,
                record_len_s=record_len_s,
                patient=patient,
                period=period,
                bin_s=coactivity_bins_s,
                stride_s=stride_s,
                run_params=run_params,
                presentation_mode=presentation_mode,
                allowed_clusters=filtered_clusters,
                emg_t_s=t_emg,
                emg_traces_L=emg_traces_L,
                emg_traces_R=emg_traces_R,
                emg_downsample=100,
                emg_panel_labels=None,
                emg_y_range=None,
                network_windows_L=None,
                network_windows_R=None,
                network_bars_L=net_bars_L,
                network_bars_R=net_bars_R,
                region_windows_by_region_L=region_windows_L,
                region_windows_by_region_R=region_windows_R,
                region_bars_by_region_L=reg_bars_L,
                region_bars_by_region_R=reg_bars_R,
                disable_bursts=disable_bursts,
                fr_df_L=fr_perm_L,
                fr_df_R=fr_perm_R,
                show_firing_rate=PLOT_FR,
            )
            raster_perm_dir = perm_dir / "raster_plots"
            raster_perm_dir.mkdir(parents=True, exist_ok=True)
            save_fig_interactive(fig_perm, raster_perm_dir / "sangerlab_presentation_LR_coact_EMG")

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


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    if sum([maxisi_toggle, alpha_meanisi_toggle, rankSurprise_toggle]) != 1:
        raise ValueError("Select a SINGLE threshold")

    RS_ALPHA_SETS = [
        # (stage1,  region,  network)
        (0.05,    0.03,    0.02),
        (0.03,    0.02,    0.01),
    ]

    for a_stage1, a_region, a_network in RS_ALPHA_SETS:
        print(f"\n=== RS alphas: stage1={a_stage1}, region={a_region}, network={a_network} ===")
        for mod in (cfg, detection, region_exclusion, utils):
            mod.RS_alpha_percentage_stage1 = a_stage1
            mod.RS_alpha_stage1 = -np.log(a_stage1)
            mod.RS_alpha_percentage_region = a_region
            mod.RS_alpha_region = -np.log(a_region)
            mod.RS_alpha_percentage_network = a_network
            mod.RS_alpha_network = -np.log(a_network)

        for mat_file in SpikeTime_Mat_File:
            patient, period = dataset_labels(mat_file)

            print(f"\nRunning {patient}, {period}")
            print("Spike :", mat_file)

            EMG_MAT = None
            NOTES_TXT = None
            if PLOT_EMG:
                emg_candidate, notes_candidate = derive_emg_notes_from_spiketime(mat_file)
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
