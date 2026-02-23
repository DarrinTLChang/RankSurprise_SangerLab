# main.py
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import scipy.io as spio
import pandas as pd

from config import *
from pipeline.emg import *
from pipeline.detection import *
from pipeline.utils import *
from pipeline.stats import *
from pipeline.coactivity import *
from pipeline.plotting import *
from pipeline.isi import *
from pipeline.region_exclusion_study import (
    load_gt_spans,
    iter_permutation_dirs,
    run_network_with_excluded_regions,
    get_permutation_data_for_plotting,
    mean_iou_vs_gt,
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
                for side, gt_spans, bursts in [
                    ("left", gt_left, bursts_left),
                    ("right", gt_right, bursts_right),
                ]:
                    network_rows, pred_spans, _ = run_network_with_excluded_regions(
                        bursts, excluded_set
                    )
                    pd.DataFrame(network_rows).to_csv(
                        perm_dir / f"network_bursts_RS_{side}.csv", index=False
                    )
                    mean_iou, n_matched = mean_iou_vs_gt(gt_spans, pred_spans)
                    K_excluded = len(excluded_set)
                    rows.append({
                        "Side": side,
                        "K_excluded": K_excluded,
                        "Regions_included": included_str,
                        "N_gt_spans": len(gt_spans),
                        "N_pred_spans": len(pred_spans),
                        "Mean_IoU": mean_iou,
                        "N_gt_matched": n_matched,
                    })
            if rows:
                df_out = pd.DataFrame(rows)
                # All Left rows first, then all Right (sort by Side: left < right)
                df_out = df_out.sort_values("Side", kind="stable")
                df_out.to_csv(study_base / "results.csv", index=False)
                print(f"• Region exclusion study: wrote {len(rows)} rows → {study_base / 'results.csv'}")

    # --------------------------------------------------
    # EMG
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

        emg_traces_L = build_emg_traces(IDX_L, "L", emg_processed=emg_processed, selected_channel_names=selected_channel_names, mask=mask)
        emg_traces_R = build_emg_traces(IDX_R, "R", emg_processed=emg_processed, selected_channel_names=selected_channel_names, mask=mask)
        emg_traces_LR = build_emg_traces(IDX_LR, "LR", emg_processed=emg_processed, selected_channel_names=selected_channel_names, mask=mask)

        if emg_traces_L is not None:
            emg_traces_L = collapse_emg_traces(emg_traces_L, mode="sum", label="EMG summed L")
        if emg_traces_R is not None:
            emg_traces_R = collapse_emg_traces(emg_traces_R, mode="sum", label="EMG summed R")
        if emg_traces_LR is not None:
            emg_traces_LR = collapse_emg_traces(emg_traces_LR, mode="sum", label="EMG summed")

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
    # Main presentation figure
    # --------------------------------------------------
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
        emg_downsample=100,
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
