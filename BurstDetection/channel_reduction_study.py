#!/usr/bin/env python3
"""
Per-region channel reduction study: compare region burst onset when using 2 electrodes
vs full (all electrodes in that region). Channel = one Electrode (physical contact);
each electrode can have multiple clusters (unit bursts). For each region with ≥2
electrodes we sample 2 electrodes and use all unit bursts (all clusters) on those 2;
run the same region-level RS detection; compute mean/median onset error (full → nearest 2-electrode onset).

Usage:
  python channel_reduction_study.py --run-dir OUTPUTS_RS_BURST/patient/period/method/run_tag
  python channel_reduction_study.py --root OUTPUTS_RS_BURST  # discover all run dirs
  python channel_reduction_study.py --root /Volumes/D_Drive/SangerLabBursts/outputs_region_exclusion  # scans outputs_RS_burst_onset_length, saves to .../outputs_channel_reduction
  python channel_reduction_study.py --run-dir ... --n-draws 20 --seed 0
  python channel_reduction_study.py --run-dir ... --save-raster --mat /path/to/data.mat  # rasters use existing presentation figure, data limited to 2 channels
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    MIN_SPIKES_IN_BURST,
    MIN_UNIQUE_CHANNELS_REGION,
    RS_Limit_region,
    RS_Percentile_Limit_region,
    RS_alpha_region,
)
from pipeline.burst_paths import BURST_TIMINGS_SUBDIR, burst_network_csv_path, burst_region_csv_path, burst_unit_csv_path
from pipeline.detection import RS_detect_burst, compute_spans_and_bars
from pipeline.utils import infer_region, compute_recording_duration_s, split_spike_struct_by_side

try:
    import scipy.io as spio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pipeline.plotting import make_sangerlab_presentation_figure, save_fig_interactive
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    make_sangerlab_presentation_figure = None
    save_fig_interactive = None


def _load_stats_from_run_dir(run_dir: Path) -> pd.DataFrame | None:
    """Load STATS (Electrode, Cluster index) from run_dir/cluster_stats.csv if present."""
    path = Path(run_dir) / "cluster_stats.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "Electrode" not in df.columns or "Cluster" not in df.columns:
        return None
    df = df.set_index(["Electrode", "Cluster"])
    return df


def _save_two_electrode_raster_presentation(
    two_ch_bursts: list[dict],
    chosen_electrodes: tuple[str, ...],
    side: str,
    reg: str,
    patient: str,
    period: str,
    run_label: str,
    raster_path: Path,
    spike_struct_L,
    spike_struct_R,
    STATS: pd.DataFrame,
    record_len_s: float,
) -> None:
    """
    Save the same sangerlab_presentation raster figure but with data limited to the 2 chosen
    electrodes (allowed_clusters + filtered bursts). Uses make_sangerlab_presentation_figure.
    """
    if not HAS_PLOTLY or make_sangerlab_presentation_figure is None or save_fig_interactive is None:
        return
    if not two_ch_bursts or STATS is None or STATS.empty:
        return
    allowed_clusters = {(str(b["Electrode"]), int(b.get("Cluster", -1))) for b in two_ch_bursts}
    allowed_clusters = {k for k in allowed_clusters if k in STATS.index}
    if not allowed_clusters:
        return
    if side == "left":
        bursts_L, bursts_R = two_ch_bursts, []
    else:
        bursts_L, bursts_R = [], two_ch_bursts
    run_params = {"run_tag": run_label}
    fig = make_sangerlab_presentation_figure(
        spike_struct_L=spike_struct_L,
        spike_struct_R=spike_struct_R,
        STATS=STATS,
        all_bursts_L=bursts_L,
        all_bursts_R=bursts_R,
        record_len_s=record_len_s,
        patient=patient,
        period=period,
        bin_s=0.1,
        stride_s=0.05,
        run_params=run_params,
        allowed_clusters=allowed_clusters,
        emg_t_s=None,
        emg_traces_L=None,
        emg_traces_R=None,
        network_windows_L=None,
        network_windows_R=None,
        network_bars_L=None,
        network_bars_R=None,
        region_windows_by_region_L=None,
        region_windows_by_region_R=None,
        region_bars_by_region_L=None,
        region_bars_by_region_R=None,
        disable_bursts=False,
        fr_df_L=pd.DataFrame(),
        fr_df_R=pd.DataFrame(),
        show_firing_rate=False,
    )
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig_interactive(fig, raster_path.with_suffix(""))
    print(f"  Raster → {raster_path.with_suffix('')}.html")


def _load_unit_bursts(csv_path: Path) -> list[dict]:
    """Load unit bursts from CSV; return list of dicts with Start_ms, End_ms, Electrode, Cluster."""
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    # Handle newer burst_* schema first, then older onset_* or Start/End
    if "burst_start_ms" in df.columns and "burst_end_ms" in df.columns:
        start_col = "burst_start_ms"
        end_col = "burst_end_ms"
    else:
        start_col = "onset_start_ms" if "onset_start_ms" in df.columns else "Start_ms"
        end_col = "onset_end_ms" if "onset_end_ms" in df.columns else "End_ms"
    if start_col not in df.columns or end_col not in df.columns:
        return []
    bursts = []
    for _, row in df.iterrows():
        s = pd.to_numeric(row[start_col], errors="coerce")
        e = pd.to_numeric(row[end_col], errors="coerce")
        if not (np.isfinite(s) and np.isfinite(e) and e > s):
            continue
        bursts.append({
            "Start_ms": float(s),
            "End_ms": float(e),
            "Electrode": str(row.get("Electrode", "")),
            "Cluster": int(pd.to_numeric(row.get("Cluster", -1), errors="coerce") or -1),
        })
    return bursts


def _load_region_bursts_by_region(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Load region_bursts CSV; return dict region -> list of (start_ms, end_ms).
    Norm (new schema): interval = [burst_start_ms, burst_end_ms] when present.
    For backward compatibility, falls back to onset_* / span_* columns.
    Duplicate (t0, t1) rows are collapsed to one.
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty or "Region" not in df.columns:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        reg = str(row["Region"]).strip()
        # Prefer new burst_* columns if available
        if "burst_start_ms" in df.columns and "burst_end_ms" in df.columns:
            t0 = pd.to_numeric(row.get("burst_start_ms"), errors="coerce")
            t1 = pd.to_numeric(row.get("burst_end_ms"), errors="coerce")
        else:
            # Fallback to onset_* + span_duration_ms or onset_end_ms/span_end_ms
            t0 = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            if not np.isfinite(t0):
                continue
            t0 = float(t0)
            span_dur = pd.to_numeric(row.get("span_duration_ms"), errors="coerce")
            if np.isfinite(span_dur) and span_dur > 0:
                t1 = t0 + float(span_dur)
            else:
                # last resort: explicit end column if present
                t1 = pd.to_numeric(
                    row.get("onset_end_ms", row.get("span_end_ms", t0)),
                    errors="coerce",
                )
        if not (np.isfinite(t0) and np.isfinite(t1) and float(t1) > float(t0)):
            continue
        t0, t1 = float(t0), float(t1)
        pair = (t0, t1)
        lst = out.setdefault(reg, [])
        if pair not in lst:
            lst.append(pair)
    for reg in out:
        out[reg] = sorted(out[reg], key=lambda p: p[0])
    return out


def _load_network_burst_windows(run_dir: Path, side: str) -> list[tuple[float, float]]:
    """
    Load network burst windows (start_ms, end_ms) from ``burst_timings/network_bursts_L.csv`` (or R).
    Norm (new schema): interval = [burst_start_ms, burst_end_ms] when present.
    For backward compatibility, falls back to onset_* / span_* columns.
    """
    path = burst_network_csv_path(run_dir, side)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # File exists but has no header/rows → treat as no network bursts
        return []
    if df.empty:
        return []
    windows: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        # Prefer new burst_* columns if available
        if "burst_start_ms" in df.columns and "burst_end_ms" in df.columns:
            t0 = pd.to_numeric(row.get("burst_start_ms"), errors="coerce")
            t1 = pd.to_numeric(row.get("burst_end_ms"), errors="coerce")
        else:
            t0 = pd.to_numeric(row.get("onset_start_ms"), errors="coerce")
            if not np.isfinite(t0):
                continue
            t0 = float(t0)
            span_dur = pd.to_numeric(row.get("span_duration_ms"), errors="coerce")
            if np.isfinite(span_dur) and span_dur > 0:
                t1 = t0 + float(span_dur)
            else:
                end_col = "onset_end_ms" if "onset_end_ms" in df.columns else "span_end_ms"
                t1 = pd.to_numeric(row.get(end_col, t0 + 1), errors="coerce")
        if not (np.isfinite(t0) and np.isfinite(t1) and float(t1) > float(t0)):
            continue
        windows.append((float(t0), float(t1)))
    return sorted(windows)


def _intervals_overlap(r0: float, r1: float, a: float, b: float) -> bool:
    """True if region burst [r0, r1] overlaps network window [a, b] (intervals overlap)."""
    return not (r1 < a or r0 > b)


def _filter_region_bursts_overlapping_network(
    region_bursts: list[tuple[float, float]],
    network_windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Return only region bursts whose interval [start, end] overlaps at least one
    network burst window [a, b]. Overlap = intervals intersect (not (r1 < a or r0 > b)).
    Each distinct burst (r0, r1) is counted at most once even if it appears multiple
    times in region_bursts or overlaps multiple network windows.
    """
    if not network_windows or not region_bursts:
        return []
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for r0, r1 in region_bursts:
        key = (r0, r1)
        if key in seen:
            continue
        for a, b in network_windows:
            if _intervals_overlap(r0, r1, a, b):
                out.append(key)
                seen.add(key)
                break
    return sorted(out, key=lambda p: p[0])


def _run_region_detection_on_bursts(
    unit_bursts_in_reg: list[dict],
    infer_region_fn,
) -> list[tuple[float, float]]:
    """
    Run region-level RS detection on the given unit bursts (single region).
    Returns list of (onset_start_ms, onset_end_ms) for each region burst that passed
    the channel filter — same duration/spans as for full-channel; used for
    interval-overlap with network bursts.
    """
    if len(unit_bursts_in_reg) < MIN_SPIKES_IN_BURST:
        return []
    onsets_ms = np.sort(np.asarray([b["Start_ms"] for b in unit_bursts_in_reg], dtype=float))
    onsets_ms = onsets_ms[np.isfinite(onsets_ms)]
    if onsets_ms.size < MIN_SPIKES_IN_BURST:
        return []

    start_idx_r, length_r, RSvals_r = RS_detect_burst(
        onsets_ms,
        limit=RS_Limit_region,
        RSalpha=RS_alpha_region,
        min_spikes=MIN_SPIKES_IN_BURST,
        RS_Percentile_Limit=RS_Percentile_Limit_region,
    )

    windows: list[tuple[float, float]] = []
    for s, L, rs in zip(start_idx_r, length_r, RSvals_r):
        s, L = int(s), int(L)
        e = s + L - 1
        if e < s or e >= onsets_ms.size:
            continue
        t0 = float(onsets_ms[s])
        t1 = float(onsets_ms[e])
        if t1 <= t0:
            continue
        windows.append((t0, t1))

    if not windows:
        return []

    # Channel filter: require at least min_unique_channels (with 2 channels we use 1)
    min_ch = min(MIN_UNIQUE_CHANNELS_REGION, 2)
    spans_ms, _ = compute_spans_and_bars(
        windows,
        unit_bursts_in_reg,
        min_unique_channels=min_ch,
        channel_mode="elec_cluster",
    )

    result: list[tuple[float, float]] = []
    for w, span in zip(windows, spans_ms):
        if np.isfinite(span[0]) and np.isfinite(span[1]) and span[1] > span[0]:
            onset_start = float(w[0])
            span_duration = float(span[1] - span[0])
            result.append((onset_start, onset_start + span_duration))
    return sorted(result, key=lambda p: p[0])


def _onset_error_full_to_nearest_pred(full_onsets: list[float], pred_onsets: list[float]) -> tuple[float, float]:
    """Mean and median of min |full - pred| over full_onsets (ms)."""
    if not full_onsets:
        return float("nan"), float("nan")
    if not pred_onsets:
        return float("nan"), float("nan")
    full_arr = np.asarray(full_onsets, dtype=float)
    pred_arr = np.asarray(pred_onsets, dtype=float)
    # per full: min |full - pred|
    diffs = np.min(np.abs(full_arr[:, np.newaxis] - pred_arr), axis=1)
    return float(np.mean(diffs)), float(np.median(diffs))


def run_study_for_run_dir(
    run_dir: Path,
    n_draws: int = 10,
    seed: int | None = None,
    infer_region_fn=None,
    save_raster: bool = False,
    raster_dir: Path | None = None,
    raster_context: tuple | None = None,
) -> list[dict]:
    """
    For one run_dir (single patient/period/method/run_tag), compute per-region
    2-channel vs full-channel onset error. Only full (all-channel) region bursts that
    overlap at least one network burst are compared; region bursts that do not
    overlap any network burst are ignored (we assess whether 2ch can replicate
    network-relevant region activity).

    raster_context: when saving rasters, (spike_struct_L, spike_struct_R, record_len_s)
    from the .mat file; STATS is loaded from run_dir/cluster_stats.csv per run_dir.
    """
    run_dir = Path(run_dir).resolve()
    if infer_region_fn is None:
        infer_region_fn = infer_region

    rows: list[dict] = []
    run_label = run_dir.name  # run_tag
    parent = run_dir.parent
    period = parent.parent.name if parent and parent.parent else ""
    patient = parent.parent.parent.name if parent and parent.parent and parent.parent.parent else ""

    for side in ("left", "right"):
        unit_path = burst_unit_csv_path(run_dir, side)
        region_path = burst_region_csv_path(run_dir, side)
        unit_bursts = _load_unit_bursts(unit_path)
        full_by_region = _load_region_bursts_by_region(region_path)
        network_windows = _load_network_burst_windows(run_dir, side)

        if not unit_bursts:
            continue

        # Add region to each burst
        for b in unit_bursts:
            b["_Region"] = infer_region_fn(str(b["Electrode"]))

        # Electrodes (channels) per region: one Electrode = one physical channel (may have multiple clusters)
        region_to_electrodes: dict[str, list[str]] = {}
        for b in unit_bursts:
            reg = b["_Region"]
            elec = str(b["Electrode"])
            if reg not in region_to_electrodes:
                region_to_electrodes[reg] = []
            if elec not in region_to_electrodes[reg]:
                region_to_electrodes[reg].append(elec)

        rng = random.Random(seed)

        for reg, electrodes in region_to_electrodes.items():
            if len(electrodes) < 2:
                continue
            full_region_bursts = full_by_region.get(reg, [])  # list of (start_ms, end_ms)
            # Only count full region bursts whose interval overlaps at least one network burst
            full_bursts_net = _filter_region_bursts_overlapping_network(full_region_bursts, network_windows)
            full_onsets_net = [b[0] for b in full_bursts_net]  # onset times for error computation
            if not full_onsets_net:
                continue

            for draw in range(n_draws):
                chosen_electrodes = tuple(rng.sample(electrodes, 2))
                chosen_set = set(chosen_electrodes)
                # All unit bursts (all clusters) on those 2 electrodes
                two_ch_bursts = [
                    b for b in unit_bursts
                    if b["_Region"] == reg and str(b["Electrode"]) in chosen_set
                ]
                two_ch_windows = _run_region_detection_on_bursts(two_ch_bursts, infer_region_fn)
                two_ch_bursts_net = _filter_region_bursts_overlapping_network(two_ch_windows, network_windows)
                two_ch_onsets_net = [b[0] for b in two_ch_bursts_net]
                mean_err_2ch_net, median_err_2ch_net = _onset_error_full_to_nearest_pred(
                    two_ch_onsets_net, full_onsets_net
                )

                if save_raster and raster_dir is not None and draw == 0 and raster_context is not None:
                    safe_reg = reg.replace("/", "-").replace("|", "_")
                    raster_path = raster_dir / f"{side}_{safe_reg}_2electrode_raster.html"
                    sstruct_L, sstruct_R, rec_len_s = raster_context
                    STATS = _load_stats_from_run_dir(run_dir)
                    if STATS is not None:
                        _save_two_electrode_raster_presentation(
                            two_ch_bursts,
                            chosen_electrodes,
                            side,
                            reg,
                            patient,
                            period,
                            run_label,
                            raster_path,
                            sstruct_L,
                            sstruct_R,
                            STATS,
                            rec_len_s,
                        )

                rows.append({
                    "patient": patient,
                    "period": period,
                    "run_tag": run_label,
                    "side": side,
                    "region": reg,
                    "n_channels_in_region": len(electrodes),
                    "n_full_bursts_total": len(full_region_bursts),
                    "n_full_bursts_overlapping_network": len(full_bursts_net),
                    "n_two_ch_bursts": len(two_ch_windows),
                    "n_two_ch_bursts_overlapping_network": len(two_ch_bursts_net),
                    "draw": draw,
                    "mean_onset_error_2ch_net_ms": mean_err_2ch_net,
                    "median_onset_error_2ch_net_ms": median_err_2ch_net,
                })

    return rows


def _infer_rs_burst_root_from_region_exclusion_root(root: Path) -> Path:
    """
    Infer RS burst output root from region-exclusion root (same layout as main.py).
    NOTE (new layout): region exclusion results now live under each run_dir as
    run_dir/region_exclusion_study/..., and burst CSVs live directly in run_dir.
    This helper is kept only for legacy paths; for new runs you can just pass
    the method root (e.g. F:\\SangerLabBursts_RS) to --root.
    """
    root = Path(root).resolve()
    name = root.name
    if "outputs_region_exclusion_onset_length" in name:
        name = name.replace("outputs_region_exclusion_onset_length", "outputs_RS_burst_onset_length")
    elif "outputs_region_exclusion" in name:
        name = name.replace("outputs_region_exclusion", "outputs_RS_burst")
    return root.parent / name


def find_run_dirs(root: Path) -> list[Path]:
    """Find all run dirs under root that contain ``burst_timings/unit_bursts_L.csv``."""
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    run_dirs = []
    for f in root.rglob("unit_bursts_L.csv"):
        if f.parent.name == BURST_TIMINGS_SUBDIR:
            run_dirs.append(f.parent.parent)
    return sorted(set(run_dirs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-region 2-channel vs full-channel region burst onset comparison."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Single run directory (new layout: <METHOD_ROOT>/patient/PeriodN/run_tag).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root to scan for run dirs (new layout: <METHOD_ROOT>, e.g. F:\\SangerLabBursts_RS). Used if --run-dir not set.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV and rasters (default: when --root set, root.parent/outputs_channel_reduction; else cwd).",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=10,
        help="Number of random 2-channel samples per region (default 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for 2-channel sampling (default 0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: when --out-dir set, out_dir/channel_reduction_study_results.csv; else channel_reduction_study_results.csv).",
    )
    parser.add_argument(
        "--save-raster",
        action="store_true",
        help="Save a burst raster (draw 0) per side/region for the 2 chosen electrodes.",
    )
    parser.add_argument(
        "--raster-dir",
        type=Path,
        default=None,
        help="Directory for raster HTML files when --save-raster (default: when --out-dir set, out_dir/channel_reduction_rasters; else channel_reduction_rasters).",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=None,
        help="Path to .mat file (required when --save-raster). Used to plot rasters with existing presentation figure (data limited to 2 channels).",
    )
    args = parser.parse_args()

    # Resolve output directory: --out-dir, or when --root set use root.parent/outputs_channel_reduction
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    elif args.root is not None:
        out_dir = Path(args.root).resolve().parent / "outputs_channel_reduction"
    else:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out is not None:
        out_path = Path(args.out).resolve()
    else:
        out_path = out_dir / "channel_reduction_study_results.csv"
    if args.raster_dir is not None:
        raster_dir_base = Path(args.raster_dir).resolve()
    else:
        raster_dir_base = out_dir / "channel_reduction_rasters"

    if args.save_raster and (args.mat is None or not Path(args.mat).exists()):
        print("When using --save-raster you must provide --mat with a valid .mat path.")
        return

    if args.run_dir is not None:
        run_dirs = [Path(args.run_dir).resolve()]
        if not run_dirs[0].exists():
            print(f"Run dir does not exist: {run_dirs[0]}")
            return
    elif args.root is not None:
        root = Path(args.root).resolve()
        # If user pointed at outputs_region_exclusion*, burst CSVs live under outputs_RS_burst*; infer scan root
        if "outputs_region_exclusion" in root.name and "outputs_RS_burst" not in root.name:
            scan_root = _infer_rs_burst_root_from_region_exclusion_root(root)
            print(f"Root {root} → scanning RS burst tree: {scan_root}")
        else:
            scan_root = root
        run_dirs = find_run_dirs(scan_root)
        if not run_dirs:
            print(f"No run dirs found under {scan_root}")
            return
        print(f"Found {len(run_dirs)} run dir(s)")
    else:
        print("Provide either --run-dir or --root")
        return

    raster_context = None
    if args.save_raster and args.mat is not None:
        if not HAS_SCIPY:
            print("scipy is required to load .mat for --save-raster; rasters will be skipped.")
        else:
            mat_path = Path(args.mat).resolve()
            if not mat_path.exists():
                print(f"Mat file not found: {mat_path}")
                return
            try:
                mat = spio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
                spike_struct = mat.get("spike_struct")
                if spike_struct is None:
                    print("Mat file has no 'spike_struct'.")
                    return
                record_len_s = compute_recording_duration_s(mat)
                spike_struct_L, spike_struct_R = split_spike_struct_by_side(spike_struct)
                raster_context = (spike_struct_L, spike_struct_R, record_len_s)
            except Exception as e:
                print(f"Failed to load .mat for raster context: {e}")
                return

    # Run per run_dir and write one CSV per patient/period/run_tag,
    # plus an aggregated CSV at out_path.
    all_rows: list[dict] = []
    any_rows = False
    for rd in run_dirs:
        raster_dir = (raster_dir_base / rd.parent.parent.parent.name / rd.parent.parent.name / rd.parent.name / rd.name) if args.save_raster else None
        rows = run_study_for_run_dir(
            rd,
            n_draws=args.n_draws,
            seed=args.seed,
            save_raster=args.save_raster,
            raster_dir=raster_dir,
            raster_context=raster_context,
        )
        if not rows:
            continue

        any_rows = True
        all_rows.extend(rows)

        # Derive patient/period/run_tag for this run to build an outputs-like tree:
        # out_dir/patient/period/run_tag_channel_reduction_study_results.csv
        sample = rows[0]
        patient = str(sample.get("patient", rd.parent.parent.parent.name))
        period = str(sample.get("period", rd.parent.parent.name)).replace(" ", "")
        run_tag = str(sample.get("run_tag", rd.name))

        per_out_dir = out_dir / patient / period
        per_out_dir.mkdir(parents=True, exist_ok=True)
        per_out_path = per_out_dir / f"{run_tag}_channel_reduction_study_results.csv"

        df_run = pd.DataFrame(rows)
        df_run.to_csv(per_out_path, index=False)
        print(f"Wrote {len(df_run)} rows → {per_out_path}")

    if not any_rows:
        print("No results (no regions with ≥2 channels or missing CSVs).")
        return

    # Aggregated CSV across all runs
    df_all = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"Wrote {len(df_all)} rows → {out_path}")


if __name__ == "__main__":
    main()
