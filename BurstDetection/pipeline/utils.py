# utils.py
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from config import *
from .detection import *
from .stats import snr_fails_filter
from collections import defaultdict


def display_region_name(region: str) -> str:
    """Map a raw region code to a presentation label (slashes, casing, etc.)."""
    m = re.fullmatch(r"shank(\d+)", str(region), flags=re.IGNORECASE)
    if m:
        return f"Shank {m.group(1)}"
    return REGION_DISPLAY.get(region, region)


def dataset_labels(mat_path: str) -> tuple[str, str]:
    p = Path(mat_path)
    patient = p.parent.name or "unknown"
    m = re.search(r"_p(\d+)", p.stem)
    period = f"Period {m.group(1)}" if m else "Period ?"
    return patient, period


def dataset_labels_any(spec: str) -> tuple[str, str]:
    """Like dataset_labels but supports ``kilosort:`` and ``kilosortset:`` entries (see kilosort_loader)."""
    from .kilosort_loader import dataset_labels_kilosort, parse_dataset_spec

    kind, path = parse_dataset_spec(spec)
    if kind == "kilosort":
        patient, period = dataset_labels_kilosort(path)
        if KILOSORT_RASTER_GROUP_BY_SHANK:
            patient = f"{patient}_separated"
        return patient, period
    if kind == "kilosortset":
        first = path.split(";", 1)[0].strip()
        patient, period = dataset_labels_kilosort(first)
        # patient ≈ m360_shank0_imec0 → output folder m360_imec0 (session + probe)
        parts = patient.split("_")
        if len(parts) >= 3:
            patient = f"{parts[0]}_{parts[-1]}"
        if KILOSORT_RASTER_GROUP_BY_SHANK:
            patient = f"{patient}_separated"
        return patient, period
    return dataset_labels(path)


def compute_recording_duration_s(spike_struct) -> float:
    ch0 = np.ravel(spike_struct)[0]
    return float(ch0.dataSegmentLength)


def parse_electrode(elec: str, combine_numbered_regions: bool | None = None) -> dict:
    """Parse electrode string. If combine_numbered_regions is None, uses config COMBINE_NUMBERED_REGIONS."""
    if combine_numbered_regions is None:
        combine_numbered_regions = COMBINE_NUMBERED_REGIONS
    base = str(elec).replace("_CommonFiltered", "")
    parts = base.split("_")

    head_raw = parts[0] if parts else base
    head = head_raw[5:] if head_raw.lower().startswith("micro") else head_raw

    side = parts[1] if len(parts) > 1 and parts[1] in ("L", "R") else ""
    idx = parts[2] if len(parts) > 2 and parts[2].isdigit() else (parts[2] if len(parts) > 2 else "")

    region = re.sub(r"\d+$", "", head) if combine_numbered_regions else head

    return {"base": base, "head": head, "side": side, "idx": idx, "region": region}


def infer_shank(elec: str) -> str | None:
    """Parse ``shankN`` from Kilosort-style names like ``rat_*_shank0_imec0_u12_L_0_...``."""
    m = re.search(r"(shank\d+)", str(elec), re.IGNORECASE)
    return m.group(1).lower() if m else None


def infer_depth_um(elec: str) -> float | None:
    """Parse depth tag from synthetic Kilosort names like ``..._d1234um_CommonFiltered``."""
    m = re.search(r"_d(-?\d+)um", str(elec), re.IGNORECASE)
    return float(m.group(1)) if m else None


def infer_region(elec: str, combine_numbered_regions: bool | None = None) -> str:
    """Return region for electrode. If combine_numbered_regions is None, uses config COMBINE_NUMBERED_REGIONS (True = GPi1+GPi2→GPi).

    When ``KILOSORT_COLOR_BY_SHANK`` and the name looks like ``rat_*_shankN_*``, returns ``shankN`` so plots match human region grouping.
    """
    if KILOSORT_COLOR_BY_SHANK and str(elec).startswith("rat_"):
        sk = infer_shank(elec)
        if sk is not None:
            return sk
    return parse_electrode(elec, combine_numbered_regions=combine_numbered_regions)["region"]


def color_for_plot_group(name: str) -> str:
    """Line / bar color for a coactivity column or burst group (brain region or shank id)."""
    if name in SHANK_COLORS:
        return SHANK_COLORS[name]
    if str(name).lower().startswith("shank") and str(name)[5:].isdigit():
        return "rgb(140, 140, 160)"
    return REGION_COLORS.get(name, "rgb(30,30,200)")


def pretty_channel_label(elec: str, drop_side: bool) -> str:
    p = parse_electrode(elec)
    head, side, idx = p["head"], p["side"], p["idx"]

    if idx:
        if drop_side or not side:
            return f"{head}_{idx}"
        return f"{head}_{side}_{idx}"

    return p["base"]


def color_for_nonburst(elec: str) -> str:
    region = infer_region(elec)
    if region in SHANK_COLORS:
        return SHANK_COLORS[region]
    if str(region).lower().startswith("shank") and str(region)[5:].isdigit():
        return "rgb(140, 140, 160)"
    for key, col in REGION_COLORS.items():
        if region.startswith(key):
            return col
    return "rgb(30,30,200)"


def iter_units_from_stats(spike_struct, STATS):
    """Yields (elec, cl, spikes_ms_sorted) for each unit in STATS.index."""
    ch_by_elec = {str(ch.electrode): ch for ch in np.ravel(spike_struct)}

    for (elec, cl) in STATS.index:
        elec = str(elec)
        cl = int(cl)

        ch = ch_by_elec.get(elec)
        if ch is None:
            continue

        try:
            arr = ch.time[cl]
        except Exception:
            arr = np.ravel(ch.time)[cl]

        if np.isscalar(arr):
            continue

        spk = np.asarray(arr).ravel().astype(float)
        spk = spk[np.isfinite(spk)]
        if spk.size < 2:
            continue

        yield elec, cl, np.sort(spk)


def split_spike_struct_by_side(spike_struct):
    left = []
    right = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        if "_L_" in elec:
            left.append(ch)
        elif "_R_" in elec:
            right.append(ch)

    return np.array(left, dtype=object), np.array(right, dtype=object)


def build_spike_labels_df(spike_struct, STATS, allowed_clusters, ibi_merge_factor):
    rows = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)

        for cl, arr in enumerate(np.ravel(ch.time)):
            if (elec, cl) not in allowed_clusters:
                continue

            spk = np.asarray(arr).flatten().astype(float)
            if spk.size == 0:
                continue
            spk = np.sort(spk)

            bursts = detect_bursts(spk, elec, cl, STATS, ibi_merge_factor=ibi_merge_factor)

            burst_id = np.full(spk.shape, -1, dtype=int)

            bursts_sorted = sorted(bursts, key=lambda b: float(b["Start_ms"]))
            j = 0
            for b_idx, b in enumerate(bursts_sorted):
                s = float(b["Start_ms"]); e = float(b["End_ms"])
                while j < spk.size and spk[j] < s:
                    j += 1
                k = j
                while k < spk.size and spk[k] <= e:
                    burst_id[k] = b_idx
                    k += 1
                j = k

            is_burst = burst_id >= 0

            for t, ib, bid in zip(spk, is_burst, burst_id):
                rows.append(
                    dict(
                        Electrode=elec,
                        Cluster=int(cl),
                        Spike_ms=float(t),
                        IsBurst=bool(ib),
                        BurstIndex=int(bid),
                    )
                )

    return pd.DataFrame(rows)


def pool_spike_struct_per_channel(spike_struct, record_len_s):
    class PooledCh:
        pass

    pooled_struct = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        snr_arr = np.asarray(getattr(ch, "snr", [])).flatten()

        pooled = []
        for cl, arr in enumerate(np.ravel(ch.time)):
            spk = np.asarray(arr).flatten().astype(float)
            if spk.size < 2:
                continue
            fr_hz = float(spk.size / record_len_s) if record_len_s > 0 else float("nan")
            if fr_hz < FR_MIN_HZ:
                continue
            snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
            if snr_fails_filter(snr_val):
                continue
            pooled.append(spk)

        if not pooled:
            continue

        pooled_spk = np.sort(np.concatenate(pooled))
        if pooled_spk.size < 2:
            continue

        pc = PooledCh()
        pc.electrode = elec
        pc.time = np.empty((1,), dtype=object)
        pc.time[0] = pooled_spk
        pc.snr = np.array([np.nan], dtype=float)

        pooled_struct.append(pc)

    return np.array(pooled_struct, dtype=object)


def standardize_output_df(
    df: pd.DataFrame, *,
    pretty_col: str = "Channel",
    drop_side: bool = False,
    replace_electrode: bool = True,
    put_pretty_first: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or "Electrode" not in df.columns:
        return df

    df = df.copy()
    df[pretty_col] = df["Electrode"].apply(lambda e: pretty_channel_label(str(e), drop_side=drop_side))

    if replace_electrode:
        df["Electrode"] = df[pretty_col]
        df.drop(columns=[pretty_col], inplace=True)
        return df

    if put_pretty_first:
        cols = [pretty_col] + [c for c in df.columns if c != pretty_col]
        df = df[cols]

    return df


def get_thresholding_method_name() -> tuple[str, int]:
    if maxisi_toggle:
        return "maxISI", 1
    if alpha_meanisi_toggle:
        return "alpha_meanISI", 2
    if rankSurprise_toggle:
        return "rankSurprise", 3
    return "unknown method", 0


def build_run_params(method_name: str, base_thr: float) -> dict:
    """Build a dict of all run parameters for the active method.

    Used to generate directory names, figure titles, and saved JSON.
    """
    common = {
        "method": method_name,
        "SNR_MIN": SNR_MIN,
        "SNR_MAX": SNR_MAX,
        "FR_MIN_HZ": FR_MIN_HZ,
        "MIN_SPIKES_IN_BURST": MIN_SPIKES_IN_BURST,
        "MIN_BURST_DURATION": MIN_BURST_DURATION,
        "pooling": pooling_toggle,
        "MIN_UNIQUE_CHANNELS_NETWORK": MIN_UNIQUE_CHANNELS_NETWORK,
    }

    if method_name == "maxISI":
        common.update({
            "base_thr": base_thr,
            "ibi_merge_factor": ibi_merge_factor,
            "MIN_BURST_DURATION": MIN_BURST_DURATION,
        })
    elif method_name == "alpha_meanISI":
        common.update({
            "alpha": ALPHA_MAXISI,
            "min_thr_ms": MIN_MAXISI_MS,
            "max_thr_ms": MAX_MAXISI_MS,
            "ibi_merge_factor": ibi_merge_factor,
            "MIN_BURST_DURATION": MIN_BURST_DURATION,
        })
    elif method_name == "rankSurprise":
        common.update({
            "RS_alpha_cluster": RS_alpha_percentage_stage1,
            "RS_limit_cluster": RS_Percentile_Limit_stage1,
            "RS_alpha_region": RS_alpha_percentage_region,
            "RS_limit_region": RS_Percentile_Limit_region,
            "RS_alpha_network": RS_alpha_percentage_network,
            "RS_limit_network": RS_Percentile_Limit_network,
            "region_burst": RS_region_burst_toggle,
            "network_burst": RS_NETWORK_ONSETS_TOGGLE,
            "RS_offset_null": bool(RS_OFFSET_NULL_ENABLE),
            "RS_offset_null_seed": int(RS_OFFSET_NULL_SEED),
            "RS_offset_null_max_offset_ms": RS_OFFSET_NULL_MAX_OFFSET_MS,
        })

    return common


def run_tag_from_params(params: dict) -> str:
    """Short, filesystem-safe directory name derived from run parameters."""
    m = params["method"]
    # Keep tags compact and stable over time. Use underscores for small joins and
    # double-underscores for major sections (easy to visually scan).
    #
    # Common suffix used by all methods (requested: keep SNR range + FR).
    # Match desired folder naming: FR value without "Hz" suffix.
    common_suffix = f"SNR={params['SNR_MIN']}-{params['SNR_MAX']}_FR={params['FR_MIN_HZ']}"
    parts: list[str] = []

    if m == "maxISI":
        parts += [
            f"thr={params['base_thr']}",
            f"ibi={params['ibi_merge_factor']}",
            f"minSpk={params['MIN_SPIKES_IN_BURST']}",
            f"minDur={params['MIN_BURST_DURATION']}ms",
            common_suffix,
        ]
        if params["pooling"]:
            parts.append("pooled")
    elif m == "alpha_meanISI":
        parts += [
            f"alpha={params['alpha']}",
            f"range={params['min_thr_ms']}-{params['max_thr_ms']}ms",
            f"ibi={params['ibi_merge_factor']}",
            f"minSpk={params['MIN_SPIKES_IN_BURST']}",
            f"minDur={params['MIN_BURST_DURATION']}ms",
            common_suffix,
        ]
        if params["pooling"]:
            parts.append("pooled")
    elif m == "rankSurprise":
        ac = float(params["RS_alpha_cluster"])
        ar = float(params["RS_alpha_region"])
        an = float(params["RS_alpha_network"])
        a_pct = (int(round(100 * ac)), int(round(100 * ar)), int(round(100 * an)))
        lims = (int(params["RS_limit_cluster"]), int(params["RS_limit_region"]), int(params["RS_limit_network"]))

        head = f"RS=({a_pct[0]},{a_pct[1]},{a_pct[2]})_({lims[0]},{lims[1]},{lims[2]})"
        mins = (
            f"minSpk={params['MIN_SPIKES_IN_BURST']}"
            f"__minDur={params['MIN_BURST_DURATION']}ms"
            f"__minCh={params['MIN_UNIQUE_CHANNELS_NETWORK']}"
        )
        # Preferred style: ..._SNR=..._FR=..._region__network
        toggles = ""
        if params.get("region_burst"):
            toggles += "_region"
        if params.get("network_burst"):
            toggles += "__network" if toggles else "network"
        if params.get("RS_offset_null"):
            toggles += "__offNull" if toggles else "offNull"
        tag = f"{head}_{mins}_{common_suffix}{toggles}"
        parts += [tag]
        if params["pooling"]:
            parts.append("pooled")

    # For RS we intentionally keep one large head string, then append optional toggles
    # as __region and __network (matching existing conventions in scripts).
    return "__".join(str(p) for p in parts)


def run_tag_from_params_kilosort(params: dict, *, good_only: bool, sep_shank: bool) -> str:
    """
    Kilosort output folders use a shorter, KS-specific tag:
    - Keep RS alpha/limits + minSpk/minDur/minCh + region/network toggles
    - Drop SNR/FR (KS clusters use NaN SNR; FR filtering is implicit in stats)
    - Append optional flags: GoodOnly, SepShank
    """
    m = params["method"]
    if m != "rankSurprise":
        # For now, only RS is used for KS runs; fall back to default naming.
        return run_tag_from_params(params)

    ac = float(params["RS_alpha_cluster"])
    ar = float(params["RS_alpha_region"])
    an = float(params["RS_alpha_network"])
    a_pct = (int(round(100 * ac)), int(round(100 * ar)), int(round(100 * an)))
    lims = (int(params["RS_limit_cluster"]), int(params["RS_limit_region"]), int(params["RS_limit_network"]))
    head = f"RS=({a_pct[0]},{a_pct[1]},{a_pct[2]})_({lims[0]},{lims[1]},{lims[2]})"

    flags = []
    if good_only:
        flags.append("GoodOnly")
    if sep_shank:
        flags.append("SepShank")
    flags_str = ("_" + "_".join(flags)) if flags else ""

    mins = (
        f"minSpk={params['MIN_SPIKES_IN_BURST']}"
        f"__minDur={params['MIN_BURST_DURATION']}ms"
        f"__minCh={params['MIN_UNIQUE_CHANNELS_NETWORK']}"
    )

    toggles = ""
    if params.get("region_burst"):
        toggles += "_region"
    if params.get("network_burst"):
        toggles += "__network" if toggles else "_network"

    return f"{head}{flags_str}_{mins}{toggles}"


def figure_title_from_params(params: dict, patient: str, period: str) -> str:
    """Human-readable figure title from run parameters."""
    m = params["method"]
    header = f"{patient} \u2022 {period} \u2014 {m}"

    if m == "maxISI":
        detail = (
            f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  FR\u2265{params['FR_MIN_HZ']}Hz  "
            f"thr={params['base_thr']}  IBI\u00d7{params['ibi_merge_factor']}  "
            f"spikes\u2265{params['MIN_SPIKES_IN_BURST']}  dur\u2265{params['MIN_BURST_DURATION']}ms"
        )
    elif m == "alpha_meanISI":
        detail = (
            f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  FR\u2265{params['FR_MIN_HZ']}Hz  "
            f"\u03b1={params['alpha']}  thr=[{params['min_thr_ms']},{params['max_thr_ms']}]ms  "
            f"IBI\u00d7{params['ibi_merge_factor']}  "
            f"spikes\u2265{params['MIN_SPIKES_IN_BURST']}  dur\u2265{params['MIN_BURST_DURATION']}ms"
        )
    elif m == "rankSurprise":
        detail = (
            f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  FR\u2265{params['FR_MIN_HZ']}Hz  "
            f"\u03b1c={params['RS_alpha_cluster']:.0%}  "
            f"\u03b1r={params['RS_alpha_region']:.0%}  "
            f"\u03b1n={params['RS_alpha_network']:.0%}  "
            f"spikes\u2265{params['MIN_SPIKES_IN_BURST']}  "
            f"region={'on' if params['region_burst'] else 'off'}  "
            f"network={'on' if params['network_burst'] else 'off'}"
        )
        if params.get("RS_offset_null"):
            mx = params.get("RS_offset_null_max_offset_ms")
            mx_txt = f"{float(mx):.0f}ms" if mx is not None else "auto"
            detail += f"  offsetNull=on(seed={params.get('RS_offset_null_seed')}, max={mx_txt})"
    else:
        detail = ""

    if params.get("pooling"):
        detail += "  [POOLED]"

    return f"{header}:  {detail}" if detail else header


def save_run_params(params: dict, run_dir: Path) -> None:
    """Write run parameters as a JSON file inside the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)


def filter_bursts_by_spike_struct(all_bursts, spike_struct):
    allowed_elecs = {str(ch.electrode) for ch in np.ravel(spike_struct)}
    return [b for b in all_bursts if b["Electrode"] in allowed_elecs]


def burst_in_window(t_ms: float, window: list) -> bool:
    for t0, t1 in window:
        if t0 <= t_ms <= t1:
            return True
    return False
