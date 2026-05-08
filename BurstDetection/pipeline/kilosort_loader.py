# kilosort_loader.py
"""
Load Kilosort 4 folders into the pipeline's in-memory ``spikeTime`` layout.

What this loader uses (typical KS4 folder)
-----------------------------------------
- **spike_times.npy** — spike times in **sample indices** (converted to ms using ``sample_rate`` from ``params.py``).
- **spike_clusters.npy** — cluster id per spike (same length as spike_times).
- **params.py** — must define ``sample_rate`` (Hz) for time conversion.
- **cluster_group.tsv** (optional) — if ``KILOSORT_GOOD_ONLY`` is True in config, only clusters whose **KSLabel**
  column equals ``good`` (case-insensitive) are kept. Set ``KILOSORT_GOOD_ONLY = False`` to include every cluster
  that has spikes (still need ≥2 spikes per unit for the pipeline).

What we do *not* do here
-------------------------
- No SNR computed from Kilosort amplitudes. Per-unit **snr** is set to **NaN**; use **KILOSORT_SKIP_PIPELINE_SNR_FILTER**
  in config so ``build_cache`` does not drop units based on ``SNR_MIN`` / ``SNR_MAX`` (see config.py).

Other Kilosort files you may use *outside* this script
------------------------------------------------------
KS4 often also writes **templates.npy**, **amplitudes.npy**, **pc_features.npy**, **cluster_ContamPct.tsv**,
**cluster_KSLabel.tsv**, drift plots, etc. You can use those in Phy / custom notebooks for manual curation or
extra QC before running this pipeline; this loader only needs the npy files above (+ optional cluster_group for KSLabel).

Region labeling in this codebase (important for rat data)
---------------------------------------------------------
Human DBS data use electrode strings like ``microGPi1_L_1_...``; **infer_region** parses the **head** field
(GPi, VIM, …) and **split_spike_struct_by_side** uses ``_L_`` vs ``_R_`` in the string.

Kilosort units get synthetic names like ``rat_m360_shank0_imec0_u{cid}_L_0_CommonFiltered`` (session/shank/imec
from the folder path) so all units land on the **left** side for plotting. With **KILOSORT_COLOR_BY_SHANK** in
config, **infer_region** returns ``shank0``, ``shank1``, … parsed from the electrode string, and **SHANK_COLORS**
controls raster / coactivity / regional-burst colors (like brain regions for human data).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """
    Returns (kind, path) where kind is:
      - 'mat' for MATLAB spikeTime .mat paths
      - 'kilosort' for a single Kilosort output folder (prefix ``kilosort:``)
      - 'kilosortset' for multiple Kilosort folders combined (prefix ``kilosortset:``)

    ``kilosortset:`` format: a semicolon-separated list of Kilosort output folders.
    """
    s = spec.strip()
    low = s.lower()
    if low.startswith("kilosort:"):
        return "kilosort", s.split(":", 1)[1].strip()
    if low.startswith("kilosortset:"):
        return "kilosortset", s.split(":", 1)[1].strip()
    return "mat", s


def dataset_labels_kilosort(ks_dir: str | Path) -> tuple[str, str]:
    """
    Derive (patient, period) labels from a path like .../m360/shank0/imec0/kilosort4.
    patient ≈ m360_shank0_imec0; period is fixed for KS exports.

    Flat layout (mouse): .../<line>/<gate_folder>/<imec_folder>/ with KS files in
    imec_folder (no kilosort4). patient ≈ f"{gate}_{imec_leaf}" to align with output
    folder tokens and synthetic electrode prefixes.
    """
    p = Path(ks_dir).resolve()
    parts = p.parts
    patient = "kilosort_session"
    try:
        if p.name.lower() == "kilosort4" and len(parts) >= 4:
            imec = parts[-2]
            shank = parts[-3]
            session = parts[-4]
            patient = f"{session}_{shank}_{imec}"
        elif len(parts) >= 2 and "imec" in p.name.lower():
            imec_leaf = p.name
            gate = p.parent.name
            patient = f"{gate}_{imec_leaf}"
    except Exception:
        patient = p.parent.name + "_" + p.name
    return patient, "Period 1"


def _read_sample_rate_hz(params_py: Path) -> float:
    """Read sample_rate from Kilosort params.py (simple assignment file)."""
    text = params_py.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*sample_rate\s*=\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"Could not parse sample_rate from {params_py}")
    return float(m.group(1))


def _load_cluster_table(ks_dir: Path, name: str) -> pd.DataFrame | None:
    path = ks_dir / name
    if not path.is_file():
        return None
    return pd.read_csv(path, sep="\t")


def _depth_um_by_cluster_from_spike_positions(
    ks_dir: Path,
    *,
    spike_clusters: np.ndarray,
    keep_mask: np.ndarray | None,
) -> dict[int, float]:
    """
    Compute per-cluster depth (um) using median y from spike_positions.npy when present.

    Returns {} when file is missing or shape is unexpected.
    """
    sp_path = ks_dir / "spike_positions.npy"
    if not sp_path.is_file():
        return {}
    try:
        sp = np.load(sp_path)
    except Exception:
        return {}
    if sp.ndim != 2 or sp.shape[0] != spike_clusters.shape[0] or sp.shape[1] < 2:
        return {}
    y_um_all = sp[:, 1].astype(np.float64, copy=False)

    out: dict[int, float] = {}
    for cid in np.unique(spike_clusters):
        cid_i = int(cid)
        sel = spike_clusters == cid_i
        if keep_mask is not None:
            sel = sel & keep_mask
        yy = y_um_all[sel]
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            continue
        out[cid_i] = float(np.median(yy))
    return out


def _depth_um_by_cluster_from_templates(
    ks_dir: Path,
    *,
    keep_cids: set[int] | None,
) -> dict[int, float]:
    """
    Fallback unit depth (um) using template peak channel y-coordinate.

    Uses:
      - templates.npy (n_templates, nt, n_channels)
      - channel_positions.npy (n_channels, 2) where column 1 is y (um)

    Assumption: cluster id corresponds to template index (standard KS4 export).
    Returns {} if required files missing/unexpected.
    """
    t_path = ks_dir / "templates.npy"
    cp_path = ks_dir / "channel_positions.npy"
    if not t_path.is_file() or not cp_path.is_file():
        return {}
    try:
        templates = np.load(t_path)
        chan_pos = np.load(cp_path)
    except Exception:
        return {}
    if templates.ndim != 3 or chan_pos.ndim != 2 or chan_pos.shape[1] < 2:
        return {}
    n_templates, _, n_ch = templates.shape
    if chan_pos.shape[0] != n_ch:
        return {}
    y_um = chan_pos[:, 1].astype(np.float64, copy=False)

    out: dict[int, float] = {}
    # Compute per-template "energy" per channel: peak-to-peak across time.
    # templates[t] shape (nt, n_ch) -> ptp over time -> (n_ch,)
    for cid in range(n_templates):
        if keep_cids is not None and int(cid) not in keep_cids:
            continue
        try:
            amp = np.ptp(templates[cid].astype(np.float64, copy=False), axis=0)
        except Exception:
            continue
        if amp.size != n_ch:
            continue
        peak_ch = int(np.nanargmax(amp))
        y = float(y_um[peak_ch])
        if np.isfinite(y):
            out[int(cid)] = y
    return out


def load_kilosort_dir(
    ks_dir: str | Path,
    *,
    good_only: bool = True,
    max_duration_s: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build spikeTime-compatible channel array from a Kilosort 4 output directory.

    One pipeline channel per Kilosort cluster (cluster index 0 on each channel).
    **snr** is set to **NaN** (no amplitude-based SNR).

    Parameters
    ----------
    ks_dir
        Directory containing spike_times.npy, spike_clusters.npy, params.py.
    good_only
        If True, keep only clusters with **KSLabel** ``good`` in **cluster_group.tsv**.
        If False, keep all clusters (subject to ≥2 spikes per unit below).
    max_duration_s
        If set, keep only spikes with time in ``[0, max_duration_s]`` seconds (wall time).
        ``dataSegmentLength`` / firing-rate denominators use this clipped duration.

    Returns
    -------
    spike_struct
        1D object array of channel objects (.electrode, .snr, .time, .dataSegmentLength).
    meta
        Dict with sample_rate_hz, n_channels, ks_dir, record_len_s.
    """
    ks_dir = Path(ks_dir).resolve()
    if not ks_dir.is_dir():
        raise FileNotFoundError(
            f"Kilosort folder not found (expected a directory with spike_times.npy, etc.):\n  {ks_dir}\n"
            "Fix the path in config, or copy/mount the data so this folder exists."
        )

    st_path = ks_dir / "spike_times.npy"
    sc_path = ks_dir / "spike_clusters.npy"
    params_path = ks_dir / "params.py"
    if not st_path.is_file() or not sc_path.is_file():
        raise FileNotFoundError(f"Missing spike_times.npy or spike_clusters.npy in {ks_dir}")
    if not params_path.is_file():
        raise FileNotFoundError(f"Missing params.py in {ks_dir}")

    fs_hz = _read_sample_rate_hz(params_path)
    spike_times = np.load(st_path)
    spike_clusters_full = np.load(sc_path)
    spike_clusters = spike_clusters_full
    if spike_times.shape != spike_clusters.shape:
        raise ValueError(f"spike_times and spike_clusters shape mismatch in {ks_dir}")

    # Samples -> ms (pipeline uses ms throughout)
    t_ms_all = spike_times.astype(np.float64) / fs_hz * 1000.0

    keep_mask: np.ndarray | None = None
    if max_duration_s is not None:
        if max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        t_end_ms = float(max_duration_s) * 1000.0
        keep_mask = t_ms_all <= t_end_ms
        t_ms_all = t_ms_all[keep_mask]
        spike_clusters = spike_clusters[keep_mask]

    depth_um_by_cid = _depth_um_by_cluster_from_spike_positions(
        ks_dir=ks_dir,
        spike_clusters=spike_clusters_full,
        keep_mask=keep_mask,
    )

    good_ids: set[int] | None = None
    cg = _load_cluster_table(ks_dir, "cluster_group.tsv")
    if cg is not None and good_only:
        label_col = "KSLabel" if "KSLabel" in cg.columns else None
        if label_col is None:
            for c in cg.columns:
                if "label" in c.lower() or c == "group":
                    label_col = c
                    break
        id_col = "cluster_id" if "cluster_id" in cg.columns else cg.columns[0]
        if label_col is not None:
            mask_good = cg[label_col].astype(str).str.lower().eq("good")
            good_ids = set(cg.loc[mask_good, id_col].astype(int).tolist())

    # If spike_positions didn't give depths, fall back to template peak channel depth.
    if not depth_um_by_cid:
        depth_um_by_cid = _depth_um_by_cluster_from_templates(ks_dir, keep_cids=good_ids)

    patient, _ = dataset_labels_kilosort(ks_dir)
    prefix = patient.replace(" ", "_")

    unique_clusters = np.unique(spike_clusters)
    channels: list[Any] = []

    if max_duration_s is not None:
        record_len_s = float(max_duration_s)
    else:
        t_max_ms = float(np.max(t_ms_all)) if t_ms_all.size else 0.0
        record_len_s = max(t_max_ms / 1000.0, 1e-6)

    for cid in unique_clusters:
        cid = int(cid)
        if good_ids is not None and cid not in good_ids:
            continue
        sel = spike_clusters == cid
        if not np.any(sel):
            continue
        spk = np.sort(t_ms_all[sel])
        if spk.size < 2:
            continue

        depth_um = depth_um_by_cid.get(cid, float("nan"))
        depth_tag = f"_d{int(round(depth_um))}um" if np.isfinite(depth_um) else ""
        elec = f"rat_{prefix}_u{cid}_L_0{depth_tag}_CommonFiltered"
        time_obj = np.empty(1, dtype=object)
        time_obj[0] = spk.astype(np.float64)
        snr_arr = np.array([np.nan], dtype=np.float64)

        ch = _SpikeChannel(
            electrode=elec,
            snr=snr_arr,
            time=time_obj,
            dataSegmentLength=record_len_s,
        )
        channels.append(ch)

    if not channels:
        raise RuntimeError(
            f"No units left in {ks_dir} (good_only={good_only}). "
            "Try good_only=False or fix cluster_group.tsv."
        )

    spike_struct = np.array(channels, dtype=object)
    meta = {
        "sample_rate_hz": fs_hz,
        "n_channels": len(channels),
        "ks_dir": str(ks_dir),
        "record_len_s": record_len_s,
        "max_duration_s": max_duration_s,
    }
    return spike_struct, meta


def load_kilosort_dirs(
    ks_dirs: list[str | Path],
    *,
    good_only: bool = True,
    max_duration_s: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load multiple Kilosort folders and combine into one spike_struct (for ``kilosortset:``)."""
    if not ks_dirs:
        raise ValueError("ks_dirs must be non-empty")

    all_channels: list[Any] = []
    record_len_s = 0.0
    fs_hz: float | None = None
    resolved: list[str] = []

    for d in ks_dirs:
        st, meta = load_kilosort_dir(d, good_only=good_only, max_duration_s=max_duration_s)
        all_channels.extend(list(np.ravel(st)))
        record_len_s = max(record_len_s, float(meta.get("record_len_s", 0.0)))
        if fs_hz is None:
            fs_hz = float(meta.get("sample_rate_hz", 0.0))
        resolved.append(str(Path(d).resolve()))

    spike_struct = np.array(all_channels, dtype=object)
    meta = {
        "sample_rate_hz": float(fs_hz or 0.0),
        "n_channels": int(spike_struct.size),
        "ks_dirs": resolved,
        "record_len_s": float(record_len_s),
        "max_duration_s": max_duration_s,
    }
    return spike_struct, meta


def save_spiketime_mat(out_path: str | Path, spike_struct: np.ndarray) -> None:
    """
    Write spikeTime in a MATLAB .mat format readable by scipy.io.loadmat
    (same layout as the pipeline expects: variable ``spikeTime``, struct array).
    """
    import scipy.io as spio

    out_path = Path(out_path)
    flat = np.ravel(spike_struct)
    n = flat.size
    dt = np.dtype(
        [
            ("electrode", object),
            ("snr", object),
            ("time", object),
            ("dataSegmentLength", object),
        ]
    )
    arr = np.empty((n, 1), dtype=dt)
    for i, ch in enumerate(flat):
        arr[i, 0]["electrode"] = ch.electrode
        arr[i, 0]["snr"] = np.asarray(ch.snr)
        arr[i, 0]["time"] = ch.time
        arr[i, 0]["dataSegmentLength"] = ch.dataSegmentLength
    spio.savemat(str(out_path), {"spikeTime": arr}, format="5", do_compression=True)


class _SpikeChannel:
    __slots__ = ("electrode", "snr", "time", "dataSegmentLength")

    def __init__(self, electrode: str, snr: np.ndarray, time: np.ndarray, dataSegmentLength: float):
        self.electrode = electrode
        self.snr = snr
        self.time = time
        self.dataSegmentLength = float(dataSegmentLength)
