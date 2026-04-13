#!/usr/bin/env python3
"""
Region exclusion summary per patient: for each patient folder under root, load results.csv,
compute mean IoU by region combo and Side (L/R separate), and write CSVs under root/<patient>/.
Runs for every patient folder found; use --patient to run for a single patient only.

Single-patient summaries keep original region labels; bucket normalization (e.g. VoSTN/VoSTNSNr → VOSTN) is for cross-patient use only.

Column meanings in output CSVs:
  N_periods: Number of unique periods that contributed (e.g. Period1, Period2).

Usage:
  python cross_patient_region_exclusion.py [--root OUTPUTS_REGION_EXCLUSION]
  python cross_patient_region_exclusion.py --patient PATIENT [--root ...]
  # Run separately for separate GPi vs combined (no script overwrites; use different outputs):
  python cross_patient_region_exclusion.py --root outputs_region_exclusion --run-tag-prefix separateGPi --out-suffix _separateGPi
  python cross_patient_region_exclusion.py --root outputs_region_exclusion --run-tag-exclude-prefix separateGPi --out-suffix _combined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objs as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from pipeline.burst_paths import burst_network_csv_path, network_burst_csv_basename

try:
    from config import CROSS_PATIENT_EXCLUDE_PATIENTS
except ImportError:
    CROSS_PATIENT_EXCLUDE_PATIENTS = []

try:
    from pipeline.region_exclusion import (
        mean_onset_start_error_pred_to_gt_ms,
        matched_pairs_metrics,
    )
except ImportError:
    mean_onset_start_error_pred_to_gt_ms = None
    matched_pairs_metrics = None

# Region → bucket mapping for cross-patient aggregation (single-patient keeps original labels).
# GPi1, GPi2, GPi3 always bucket to GPi for cross-patient.
REGION_TO_BUCKET = {
    "vim": "VIMPPN",
    "ppn": "VIMPPN",
    "vimppn": "VIMPPN",
    "vostn": "VOSTN",
    "vostnsnr": "VOSTN",
    "va": "VA",
    "gpi": "GPi",
    "gpi1": "GPi",
    "gpi2": "GPi",
    "gpi3": "GPi",
    "cmcl": "CMCL",
    "ant": "ANT",
}


def region_to_bucket(name: str) -> str:
    """Map a single region name to its canonical bucket (unknown names returned as-is)."""
    key = name.strip()
    if not key:
        return key
    return REGION_TO_BUCKET.get(key.lower(), key)


def normalize_regions_included(regions_str: str) -> str:
    """
    Convert a pipe-separated Regions_included string to a normalized bucket combo.
    E.g. 'CMCL|GPi|VoSTNSNr' -> 'CMCL|GPi|VOSTN' so VoSTN and VoSTNSNr combine.
    """
    if not regions_str or not isinstance(regions_str, str):
        return regions_str
    parts = [p.strip() for p in regions_str.split("|") if p.strip()]
    buckets = sorted({region_to_bucket(p) for p in parts})
    return "|".join(buckets)


def _has_multiple_gpi_variants(regions_str: str) -> bool:
    """
    Return True if Regions_included (original, unbucketed) contains >1 distinct GPi variant
    (e.g. GPi1 and GPi2 together). Used only for cross-patient filtering so that
    combos with multiple GPi* are excluded from cross-patient summaries.
    """
    if not regions_str or not isinstance(regions_str, str):
        return False
    gpi_labels = {"gpi", "gpi1", "gpi2", "gpi3"}
    parts = [p.strip().lower() for p in regions_str.split("|") if p.strip()]
    present = {p for p in parts if p in gpi_labels}
    return len(present) > 1


def _run_tag_matches(
    path: Path, root: Path,
    run_tag_include_prefix: str | None,
    run_tag_exclude_prefix: str | None,
) -> bool:
    """True if path should be included given optional run_tag filters (path relative to root: patient/period/method/run_tag/...)."""
    if not run_tag_include_prefix and not run_tag_exclude_prefix:
        return True
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            return True
        run_tag = parts[3]
        if run_tag_include_prefix and not run_tag.startswith(run_tag_include_prefix):
            return False
        if run_tag_exclude_prefix and run_tag.startswith(run_tag_exclude_prefix):
            return False
        return True
    except ValueError:
        return True


def find_all_patients(
    root: Path,
    run_tag_include_prefix: str | None = None,
    run_tag_exclude_prefix: str | None = None,
) -> list[str]:
    """Return sorted list of patient folder names that contain at least one results.csv (optionally filtered by run_tag)."""
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        return []
    patients = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            for p in entry.rglob("results.csv"):
                if _run_tag_matches(p, root, run_tag_include_prefix, run_tag_exclude_prefix):
                    patients.append(entry.name)
                    break
    return sorted(patients)


def find_patient_results_csv(
    root: Path,
    patient: str,
    run_tag_include_prefix: str | None = None,
    run_tag_exclude_prefix: str | None = None,
) -> list[Path]:
    """Return list of results.csv paths under root/patient/ (recursive), optionally filtered by run_tag."""
    root = Path(root).resolve()
    patient_dir = root / patient
    if not patient_dir.exists() or not patient_dir.is_dir():
        return []
    paths = sorted(patient_dir.rglob("results.csv"))
    if run_tag_include_prefix or run_tag_exclude_prefix:
        paths = [
            p for p in paths
            if _run_tag_matches(p, root, run_tag_include_prefix, run_tag_exclude_prefix)
        ]
    return paths


def load_and_annotate(path: Path, root: Path, patient: str) -> pd.DataFrame | None:
    """
    Read results.csv and add columns: patient, period, method, run_tag from path.
    Path is .../outputs_region_exclusion/patient/period/method/run_tag/.../results.csv
    If Pct_GT_matched is missing but N_gt_matched and N_gt_spans exist, compute it for backward compatibility.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.copy()
    if "Pct_GT_matched" not in df.columns and "N_gt_matched" in df.columns and "N_gt_spans" in df.columns:
        n_gt = pd.to_numeric(df["N_gt_spans"], errors="coerce")
        n_matched = pd.to_numeric(df["N_gt_matched"], errors="coerce")
        df["Pct_GT_matched"] = np.where(n_gt > 0, 100.0 * n_matched / n_gt, float("nan"))
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        df["patient"] = patient
        if len(parts) < 4:
            return df
        period = parts[1]
        method = parts[2]
        run_tag = parts[3]
        df["period"] = period
        df["method"] = method
        df["run_tag"] = run_tag
        df["session"] = f"{patient}_{period}"
        return df
    except ValueError:
        df["patient"] = patient
        return df


def load_patient(
    root: Path,
    patient: str,
    run_tag_include_prefix: str | None = None,
    run_tag_exclude_prefix: str | None = None,
) -> pd.DataFrame:
    """Load all results.csv for one patient under root/patient/ into one DataFrame."""
    paths = find_patient_results_csv(
        root, patient,
        run_tag_include_prefix=run_tag_include_prefix,
        run_tag_exclude_prefix=run_tag_exclude_prefix,
    )
    if not paths:
        return pd.DataFrame()
    root = Path(root).resolve()
    frames = []
    for p in paths:
        df = load_and_annotate(p, root, patient)
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all(
    root: Path,
    patients: list[str] | None = None,
    run_tag_include_prefix: str | None = None,
    run_tag_exclude_prefix: str | None = None,
) -> pd.DataFrame:
    """Load all results.csv for all (or given) patients into one DataFrame."""
    root = Path(root).resolve()
    if patients is None:
        patients = find_all_patients(
            root,
            run_tag_include_prefix=run_tag_include_prefix,
            run_tag_exclude_prefix=run_tag_exclude_prefix,
        )
    if not patients:
        return pd.DataFrame()
    frames = [
        load_patient(
            root, p,
            run_tag_include_prefix=run_tag_include_prefix,
            run_tag_exclude_prefix=run_tag_exclude_prefix,
        )
        for p in patients
    ]
    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _regions_included_to_perm_folder(regions_str: str) -> str:
    """Convert Regions_included (pipe-separated) to permutation folder name (matches pipeline)."""
    if not regions_str or not isinstance(regions_str, str):
        return ""
    parts = [p.strip() for p in str(regions_str).split("|") if p.strip()]
    return "_".join(sorted(parts)).replace(" ", "_").replace("/", "-")


def _infer_rs_burst_root(region_exclusion_root: Path) -> Path:
    """Infer RS burst output root from region-exclusion root (same layout as main.py)."""
    root = Path(region_exclusion_root).resolve()
    name = root.name.replace("outputs_region_exclusion", "outputs_RS_burst")
    return root.parent / name


def enrich_pred_to_gt_from_onsets(
    df: pd.DataFrame,
    root: Path,
    *,
    rs_burst_root: Path | None = None,
) -> pd.DataFrame:
    """
    Compute Mean_onset_start_error_pred_to_gt_ms from existing network_bursts CSVs when
    the column is missing, so the full pipeline does not need to be re-run.
    GT onsets: rs_burst_root/patient/period/run_tag/burst_timings/network_bursts_L.csv (or R)
    Pred onsets: root/patient/period/run_tag/{perm_folder}/network_bursts_L.csv
    """
    if mean_onset_start_error_pred_to_gt_ms is None:
        return df
    need = {"patient", "period", "method", "run_tag", "Regions_included", "Side"}
    if not need.issubset(df.columns):
        return df
    root = Path(root).resolve()
    rs = Path(rs_burst_root).resolve() if rs_burst_root else _infer_rs_burst_root(root)

    gt_cache: dict[tuple[str, str, str, str, str], list[float]] = {}
    pred_cache: dict[tuple[str, str, str, str, str, str], list[float]] = {}

    def load_onset_starts(csv_path: Path) -> list[float]:
        if not csv_path.exists():
            return []
        try:
            tab = pd.read_csv(csv_path)
            if "onset_start_ms" not in tab.columns:
                return []
            return pd.to_numeric(tab["onset_start_ms"], errors="coerce").dropna().tolist()
        except Exception:
            return []

    out = df.copy()
    if "Mean_onset_start_error_pred_to_gt_ms" not in out.columns:
        out["Mean_onset_start_error_pred_to_gt_ms"] = float("nan")

    for i, row in out.iterrows():
        if pd.notna(out.at[i, "Mean_onset_start_error_pred_to_gt_ms"]):
            continue
        patient = str(row["patient"])
        period = str(row["period"])
        method = str(row["method"])
        run_tag = str(row["run_tag"])
        side = str(row["Side"]).strip().lower()
        side_file = "left" if side == "left" else "right"
        regions = str(row["Regions_included"])
        perm_folder = _regions_included_to_perm_folder(regions)
        if not perm_folder:
            continue
        key_gt = (patient, period, method, run_tag, side_file)
        if key_gt not in gt_cache:
            gt_path = burst_network_csv_path(rs / patient / period / run_tag, side_file)
            gt_cache[key_gt] = load_onset_starts(gt_path)
        key_pred = (patient, period, method, run_tag, perm_folder, side_file)
        if key_pred not in pred_cache:
            pred_path = root / patient / period / run_tag / perm_folder / network_burst_csv_basename(side_file)
            pred_cache[key_pred] = load_onset_starts(pred_path)
        gt_starts = gt_cache[key_gt]
        pred_starts = pred_cache[key_pred]
        out.at[i, "Mean_onset_start_error_pred_to_gt_ms"] = mean_onset_start_error_pred_to_gt_ms(
            gt_starts, pred_starts
        )

    return out


def _load_network_bursts_spans_and_onsets(csv_path: Path) -> tuple[list[tuple[float, float]], list[float]]:
    """Load (span_start_ms, span_end_ms) and onset_start_ms from a network_bursts CSV. Returns (spans, onsets)."""
    if not csv_path.exists():
        return [], []
    try:
        tab = pd.read_csv(csv_path)
        if "onset_start_ms" not in tab.columns:
            return [], []
        spans: list[tuple[float, float]] = []
        onsets: list[float] = []
        has_span = "span_start_ms" in tab.columns and "span_end_ms" in tab.columns
        for _, row in tab.iterrows():
            o = pd.to_numeric(row["onset_start_ms"], errors="coerce")
            if not np.isfinite(o):
                continue
            onsets.append(float(o))
            if has_span:
                s = pd.to_numeric(row["span_start_ms"], errors="coerce")
                e = pd.to_numeric(row["span_end_ms"], errors="coerce")
                if np.isfinite(s) and np.isfinite(e) and e > s:
                    spans.append((float(s), float(e)))
                else:
                    e2 = pd.to_numeric(row.get("onset_end_ms", o + 1), errors="coerce")
                    spans.append((float(o), float(e2) if np.isfinite(e2) else float(o) + 1.0))
            else:
                e2 = pd.to_numeric(row.get("onset_end_ms", o + 1), errors="coerce")
                spans.append((float(o), float(e2) if np.isfinite(e2) else float(o) + 1.0))
        return spans, onsets
    except Exception:
        return [], []


def enrich_matched_pairs_accuracy(
    df: pd.DataFrame,
    root: Path,
    *,
    rs_burst_root: Path | None = None,
) -> pd.DataFrame:
    """
    Add Mean_onset_error_matched_pairs_ms and N_matched_pairs: for each GT we keep only
    the overlapping pred with smallest onset error; other preds are discarded. Measures
    accuracy of the predictions we do make (even if we make fewer of them).
    """
    if matched_pairs_metrics is None:
        return df
    need = {"patient", "period", "method", "run_tag", "Regions_included", "Side"}
    if not need.issubset(df.columns):
        return df
    root = Path(root).resolve()
    rs = Path(rs_burst_root).resolve() if rs_burst_root else _infer_rs_burst_root(root)

    gt_cache: dict[tuple[str, str, str, str, str], tuple[list[tuple[float, float]], list[float]]] = {}
    pred_cache: dict[tuple[str, str, str, str, str, str], tuple[list[tuple[float, float]], list[float]]] = {}

    out = df.copy()
    out["Mean_onset_error_matched_pairs_ms"] = float("nan")
    out["N_matched_pairs"] = 0
    out["N_pred_kept"] = 0
    out["N_pred_overlapping_any_gt"] = 0
    out["Pct_pred_matched_matched_pairs"] = float("nan")

    for i, row in out.iterrows():
        patient = str(row["patient"])
        period = str(row["period"])
        method = str(row["method"])
        run_tag = str(row["run_tag"])
        side = str(row["Side"]).strip().lower()
        side_file = "left" if side == "left" else "right"
        regions = str(row["Regions_included"])
        perm_folder = _regions_included_to_perm_folder(regions)
        if not perm_folder:
            continue
        key_gt = (patient, period, method, run_tag, side_file)
        if key_gt not in gt_cache:
            gt_path = burst_network_csv_path(rs / patient / period / run_tag, side_file)
            gt_cache[key_gt] = _load_network_bursts_spans_and_onsets(gt_path)
        key_pred = (patient, period, method, run_tag, perm_folder, side_file)
        if key_pred not in pred_cache:
            pred_path = root / patient / period / run_tag / perm_folder / network_burst_csv_basename(side_file)
            pred_cache[key_pred] = _load_network_bursts_spans_and_onsets(pred_path)
        gt_spans, gt_onsets = gt_cache[key_gt]
        pred_spans, pred_onsets = pred_cache[key_pred]
        mean_err, n_pairs, n_pred_kept, n_pred_overlapping_any_gt = matched_pairs_metrics(
            gt_spans, pred_spans, gt_onsets, pred_onsets
        )
        out.at[i, "Mean_onset_error_matched_pairs_ms"] = mean_err
        out.at[i, "N_matched_pairs"] = n_pairs
        out.at[i, "N_pred_kept"] = n_pred_kept
        out.at[i, "N_pred_overlapping_any_gt"] = n_pred_overlapping_any_gt
        # Exclude overlapping-but-discarded preds from denominator (don't punish for them)
        n_pred_total = len(pred_spans)
        n_discarded = n_pred_overlapping_any_gt - n_pred_kept  # overlapped but not kept
        denom = n_pred_total - n_discarded
        out.at[i, "Pct_pred_matched_matched_pairs"] = (100.0 * n_pred_kept / denom) if denom > 0 else float("nan")

    return out


def summary_matched_pairs_accuracy_cross_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-patient summary for matched-pairs accuracy: one row per (Regions_included_bucket, Side).
    N_pred_kept = preds that were the kept match for a GT (overlap + smallest onset error).
    Overlapping-but-discarded preds (overlapped a GT but not smallest onset error) are excluded
    from the percentage denominator so we don't punish for them. So pct = 100 * N_pred_kept / (N_pred_spans - N_discarded).
    Percentage reported is mean of per-patient percentages.
    """
    if df.empty or "Regions_included" not in df.columns or "patient" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "Side" not in df.columns:
        df["Side"] = "left"
    df["Regions_included"] = df["Regions_included"].astype(str).map(normalize_regions_included)
    if "Mean_onset_error_matched_pairs_ms" not in df.columns or "N_matched_pairs" not in df.columns:
        return pd.DataFrame()
    if "N_pred_kept" not in df.columns or "N_pred_spans" not in df.columns:
        return pd.DataFrame()
    if "N_pred_overlapping_any_gt" not in df.columns:
        return pd.DataFrame()
    # Per patient: denominator excludes overlapping-but-discarded preds
    by_patient = df.groupby(["Regions_included", "Side", "patient"]).agg(
        N_pred_kept=("N_pred_kept", "sum"),
        N_pred_spans=("N_pred_spans", "sum"),
        N_pred_overlapping_any_gt=("N_pred_overlapping_any_gt", "sum"),
    ).reset_index()
    n_discarded = by_patient["N_pred_overlapping_any_gt"] - by_patient["N_pred_kept"]
    denom = by_patient["N_pred_spans"] - n_discarded
    by_patient["Pct_pred_patient"] = np.where(
        denom > 0,
        100.0 * by_patient["N_pred_kept"] / denom,
        float("nan"),
    )
    # Cross-patient: mean and median of those per-patient percentages, and other aggregates
    grouped = df.groupby(["Regions_included", "Side"])
    pct_agg = by_patient.groupby(["Regions_included", "Side"])["Pct_pred_patient"].agg(["mean", "median"]).reset_index()
    pct_agg = pct_agg.rename(columns={"mean": "mean_Pct_pred_matched_matched_pairs", "median": "median_Pct_pred_matched_matched_pairs"})
    agg = grouped.agg({
        "session": "nunique",
        "patient": "nunique",
        "Mean_onset_error_matched_pairs_ms": ["mean", "median"],
        "N_matched_pairs": "sum",
        "K_excluded": "min",
    })
    agg.columns = [
        "N_periods", "N_patients",
        "mean_Mean_onset_error_matched_pairs_ms", "median_Mean_onset_error_matched_pairs_ms",
        "total_N_matched_pairs", "K_excluded",
    ]
    if "Pct_GT_matched" in df.columns:
        agg["mean_Pct_GT_matched"] = grouped["Pct_GT_matched"].mean().values
        agg["median_Pct_GT_matched"] = grouped["Pct_GT_matched"].median().values
    agg = agg.reset_index()
    agg = agg.merge(pct_agg, on=["Regions_included", "Side"], how="left")
    agg["patients"] = grouped["patient"].apply(
        lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))
    ).values
    agg = agg.sort_values(["K_excluded", "Regions_included", "Side"]).reset_index(drop=True)
    return agg


def summary_by_region_combo_single_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    For one patient's df: for each (Regions_included, Side), compute N_periods,
    mean_Mean_IoU, median_Mean_IoU, and if present mean_Pct_GT_matched, median_Pct_GT_matched.
    Left and right always separated.
    """
    if df.empty or "Regions_included" not in df.columns:
        return pd.DataFrame()
    if "Side" not in df.columns:
        df = df.copy()
        df["Side"] = "left"
    if "session" not in df.columns or "Mean_IoU" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(["Regions_included", "Side"])
    agg = grouped.agg({"session": "nunique", "Mean_IoU": ["mean", "median"]})
    agg.columns = ["N_periods", "mean_Mean_IoU", "median_Mean_IoU"]
    agg = agg.reset_index()
    if "Pct_GT_matched" in df.columns:
        agg["mean_Pct_GT_matched"] = grouped["Pct_GT_matched"].mean().values
        agg["median_Pct_GT_matched"] = grouped["Pct_GT_matched"].median().values
    if "Pct_pred_matched" in df.columns:
        agg["mean_Pct_pred_matched"] = grouped["Pct_pred_matched"].mean().values
        agg["median_Pct_pred_matched"] = grouped["Pct_pred_matched"].median().values
    if "Mean_onset_start_error_ms" in df.columns:
        agg["mean_Mean_onset_start_error_ms"] = grouped["Mean_onset_start_error_ms"].mean().values
        agg["median_Mean_onset_start_error_ms"] = grouped["Mean_onset_start_error_ms"].median().values
    if "Mean_onset_start_error_pred_to_gt_ms" in df.columns:
        agg["mean_Mean_onset_start_error_pred_to_gt_ms"] = grouped["Mean_onset_start_error_pred_to_gt_ms"].mean().values
        agg["median_Mean_onset_start_error_pred_to_gt_ms"] = grouped["Mean_onset_start_error_pred_to_gt_ms"].median().values
    if "Pct_pred_matched_matched_pairs" in df.columns:
        agg["mean_Pct_pred_matched_matched_pairs"] = grouped["Pct_pred_matched_matched_pairs"].mean().values
        agg["median_Pct_pred_matched_matched_pairs"] = grouped["Pct_pred_matched_matched_pairs"].median().values
    if "Mean_onset_error_matched_pairs_ms" in df.columns:
        agg["mean_Mean_onset_error_matched_pairs_ms"] = grouped["Mean_onset_error_matched_pairs_ms"].mean().values
        agg["median_Mean_onset_error_matched_pairs_ms"] = grouped["Mean_onset_error_matched_pairs_ms"].median().values
    if "N_matched_pairs" in df.columns:
        agg["total_N_matched_pairs"] = grouped["N_matched_pairs"].sum().values
    k_map = df.groupby("Regions_included")["K_excluded"].first()
    agg["K_excluded"] = agg["Regions_included"].map(k_map)
    agg = agg.sort_values(["K_excluded", "Regions_included", "Side"]).reset_index(drop=True)
    return agg


def summary_by_region_combo_cross_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-patient: for each (Regions_included_bucket, Side), compute N_periods,
    N_patients, patients (comma-sep), mean_Mean_IoU, median_Mean_IoU, and if present
    mean_Pct_GT_matched, median_Pct_GT_matched.
    Uses bucket normalization so e.g. VoSTN and VoSTNSNr combine.
    """
    if df.empty or "Regions_included" not in df.columns or "patient" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "Side" not in df.columns:
        df["Side"] = "left"
    df["Regions_included"] = df["Regions_included"].astype(str).map(normalize_regions_included)
    grouped = df.groupby(["Regions_included", "Side"])
    agg = grouped.agg({
        "session": "nunique",
        "patient": "nunique",
        "Mean_IoU": ["mean", "median"],
        "K_excluded": "min",
    })
    agg.columns = ["N_periods", "N_patients", "mean_Mean_IoU", "median_Mean_IoU", "K_excluded"]
    agg = agg.reset_index()
    agg["patients"] = grouped["patient"].apply(
        lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))
    ).values
    if "Pct_GT_matched" in df.columns:
        agg["mean_Pct_GT_matched"] = grouped["Pct_GT_matched"].mean().values
        agg["median_Pct_GT_matched"] = grouped["Pct_GT_matched"].median().values
    if "Pct_pred_matched" in df.columns:
        agg["mean_Pct_pred_matched"] = grouped["Pct_pred_matched"].mean().values
        agg["median_Pct_pred_matched"] = grouped["Pct_pred_matched"].median().values
    if "Mean_onset_start_error_ms" in df.columns:
        agg["mean_Mean_onset_start_error_ms"] = grouped["Mean_onset_start_error_ms"].mean().values
        agg["median_Mean_onset_start_error_ms"] = grouped["Mean_onset_start_error_ms"].median().values
    if "Mean_onset_start_error_pred_to_gt_ms" in df.columns:
        agg["mean_Mean_onset_start_error_pred_to_gt_ms"] = grouped["Mean_onset_start_error_pred_to_gt_ms"].mean().values
        agg["median_Mean_onset_start_error_pred_to_gt_ms"] = grouped["Mean_onset_start_error_pred_to_gt_ms"].median().values
    agg = agg.sort_values(["K_excluded", "Regions_included", "Side"]).reset_index(drop=True)
    return agg


def _boxplot_one_side(
    df: pd.DataFrame,
    side: str,
    out_path: Path | None,
    *,
    title_suffix: str,
    value_col: str = "Mean_IoU",
    plot_title: str = "Mean IoU by region combo (bucket)",
    yaxis_title: str = "Mean IoU",
) -> None:
    """Single Plotly boxplot for one side: x = bucket, y = value_col."""
    if value_col not in df.columns:
        return
    side_lower = str(side).lower()
    df_side = df[df["Side"].astype(str).str.lower() == side_lower].copy()
    if df_side.empty:
        return
    df_side["_bucket"] = df_side["Regions_included"].astype(str).map(normalize_regions_included)
    order_df = (
        df_side[["_bucket", "K_excluded"]]
        .drop_duplicates()
        .sort_values(["K_excluded", "_bucket"])
    )
    bucket_order = order_df["_bucket"].tolist()[::-1]  # descending left to right
    if not bucket_order:
        return
    fig = go.Figure()
    for bucket in bucket_order:
        vals = df_side.loc[df_side["_bucket"] == bucket, value_col].dropna()
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Box(
                y=vals.values,
                name=bucket,
                boxpoints="outliers",
                marker_color="lightsteelblue",
                line_color="rgb(70,100,140)",
            )
        )
    fig.update_layout(
        title=f"Cross-patient: {plot_title} — {title_suffix}",
        xaxis_title="Region combo (bucket)",
        yaxis_title=yaxis_title,
        showlegend=False,
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(b=120),
    )
    if out_path:
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")


def plot_cross_patient_boxplot_by_bucket(
    df: pd.DataFrame,
    out_dir: Path | None = None,
) -> None:
    """
    Plotly boxplots: Left and Right for Mean IoU, and if Pct_GT_matched exists, for % GT matched too.
    Saves HTML to out_dir if provided.
    """
    if not HAS_PLOTLY:
        return
    if df.empty or "Mean_IoU" not in df.columns:
        return
    df = df.copy()
    if "Side" not in df.columns:
        df["Side"] = "left"
    # IoU boxplots
    left_path = Path(out_dir) / "mean_iou_boxplot_left.html" if out_dir else None
    right_path = Path(out_dir) / "mean_iou_boxplot_right.html" if out_dir else None
    _boxplot_one_side(df, "left", left_path, title_suffix="Left")
    _boxplot_one_side(df, "right", right_path, title_suffix="Right")
    # % GT matched boxplots (percentage of ground-truth network bursts that are matched)
    if "Pct_GT_matched" in df.columns:
        pct_left = Path(out_dir) / "pct_gt_matched_boxplot_left.html" if out_dir else None
        pct_right = Path(out_dir) / "pct_gt_matched_boxplot_right.html" if out_dir else None
        _boxplot_one_side(
            df, "left", pct_left, title_suffix="Left",
            value_col="Pct_GT_matched",
            plot_title="% of GT network bursts matched",
            yaxis_title="% GT matched",
        )
        _boxplot_one_side(
            df, "right", pct_right, title_suffix="Right",
            value_col="Pct_GT_matched",
            plot_title="% of GT network bursts matched",
            yaxis_title="% GT matched",
        )
    # % pred matched boxplots (percentage of predicted bursts that overlap ≥1 GT)
    if "Pct_pred_matched" in df.columns:
        pp_left = Path(out_dir) / "pct_pred_matched_boxplot_left.html" if out_dir else None
        pp_right = Path(out_dir) / "pct_pred_matched_boxplot_right.html" if out_dir else None
        _boxplot_one_side(
            df, "left", pp_left, title_suffix="Left",
            value_col="Pct_pred_matched",
            plot_title="% of predicted network bursts matched (overlap)",
            yaxis_title="% pred matched",
        )
        _boxplot_one_side(
            df, "right", pp_right, title_suffix="Right",
            value_col="Pct_pred_matched",
            plot_title="% of predicted network bursts matched (overlap)",
            yaxis_title="% pred matched",
        )
    # Onset start error (ms) boxplots: GT→nearest pred
    if "Mean_onset_start_error_ms" in df.columns:
        onset_left = Path(out_dir) / "mean_onset_start_error_boxplot_left.html" if out_dir else None
        onset_right = Path(out_dir) / "mean_onset_start_error_boxplot_right.html" if out_dir else None
        _boxplot_one_side(
            df, "left", onset_left, title_suffix="Left",
            value_col="Mean_onset_start_error_ms",
            plot_title="Mean onset start error (GT→nearest pred)",
            yaxis_title="Onset start error (ms)",
        )
        _boxplot_one_side(
            df, "right", onset_right, title_suffix="Right",
            value_col="Mean_onset_start_error_ms",
            plot_title="Mean onset start error (GT→nearest pred)",
            yaxis_title="Onset start error (ms)",
        )
    # Onset start error (ms) boxplots: pred→nearest GT
    if "Mean_onset_start_error_pred_to_gt_ms" in df.columns:
        p2g_left = Path(out_dir) / "mean_onset_start_error_pred_to_gt_boxplot_left.html" if out_dir else None
        p2g_right = Path(out_dir) / "mean_onset_start_error_pred_to_gt_boxplot_right.html" if out_dir else None
        _boxplot_one_side(
            df, "left", p2g_left, title_suffix="Left",
            value_col="Mean_onset_start_error_pred_to_gt_ms",
            plot_title="Mean onset start error (pred→nearest GT)",
            yaxis_title="Onset start error (ms)",
        )
        _boxplot_one_side(
            df, "right", p2g_right, title_suffix="Right",
            value_col="Mean_onset_start_error_pred_to_gt_ms",
            plot_title="Mean onset start error (pred→nearest GT)",
            yaxis_title="Onset start error (ms)",
        )


def process_patient(
    root: Path,
    patient: str,
    *,
    run_tag_include_prefix: str | None = None,
    run_tag_exclude_prefix: str | None = None,
    out_suffix: str = "",
    rs_burst_root: Path | None = None,
) -> None:
    """Load one patient's results, compute summaries, and write CSVs under root/patient/."""
    df = load_patient(
        root, patient,
        run_tag_include_prefix=run_tag_include_prefix,
        run_tag_exclude_prefix=run_tag_exclude_prefix,
    )
    if df.empty:
        print(f"  No results.csv under {root / patient}; skipping.")
        return
    df = enrich_pred_to_gt_from_onsets(df, root, rs_burst_root=rs_burst_root)
    df = enrich_matched_pairs_accuracy(df, root, rs_burst_root=rs_burst_root)

    print(f"  {patient}: {len(df)} rows, {df['session'].nunique()} sessions")

    summary_combo = summary_by_region_combo_single_patient(df)
    if not summary_combo.empty:
        summary_path = root / patient / f"summary_by_region_combo{out_suffix}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_combo.to_csv(summary_path, index=False)
        print(f"    → {summary_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Region exclusion summary per patient: Mean IoU by region combo and Side (L/R separate), write CSVs under root/patient/. Runs for every patient folder by default."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="If set, process only this patient; otherwise process every folder that contains results.csv",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs_region_exclusion"),
        help="Root folder containing patient/period/method/run_tag/results.csv",
    )
    parser.add_argument(
        "--run-tag-prefix",
        type=str,
        default=None,
        help="If set, only include results.csv under run_tag folders whose name starts with this (e.g. separateGPi).",
    )
    parser.add_argument(
        "--run-tag-exclude-prefix",
        type=str,
        default=None,
        help="If set, exclude results.csv under run_tag folders whose name starts with this (e.g. separateGPi).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Suffix for output dir and files (e.g. _separateGPi, _combined) so multiple runs do not overwrite.",
    )
    parser.add_argument(
        "--rs-burst-root",
        type=Path,
        default=None,
        help="Root for GT network_bursts CSVs (default: inferred from --root, e.g. outputs_RS_burst_onset_length).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return

    run_tag_include = args.run_tag_prefix
    run_tag_exclude = args.run_tag_exclude_prefix
    out_suffix = args.out_suffix or ""
    rs_burst_root = Path(args.rs_burst_root).resolve() if args.rs_burst_root else None

    if args.patient:
        patients = [args.patient]
        if not (root / args.patient).exists():
            print(f"Patient folder does not exist: {root / args.patient}")
            return
        for patient in patients:
            process_patient(
                root, patient,
                run_tag_include_prefix=run_tag_include,
                run_tag_exclude_prefix=run_tag_exclude,
                out_suffix=out_suffix,
                rs_burst_root=rs_burst_root,
            )
        return

    patients = find_all_patients(
        root,
        run_tag_include_prefix=run_tag_include,
        run_tag_exclude_prefix=run_tag_exclude,
    )
    if not patients:
        print(f"No patient folders with results.csv found under {root}")
        if "outputs" in str(root) and "region_exclusion" not in str(root):
            print("Hint: region-exclusion results.csv are written to outputs_region_exclusion by main.py, not under outputs. Try: --root outputs_region_exclusion")
        return
    print(f"Found {len(patients)} patient(s): {', '.join(patients)}")

    exclude_set = set(CROSS_PATIENT_EXCLUDE_PATIENTS or [])
    patients_for_cross = [p for p in patients if p not in exclude_set]
    if exclude_set:
        excluded = [p for p in patients if p in exclude_set]
        print(f"Excluding from cross-patient (config): {', '.join(sorted(excluded))} → {len(patients_for_cross)} patient(s) for cross-patient.")

    for patient in patients:
        process_patient(
            root, patient,
            run_tag_include_prefix=run_tag_include,
            run_tag_exclude_prefix=run_tag_exclude,
            out_suffix=out_suffix,
            rs_burst_root=rs_burst_root,
        )

    # Cross-patient: bucket summary + boxplot, in a subfolder per run_tag (excluded patients omitted)
    df_all = load_all(
        root, patients_for_cross,
        run_tag_include_prefix=run_tag_include,
        run_tag_exclude_prefix=run_tag_exclude,
    )
    if df_all.empty:
        print("No data for cross-patient summary.")
        return
    df_all = enrich_pred_to_gt_from_onsets(df_all, root, rs_burst_root=rs_burst_root)
    df_all = enrich_matched_pairs_accuracy(df_all, root, rs_burst_root=rs_burst_root)

    # Exclude rows where the original Regions_included had multiple GPi variants together
    # (e.g. GPi1+GPi2). These still count for per-patient summaries but are ignored
    # in cross-patient aggregation to avoid mixed-GPi combos driving the buckets.
    if "Regions_included" in df_all.columns:
        mask_multi_gpi = df_all["Regions_included"].apply(_has_multiple_gpi_variants)
        if mask_multi_gpi.any():
            n_drop = int(mask_multi_gpi.sum())
            print(f"Excluding {n_drop} row(s) with multiple GPi variants in Regions_included from cross-patient summary.")
            df_all = df_all[~mask_multi_gpi]
            if df_all.empty:
                print("No data left for cross-patient summary after GPi filtering.")
                return

    base_out = root / f"cross_patient{out_suffix}"
    base_out.mkdir(parents=True, exist_ok=True)

    if "run_tag" in df_all.columns and df_all["run_tag"].notna().any():
        run_tags = sorted(df_all["run_tag"].dropna().astype(str).unique())
    else:
        run_tags = [None]  # no run_tag column or all NaN: write once under base_out

    for rtag in run_tags:
        if rtag is None:
            df_sub = df_all
            out_dir = base_out
        else:
            df_sub = df_all[df_all["run_tag"].astype(str) == rtag]
            if df_sub.empty:
                continue
            out_dir = base_out / rtag
            out_dir.mkdir(parents=True, exist_ok=True)

        if "patient" not in df_sub.columns or df_sub["patient"].nunique() < 2:
            print(f"Skipping cross-patient save for [{rtag or 'all'}]: only one patient.")
            continue
        summary_cp = summary_by_region_combo_cross_patient(df_sub)
        if summary_cp.empty:
            continue
        # Keep only region combos that appear in at least 2 patients (true cross-patient)
        summary_cp = summary_cp[summary_cp["N_patients"] >= 2]
        if summary_cp.empty:
            print(f"No region combos with ≥2 patients for [{rtag or 'all'}]; skipping cross-patient save.")
            continue
        for side_name, side_val in [("left", "left"), ("right", "right")]:
            sub = summary_cp[summary_cp["Side"].astype(str).str.strip().str.lower() == side_val]
            if not sub.empty:
                summary_path = out_dir / f"summary_by_region_combo_{side_name}{out_suffix}.csv"
                sub.to_csv(summary_path, index=False)
                print(f"Cross-patient summary {side_name} ({len(sub)} bucket rows) [{rtag or 'all'}] → {summary_path}")

        # Matched-pairs accuracy: of the predictions we keep (one per GT, smallest onset error), how accurate?
        summary_mp = summary_matched_pairs_accuracy_cross_patient(df_sub)
        if not summary_mp.empty:
            summary_mp = summary_mp[summary_mp["N_patients"] >= 2]
        if not summary_mp.empty:
            for side_name, side_val in [("left", "left"), ("right", "right")]:
                sub = summary_mp[summary_mp["Side"].astype(str).str.strip().str.lower() == side_val]
                if not sub.empty:
                    acc_path = out_dir / f"accuracy_matched_pairs_{side_name}{out_suffix}.csv"
                    sub.to_csv(acc_path, index=False)
                    print(f"Matched-pairs accuracy {side_name} ({len(sub)} bucket rows) [{rtag or 'all'}] → {acc_path}")

        if HAS_PLOTLY:
            # Restrict boxplots to (bucket, Side) combos that have ≥2 patients
            df_plot = df_sub.copy()
            df_plot["_bucket"] = df_plot["Regions_included"].astype(str).map(normalize_regions_included)
            n_pat = df_plot.groupby(["_bucket", "Side"])["patient"].nunique()
            valid_pairs = set(n_pat[n_pat >= 2].index)
            df_plot = df_plot[df_plot.apply(lambda r: (r["_bucket"], r["Side"]) in valid_pairs, axis=1)].drop(columns=["_bucket"])
            plot_cross_patient_boxplot_by_bucket(df_plot, out_dir=out_dir)
        else:
            print("Install plotly to generate cross-patient boxplots: pip install plotly")


if __name__ == "__main__":
    main()
