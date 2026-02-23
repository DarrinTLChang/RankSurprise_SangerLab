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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objs as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Region → bucket mapping for cross-patient aggregation (single-patient keeps original labels).
# GPi1 and GPi2 always bucket to GPi for cross-patient.
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


def find_all_patients(root: Path) -> list[str]:
    """Return sorted list of patient folder names that contain at least one results.csv."""
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        return []
    patients = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            if any(entry.rglob("results.csv")):
                patients.append(entry.name)
    return sorted(patients)


def find_patient_results_csv(root: Path, patient: str) -> list[Path]:
    """Return list of results.csv paths under root/patient/ (recursive)."""
    root = Path(root).resolve()
    patient_dir = root / patient
    if not patient_dir.exists() or not patient_dir.is_dir():
        return []
    return sorted(patient_dir.rglob("results.csv"))


def load_and_annotate(path: Path, root: Path, patient: str) -> pd.DataFrame | None:
    """
    Read results.csv and add columns: patient, period, method, run_tag from path.
    Path is .../outputs_region_exclusion/patient/period/method/run_tag/.../results.csv
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        df = df.copy()
        df["patient"] = patient
        if len(parts) < 4:
            return df
        else:
            df = df.copy()
            period = parts[1]
            method = parts[2]
            run_tag = parts[3]
            df["period"] = period
            df["method"] = method
            df["run_tag"] = run_tag
            df["session"] = f"{patient}_{period}"
        return df
    except ValueError:
        df = df.copy()
        df["patient"] = patient
        return df


def load_patient(root: Path, patient: str) -> pd.DataFrame:
    """Load all results.csv for one patient under root/patient/ into one DataFrame."""
    paths = find_patient_results_csv(root, patient)
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


def load_all(root: Path, patients: list[str] | None = None) -> pd.DataFrame:
    """Load all results.csv for all (or given) patients into one DataFrame."""
    root = Path(root).resolve()
    if patients is None:
        patients = find_all_patients(root)
    if not patients:
        return pd.DataFrame()
    frames = [load_patient(root, p) for p in patients]
    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summary_by_region_combo_single_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    For one patient's df: for each (Regions_included, Side), compute N_periods,
    mean_Mean_IoU, median_Mean_IoU. Uses original region labels (no bucketing).
    Left and right always separated.
    """
    if df.empty or "Regions_included" not in df.columns:
        return pd.DataFrame()
    if "Side" not in df.columns:
        df = df.copy()
        df["Side"] = "left"
    grouped = df.groupby(["Regions_included", "Side"], as_index=False)
    agg = grouped.agg(
        N_periods=("session", "nunique"),
        mean_Mean_IoU=("Mean_IoU", "mean"),
        median_Mean_IoU=("Mean_IoU", "median"),
    )
    k_map = df.groupby("Regions_included")["K_excluded"].first()
    agg["K_excluded"] = agg["Regions_included"].map(k_map)
    agg = agg.sort_values(["K_excluded", "Regions_included", "Side"]).reset_index(drop=True)
    return agg


def summary_by_region_combo_cross_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-patient: for each (Regions_included_bucket, Side), compute N_periods,
    N_patients, patients (comma-sep), mean_Mean_IoU, median_Mean_IoU.
    Uses bucket normalization so e.g. VoSTN and VoSTNSNr combine.
    """
    if df.empty or "Regions_included" not in df.columns or "patient" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "Side" not in df.columns:
        df["Side"] = "left"
    df["Regions_included"] = df["Regions_included"].astype(str).map(normalize_regions_included)
    grouped = df.groupby(["Regions_included", "Side"], as_index=False)
    agg = grouped.agg(
        N_periods=("session", "nunique"),
        N_patients=("patient", "nunique"),
        patients=("patient", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
        mean_Mean_IoU=("Mean_IoU", "mean"),
        median_Mean_IoU=("Mean_IoU", "median"),
        K_excluded=("K_excluded", "min"),
    )
    agg = agg.sort_values(["K_excluded", "Regions_included", "Side"]).reset_index(drop=True)
    return agg


def _boxplot_one_side(
    df: pd.DataFrame,
    side: str,
    out_path: Path | None,
    *,
    title_suffix: str,
) -> None:
    """Single Plotly boxplot for one side (left or right): x = bucket, y = Mean_IoU."""
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
    bucket_order = order_df["_bucket"].tolist()
    if not bucket_order:
        return
    fig = go.Figure()
    for bucket in bucket_order:
        vals = df_side.loc[df_side["_bucket"] == bucket, "Mean_IoU"].dropna()
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
        title=f"Cross-patient: Mean IoU by region combo (bucket) — {title_suffix}",
        xaxis_title="Region combo (bucket)",
        yaxis_title="Mean IoU",
        showlegend=False,
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(b=120),
    )
    if out_path:
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")
    fig.show()


def plot_cross_patient_boxplot_by_bucket(
    df: pd.DataFrame,
    out_dir: Path | None = None,
) -> None:
    """
    Two separate Plotly interactive boxplots: one for Left, one for Right.
    x = bucket name, y = Mean_IoU. Saves HTML to out_dir if provided.
    """
    if not HAS_PLOTLY:
        return
    if df.empty or "Mean_IoU" not in df.columns:
        return
    df = df.copy()
    if "Side" not in df.columns:
        df["Side"] = "left"
    left_path = Path(out_dir) / "mean_iou_boxplot_left.html" if out_dir else None
    right_path = Path(out_dir) / "mean_iou_boxplot_right.html" if out_dir else None
    _boxplot_one_side(df, "left", left_path, title_suffix="Left")
    _boxplot_one_side(df, "right", right_path, title_suffix="Right")


def process_patient(root: Path, patient: str) -> None:
    """Load one patient's results, compute summaries, and write CSVs under root/patient/."""
    df = load_patient(root, patient)
    if df.empty:
        print(f"  No results.csv under {root / patient}; skipping.")
        return

    print(f"  {patient}: {len(df)} rows, {df['session'].nunique()} sessions")

    summary_combo = summary_by_region_combo_single_patient(df)
    if not summary_combo.empty:
        summary_path = root / patient / "summary_by_region_combo.csv"
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
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return

    if args.patient:
        patients = [args.patient]
        if not (root / args.patient).exists():
            print(f"Patient folder does not exist: {root / args.patient}")
            return
        for patient in patients:
            process_patient(root, patient)
        return

    patients = find_all_patients(root)
    if not patients:
        print(f"No patient folders with results.csv found under {root}")
        return
    print(f"Found {len(patients)} patient(s): {', '.join(patients)}")

    for patient in patients:
        process_patient(root, patient)

    # Cross-patient: bucket summary + boxplot
    df_all = load_all(root, patients)
    if df_all.empty:
        print("No data for cross-patient summary.")
        return
    summary_cp = summary_by_region_combo_cross_patient(df_all)
    if summary_cp.empty:
        return
    out_dir = root / "cross_patient"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_by_region_combo.csv"
    summary_cp.to_csv(summary_path, index=False)
    print(f"Cross-patient summary ({len(summary_cp)} bucket rows) → {summary_path}")

    if HAS_PLOTLY:
        plot_cross_patient_boxplot_by_bucket(df_all, out_dir=out_dir)
    else:
        print("Install plotly to generate cross-patient boxplots: pip install plotly")


if __name__ == "__main__":
    main()
