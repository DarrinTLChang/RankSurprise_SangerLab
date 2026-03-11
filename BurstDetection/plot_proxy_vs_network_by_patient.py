#!/usr/bin/env python3
"""
Plot proxy–network correlation (fast proxy activity vs network bursts) with
one pair of bars per patient (Left + Right) + ALL.

HOW TO CALL
-----------
  # Uses paths and RUN_TAGS at top of this file; plots go to D_Drive outputs_fast_proxy/proxy_network_plots:
  python plot_proxy_vs_network_by_patient.py

  # Run 300ms and 400ms independently (each reads from and saves to its own folder):
  python plot_proxy_vs_network_by_patient.py --onset-fixed-duration 300
  python plot_proxy_vs_network_by_patient.py --onset-fixed-duration 400

  # Or plot all onset-fixed folders in one go (from ONSET_FIXED_ROOTS):
  python plot_proxy_vs_network_by_patient.py --plot-onset-fixed

  # Single onset-fixed folder by path:
  python plot_proxy_vs_network_by_patient.py --root /path/to/outputs_fast_proxy_onset_fixed_300ms --metric correlation_onset_fixed

  # Override root or run_tags:
  python plot_proxy_vs_network_by_patient.py --root /path/to/outputs_fast_proxy
  python plot_proxy_vs_network_by_patient.py --run-tag "separateGPi__..." "another_tag"
  python plot_proxy_vs_network_by_patient.py --metric mean_hemi_proxy_inside --use-std

Plots are saved under <root>/proxy_network_plots (so duration-based and onset-fixed stay in their respective folders).
Paths and RUN_TAGS at top of file (edit like config.py). ONSET_FIXED_ROOTS for 300ms, 400ms, etc.
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

# ============================================================
# PATHS  (edit these like the dataset list in config.py)
# ============================================================
DEFAULT_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_fast_proxy")
# Onset-fixed results (from compare_proxy_vs_network.py). With --plot-onset-fixed, plots all of these.
BASE_D_Drive = Path("/Volumes/D_Drive/SangerLabBursts")
ONSET_FIXED_ROOTS = [
    BASE_D_Drive / "outputs_fast_proxy_onset_fixed_300ms",
    BASE_D_Drive / "outputs_fast_proxy_onset_fixed_400ms",
]
# When --out-dir not given, plots go to <root>/proxy_network_plots (so onset-fixed root → onset-fixed plots)

# ============================================================
# RUN TAGS  (comment / uncomment to select which run_tags to include)
# ============================================================
RUN_TAGS = [
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=0ms__minCh=1__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=50ms__minCh=1__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=3%__limClust=75__aReg=2%__limReg=75__aNet=1%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=3%__limReg=75__aNet=2%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=8%__limClust=75__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=4%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=5__region__network",
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=5%__limClust=75__aReg=3%__limReg=75__aNet=2%__limNet=75__minSpk=3__minDur=50ms__minCh=5__region__network",
]


def _sanitize_run_tag_for_filename(run_tag: str) -> str:
    """Filename-safe version of full run_tag: only replace characters invalid in filenames."""
    s = run_tag.replace("%", "pct").replace("=", "_").replace(" ", "_")
    for c in "/\\:*?\"<>|":
        s = s.replace(c, "_")
    return s or "run"


def load_all_proxy_network_results(root: Path, run_tags: list[str] | None = None) -> pd.DataFrame:
    """
    Load all proxy_vs_network_results_left.csv and _right.csv under root into one DataFrame.
    If run_tags is set, only include CSVs whose parent directory name is in the list.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return pd.DataFrame()

    tag_set = set(run_tags) if run_tags else None
    rows = []
    for path in root.rglob("proxy_vs_network_results_*.csv"):
        if tag_set is not None and path.parent.name not in tag_set:
            continue
        side = "left" if path.name.endswith("_left.csv") else "right"
        df = pd.read_csv(path)
        df["side"] = side
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_region_proxy_results(
    root: Path,
    subdir: str,
    run_tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load region_proxy_to_*_burst/results_*.csv under root into one DataFrame.
    subdir is "region_proxy_to_hemi_burst" or "region_proxy_to_region_burst".
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return pd.DataFrame()

    tag_set = set(run_tags) if run_tags else None
    rows = []
    pattern = f"{subdir}/results_*.csv"
    for path in root.rglob(pattern):
        # Path layout: root/patient/PeriodN/rankSurprise/run_tag/subdir/results_side.csv
        try:
            run_tag = path.parent.parent.name  # parent=subdir, parent.parent=run_tag
        except Exception:
            continue
        if tag_set is not None and run_tag not in tag_set:
            continue
        side = "left" if path.name.endswith("_left.csv") else "right"
        df = pd.read_csv(path)
        if df.empty or "correlation" not in df.columns:
            continue
        df["side"] = side
        df["run_tag"] = run_tag
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_box_by_side(
    df: pd.DataFrame,
    *,
    value_col: str = "correlation",
    y_title: str | None = None,
    title: str | None = None,
    out_path: Path | None = None,
) -> None:
    """
    Simple boxplot: one box per side (Left, Right) over all rows in df.
    """
    if not HAS_PLOTLY:
        print("Install plotly: pip install plotly")
        return
    if df.empty or value_col not in df.columns or "side" not in df.columns:
        return

    sides = [s for s in ("left", "right") if s in set(df["side"].unique())]
    if not sides:
        return

    fig = go.Figure()
    for side in sides:
        vals = df.loc[df["side"] == side, value_col].dropna()
        if vals.empty:
            continue
        fig.add_trace(
            go.Box(
                y=vals,
                name=side.capitalize(),
                boxmean="sd",
            )
        )

    if not fig.data:
        return

    y_title = y_title or value_col
    chart_title = title or f"Distribution of {value_col}"
    fig.update_layout(
        title=dict(text=chart_title),
        xaxis_title="Side",
        yaxis_title=y_title,
        boxmode="group",
        height=500,
        margin=dict(b=120),
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")


def plot_one_side(
    df: pd.DataFrame,
    side: str,
    *,
    value_col: str = "correlation",
    y_title: str | None = None,
    out_path: Path | None = None,
    title: str | None = None,
    use_sem: bool = True,
) -> None:
    """
    Bar chart: one bar per patient (mean of value_col) + error bar (SE or SD), then ALL.
    """
    if not HAS_PLOTLY:
        print("Install plotly: pip install plotly")
        return
    df_side = df.loc[df["side"] == side]
    if df_side.empty:
        return

    by_patient = df_side.groupby("patient")[value_col]
    means = by_patient.mean()
    if use_sem:
        errs = by_patient.sem()
    else:
        errs = by_patient.std()
    counts = by_patient.count()

    patients = means.index.tolist()
    # Sort by patient id (e.g. s508, s509, ...)
    patients = sorted(patients, key=lambda p: (p.replace("s", "").zfill(4), p))
    y = [float(means.loc[p]) for p in patients]
    error_y = []
    for p in patients:
        e = errs.loc[p]
        if counts.loc[p] > 1 and pd.notna(e):
            error_y.append(float(e))
        else:
            error_y.append(0.0)

    # ALL bar
    all_mean = float(df_side[value_col].mean())
    all_err_val = df_side[value_col].sem() if use_sem else df_side[value_col].std()
    all_err = 0.0 if (len(df_side) <= 1 or pd.isna(all_err_val)) else float(all_err_val)
    patients = patients + ["ALL"]
    y = y + [all_mean]
    error_y = error_y + [all_err]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=patients,
            y=y,
            error_y=dict(type="data", array=error_y, visible=True),
            marker_color="rgb(70, 130, 180)",
            marker_line_color="rgb(40, 80, 120)",
        )
    )
    y_title = y_title or value_col
    chart_title = title or f"Fast proxy vs network burst — {value_col}"
    fig.update_layout(
        title=dict(text=chart_title),
        xaxis_title="Patient",
        yaxis_title=y_title,
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(b=120),
        showlegend=False,
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")


def plot_left_right_grouped(
    df: pd.DataFrame,
    *,
    value_col: str = "correlation",
    y_title: str | None = None,
    out_path: Path | None = None,
    title: str | None = None,
    use_sem: bool = True,
) -> None:
    """
    One chart: two bars per patient (Left, Right) + ALL, for correlation of
    fast proxy activity and network bursts. Matches "Proxy vs firing rate
    correlation (by subject)" layout.
    """
    if not HAS_PLOTLY:
        print("Install plotly: pip install plotly")
        return
    if df.empty:
        return

    patients = sorted(df["patient"].unique(), key=lambda p: (p.replace("s", "").zfill(4), p))
    patients = patients + ["ALL"]

    def series_for_side(side: str):
        d = df.loc[df["side"] == side]
        if d.empty:
            return [None] * len(patients), [None] * len(patients)
        by_p = d.groupby("patient")[value_col]
        means = by_p.mean()
        errs = by_p.sem() if use_sem else by_p.std()
        counts = by_p.count()
        y_list = []
        err_list = []
        for p in patients:
            if p == "ALL":
                y_list.append(float(d[value_col].mean()))
                v = d[value_col].sem() if use_sem else d[value_col].std()
                err_list.append(0.0 if (len(d) <= 1 or pd.isna(v)) else float(v))
            else:
                if p in means.index:
                    y_list.append(float(means.loc[p]))
                    e = errs.loc[p]
                    err_list.append(float(e) if counts.loc[p] > 1 and pd.notna(e) else 0.0)
                else:
                    y_list.append(None)
                    err_list.append(0.0)
        return y_list, err_list

    left_y, left_err = series_for_side("left")
    right_y, right_err = series_for_side("right")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=patients,
            y=left_y,
            error_y=dict(type="data", array=left_err, visible=True),
            name="Left",
            marker_color="rgb(70, 130, 180)",
            marker_line_color="rgb(40, 80, 120)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=patients,
            y=right_y,
            error_y=dict(type="data", array=right_err, visible=True),
            name="Right",
            marker_color="rgb(255, 165, 0)",
            marker_line_color="rgb(200, 120, 0)",
        )
    )
    y_title = y_title or "Correlation"
    chart_title = title or "Proxy vs network burst correlation (by subject)"
    fig.update_layout(
        title=dict(text=chart_title),
        xaxis_title="Subject",
        yaxis_title=y_title,
        xaxis=dict(tickangle=-45),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(b=120),
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")


def plot_box_by_region(
    df: pd.DataFrame,
    *,
    value_col: str = "correlation",
    y_title: str | None = None,
    title: str | None = None,
    out_path: Path | None = None,
    patient: str | None = None,
) -> None:
    """
    Box plot with one box per region. If patient is set, filter to that patient first.
    """
    if not HAS_PLOTLY:
        print("Install plotly: pip install plotly")
        return
    if df.empty or value_col not in df.columns or "region" not in df.columns:
        return

    data = df.loc[df["patient"] == patient] if patient else df
    if data.empty:
        return

    regions = sorted(data["region"].unique())
    if not regions:
        return

    fig = go.Figure()
    for reg in regions:
        vals = data.loc[data["region"] == reg, value_col].dropna()
        if vals.empty:
            continue
        fig.add_trace(
            go.Box(
                y=vals,
                name=reg,
                boxmean="sd",
            )
        )

    if not fig.data:
        return

    y_title = y_title or value_col
    chart_title = title or f"Distribution of {value_col} by region"
    if patient:
        chart_title = f"{chart_title} — {patient}"
    fig.update_layout(
        title=dict(text=chart_title),
        xaxis_title="Region",
        yaxis_title=y_title,
        xaxis=dict(tickangle=-45),
        boxmode="group",
        height=500,
        margin=dict(b=150),
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot proxy–network correlation (or other metric) with one bar per patient + ALL."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory containing patient/PeriodN/rankSurprise/run_tag/ (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        nargs="*",
        default=None,
        help="Only load results from these run_tag folder(s). "
             "Defaults to the RUN_TAGS list defined at the top of this file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save HTML plots (default: <root>/proxy_network_plots).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="correlation",
        choices=[
            "correlation",
            "correlation_onset_fixed",
            "mean_hemi_proxy_inside",
            "mean_hemi_proxy_outside",
            "median_hemi_proxy_inside",
            "median_hemi_proxy_outside",
            "mean_hemi_proxy_inside_onset_fixed",
            "mean_hemi_proxy_outside_onset_fixed",
            "median_hemi_proxy_inside_onset_fixed",
            "median_hemi_proxy_outside_onset_fixed",
        ],
        help="Metric to plot (default: correlation). Use correlation_onset_fixed for onset-fixed folder.",
    )
    parser.add_argument(
        "--use-std",
        action="store_true",
        help="Use standard deviation for error bars instead of standard error of the mean.",
    )
    parser.add_argument(
        "--plot-onset-fixed",
        action="store_true",
        help="Run plotting for each folder in ONSET_FIXED_ROOTS (300ms, 400ms, ...) with metric=correlation_onset_fixed.",
    )
    parser.add_argument(
        "--onset-fixed-duration",
        type=int,
        default=None,
        metavar="MS",
        help="Run for a single onset-fixed folder only (e.g. 300 or 400). Reads from and saves to outputs_fast_proxy_onset_fixed_<MS>ms. Use with BASE_D_Drive in script.",
    )
    args = parser.parse_args()

    if not HAS_PLOTLY:
        print("Install plotly: pip install plotly")
        return

    active_tags = args.run_tag if args.run_tag is not None else RUN_TAGS
    if not active_tags:
        print("No run tags specified (RUN_TAGS is empty or --run-tag not given).")
        return

    # When --onset-fixed-duration N: run only for outputs_fast_proxy_onset_fixed_Nms (one folder, independent run)
    # When --plot-onset-fixed: run once per root in ONSET_FIXED_ROOTS
    roots_to_plot: list[tuple[Path, str]] = []  # (root, metric)
    if args.onset_fixed_duration is not None:
        root_one = BASE_D_Drive / f"outputs_fast_proxy_onset_fixed_{args.onset_fixed_duration}ms"
        roots_to_plot.append((Path(root_one).resolve(), "correlation_onset_fixed"))
    elif args.plot_onset_fixed:
        for r in ONSET_FIXED_ROOTS:
            roots_to_plot.append((Path(r).resolve(), "correlation_onset_fixed"))
    else:
        roots_to_plot.append((Path(args.root).resolve(), args.metric))

    base_title = "Proxy vs network burst correlation (by subject)"
    n_written = 0
    for root_path, metric in roots_to_plot:
        root_path = Path(root_path).resolve()
        if not root_path.is_dir():
            if args.plot_onset_fixed or args.onset_fixed_duration is not None:
                print(f"  Skip (not found): {root_path}")
            continue
        if args.plot_onset_fixed or args.onset_fixed_duration is not None:
            print(f"\n=== Onset-fixed: {root_path.name} (plots → {root_path / 'proxy_network_plots'}) ===\n")
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (root_path / "proxy_network_plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        y_title = metric.replace("_", " ").title()
        if metric == "correlation":
            y_title = "Correlation (fast proxy vs network burst)"
        elif metric == "correlation_onset_fixed":
            y_title = "Correlation (onset-fixed window)"

        for run_tag in active_tags:
            df = load_all_proxy_network_results(root_path, run_tags=[run_tag])
            if df.empty:
                print(f"  Skip {run_tag[:50]}...: no hemi_proxy→hemi_burst data")
                continue
            tag_safe = _sanitize_run_tag_for_filename(run_tag)
            # 1) Bar chart by patient for hemi_proxy→hemi_burst (existing behavior)
            hemi_dir = out_dir / "hemi_proxy_to_hemi_burst"
            hemi_dir.mkdir(parents=True, exist_ok=True)
            out_path = hemi_dir / f"proxy_network_{metric}_by_patient_{tag_safe}.html"
            title = f"{base_title} — {tag_safe}"
            plot_left_right_grouped(
                df,
                value_col=metric,
                y_title=y_title,
                out_path=out_path,
                title=title,
                use_sem=not args.use_std,
            )
            n_written += 1

            # 2) Boxplot for hemi_proxy→hemi_burst correlations
            hemi_box_path = hemi_dir / f"proxy_network_{metric}_box_{tag_safe}.html"
            plot_box_by_side(
                df,
                value_col=metric,
                y_title=y_title,
                title=f"Hemi proxy ↔ hemi bursts — {tag_safe}",
                out_path=hemi_box_path,
            )

            # 3) Region_proxy→hemi_burst: master box (one box per region, all patients) + per-patient box (one box per region)
            df_region_hemi = _load_region_proxy_results(
                root_path,
                subdir="region_proxy_to_hemi_burst",
                run_tags=[run_tag],
            )
            if not df_region_hemi.empty:
                reg_hemi_dir = out_dir / "region_proxy_to_hemi_burst"
                reg_hemi_dir.mkdir(parents=True, exist_ok=True)
                col = metric if metric in df_region_hemi.columns else "correlation"
                y_reg = y_title if col == metric else "Correlation (region proxy vs hemi burst)"
                # Master: one box per region (all patients/periods/sides)
                master_path = reg_hemi_dir / f"region_proxy_to_hemi_burst_{metric}_box_{tag_safe}_master.html"
                plot_box_by_region(
                    df_region_hemi,
                    value_col=col,
                    y_title=y_reg,
                    title=f"Region proxy ↔ hemisphere bursts (all) — {tag_safe}",
                    out_path=master_path,
                    patient=None,
                )
                n_written += 1
                # Per patient: one box per region
                by_patient_dir = reg_hemi_dir / "by_patient"
                by_patient_dir.mkdir(parents=True, exist_ok=True)
                for p in sorted(df_region_hemi["patient"].unique()):
                    patient_path = by_patient_dir / f"region_proxy_to_hemi_burst_{metric}_box_{tag_safe}_{p}.html"
                    plot_box_by_region(
                        df_region_hemi,
                        value_col=col,
                        y_title=y_reg,
                        title=f"Region proxy ↔ hemisphere bursts — {tag_safe}",
                        out_path=patient_path,
                        patient=p,
                    )
                    n_written += 1

            # 4) Region_proxy→region_burst: master box + per-patient box by region
            df_region_region = _load_region_proxy_results(
                root_path,
                subdir="region_proxy_to_region_burst",
                run_tags=[run_tag],
            )
            if not df_region_region.empty:
                reg_reg_dir = out_dir / "region_proxy_to_region_burst"
                reg_reg_dir.mkdir(parents=True, exist_ok=True)
                col = metric if metric in df_region_region.columns else "correlation"
                y_reg = y_title if col == metric else "Correlation (region proxy vs region bursts)"
                master_path = reg_reg_dir / f"region_proxy_to_region_burst_{metric}_box_{tag_safe}_master.html"
                plot_box_by_region(
                    df_region_region,
                    value_col=col,
                    y_title=y_reg,
                    title=f"Region proxy ↔ region bursts (all) — {tag_safe}",
                    out_path=master_path,
                    patient=None,
                )
                n_written += 1
                by_patient_dir = reg_reg_dir / "by_patient"
                by_patient_dir.mkdir(parents=True, exist_ok=True)
                for p in sorted(df_region_region["patient"].unique()):
                    patient_path = by_patient_dir / f"region_proxy_to_region_burst_{metric}_box_{tag_safe}_{p}.html"
                    plot_box_by_region(
                        df_region_region,
                        value_col=col,
                        y_title=y_reg,
                        title=f"Region proxy ↔ region bursts — {tag_safe}",
                        out_path=patient_path,
                        patient=p,
                    )
                    n_written += 1

    if n_written == 0:
        print(f"No proxy_vs_network_results_*.csv found for any of {len(active_tags)} run_tag(s).")
    else:
        print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
