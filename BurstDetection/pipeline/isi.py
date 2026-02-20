# isi.py
from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import *


# =====================================================================================
# ISI ROW BUILDERS
# =====================================================================================

def append_burst_isi_rows(
    isi_burst_rows: list[dict[str, Any]],
    spikes_ms: np.ndarray,
    bursts: list[dict[str, Any]],
    elec: str,
    cl: int,
) -> None:
    if not bursts:
        return

    spk = np.asarray(spikes_ms, dtype=float)

    for b in bursts:
        s = float(b["Start_ms"])
        e = float(b["End_ms"])

        spk_b = spk[(spk >= s) & (spk <= e)]
        if spk_b.size < 2:
            continue

        isis = np.diff(spk_b)

        for isi in isis:
            isi_burst_rows.append({
                "Electrode": elec,
                "Cluster": int(cl),
                "Burst_Start_ms": s,
                "Burst_End_ms": e,
                "Num_Spikes": int(b["Num_Spikes"]),
                "ISI_ms": float(isi),
            })


def append_nonburst_isi_rows(
    isi_nonburst_rows: list[dict[str, Any]],
    spikes_ms: np.ndarray,
    bursts: list[dict[str, Any]],
    elec: str,
    cl: int,
) -> None:
    spk = np.asarray(spikes_ms, dtype=float)
    if spk.size < 2:
        return

    isis = np.diff(spk)

    in_burst = np.zeros(len(isis), dtype=bool)
    for b in bursts:
        s = float(b["Start_ms"])
        e = float(b["End_ms"])
        in_burst |= (spk[:-1] >= s) & (spk[1:] <= e)

    for i in np.where(~in_burst)[0]:
        isi_nonburst_rows.append({
            "Electrode": elec,
            "Cluster": int(cl),
            "ISI_start_ms": float(spk[i]),
            "ISI_end_ms": float(spk[i + 1]),
            "ISI_ms": float(isis[i]),
        })


# =====================================================================================
# SAVERS
# =====================================================================================

def save_burst_events(
    all_bursts: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "burst_events.csv",
    col_order: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    if not all_bursts:
        return None

    burst_df = pd.DataFrame(all_bursts)
    if metadata:
        for k, v in metadata.items():
            burst_df[k] = v
    if col_order is not None:
        existing = [c for c in col_order if c in burst_df.columns]
        extras = [c for c in burst_df.columns if c not in existing]
        burst_df = burst_df[existing + extras]
    if "Channel" in burst_df.columns:
        burst_df = burst_df[["Channel"] + [c for c in burst_df.columns if c != "Channel"]]
    out_path = out_dir / filename
    burst_df = standardize_output_df(burst_df)
    burst_df.to_csv(out_path, index=False)
    return out_path


def save_isi_tables(
    isi_burst_rows: list[dict[str, Any]],
    isi_nonburst_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if isi_burst_rows:
        pd.DataFrame(isi_burst_rows).to_csv(
            out_dir / "isi_in_burst.csv", index=False
        )
    if isi_nonburst_rows:
        pd.DataFrame(isi_nonburst_rows).to_csv(
            out_dir / "isi_outside_burst.csv", index=False
        )


# =====================================================================================
# PLOTTING VECTORS
# =====================================================================================

def extract_isi_vectors(
    isi_burst_rows: list[dict[str, Any]],
    isi_nonburst_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    isi_burst = np.asarray(
        [r["ISI_ms"] for r in isi_burst_rows], dtype=float
    ) if isi_burst_rows else np.array([])

    isi_non = np.asarray(
        [r["ISI_ms"] for r in isi_nonburst_rows], dtype=float
    ) if isi_nonburst_rows else np.array([])

    return isi_burst, isi_non


# =====================================================================================
# POPULATION SUMMARY
# =====================================================================================

def save_isi_cluster_summary(
    isi_burst_rows: list[dict[str, Any]],
    isi_nonburst_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    df_burst = pd.DataFrame(isi_burst_rows)
    df_non = pd.DataFrame(isi_nonburst_rows)

    cluster_order = (
        df_burst[["Electrode", "Cluster"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cluster_order["order"] = np.arange(len(cluster_order))

    burst_stats = (
        df_burst.groupby(["Electrode", "Cluster"])["ISI_ms"]
        .agg(
            burst_isi_count="count",
            burst_mean="mean",
            burst_median="median",
            burst_std="std",
        )
        .reset_index()
    )

    non_stats = (
        df_non.groupby(["Electrode", "Cluster"])["ISI_ms"]
        .agg(
            nonburst_isi_count="count",
            non_mean="mean",
            non_median="median",
            non_std="std",
        )
        .reset_index()
    )

    summary = burst_stats.merge(non_stats, on=["Electrode", "Cluster"], how="left")
    summary = summary.merge(cluster_order, on=["Electrode", "Cluster"], how="left").sort_values("order")
    summary = summary.drop(columns="order")

    summary.to_csv(out_path, index=False)
    return out_path


def save_isi_region_summary(
    isi_burst_rows: list[dict[str, Any]],
    isi_nonburst_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    df_burst = pd.DataFrame(isi_burst_rows)
    df_non = pd.DataFrame(isi_nonburst_rows)

    if df_burst.empty and df_non.empty:
        pd.DataFrame(columns=[
            "Region", "Side",
            "burst_isi_count", "burst_mean", "burst_median", "burst_std",
            "nonburst_isi_count", "non_mean", "non_median", "non_std",
        ]).to_csv(out_path, index=False)
        return

    if not df_burst.empty:
        parsed = df_burst["Electrode"].apply(parse_electrode)
        df_burst["Region"] = parsed.apply(lambda d: d["region"])
        df_burst["Side"] = parsed.apply(lambda d: d["side"])

    if not df_non.empty:
        parsed = df_non["Electrode"].apply(parse_electrode)
        df_non["Region"] = parsed.apply(lambda d: d["region"])
        df_non["Side"] = parsed.apply(lambda d: d["side"])

    if df_burst.empty:
        burst_stats = pd.DataFrame(columns=[
            "Region", "Side", "burst_isi_count", "burst_mean", "burst_median", "burst_std"
        ])
    else:
        burst_stats = (
            df_burst.groupby(["Region", "Side"], dropna=False)["ISI_ms"]
            .agg(burst_isi_count="count", burst_mean="mean", burst_median="median", burst_std="std")
            .reset_index()
        )

    if df_non.empty:
        non_stats = pd.DataFrame(columns=[
            "Region", "Side", "nonburst_isi_count", "non_mean", "non_median", "non_std"
        ])
    else:
        non_stats = (
            df_non.groupby(["Region", "Side"], dropna=False)["ISI_ms"]
            .agg(nonburst_isi_count="count", non_mean="mean", non_median="median", non_std="std")
            .reset_index()
        )

    summary = pd.merge(burst_stats, non_stats, on=["Region", "Side"], how="outer")

    base_df = df_burst if not df_burst.empty else df_non
    order_df = (
        base_df[["Region", "Side"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    order_df["order"] = np.arange(len(order_df))

    summary = (
        summary.merge(order_df, on=["Region", "Side"], how="left")
        .sort_values("order", na_position="last")
        .drop(columns="order")
    )

    summary.to_csv(out_path, index=False)
