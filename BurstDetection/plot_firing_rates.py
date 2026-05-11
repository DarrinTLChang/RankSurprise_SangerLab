"""
Standalone firing-rate export (CSV only).

Resolves datasets the same way as plot_isi_hist.py and loads spike data via the
same helpers. Writes per-cluster and per-region summary CSVs.

Run from repo root, e.g.:
  python BurstDetection/plot_firing_rates.py --dataset-index 0 --output-dir BurstDetection/fr_plots
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse loading / iteration from plot_isi_hist so behavior stays aligned with main.py.
from plot_isi_hist import (
    _infer_region_from_electrode,
    _iter_units,
    _load_spike_struct,
    _load_spike_times_npy_as_unit,
    _load_spiketimeset_regions,
    _parse_dataset_kind,
    _select_units,
)


def _sanitize_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_") or "run"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export per-cluster and per-region firing rates (Hz) to CSV; dataset selection matches plot_isi_hist."
    )

    src = p.add_argument_group("dataset selection")
    src.add_argument("--dataset-index", type=int, default=0, help="Index into config.SpikeTime_Mat_File.")
    src.add_argument(
        "--dataset-spec",
        type=str,
        default=None,
        help='Override config (supports "kilosort:" / "kilosortset:" / paths to .mat or .npy).',
    )
    src.add_argument(
        "--dataset-spec-set",
        type=str,
        default=None,
        help="Multi-input spec (e.g. spiketimeset:...); same as plot_isi_hist.",
    )
    src.add_argument("--only-datasets", type=str, default=None, help='Comma-separated indices (e.g. "0,4,7").')
    src.add_argument(
        "--only-dataset-contains",
        action="append",
        default=[],
        help="Substring filter on dataset path (repeatable; AND).",
    )
    src.add_argument(
        "--only-dataset-kind",
        type=str,
        choices=("mat", "kilosort", "kilosortset"),
        default=None,
        help="Filter config list by kind.",
    )
    src.add_argument("--good-only", action="store_true", default=False, help="Kilosort: KSLabel good only.")
    src.add_argument("--max-duration-s", type=float, default=None, help="Kilosort: clip spikes to this duration (s).")

    sel = p.add_argument_group("unit selection")
    sel.add_argument("--electrode", type=str, default=None, help="Regex to match electrode string.")
    sel.add_argument("--cluster", type=int, default=None, help="Cluster index to include.")

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV output (default: BurstDetection/fr_plots/<dataset_token>).",
    )
    return p.parse_args()


def _resolve_dataset_spec(args: argparse.Namespace) -> str:
    if args.dataset_spec_set is not None:
        return str(args.dataset_spec_set)

    if args.dataset_spec is None:
        from config import SpikeTime_Mat_File

        specs = list(SpikeTime_Mat_File)

        if args.only_datasets:
            try:
                idxs = [int(x.strip()) for x in str(args.only_datasets).split(",") if x.strip() != ""]
            except Exception:
                raise SystemExit(
                    f"Could not parse --only-datasets {args.only_datasets!r} (expected comma-separated ints)."
                )
            picked: list[str] = []
            for i in idxs:
                if i < 0 or i >= len(specs):
                    raise SystemExit(f"--only-datasets index out of range: {i} (0..{len(specs)-1})")
                picked.append(specs[i])
            specs = picked

        if args.only_dataset_kind is not None:
            keep = []
            for s in specs:
                k, _raw = _parse_dataset_kind(s)
                if k == str(args.only_dataset_kind):
                    keep.append(s)
            specs = keep

        contains = [str(t) for t in (args.only_dataset_contains or []) if str(t).strip() != ""]
        if contains:
            keep = []
            for s in specs:
                ok = all(tok in str(s) for tok in contains)
                if ok:
                    keep.append(s)
            specs = keep

        if not specs:
            raise SystemExit("No datasets matched your filters in config.SpikeTime_Mat_File.")

        used_filters = bool(args.only_datasets or args.only_dataset_kind or contains)
        if used_filters:
            if len(specs) > 1:
                print("Matched datasets (using the first):")
                for i, s in enumerate(specs[:20]):
                    print(f"  [{i}] {s}")
                if len(specs) > 20:
                    print(f"  ... ({len(specs) - 20} more)")
            return specs[0]

        if args.dataset_index < 0 or args.dataset_index >= len(specs):
            raise SystemExit(f"--dataset-index out of range: {args.dataset_index} (0..{len(specs) - 1})")
        return specs[int(args.dataset_index)]

    return str(args.dataset_spec)


def _compute_recording_duration_s(spike_struct: np.ndarray) -> float:
    """Match pipeline.utils.compute_recording_duration_s (avoids importing utils, which pulls pandas)."""
    ch0 = np.ravel(spike_struct)[0]
    return float(ch0.dataSegmentLength)


def _record_len_from_spike_span_s(spk_ms: np.ndarray) -> float:
    spk = np.asarray(spk_ms, dtype=float).ravel()
    spk = spk[np.isfinite(spk)]
    if spk.size < 2:
        return 1e-9
    spk = np.sort(spk)
    return max(float(spk[-1] - spk[0]) / 1000.0, 1e-9)


def _rows_from_spike_struct(
    spike_struct: np.ndarray,
    *,
    label: str,
    electrode_regex: str | None,
    cluster: int | None,
) -> tuple[list[dict], float]:
    """Returns (rows, record_len_s) using pipeline recording length."""
    record_len_s = float(_compute_recording_duration_s(spike_struct))
    if not np.isfinite(record_len_s) or record_len_s <= 0:
        record_len_s = 1e-9

    units = _select_units(_iter_units(spike_struct), electrode_regex=electrode_regex, cluster=cluster)
    rows: list[dict] = []
    for k, spk in units:
        n = int(spk.size)
        fr = float(n) / record_len_s
        reg = _infer_region_from_electrode(k.electrode)
        rows.append(
            {
                "DatasetLabel": label,
                "Electrode": k.electrode,
                "Cluster": int(k.cluster),
                "Region": reg,
                "Num_Spikes": n,
                "RecordingDuration_s": record_len_s,
                "FR_Hz": fr,
            }
        )
    return rows, record_len_s


def _rows_from_spiketimeset(
    raw: str,
    *,
    label: str,
    electrode_regex: str | None,
    cluster: int | None,
) -> tuple[list[dict], float | None]:
    """
    One pseudo-unit per .npy file; duration from spike span per file.
    Global record_len_s is None (region pooled FR uses per-file duration only in pooled column — use NaN or skip).
    """
    regions, _lbl = _load_spiketimeset_regions(raw)
    rows: list[dict] = []
    for reg_name, unit_list in regions.items():
        for k, spk in unit_list:
            if electrode_regex is not None and not re.compile(electrode_regex).search(k.electrode):
                continue
            if cluster is not None and int(k.cluster) != int(cluster):
                continue
            rec = _record_len_from_spike_span_s(spk)
            n = int(spk.size)
            fr = float(n) / rec
            reg = _infer_region_from_electrode(k.electrode)
            rows.append(
                {
                    "DatasetLabel": label,
                    "Electrode": k.electrode,
                    "Cluster": int(k.cluster),
                    "Region": reg,
                    "Num_Spikes": n,
                    "RecordingDuration_s": rec,
                    "FR_Hz": fr,
                }
            )
    return rows, None


def _rows_from_spiketime_npy(raw: str, *, label: str) -> tuple[list[dict], float | None]:
    k, spk = _load_spike_times_npy_as_unit(raw)
    rec = _record_len_from_spike_span_s(spk)
    n = int(spk.size)
    fr = float(n) / rec
    reg = _infer_region_from_electrode(k.electrode)
    row = {
        "DatasetLabel": label,
        "Electrode": k.electrode,
        "Cluster": int(k.cluster),
        "Region": reg,
        "Num_Spikes": n,
        "RecordingDuration_s": rec,
        "FR_Hz": fr,
    }
    return [row], None


def _median_sorted(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2:
        return float(ys[mid])
    return float(ys[mid - 1] + ys[mid]) / 2.0


def _pop_std(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    v = sum((x - mean) ** 2 for x in xs) / float(len(xs))
    return float(math.sqrt(v))


def _region_summary_rows(rows: list[dict], *, global_record_len_s: float | None) -> list[dict]:
    if not rows:
        return []

    by_reg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_reg[str(r.get("Region", ""))].append(r)

    out: list[dict] = []
    for reg in sorted(by_reg.keys(), key=lambda x: (x is None, str(x))):
        rs = by_reg[reg]
        frs = [float(r["FR_Hz"]) for r in rs]
        n_units = len(frs)
        sum_spikes = int(sum(int(r["Num_Spikes"]) for r in rs))
        mean_fr = sum(frs) / float(n_units) if n_units else float("nan")
        med_fr = _median_sorted(frs)
        std_fr = _pop_std(frs, mean_fr)
        min_fr = min(frs) if frs else float("nan")
        max_fr = max(frs) if frs else float("nan")
        if global_record_len_s is not None and np.isfinite(global_record_len_s) and global_record_len_s > 0:
            pooled = float(sum_spikes) / float(global_record_len_s)
        else:
            pooled = float("nan")
        out.append(
            {
                "Region": reg,
                "mean_FR_Hz": mean_fr,
                "median_FR_Hz": med_fr,
                "std_FR_Hz": std_fr,
                "min_FR_Hz": min_fr,
                "max_FR_Hz": max_fr,
                "n_units": n_units,
                "sum_spikes": sum_spikes,
                "pooled_FR_Hz": pooled,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out_row = {}
            for k in fieldnames:
                v = r.get(k, "")
                if v is None:
                    out_row[k] = ""
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out_row[k] = ""
                else:
                    out_row[k] = v
            w.writerow(out_row)


def main() -> None:
    args = _parse_args()
    spec = _resolve_dataset_spec(args)
    kind, raw = _parse_dataset_kind(spec)

    label = str(spec)
    rows: list[dict]
    global_rec: float | None

    if kind == "spiketimeset":
        rows, global_rec = _rows_from_spiketimeset(
            raw,
            label=label,
            electrode_regex=args.electrode,
            cluster=args.cluster,
        )
    elif kind == "spiketime_npy":
        rows, global_rec = _rows_from_spiketime_npy(raw, label=label)
        if rows and (args.electrode is not None or args.cluster is not None):
            r0 = rows[0]
            if args.cluster is not None and int(r0["Cluster"]) != int(args.cluster):
                rows = []
            elif args.electrode is not None and not re.search(args.electrode, str(r0["Electrode"])):
                rows = []
            else:
                rows = [r0]
    else:
        st, resolved_label = _load_spike_struct(
            spec,
            good_only=bool(args.good_only),
            max_duration_s=args.max_duration_s,
        )
        label = resolved_label
        rows, global_rec = _rows_from_spike_struct(
            st,
            label=label,
            electrode_regex=args.electrode,
            cluster=args.cluster,
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = Path(__file__).resolve().parent / "fr_plots"
        out_dir = base / _sanitize_token(label)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_path = out_dir / "firing_rates_per_cluster.csv"
    reg_path = out_dir / "firing_rates_by_region.csv"

    per_fields = [
        "DatasetLabel",
        "Electrode",
        "Cluster",
        "Region",
        "Num_Spikes",
        "RecordingDuration_s",
        "FR_Hz",
    ]
    _write_csv(per_path, rows, per_fields)
    reg_rows = _region_summary_rows(rows, global_record_len_s=global_rec)
    reg_fields = [
        "Region",
        "mean_FR_Hz",
        "median_FR_Hz",
        "std_FR_Hz",
        "min_FR_Hz",
        "max_FR_Hz",
        "n_units",
        "sum_spikes",
        "pooled_FR_Hz",
    ]
    _write_csv(reg_path, reg_rows, reg_fields)

    print(f"Wrote {per_path} ({len(rows)} rows)")
    print(f"Wrote {reg_path}")


if __name__ == "__main__":
    main()
