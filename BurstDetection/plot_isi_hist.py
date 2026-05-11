from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as spio

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None  # type: ignore[assignment]


class UnitKey:
    __slots__ = ("electrode", "cluster")

    def __init__(self, electrode: str, cluster: int):
        self.electrode = str(electrode)
        self.cluster = int(cluster)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone ISI histogram explorer. Loads a dataset (MAT spikeTime or kilosort: / "
            "kilosortset: spec), computes ISIs, and plots histograms (optionally log(ISI)) "
            "with optional Gaussian overlay."
        )
    )

    src = p.add_argument_group("dataset selection")
    src.add_argument("--dataset-index", type=int, default=0, help="Index into config.SpikeTime_Mat_File.")
    src.add_argument(
        "--dataset-spec",
        type=str,
        default=None,
        help='Override config and load this spec/path directly (supports "kilosort:" / "kilosortset:").',
    )
    src.add_argument(
        "--dataset-spec-set",
        type=str,
        default=None,
        help=(
            "Multi-input spec. Supported: "
            "spiketimeset:<spike_times.npy>;<spike_times.npy>;... "
            "(each file treated as one region for --by-region)."
        ),
    )
    src.add_argument(
        "--only-datasets",
        type=str,
        default=None,
        help='Comma-separated indices into config.SpikeTime_Mat_File (e.g. "0,4,7").',
    )
    src.add_argument(
        "--only-dataset-contains",
        action="append",
        default=[],
        help="Substring filter on dataset spec/path (repeatable; AND semantics).",
    )
    src.add_argument(
        "--only-dataset-kind",
        type=str,
        choices=("mat", "kilosort", "kilosortset"),
        default=None,
        help="Filter config.SpikeTime_Mat_File by kind.",
    )
    src.add_argument(
        "--good-only",
        action="store_true",
        default=False,
        help="Kilosort only: keep only KSLabel==good clusters (requires cluster_group.tsv).",
    )
    src.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        help="Kilosort only: clip spikes to [0, max_duration_s] seconds.",
    )

    sel = p.add_argument_group("unit selection")
    sel.add_argument(
        "--electrode",
        type=str,
        default=None,
        help="Regex to match electrode string (e.g. 'GPi.*_L_'). If omitted, include all electrodes.",
    )
    sel.add_argument("--cluster", type=int, default=None, help="Cluster index to include (per electrode).")

    shape = p.add_argument_group("ISI shaping")
    shape.add_argument("--min-isi-ms", type=float, default=0.0, help="Drop ISIs < min_isi_ms.")
    shape.add_argument("--max-isi-ms", type=float, default=None, help="Drop ISIs > max_isi_ms (optional).")

    plot = p.add_argument_group("plot")
    plot.add_argument(
        "--pool",
        type=str,
        choices=("all", "per-unit"),
        default="all",
        help="Pool ISIs across selected units, or plot one histogram per unit.",
    )
    plot.add_argument("--log", action="store_true", default=False, help="Also plot histogram of log(ISI_ms).")
    plot.add_argument("--bins", type=int, default=None, help="Number of histogram bins.")
    plot.add_argument("--bin-width-ms", type=float, default=None, help="Histogram bin width in ms (linear plot).")
    plot.add_argument(
        "--overlay-gaussian",
        type=str,
        choices=("none", "isi", "logisi"),
        default="none",
        help="Overlay a fitted Gaussian curve on the chosen domain.",
    )
    plot.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="If set, write Plotly figure HTML to this path instead of opening an interactive window.",
    )
    plot.add_argument(
        "--by-region",
        action="store_true",
        default=False,
        help=(
            "Generate region-level outputs: histogram per region, combined histogram across regions, "
            "and an HTML plot for the first (cluster 0) unit of each region."
        ),
    )
    plot.add_argument(
        "--region-output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write region HTML outputs (defaults to alongside --output-html if provided, "
            "otherwise ./isi_region_plots)."
        ),
    )
    plot.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Convenience root for multi-output runs. When using spiketimeset: with --by-region and "
            "no --region-output-dir, defaults to BurstDetection/isi_hist and writes per-file outputs."
        ),
    )
    plot.add_argument(
        "--combine-spiketimeset",
        action="store_true",
        default=False,
        help="Only for spiketimeset:. If set, also write a combined set of plots across all inputs.",
    )

    return p.parse_args()


def _load_spike_struct(spec: str, *, good_only: bool, max_duration_s: float | None) -> tuple[np.ndarray, str]:
    s = str(spec).strip()
    low = s.lower()
    if low.startswith("kilosort:"):
        kind, raw = "kilosort", s.split(":", 1)[1].strip()
    elif low.startswith("kilosortset:"):
        kind, raw = "kilosortset", s.split(":", 1)[1].strip()
    else:
        kind, raw = "mat", s

    base_dir = Path(__file__).resolve().parent

    def _resolve_one(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        cand = base_dir / pp
        return str(cand) if cand.exists() else str(pp)

    if kind == "mat":
        raw = _resolve_one(raw)

    if kind == "kilosort":
        try:
            from pipeline.kilosort_loader import load_kilosort_dir
        except Exception as e:
            raise SystemExit(
                "Kilosort loading requires extra dependencies (notably pandas). "
                "Install repo deps (see README) or use a .mat spikeTime dataset.\n"
                f"Import error: {e}"
            )
        raw_resolved = _resolve_one(raw)
        st, _meta = load_kilosort_dir(raw_resolved, good_only=good_only, max_duration_s=max_duration_s)
        return st, f"kilosort:{raw_resolved}"
    if kind == "kilosortset":
        ks_dirs_raw = [p.strip() for p in raw.split(";") if p.strip()]
        ks_dirs = [_resolve_one(p) for p in ks_dirs_raw]
        try:
            from pipeline.kilosort_loader import load_kilosort_dirs
        except Exception as e:
            raise SystemExit(
                "Kilosortset loading requires extra dependencies (notably pandas). "
                "Install repo deps (see README) or use a .mat spikeTime dataset.\n"
                f"Import error: {e}"
            )
        st, _meta = load_kilosort_dirs(ks_dirs, good_only=good_only, max_duration_s=max_duration_s)
        return st, f"kilosortset:{len(ks_dirs)}dirs"

    mat = spio.loadmat(raw, squeeze_me=True, struct_as_record=False)
    st = mat["spikeTime"]
    return st, str(raw)


def _parse_dataset_kind(spec: str) -> tuple[str, str]:
    s = str(spec).strip()
    low = s.lower()
    if low.startswith("kilosort:"):
        return "kilosort", s.split(":", 1)[1].strip()
    if low.startswith("kilosortset:"):
        return "kilosortset", s.split(":", 1)[1].strip()
    if low.startswith("spiketimeset:"):
        return "spiketimeset", s.split(":", 1)[1].strip()
    if low.endswith(".npy"):
        # Treat direct spike_times.npy paths as a spike-time train input.
        return "spiketime_npy", s
    return "mat", s


def _resolve_relative_to_burst_detection(p: str) -> str:
    base_dir = Path(__file__).resolve().parent
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    cand = base_dir / pp
    return str(cand) if cand.exists() else str(pp)


def _load_kilosortset_regions(
    raw: str,
    *,
    good_only: bool,
    max_duration_s: float | None,
) -> tuple[dict[str, np.ndarray], str]:
    """
    Load kilosortset spec as {region_label: spike_struct}, where each provided ks_dir is treated as one region.
    """
    ks_dirs_raw = [p.strip() for p in str(raw).split(";") if p.strip()]
    ks_dirs = [_resolve_relative_to_burst_detection(p) for p in ks_dirs_raw]
    try:
        from pipeline.kilosort_loader import load_kilosort_dir
    except Exception as e:
        raise SystemExit(
            "Kilosortset loading requires extra dependencies (notably pandas). "
            "Install repo deps (see README) or run in your analysis environment.\n"
            f"Import error: {e}"
        )

    out: dict[str, np.ndarray] = {}
    for d in ks_dirs:
        dpath = Path(d)
        # Label: prefer shank folder name when present; otherwise use last path token.
        name = dpath.name
        if name.lower() == "kilosort4" and dpath.parent is not None:
            name = dpath.parent.name
        region_label = str(name)
        st, _meta = load_kilosort_dir(d, good_only=good_only, max_duration_s=max_duration_s)
        out[region_label] = st

    label = f"kilosortset:{len(ks_dirs)}dirs"
    return out, label


def _find_params_py_for_spike_times_npy(spike_times_path: Path) -> Path | None:
    """
    Best-effort lookup for a Kilosort params.py near a spike_times.npy.
    Common layouts:
      - <ks_dir>/spike_times.npy + <ks_dir>/params.py
      - <ks_dir>/kilosort4/spike_times.npy + <ks_dir>/kilosort4/params.py
    """
    for p in (spike_times_path.parent, spike_times_path.parent.parent):
        cand = p / "params.py"
        if cand.exists():
            return cand
    return None


def _read_sample_rate_hz_from_params(params_py: Path) -> float | None:
    """
    Read Kilosort sample rate from params.py without importing the full loader (keeps this tool lightweight).
    """
    try:
        txt = params_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Typical line: sample_rate = 30000
    m = re.search(r"sample_rate\s*=\s*([0-9]*\.?[0-9]+)", txt)
    if not m:
        return None
    try:
        fs = float(m.group(1))
    except Exception:
        return None
    return fs if np.isfinite(fs) and fs > 0 else None


def _load_spike_times_npy_as_unit(spike_times_npy: str) -> tuple[UnitKey, np.ndarray]:
    """
    Load a direct spike_times.npy file as a single unit (cluster 0).
    If a nearby params.py is found, interpret values as samples and convert to ms.
    Otherwise assume values are already in ms.
    """
    p = Path(_resolve_relative_to_burst_detection(spike_times_npy)).resolve()
    if not p.exists():
        raise FileNotFoundError(f"spike_times.npy not found: {p}")
    t = np.load(str(p))
    t = np.asarray(t, dtype=float).ravel()
    t = t[np.isfinite(t)]
    if t.size < 2:
        raise RuntimeError(f"Not enough spikes in {p} to form ISIs.")

    fs = None
    params = _find_params_py_for_spike_times_npy(p)
    if params is not None:
        fs = _read_sample_rate_hz_from_params(params)
    if fs is not None:
        t_ms = t / float(fs) * 1000.0
        units = "samples->ms"
    else:
        t_ms = t
        units = "assumed_ms"

    # Region label is derived from filename stem (without extension).
    elec = p.stem
    print(f"• Loaded {p} ({units}, n_spikes={t_ms.size})")
    return UnitKey(electrode=elec, cluster=0), np.sort(t_ms.astype(float, copy=False))


def _load_spiketimeset_regions(raw: str) -> tuple[dict[str, list[tuple[UnitKey, np.ndarray]]], str]:
    """
    Parse spiketimeset:<path1>;<path2>;... and treat each file as one region containing one unit (cluster 0).
    """
    paths = [p.strip() for p in str(raw).split(";") if p.strip()]
    regions: dict[str, list[tuple[UnitKey, np.ndarray]]] = {}
    for sp in paths:
        k, spk = _load_spike_times_npy_as_unit(sp)
        # Use electrode key as the region label in this mode (one file == one region).
        regions.setdefault(str(k.electrode), []).append((k, spk))
    return regions, f"spiketimeset:{len(paths)}files"


def _sanitize_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", str(s)).strip("_") or "item"


def _infer_region_from_electrode(elec: str) -> str:
    try:
        from pipeline.utils import infer_region
    except Exception:
        return str(elec)
    try:
        return str(infer_region(str(elec)))
    except Exception:
        return str(elec)


def _group_units_by_region_from_spike_struct(units: list[tuple[UnitKey, np.ndarray]]) -> dict[str, list[tuple[UnitKey, np.ndarray]]]:
    out: dict[str, list[tuple[UnitKey, np.ndarray]]] = {}
    for k, spk in units:
        reg = _infer_region_from_electrode(k.electrode)
        out.setdefault(reg, []).append((k, spk))
    return out


def _plotly_required() -> None:
    if go is None:
        raise SystemExit(
            "This output requires Plotly (HTML requested). Activate your analysis environment "
            "or install plotly: `pip install plotly`."
        )


def _iter_units(spike_struct: np.ndarray) -> Iterable[tuple[UnitKey, np.ndarray]]:
    for ch in np.ravel(spike_struct):
        elec = str(getattr(ch, "electrode", ""))
        times = np.ravel(getattr(ch, "time", np.empty((0,), dtype=object)))
        for cl, arr in enumerate(times):
            if np.isscalar(arr):
                continue
            spk = np.asarray(arr, dtype=float).ravel()
            spk = spk[np.isfinite(spk)]
            if spk.size < 2:
                continue
            yield UnitKey(electrode=elec, cluster=int(cl)), np.sort(spk)


def _select_units(
    units: Iterable[tuple[UnitKey, np.ndarray]],
    *,
    electrode_regex: str | None,
    cluster: int | None,
) -> list[tuple[UnitKey, np.ndarray]]:
    out: list[tuple[UnitKey, np.ndarray]] = []
    rx = re.compile(electrode_regex) if electrode_regex else None
    for k, spk in units:
        if rx is not None and not rx.search(k.electrode):
            continue
        if cluster is not None and int(k.cluster) != int(cluster):
            continue
        out.append((k, spk))
    return out


def _compute_isis_ms(spk_ms: np.ndarray, *, min_isi_ms: float, max_isi_ms: float | None) -> np.ndarray:
    spk = np.asarray(spk_ms, dtype=float).ravel()
    spk = spk[np.isfinite(spk)]
    if spk.size < 2:
        return np.asarray([], dtype=float)
    spk = np.sort(spk)
    isis = np.diff(spk)
    isis = isis[np.isfinite(isis)]
    if min_isi_ms is not None:
        isis = isis[isis >= float(min_isi_ms)]
    if max_isi_ms is not None:
        isis = isis[isis <= float(max_isi_ms)]
    return isis.astype(float, copy=False)


def _fit_gaussian(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    return mu, sigma


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not (np.isfinite(mu) and np.isfinite(sigma)) or sigma <= 0:
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def _hist_params(values: np.ndarray, *, bins: int | None, bin_width: float | None) -> tuple[int | str, np.ndarray | None]:
    """
    Return (bins_arg, bin_edges_or_none) for Plotly histogram configuration.
    - If bins is set: return (bins, None)
    - Else if bin_width is set: return ("custom", edges)
    - Else: return ("auto", None)
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if bins is not None:
        return int(bins), None
    if bin_width is not None and np.isfinite(bin_width) and bin_width > 0 and v.size > 0:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo:
            return 1, None
        w = float(bin_width)
        n = int(np.ceil((hi - lo) / w))
        edges = lo + w * np.arange(n + 1, dtype=float)
        if edges[-1] < hi:
            edges = np.r_[edges, edges[-1] + w]
        return "custom", edges
    return "auto", None


def _plotly_hist_trace(values: np.ndarray, *, name: str, bins: int | None, bin_width: float | None):
    assert go is not None
    bins_arg, edges = _hist_params(values, bins=bins, bin_width=bin_width)
    kwargs = {}
    if isinstance(bins_arg, int):
        kwargs["nbinsx"] = bins_arg
    elif edges is not None:
        kwargs["xbins"] = dict(start=float(edges[0]), end=float(edges[-1]), size=float(edges[1] - edges[0]))
    return go.Histogram(x=values, name=name, opacity=0.75, **kwargs)


def _make_plotly_figure(
    *,
    isis_ms: np.ndarray,
    title: str,
    bins: int | None,
    bin_width_ms: float | None,
    plot_log: bool,
    overlay_gaussian: str,
) -> tuple["go.Figure", "go.Figure | None"]:
    assert go is not None
    fig = go.Figure()

    isis_ms = np.asarray(isis_ms, dtype=float)
    isis_ms = isis_ms[np.isfinite(isis_ms)]

    fig.add_trace(_plotly_hist_trace(isis_ms, name="ISI (ms)", bins=bins, bin_width=bin_width_ms))
    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="ISI (ms)",
        yaxis_title="Count",
        legend_title="",
    )

    if overlay_gaussian == "isi" and isis_ms.size > 0:
        mu, sigma = _fit_gaussian(isis_ms)
        x = np.linspace(float(np.min(isis_ms)), float(np.max(isis_ms)), 300)
        pdf = _gaussian_pdf(x, mu, sigma)
        scale = float(isis_ms.size) * (float(bin_width_ms) if bin_width_ms else 1.0)
        fig.add_trace(go.Scatter(x=x, y=pdf * scale, name=f"Gaussian fit (μ={mu:.3g}, σ={sigma:.3g})"))

    fig_log: "go.Figure | None" = None
    if plot_log:
        eps = np.finfo(float).tiny
        log_isi = np.log(np.maximum(isis_ms, eps))
        fig_log = go.Figure()
        fig_log.add_trace(_plotly_hist_trace(log_isi, name="log(ISI)", bins=bins, bin_width=None))
        fig_log.update_layout(
            title=title + " — log(ISI)",
            barmode="overlay",
            xaxis_title="log(ISI_ms)",
            yaxis_title="Count",
            legend_title="",
        )
        if overlay_gaussian == "logisi" and log_isi.size > 0:
            mu, sigma = _fit_gaussian(log_isi)
            x = np.linspace(float(np.min(log_isi)), float(np.max(log_isi)), 300)
            pdf = _gaussian_pdf(x, mu, sigma)
            fig_log.add_trace(go.Scatter(x=x, y=pdf * float(log_isi.size), name=f"Gaussian fit (μ={mu:.3g}, σ={sigma:.3g})"))

    return fig, fig_log


def _write_or_show_plotly(
    fig_lin: "go.Figure",
    fig_log: "go.Figure | None",
    *,
    out_html_linear: str | None,
    out_html_log: str | None,
) -> None:
    # Linear
    if out_html_linear:
        out_path = Path(out_html_linear)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_lin.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"Wrote {out_path}")
    else:
        fig_lin.show()

    # Log (optional)
    if fig_log is not None:
        if out_html_log:
            out_path = Path(out_html_log)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig_log.write_html(str(out_path), include_plotlyjs="cdn")
            print(f"Wrote {out_path}")
        else:
            fig_log.show()


def main() -> None:
    args = _parse_args()
    _plotly_required()

    if args.dataset_spec_set is not None:
        spec = str(args.dataset_spec_set)
    else:
        if args.dataset_spec is None:
            from config import SpikeTime_Mat_File

            specs = list(SpikeTime_Mat_File)

            # Optional: filter by explicit indices (like main.py)
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

            # Optional: filter by kind / contains tokens
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
                    ok = True
                    for tok in contains:
                        if tok not in str(s):
                            ok = False
                            break
                    if ok:
                        keep.append(s)
                specs = keep

            if not specs:
                raise SystemExit("No datasets matched your filters in config.SpikeTime_Mat_File.")

            # If filters were used, default to the first matching dataset; otherwise use --dataset-index.
            used_filters = bool(args.only_datasets or args.only_dataset_kind or contains)
            if used_filters:
                if len(specs) > 1:
                    print("Matched datasets (using the first):")
                    for i, s in enumerate(specs[:20]):
                        print(f"  [{i}] {s}")
                    if len(specs) > 20:
                        print(f"  ... ({len(specs)-20} more)")
                spec = specs[0]
            else:
                if args.dataset_index < 0 or args.dataset_index >= len(specs):
                    raise SystemExit(f"--dataset-index out of range: {args.dataset_index} (0..{len(specs)-1})")
                spec = specs[int(args.dataset_index)]
        else:
            spec = str(args.dataset_spec)

    kind, raw = _parse_dataset_kind(spec)

    # Region-mode: treat kilosortset as N regions (one per ks_dir). For MAT and kilosort, infer regions from electrode strings.
    if bool(args.by_region):
        base_out_root = Path(args.output_root) if args.output_root else (Path(__file__).resolve().parent / "isi_hist")

        min_isi = float(args.min_isi_ms)
        max_isi = float(args.max_isi_ms) if args.max_isi_ms is not None else None

        regions: dict[str, list[tuple[UnitKey, np.ndarray]]] = {}
        label = spec

        if kind == "kilosortset":
            region_structs, label = _load_kilosortset_regions(
                raw,
                good_only=bool(args.good_only),
                max_duration_s=args.max_duration_s,
            )
            for reg, st_r in region_structs.items():
                u = _select_units(
                    _iter_units(st_r),
                    electrode_regex=args.electrode,
                    cluster=args.cluster,
                )
                if u:
                    regions[reg] = u
        elif kind == "spiketimeset":
            regions, label = _load_spiketimeset_regions(raw)
        elif kind == "spiketime_npy":
            # Single npy file is treated as a single-region dataset.
            k, spk = _load_spike_times_npy_as_unit(raw)
            regions = {str(k.electrode): [(k, spk)]}
            label = str(raw)
        else:
            st, label = _load_spike_struct(
                spec,
                good_only=bool(args.good_only),
                max_duration_s=args.max_duration_s,
            )
            units = _select_units(
                _iter_units(st),
                electrode_regex=args.electrode,
                cluster=args.cluster,
            )
            regions = _group_units_by_region_from_spike_struct(units)

        if not regions:
            raise SystemExit("No units matched selection for any region.")

        # Output directory policy:
        # - If user explicitly provided --region-output-dir, write outputs there.
        # - Else if spiketimeset: and NOT --combine-spiketimeset, write ONLY per-file outputs under <root>/indiv/<stem>.
        # - Else: write combined outputs to <root>/combined (or alongside --output-html in legacy mode).
        if kind == "spiketimeset" and not args.combine_spiketimeset and not args.region_output_dir:
            out_dir = None
        else:
            if args.region_output_dir:
                out_dir = Path(args.region_output_dir)
            elif args.output_html:
                out_dir = Path(args.output_html).parent
            else:
                out_dir = base_out_root / "combined"
            out_dir.mkdir(parents=True, exist_ok=True)

        # If spiketimeset is not being combined, skip combined plots entirely.
        if kind == "spiketimeset" and not args.combine_spiketimeset and not args.region_output_dir:
            indiv_root = base_out_root / "indiv"
            indiv_root.mkdir(parents=True, exist_ok=True)
            from plotly.subplots import make_subplots

            for reg in sorted(regions.keys()):
                one_dir = indiv_root / _sanitize_token(reg)
                one_dir.mkdir(parents=True, exist_ok=True)

                reg_units = regions[reg]
                reg_isis = [
                    _compute_isis_ms(spk, min_isi_ms=min_isi, max_isi_ms=max_isi) for (_k, spk) in reg_units
                ]
                isis = np.concatenate(reg_isis) if reg_isis else np.asarray([], dtype=float)
                eps = np.finfo(float).tiny
                log_isis = np.log(np.maximum(isis, eps)) if bool(args.log) else None

                fig_r = make_subplots(rows=1, cols=1, subplot_titles=[reg])
                fig_r.add_trace(
                    _plotly_hist_trace(isis, name=f"{reg} ISI", bins=args.bins, bin_width=args.bin_width_ms),
                    row=1,
                    col=1,
                )
                fig_r.update_layout(title=f"ISI histogram — {reg} — {label}", barmode="overlay")
                fig_r.update_xaxes(title_text="ISI (ms)", row=1, col=1)
                fig_r.update_yaxes(title_text="Count", row=1, col=1)
                fig_r.write_html(str(one_dir / "isi_hist_by_region.html"), include_plotlyjs="cdn")
                if log_isis is not None:
                    fig_r_log = make_subplots(rows=1, cols=1, subplot_titles=[reg])
                    fig_r_log.add_trace(
                        _plotly_hist_trace(log_isis, name=f"{reg} log(ISI)", bins=args.bins, bin_width=None),
                        row=1,
                        col=1,
                    )
                    fig_r_log.update_layout(title=f"ISI histogram — {reg} — {label} — log(ISI)", barmode="overlay")
                    fig_r_log.update_xaxes(title_text="log(ISI_ms)", row=1, col=1)
                    fig_r_log.update_yaxes(title_text="Count", row=1, col=1)
                    fig_r_log.write_html(str(one_dir / "isi_hist_by_region_log.html"), include_plotlyjs="cdn")

                fig_one_all = go.Figure()
                fig_one_all.add_trace(_plotly_hist_trace(isis, name=reg, bins=args.bins, bin_width=args.bin_width_ms))
                fig_one_all.update_layout(
                    title=f"ISI histogram — {reg} — {label}",
                    xaxis_title="ISI (ms)",
                    yaxis_title="Count",
                )
                fig_one_all.write_html(str(one_dir / "isi_hist_all_regions_combined.html"), include_plotlyjs="cdn")
                if log_isis is not None:
                    fig_one_all_log = go.Figure()
                    fig_one_all_log.add_trace(_plotly_hist_trace(log_isis, name=f"{reg} log(ISI)", bins=args.bins, bin_width=None))
                    fig_one_all_log.update_layout(
                        title=f"ISI histogram — {reg} — {label} — log(ISI)",
                        xaxis_title="log(ISI_ms)",
                        yaxis_title="Count",
                    )
                    fig_one_all_log.write_html(str(one_dir / "isi_hist_all_regions_combined_log.html"), include_plotlyjs="cdn")

                # first cluster (cluster 0; here each file is cluster 0 anyway)
                fig_one_first = go.Figure()
                k0, spk0 = sorted(reg_units, key=lambda x: str(x[0].electrode))[0]
                isis0 = _compute_isis_ms(spk0, min_isi_ms=min_isi, max_isi_ms=max_isi)
                fig_one_first.add_trace(
                    _plotly_hist_trace(
                        isis0,
                        name=f"{reg}: {k0.electrode} cl={k0.cluster}",
                        bins=args.bins,
                        bin_width=args.bin_width_ms,
                    )
                )
                fig_one_first.update_layout(
                    title=f"ISI histogram — first cluster — {reg} — {label}",
                    xaxis_title="ISI (ms)",
                    yaxis_title="Count",
                )
                fig_one_first.write_html(str(one_dir / "isi_hist_first_cluster_per_region.html"), include_plotlyjs="cdn")
                if bool(args.log):
                    isis0_log = np.log(np.maximum(isis0, eps))
                    fig_one_first_log = go.Figure()
                    fig_one_first_log.add_trace(
                        _plotly_hist_trace(
                            isis0_log,
                            name=f"{reg}: {k0.electrode} cl={k0.cluster} log(ISI)",
                            bins=args.bins,
                            bin_width=None,
                        )
                    )
                    fig_one_first_log.update_layout(
                        title=f"ISI histogram — first cluster — {reg} — {label} — log(ISI)",
                        xaxis_title="log(ISI_ms)",
                        yaxis_title="Count",
                    )
                    fig_one_first_log.write_html(str(one_dir / "isi_hist_first_cluster_per_region_log.html"), include_plotlyjs="cdn")

            print(f"Wrote per-file outputs under:\n  {indiv_root}")
            return

        # 1) Histogram per region (combined mode)
        from plotly.subplots import make_subplots

        reg_names = sorted(regions.keys())
        rows = len(reg_names)
        fig_reg = make_subplots(rows=rows, cols=1, shared_xaxes=False, subplot_titles=reg_names)
        combined_isis: list[np.ndarray] = []

        for i, reg in enumerate(reg_names, start=1):
            reg_units = regions[reg]
            reg_isis = []
            for _k, spk in reg_units:
                reg_isis.append(_compute_isis_ms(spk, min_isi_ms=min_isi, max_isi_ms=max_isi))
            isis = np.concatenate(reg_isis) if reg_isis else np.asarray([], dtype=float)
            combined_isis.append(isis)
            fig_reg.add_trace(
                _plotly_hist_trace(isis, name=f"{reg} ISI", bins=args.bins, bin_width=args.bin_width_ms),
                row=i,
                col=1,
            )
            fig_reg.update_yaxes(title_text="Count", row=i, col=1)
            fig_reg.update_xaxes(title_text="ISI (ms)", row=i, col=1)

        fig_reg.update_layout(title=f"ISI histograms per region — {label}", barmode="overlay")
        out1 = out_dir / "isi_hist_by_region.html"
        fig_reg.write_html(str(out1), include_plotlyjs="cdn")

        # 2) Combined histogram across regions
        all_isis = np.concatenate([x for x in combined_isis if x.size]) if combined_isis else np.asarray([], dtype=float)
        fig_all = go.Figure()
        fig_all.add_trace(_plotly_hist_trace(all_isis, name="All regions", bins=args.bins, bin_width=args.bin_width_ms))
        fig_all.update_layout(
            title=f"ISI histogram — all regions combined — {label}",
            xaxis_title="ISI (ms)",
            yaxis_title="Count",
            barmode="overlay",
        )
        out2 = out_dir / "isi_hist_all_regions_combined.html"
        fig_all.write_html(str(out2), include_plotlyjs="cdn")

        # 3) HTML plot for the first cluster of each region (cluster 0)
        fig_first = make_subplots(rows=len(reg_names), cols=1, shared_xaxes=False, subplot_titles=reg_names)
        for i, reg in enumerate(reg_names, start=1):
            reg_units = regions[reg]
            # "first cluster" interpreted as cluster==0; fall back to first unit if no cluster 0 exists.
            cand = [(k, spk) for (k, spk) in reg_units if int(k.cluster) == 0]
            if not cand:
                cand = reg_units[:1]
            if not cand:
                continue
            k0, spk0 = sorted(cand, key=lambda x: str(x[0].electrode))[0]
            isis0 = _compute_isis_ms(spk0, min_isi_ms=min_isi, max_isi_ms=max_isi)
            fig_first.add_trace(
                _plotly_hist_trace(isis0, name=f"{reg}: {k0.electrode} cl={k0.cluster}", bins=args.bins, bin_width=args.bin_width_ms),
                row=i,
                col=1,
            )
            fig_first.update_yaxes(title_text="Count", row=i, col=1)
            fig_first.update_xaxes(title_text="ISI (ms)", row=i, col=1)

        fig_first.update_layout(title=f"ISI histogram — first cluster per region — {label}", barmode="overlay")
        out3 = out_dir / "isi_hist_first_cluster_per_region.html"
        fig_first.write_html(str(out3), include_plotlyjs="cdn")

        # For spiketimeset (combined mode), also write one subfolder per input file (each treated as a single-region dataset).
        if kind == "spiketimeset" and not args.region_output_dir:
            indiv_root = base_out_root / "indiv"
            indiv_root.mkdir(parents=True, exist_ok=True)
            for reg in sorted(regions.keys()):
                one = {reg: regions[reg]}
                one_dir = indiv_root / _sanitize_token(reg)
                one_dir.mkdir(parents=True, exist_ok=True)
                reg_names1 = [reg]

                fig_r = make_subplots(rows=1, cols=1, subplot_titles=reg_names1)
                reg_isis = []
                for _k, spk in one[reg]:
                    reg_isis.append(_compute_isis_ms(spk, min_isi_ms=min_isi, max_isi_ms=max_isi))
                isis = np.concatenate(reg_isis) if reg_isis else np.asarray([], dtype=float)
                fig_r.add_trace(_plotly_hist_trace(isis, name=f"{reg} ISI", bins=args.bins, bin_width=args.bin_width_ms), row=1, col=1)
                fig_r.update_layout(title=f"ISI histogram — {reg} — {label}", barmode="overlay")
                fig_r.update_xaxes(title_text="ISI (ms)", row=1, col=1)
                fig_r.update_yaxes(title_text="Count", row=1, col=1)
                (one_dir / "isi_hist_by_region.html").write_text(fig_r.to_html(include_plotlyjs="cdn"), encoding="utf-8")

                fig_one_all = go.Figure()
                fig_one_all.add_trace(_plotly_hist_trace(isis, name=reg, bins=args.bins, bin_width=args.bin_width_ms))
                fig_one_all.update_layout(title=f"ISI histogram — {reg} — {label}", xaxis_title="ISI (ms)", yaxis_title="Count")
                fig_one_all.write_html(str(one_dir / "isi_hist_all_regions_combined.html"), include_plotlyjs="cdn")

                fig_one_first = go.Figure()
                k0, spk0 = sorted(one[reg], key=lambda x: str(x[0].electrode))[0]
                isis0 = _compute_isis_ms(spk0, min_isi_ms=min_isi, max_isi_ms=max_isi)
                fig_one_first.add_trace(_plotly_hist_trace(isis0, name=f"{reg}: {k0.electrode} cl={k0.cluster}", bins=args.bins, bin_width=args.bin_width_ms))
                fig_one_first.update_layout(title=f"ISI histogram — first cluster — {reg} — {label}", xaxis_title="ISI (ms)", yaxis_title="Count")
                fig_one_first.write_html(str(one_dir / "isi_hist_first_cluster_per_region.html"), include_plotlyjs="cdn")

            print(f"Wrote combined:\n  {out1}\n  {out2}\n  {out3}\nWrote per-file outputs under:\n  {indiv_root}")
            return

        print(f"Wrote:\n  {out1}\n  {out2}\n  {out3}")
        return

    # Default (non-region) behavior
    if kind in ("spiketimeset", "spiketime_npy"):
        raise SystemExit("Direct .npy inputs are currently supported only with --by-region (use --dataset-spec spiketimeset:...).")
    st, label = _load_spike_struct(spec, good_only=bool(args.good_only), max_duration_s=args.max_duration_s)

    units = _select_units(_iter_units(st), electrode_regex=args.electrode, cluster=args.cluster)
    if not units:
        raise SystemExit("No units matched selection. Try a different --electrode / --cluster.")

    min_isi = float(args.min_isi_ms)
    max_isi = float(args.max_isi_ms) if args.max_isi_ms is not None else None

    if args.pool == "all":
        all_isis = []
        for _k, spk in units:
            all_isis.append(_compute_isis_ms(spk, min_isi_ms=min_isi, max_isi_ms=max_isi))
        isis = np.concatenate(all_isis) if all_isis else np.asarray([], dtype=float)

        title = (
            f"ISI histogram — {label} — units={len(units)} "
            f"(min={min_isi:g}ms"
            + (f", max={max_isi:g}ms" if max_isi is not None else "")
            + ")"
        )
        fig_lin, fig_log = _make_plotly_figure(
            isis_ms=isis,
            title=title,
            bins=args.bins,
            bin_width_ms=args.bin_width_ms,
            plot_log=bool(args.log),
            overlay_gaussian=str(args.overlay_gaussian),
        )
        if args.output_html:
            base = Path(args.output_html)
            lin_path = base
            log_path = base.with_name(f"{base.stem}_log{base.suffix}") if args.log else None
            _write_or_show_plotly(fig_lin, fig_log, out_html_linear=str(lin_path), out_html_log=str(log_path) if log_path else None)
        else:
            _write_or_show_plotly(fig_lin, fig_log, out_html_linear=None, out_html_log=None)
        return

    # per-unit mode
    for k, spk in units:
        isis = _compute_isis_ms(spk, min_isi_ms=min_isi, max_isi_ms=max_isi)
        title = f"ISI histogram — {label} — {k.electrode} cl={k.cluster} (nISI={isis.size})"
        fig_lin, fig_log = _make_plotly_figure(
            isis_ms=isis,
            title=title,
            bins=args.bins,
            bin_width_ms=args.bin_width_ms,
            plot_log=bool(args.log),
            overlay_gaussian=str(args.overlay_gaussian),
        )

        if args.output_html:
            base = Path(args.output_html)
            out_lin = base
            if base.suffix.lower() in (".html", ".htm"):
                elec_tok = _sanitize_token(k.electrode)
                out_lin = base.with_name(
                    f"{base.stem}__{elec_tok}_cl{k.cluster}.html"
                )
            out_log = None
            if args.log:
                stem = Path(out_lin).stem
                out_log = Path(out_lin).with_name(f"{stem}_log{Path(out_lin).suffix}")
            _write_or_show_plotly(fig_lin, fig_log, out_html_linear=str(out_lin), out_html_log=str(out_log) if out_log else None)
        else:
            _write_or_show_plotly(fig_lin, fig_log, out_html_linear=None, out_html_log=None)


if __name__ == "__main__":
    main()

