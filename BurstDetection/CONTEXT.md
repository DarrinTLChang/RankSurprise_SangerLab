# BurstDetection — project context

Use this file with `README.md` when orienting to the repo, running jobs, or extending the pipeline.

## Purpose

Burst detection on sorted spike data: **unit → region → network** bursts (Rank Surprise, MaxISI, or alpha-mean-ISI). Optional region-exclusion study, rasters (Plotly), co-activity, EMG/proxy panels, and standalone analysis scripts.

## Where to work

- **Primary entry:** `main.py` (run from `BurstDetection/` so relative `patient_data/...` paths resolve).
- **Parameters and datasets:** `config.py` (`SpikeTime_Mat_File`, method toggles, SNR/FR limits, output roots, RS alphas).
- **Core logic:** `pipeline/` — `detection.py`, `stats.py`, `utils.py`, `plotting.py`, `region_exclusion.py`, `kilosort_loader.py`, etc.

## Dataset entries (`SpikeTime_Mat_File`)

Each list entry is a string interpreted by `parse_dataset_spec()` in `pipeline/kilosort_loader.py`:

| Prefix / form | Kind | Meaning |
|----------------|------|---------|
| Plain path, e.g. `patient_data/s516/spikeTime_p2.mat` | `mat` | MATLAB file; variable `spikeTime` |
| `kilosort:<path>` | `kilosort` | Single Kilosort 4 output folder |
| `kilosortset:<path1>;<path2>;...` | `kilosortset` | Multiple KS folders merged for one run |

Spike times in the pipeline are in **milliseconds**. `dataSegmentLength` on channels is recording length in **seconds**.

## CLI subset selection (`main.py`)

Mutually exclusive:

- `--only-datasets INDICES` — comma-separated 0-based indices into `SpikeTime_Mat_File`.
- `--only-dataset-contains FRAG` — repeat for **AND** substring match on the full entry string (e.g. patient folder + `spikeTime_p1`).

Optional (after the above):

- `--only-dataset-kind {mat,kilosort,kilosortset}` — restrict to that kind.

Examples:

```text
python -u main.py --only-dataset-kind mat --only-dataset-contains s516_surgery --only-dataset-contains spikeTime_p1
python -u main.py --only-dataset-kind kilosort --only-dataset-contains m361 --only-dataset-contains imec1
```

Use **`python -u`** (or `PYTHONUNBUFFERED=1`) for live logs. **`conda run`** often block-buffers stdout so the terminal looks idle until the buffer flushes. Prefer `conda activate bursts` then `python -u main.py ...` from `BurstDetection/`.

## Method selection

In `config.py`, **exactly one** of `maxisi_toggle`, `alpha_meanisi_toggle`, `rankSurprise_toggle` must be `True` or `main.py` raises on startup.

`RS_ALPHA_SETS` in `config.py` drives a loop over Rank Surprise alpha triples for the main run (when using RS).

## RankSurprise multi-stage baseline (planned change)

Current RS staging uses **onset-time trains** at higher levels (region, then network), and RS is run again on those compiled onset trains. This can dilute “surprise” when many channels burst synchronously at repeated times (e.g. strong clusters at 1s, 3s, 10s): the compiled train’s own ISI structure becomes the reference and those clusters can look “not surprising”.

Planned change for **region-level RS** and **network-level RS**:

- **Observed event train (unchanged)**: compile the relevant lower-level onsets into a single event train for that region/network (as today).
- **Reference / null train (new)**: build an *independence-preserving* reference distribution by taking each contributing channel/unit (for region) or each contributing region (for network) and applying a **random constant time offset** (different per source), then merging the offset trains.
  - Offsets are sampled once per run from a reproducible RNG seed (so results are repeatable).
  - The goal is to break cross-source synchrony while preserving within-source timing statistics.
- **Scoring**: compare the observed compiled train to this offset-based reference (instead of using only the observed compiled train’s own ISI structure as its implicit baseline).

Implementation notes to keep consistent across datasets:

- **Offsets are in ms** (pipeline spike/event times are ms).
- **Reproducibility**: use a documented seed (e.g. derived from patient/period/run_tag or a fixed config seed).
- **Offset range**: choose a range large enough to decorrelate sources relative to expected burst widths/IBIs (paper should state the exact range).

## Output layout (current)

Outputs go under **per-method roots** (see `config.py`), not only legacy `outputs/` names in older docs:

- **Rank Surprise + `mat` (human-style paths):** `OUTPUT_ROOT_RS` (e.g. `F:\SangerLabBursts_RS\<patient>\<Period>\...`).
- **Rank Surprise + `kilosort` / `kilosortset`:** `OUTPUT_ROOT_RS_MOUSE` (e.g. `F:\mouse_RS\...`) with session / imec / shank layout set in `main.py`.
- **maxISI / alpha_meanISI:** `OUTPUT_ROOT_MAXISI` / `OUTPUT_ROOT_ALPHA_MEANISI`.

`BURST_ROOT` defaults to `OUTPUT_ROOT_RS` for proxy/helper scripts that still assume the human RS tree; point them at `OUTPUT_ROOT_RS_MOUSE` or pass script-specific roots when analyzing mouse Kilosort outputs.

## Kilosort-related config (abbreviated)

- `KILOSORT_GOOD_ONLY`, `KILOSORT_SKIP_PIPELINE_SNR_FILTER`, `KILOSORT_WRITE_SPIKETIME_MAT`, shank/raster grouping flags — see comments in `config.py`.
- Labels and output folder naming for KS paths: `dataset_labels_any`, `run_tag_from_params_kilosort` in `main.py` / `utils`.

## Python / Windows gotchas

- **Windows paths in `config.py`:** use **`r"..."`** raw strings or forward slashes. A normal `"C:\Users\..."` string raises `unicodeescape` errors (`\U`, etc.).
- **Parallel wrapper:** comments in `config.py` reference `run_datasets_parallel.ps1`; that script may live outside the repo. When pasting multiline PowerShell, do not type the secondary prompt `>>` — only the continuation lines (spaces + code).

## Other scripts (high level)

| Script | Role |
|--------|------|
| `compare_proxy_vs_network.py` | Proxy vs network burst comparisons |
| `channel_reduction_study.py`, `channel_burst_roc.py` | Channel / ROC style studies |
| `cross_patient_region_exclusion.py` | Aggregate region-exclusion `results.csv` across patients |
| `plot_proxy_vs_fr_by_patient.py`, `proxy_*.py` | Proxy lag, slopes, demos, onset-triggered plots |
| `kilosort_to_spiketime_mat.py` | Helper around Kilosort → `.mat` |

Each may assume paths under `BURST_ROOT`, `PROXY_ANALYSIS_ROOT`, or CLI `--burst-root` / `--run-tag` — check the script’s argparse and defaults before batch runs.

## README note

`README.md` still mentions legacy folders like `outputs/` and `outputs_RS_burst/` in places. The **authoritative** run output locations are the **`OUTPUT_ROOT_*`** directories and the `run_dir` logic in `main.py`. Prefer this `CONTEXT.md` + `config.py` for paths and layout.
