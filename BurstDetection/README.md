# Burst Detection Pipeline

A spike burst detection pipeline that runs on spike-sorted data (e.g. from `.mat` files). It detects bursts at the **unit** (single unit/cluster), **region**, and **network** level, and can optionally run a region-exclusion (leave-K-out) study and cross-patient summaries.

## What the code does

- **Input**: Paths to spike-time `.mat` files and (optionally) EMG and notes files are set in `config.py`. The code expects a spike struct with electrode and cluster information and spike times.
- **Unit-level bursts**: For each unit, bursts are detected using one of three methods (Rank Surprise, MaxISI, or alpha-mean-ISI). Cluster quality filters (e.g. SNR, firing rate) are applied before detection.
- **Region-level bursts**: Electrode names are mapped to regions (via `parse_electrode` / `infer_region` in `pipeline/utils.py`). Unit bursts are aggregated by region and a second burst pass is run on region onset times. A config flag (`COMBINE_NUMBERED_REGIONS`) controls whether suffixes like `1`/`2` are stripped (e.g. GPi1 and GPi2 → one region “GPi”) or kept as separate regions.
- **Network-level bursts**: Region-level bursts are aggregated across regions and a third pass defines network bursts. Outputs include CSV tables of unit, region, and network bursts and (optionally) network span times for comparison.
- **Region-exclusion study**: For each run, the pipeline can run a leave-K-out study: for each subset of excluded regions, it re-runs network detection on the remaining regions and compares predicted network spans to ground-truth (all-region) spans using mean IoU. Results are written per run to `outputs_region_exclusion/.../results.csv`.
- **Cross-patient summary**: A separate script (`cross_patient_region_exclusion.py`) scans `outputs_region_exclusion`, loads all `results.csv` files, and for each patient writes a per-patient summary (original region labels). When run for all patients, it also builds a cross-patient summary with region names normalized into buckets (e.g. VoSTN and VoSTNSNr → VOSTN, GPi1 and GPi2 → GPi), writes a combined CSV with patient counts and contributor IDs, and produces Plotly boxplots of mean IoU by bucket (separate left/right).
- **Visualization**: The pipeline can generate raster plots (Plotly) with unit spikes, burst overlays, region and network burst bars, co-activity/firing-rate panels, and optional EMG traces. ISI histograms and summary stats are also produced.

No assumption is made about the recording hardware (e.g. MEA); the code only relies on the structure of the spike and config inputs.

## Project structure

```
BurstDetection/
├── main.py                      # Entry point: loads data, runs detection, optional region-exclusion, plots
├── config.py                    # Dataset paths, toggles, and all detection/plotting parameters
├── cross_patient_region_exclusion.py   # Aggregates region-exclusion results per patient and across patients
├── pipeline/
│   ├── detection.py             # Unit, region, and network burst detection (RS, MaxISI, alpha-mean-ISI)
│   ├── stats.py                 # Cluster statistics, filtering, spike labeling
│   ├── coactivity.py           # Co-activity and firing rate over time bins
│   ├── emg.py                  # EMG loading, preprocessing, trace building
│   ├── isi.py                  # ISI distributions and plots
│   ├── plotting.py             # Raster, co-activity, EMG panels (Plotly)
│   ├── region_exclusion.py     # Leave-K-out study: load GT spans, re-run network with excluded regions, IoU
│   └── utils.py                # Electrode parsing, region inference, run-tag and parameter helpers
└── patient_data/               # (not tracked) input .mat and related files
```

Outputs (paths set in code):

- `outputs/` — run params, cluster stats, spike labels, ISI, raster plots
- `outputs_RS_burst/` — unit, region, and network burst CSVs per run
- `outputs_region_exclusion/` — per-run region-exclusion results; per-patient summaries; `cross_patient/` for combined summary and boxplots

## Setup

```bash
pip install -r requirements.txt
```

Typical dependencies: `numpy`, `scipy`, `pandas`, `plotly` (and optionally `matplotlib` if used elsewhere).

## Usage

1. **Configure inputs and toggles**  
   Edit `config.py`: set `SpikeTime_Mat_File` (and any EMG/notes paths), choose one detection method (`rankSurprise_toggle`, `maxisi_toggle`, or `alpha_meanisi_toggle`), and set `RUN_REGION_EXCLUSION_STUDY` if you want the leave-K-out study.

2. **Run the pipeline**  
   From the `BurstDetection/` directory:

   ```bash
   cd BurstDetection
   python main.py
   ```

   This processes each configured `.mat` file: runs detection, writes burst CSVs and (if enabled) the region-exclusion study, then generates plots and ISI outputs.

3. **Cross-patient region-exclusion summary**  
   After runs have produced `outputs_region_exclusion/.../results.csv`:

   ```bash
   python cross_patient_region_exclusion.py
   ```

   With no arguments it finds all patient folders under `outputs_region_exclusion`, writes per-patient summaries and the cross-patient bucket summary + boxplots. Use `--patient <id>` to run only for one patient, or `--root` to point to another root directory.

## Configuration (high level)

- **Dataset**: `SpikeTime_Mat_File` — list of `.mat` paths; EMG and notes paths are set per dataset in the main loop.
- **Detection method**: Exactly one of `rankSurprise_toggle`, `maxisi_toggle`, `alpha_meanisi_toggle` should be True. Rank Surprise uses alpha/percentile limits at unit, region, and network stages.
- **Region behavior**: `COMBINE_NUMBERED_REGIONS` — if True, electrode names with trailing digits (e.g. GPi1, GPi2) are mapped to one region (GPi); if False, they stay distinct. The run tag includes `combinedGPi` or `separateGPi` at the front.
- **Quality filters**: `SNR_MIN`, `FR_MIN_HZ` (and method-specific params) control which units are included.
- **Region-exclusion**: `RUN_REGION_EXCLUSION_STUDY` enables the leave-K-out study after detection; it reads existing burst CSVs and writes `outputs_region_exclusion/.../results.csv`.
- **Plotting**: `PLOT_RASTER`, `PLOT_EMG`, `PLOT_FR`, etc. control which figures are generated.

See `config.py` for full parameter lists and comments.
