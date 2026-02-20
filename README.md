# Rank Surprise Burst Detection

Spike burst detection pipeline for multi-electrode array (MEA) recordings, built for the Sanger Lab. Detects bursts at the unit, region, and network level using the **Rank Surprise** algorithm, with optional support for MaxISI and alpha-mean-ISI methods.

## Overview

The pipeline processes spike-sorted `.mat` files and produces:

- Per-unit burst detection via Rank Surprise (or MaxISI / alpha-mean-ISI)
- Region-level and network-level burst aggregation
- Interactive raster plots with burst overlays, co-activity panels, and EMG traces (via Plotly)
- ISI histograms and summary statistics
- CSV exports of cluster stats and burst labels

## Project Structure

```
BurstDetection/
├── main.py                  # Entry point — runs the full pipeline
├── config.py                # All tunable parameters and dataset paths
├── pipeline/
│   ├── detection.py         # Burst detection (Rank Surprise, MaxISI, alpha-mean-ISI)
│   ├── stats.py             # Cluster statistics, filtering, spike labeling
│   ├── coactivity.py        # Co-activity and firing rate computation
│   ├── emg.py               # EMG loading, preprocessing, and trace building
│   ├── isi.py               # ISI distribution analysis and plotting
│   ├── plotting.py          # All visualization (raster, co-activity, EMG panels)
│   └── utils.py             # Run parameter management, region inference, helpers
└── patient_data/            # (not tracked) .mat spike files and clinical notes
```

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pandas`, `plotly`

## Usage

1. Place spike-sorted `.mat` files in `BurstDetection/patient_data/`.
2. Edit `BurstDetection/config.py` to select datasets and set parameters.
3. Run from inside the `BurstDetection/` directory:

```bash
cd BurstDetection
python main.py
```

Results are written to `BurstDetection/outputs/<patient>/<period>/`.

## Configuration

Key parameters in `config.py`:

| Parameter | Description |
|---|---|
| `SpikeTime_Mat_File` | List of `.mat` files to process |
| `rankSurprise_toggle` | Enable Rank Surprise detection (default) |
| `RS_alpha_percentage_stage1` | Significance threshold for unit-level bursts |
| `RS_Percentile_Limit_stage1` | ISI percentile limit for unit-level detection |
| `RS_alpha_percentage_region` | Significance threshold for region-level bursts |
| `RS_alpha_percentage_network` | Significance threshold for network-level bursts |
| `SNR_MIN` / `FR_MIN_HZ` | Cluster quality filters |
| `coactivity_bins_s` | Time bin width for co-activity analysis |
| `PLOT_EMG` | Include EMG traces in output figures |

## Detection Methods

- **Rank Surprise**: Non-parametric method that evaluates ISI sequences against a uniform null. Runs in three stages — unit bursts, then region aggregation, then network aggregation.
- **MaxISI**: Classical method using a fixed or adaptive ISI threshold to define burst boundaries.
- **Alpha-mean-ISI**: Uses a fraction of each unit's mean ISI as the burst threshold.

Select exactly one method via the toggle flags in `config.py`.
