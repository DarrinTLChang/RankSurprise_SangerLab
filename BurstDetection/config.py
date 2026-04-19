from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.ma import nonzero

# ============================================================
# DATASETS
# ============================================================


# cd "c:\Users\Maral\Desktop\Darrin\RankSurprise_SangerLab-2\BurstDetection"
# .\run_datasets_parallel.ps1 -CondaEnv bursts -Kind mat -ContainsJobs @(
#     's516_surgery,spikeTime_p1',
#     's522,spikeTime_p2',
#     's522,spikeTime_p3',
# )

# .\run_datasets_parallel.ps1 -CondaEnv bursts -Kind kilosort -ContainsJobs @(
#     'm361,imec0',
#     'm361,imec1',
#     'm361,imec2'
#)

# .\run_datasets_parallel.ps1 -CondaEnv bursts -Kind mat -ContainsJobs @(

#     's522',
#     's523',
# )





# 
# #conda run -n bursts python main.py --only-dataset-kind mat --only-dataset-contains s531
# python -u main.py --only-dataset-kind mat --only-dataset-contains s516_surgery --only-dataset-contains spikeTime_p1
SpikeTime_Mat_File = [
     #511 on the fence

    #benzodacypin 1 week, gradually fades until the next 
    #check induce benzo and see as it gradually fades how does it gradaully fade over time and effects on bursts
    #benzo is one of the most important drugs for dystonia control

    "patient_data/s432/spikeTime_p2.mat",
    "patient_data/s432/spikeTime_p7.mat",
    "patient_data/s432/spikeTime_p9.mat",

    "patient_data/s508/spikeTime_p1.mat",
    "patient_data/s508/spikeTime_p2.mat",
    "patient_data/s508/spikeTime_p5.mat",
    "patient_data/s508/spikeTime_p6.mat",
    "patient_data/s508/spikeTime_p7.mat",
    "patient_data/s508/spikeTime_p8.mat",
    "patient_data/s508/spikeTime_p9.mat",
    "patient_data/s508/spikeTime_p10.mat",

    "patient_data/s509/spikeTime_p2.mat",
    "patient_data/s509/spikeTime_p3.mat",
    "patient_data/s509/spikeTime_p4.mat",
    "patient_data/s509/spikeTime_p5.mat",
    "patient_data/s509/spikeTime_p6.mat",
    "patient_data/s509/spikeTime_p7.mat",
    "patient_data/s509/spikeTime_p8.mat",
    "patient_data/s509/spikeTime_p9.mat",

    "patient_data/s510/spikeTime_p2.mat",
    "patient_data/s510/spikeTime_p3.mat",
    "patient_data/s510/spikeTime_p4.mat",
    "patient_data/s510/spikeTime_p5.mat",
    "patient_data/s510/spikeTime_p10.mat",
    "patient_data/s510/spikeTime_p11.mat",
    "patient_data/s510/spikeTime_p12.mat",
    "patient_data/s510/spikeTime_p13.mat",
    "patient_data/s510/spikeTime_p14.mat",
    "patient_data/s510/spikeTime_p15.mat",

    "patient_data/s511/spikeTime_p1.mat",
    "patient_data/s511/spikeTime_p2.mat",
    "patient_data/s511/spikeTime_p3.mat",
    "patient_data/s511/spikeTime_p4.mat",
    "patient_data/s511/spikeTime_p5.mat",
    "patient_data/s511/spikeTime_p6.mat",

    "patient_data/s512/spikeTime_p1.mat",
    "patient_data/s512/spikeTime_p2.mat",
    "patient_data/s512/spikeTime_p3.mat",
    "patient_data/s512/spikeTime_p4.mat",
    "patient_data/s512/spikeTime_p9.mat",
    "patient_data/s512/spikeTime_p10.mat",
    "patient_data/s512/spikeTime_p11.mat",
    "patient_data/s512/spikeTime_p12.mat",

    "patient_data/s515/spikeTime_p1.mat",
    "patient_data/s515/spikeTime_p2.mat",
    "patient_data/s515/spikeTime_p3.mat",
    "patient_data/s515/spikeTime_p4.mat",

    "patient_data/s516_surgery/spikeTime_p1.mat",
    "patient_data/s516/spikeTime_p2.mat",
    "patient_data/s516/spikeTime_p3.mat",
    "patient_data/s516/spikeTime_p4.mat",
    "patient_data/s516/spikeTime_p5.mat",
    "patient_data/s516/spikeTime_p6.mat",
    "patient_data/s516/spikeTime_p7.mat",
    "patient_data/s516/spikeTime_p8.mat",
    "patient_data/s516/spikeTime_p9.mat",

    "patient_data/s517/spikeTime_p9.mat",
    "patient_data/s517/spikeTime_p10.mat",
    "patient_data/s517/spikeTime_p11.mat",

    "patient_data/s519/spikeTime_p1.mat",
    "patient_data/s519/spikeTime_p2.mat",

    "patient_data/s520/spikeTime_p10.mat", #
    "patient_data/s520/spikeTime_p12.mat", #
    "patient_data/s520/spikeTime_p13.mat", #

    "patient_data/s521/spikeTime_p1.mat",
    "patient_data/s521/spikeTime_p2.mat",
    "patient_data/s521/spikeTime_p3.mat",
    "patient_data/s521/spikeTime_p4.mat",
    "patient_data/s521/spikeTime_p10.mat",
    "patient_data/s521/spikeTime_p11.mat",
    "patient_data/s521/spikeTime_p12.mat",
    "patient_data/s521/spikeTime_p13.mat",

    "patient_data/s522/spikeTime_p2.mat",
    "patient_data/s522/spikeTime_p3.mat",
    "patient_data/s522/spikeTime_p6.mat",
    "patient_data/s522/spikeTime_p7.mat",##
    "patient_data/s522/spikeTime_p13.mat",
    "patient_data/s522/spikeTime_p16.mat",
    "patient_data/s522/spikeTime_p17.mat",
    "patient_data/s522/spikeTime_p5.mat",  #very long

    "patient_data/s523/spikeTime_p1.mat",
    "patient_data/s523/spikeTime_p3.mat",
    "patient_data/s523/spikeTime_p7.mat",

    "patient_data/s524/spikeTime_p1.mat",
    "patient_data/s524/spikeTime_p2.mat",
    "patient_data/s524/spikeTime_p3.mat",
    "patient_data/s524/spikeTime_p4.mat",

    "patient_data/s527/spikeTime_p1.mat",
    "patient_data/s527/spikeTime_p2.mat",
    "patient_data/s527/spikeTime_p3.mat",

    "patient_data/s530/spikeTime_p1.mat",
    "patient_data/s530/spikeTime_p2.mat",
    "patient_data/s530/spikeTime_p3.mat",


    # "patient_data/s531/day1_baseline/spikeTime_p2.mat",
   
    # r"kilosort:F:\rat data\m360\shank0\imec0\kilosort4",
    # r"kilosort:F:\rat data\m360\shank1\imec0\kilosort4",
    # r"kilosort:F:\rat data\m360\shank2\imec0\kilosort4",
    # r"kilosort:F:\rat data\m360\shank3\imec0\kilosort4",

    # r"kilosort:F:\rat data\m360\shank0\imec1\kilosort4",
    # r"kilosort:F:\rat data\m360\shank1\imec1\kilosort4",
    # r"kilosort:F:\rat data\m360\shank2\imec1\kilosort4",
    # r"kilosort:F:\rat data\m360\shank3\imec1\kilosort4",

    # r"kilosort:F:\rat data\m360\shank0\imec2\kilosort4",
    # r"kilosort:F:\rat data\m360\shank1\imec2\kilosort4",
    # r"kilosort:F:\rat data\m360\shank2\imec2\kilosort4",
    # r"kilosort:F:\rat data\m360\shank3\imec2\kilosort4",


    # r"kilosort:F:\rat data\m361\shank0\imec0\kilosort4",
    # r"kilosort:F:\rat data\m361\shank1\imec0\kilosort4",
    # r"kilosort:F:\rat data\m361\shank2\imec0\kilosort4",
    # r"kilosort:F:\rat data\m361\shank3\imec0\kilosort4",

    # r"kilosort:F:\rat data\m361\shank0\imec1\kilosort4",
    # r"kilosort:F:\rat data\m361\shank1\imec1\kilosort4",
    # r"kilosort:F:\rat data\m361\shank2\imec1\kilosort4",
    # r"kilosort:F:\rat data\m361\shank3\imec1\kilosort4",

    # r"kilosort:F:\rat data\m361\shank0\imec2\kilosort4",
    # r"kilosort:F:\rat data\m361\shank1\imec2\kilosort4",
    # r"kilosort:F:\rat data\m361\shank2\imec2\kilosort4",
    # r"kilosort:F:\rat data\m361\shank3\imec2\kilosort4",

    # r"kilosortset:F:\rat data\m360\shank0\imec0\kilosort4;F:\rat data\m360\shank1\imec0\kilosort4;F:\rat data\m360\shank2\imec0\kilosort4;F:\rat data\m360\shank3\imec0\kilosort4",
    # r"kilosortset:F:\rat data\m360\shank0\imec1\kilosort4;F:\rat data\m360\shank1\imec1\kilosort4;F:\rat data\m360\shank2\imec1\kilosort4;F:\rat data\m360\shank3\imec1\kilosort4",
    # r"kilosortset:F:\rat data\m360\shank0\imec2\kilosort4;F:\rat data\m360\shank1\imec2\kilosort4;F:\rat data\m360\shank2\imec2\kilosort4;F:\rat data\m360\shank3\imec2\kilosort4",
   
    # r"kilosortset:F:\rat data\m361\shank0\imec0\kilosort4;F:\rat data\m361\shank1\imec0\kilosort4;F:\rat data\m361\shank2\imec0\kilosort4;F:\rat data\m361\shank3\imec0\kilosort4",
    # r"kilosortset:F:\rat data\m361\shank0\imec1\kilosort4;F:\rat data\m361\shank1\imec1\kilosort4;F:\rat data\m361\shank2\imec1\kilosort4;F:\rat data\m361\shank3\imec1\kilosort4",
    # r"kilosortset:F:\rat data\m361\shank0\imec2\kilosort4;F:\rat data\m361\shank1\imec2\kilosort4;F:\rat data\m361\shank2\imec2\kilosort4;F:\rat data\m361\shank3\imec2\kilosort4",


    # "patient_data/s531/day1_test/spikeTime_p2.mat",


    # "patient_data/s531/day1_test/spikeTime_p2.mat",
    # "patient_data/s531/day1_test/spikeTime_p3.mat",
    # 'patient_data/s531/day1_baseline/spikeTime_p3.mat',
    

    # "patient_data/s531/day2_baseline/spikeTime_p2.mat",
    # "patient_data/s531/day2_baseline/spikeTime_p3.mat",


    # "patient_data/s531/day4_baseline/spikeTime_p5.mat",


]

# Single CSV with bilateral proxy (main.py when PLOT_CORRELATION_GRAPH / COMPUTE_PROXY_VS_FR / PLOT_PROXY_PANEL).
# Path may be absolute or relative to the repository root. Time column is in seconds.
PROXY_CSV = r"F:\s531_binary\period2_baseline\offline\hemisphere_neo_binned.csv"  # e.g. r"F:\data\session_proxies.csv"
PROXY_TIME_COL = "time_s"
PROXY_LEFT_COL = "hemisphere_L_median_proxy"
PROXY_RIGHT_COL = "hemisphere_R_median_proxy"

# Legacy only: folder root for standalone scripts (proxy_lag_analysis, etc.) that still use the old per-side Excel layout.
PROXY_ROOT = None

# Kilosort 4 (see pipeline/kilosort_loader.py). Used when a dataset entry is ``kilosort:<path/to/kilosort4>``.
# good_only: keep only clusters with KSLabel ``good`` in cluster_group.tsv (False = include MUA/noise too).
# skip_pipeline_snr_filter: Kilosort units have NaN SNR here; True skips SNR_MIN/SNR_MAX in build_cache (FR_MIN_HZ still applies).
# max_duration_s: None = full recording; set e.g. 60.0 to use only spikes in the first 60 seconds (for quick tests).
KILOSORT_GOOD_ONLY = True
KILOSORT_SKIP_PIPELINE_SNR_FILTER = True
KILOSORT_MAX_DURATION_S = None  # e.g. 60.0 for first 60 s only
# For Kilosort synthetic names ``rat_*_shankN_*``: group/color by shank like regions (infer_region → shank0, …).
KILOSORT_COLOR_BY_SHANK = True

# Write a MATLAB spikeTime .mat alongside outputs for Kilosort datasets.
KILOSORT_WRITE_SPIKETIME_MAT = False

# Duplicate a copy of the main SangerLab HTML into one flat folder (gallery).
# The original output folder structure remains unchanged.
DUPLICATE_SANGER_HTML = True
DUPLICATE_SANGER_HTML_DIR = Path(r"F:\SangerLabBursts\sanger_html_gallery_seperated")

# ============================================================
# OUTPUT ROOTS (NEW LAYOUT)
# ============================================================
# All pipeline outputs are written under one root per method:
#   <METHOD_ROOT>\<patient>\<PeriodN>\<run_tag>\...
#
# Examples:
#   F:\SangerLabBursts_RS\s531\Period3\<run_tag>\...
#   F:\SangerLabBursts_maxISI\s531\Period3\<run_tag>\...
OUTPUT_ROOT_RS = Path(r"F:\SangerLabBursts_RS")
# RankSurprise outputs for kilosort: / kilosortset: (mouse neuropixels), separate from human paths.
OUTPUT_ROOT_RS_MOUSE = Path(r"F:\mouse_RS")
OUTPUT_ROOT_MAXISI = Path(r"F:\SangerLabBursts_maxISI")
OUTPUT_ROOT_ALPHA_MEANISI = Path(r"F:\SangerLabBursts_alphaMeanISI")

# Backwards-compat alias used by some proxy scripts as a default. Interpreted as the
# RankSurprise root under the new layout.
BURST_ROOT = OUTPUT_ROOT_RS

# Root for all proxy-vs-burst analysis outputs. Each script writes under a subdir:
#   proxy_lag_plots, proxy_detection_demo, onset_triggered_plots, proxy_slope_detection.
# Change this to redirect all proxy analysis outputs (e.g. to D_Drive).
PROXY_ANALYSIS_ROOT = Path(r"F:\SangerLabBursts\proxy_analysis")

# Run tag for proxy scripts (proxy_lag_analysis, proxy_detection_demo, onset_triggered_proxy,
# proxy_slope_detection). Must match a rankSurprise subdir under BURST_ROOT/patient/PeriodN/.
# Change here to switch burst parameter set without passing --run-tag each time.
PROXY_ANALYSIS_RUN_TAG = (
    "RS=(8,5,3)_(75,75,75)"
    "_minSpk=3__minDur=0ms__minCh=1"
    "_SNR=1.2-1000_FR=0.8Hz"
    "_region__network"
)

# ============================================================
# PIPELINE TOGGLES
# ============================================================
presentation_mode = False
PLOT_RASTER = True
PLOT_FR = False
# Fixed y-axis max for FR panel (Hz). When set (e.g. 0.2), all periods use [0, FR_Y_MAX]
# so scale is comparable; when None, each plot uses data range.
FR_Y_MAX = 0.2

# High-level figure toggles
# - PLOT_SANGER_PRESENTATION_FIG: classic SangerLab figure with raster + EMG + FR
# - PLOT_CORRELATION_GRAPH: raster + proxy (PROXY_CSV) + FR
# - COMPUTE_PROXY_VS_FR: correlate proxy with FR (hemi + region level), write CSV
PLOT_SANGER_PRESENTATION_FIG = True
PLOT_CORRELATION_GRAPH = False
COMPUTE_PROXY_VS_FR = False

pooling_toggle = False
plot_region_bar = True
disable_bursts = False
plot_network_bar = True


# If True: use proxy traces from PROXY_CSV (left/right columns) in the bottom panel
# of the SangerLab presentation figure instead of EMG. EMG is not loaded.
PLOT_PROXY_PANEL = False
PLOT_EMG = False

# When True, draw shaded rectangles (“shaders”) on the EMG/proxy panels to indicate
# network burst intervals (uses network_bars_L/R). Turn off if you want a cleaner proxy plot.
SHOW_PROXY_SHADERS: bool = False

# Method selection (exactly one must be True)
maxisi_toggle = False
alpha_meanisi_toggle = False
rankSurprise_toggle = True
RS_region_burst_toggle = True
plot_network_bursts = False
RS_NETWORK_ONSETS_TOGGLE = True
RUN_REGION_EXCLUSION_STUDY = False  # Leave-K-out span overlap vs GT (reads existing CSVs)
# If True: network burst span = onset window start + median(burst lengths). Outputs go to *_onset_length folders.
NETWORK_SPAN_ONSET_PLUS_LENGTH = True
# If True: region burst span = onset window start + median(burst lengths). Same norm for raster bars and CSVs.
REGION_SPAN_ONSET_PLUS_LENGTH = True

# Cross-patient region exclusion: patients to exclude from cross-patient summaries/boxplots (e.g. ["s527"]).
# Per-patient summaries are still run for these; they are only omitted from cross-patient aggregation.
CROSS_PATIENT_EXCLUDE_PATIENTS: list[str] = ["s510","s512","s527"]

# ============================================================
# MAXISI PARAMETERS
# ============================================================
maxISI_thresholds = [0]

# ISI detection
MIN_SPIKES_IN_BURST = 3
MIN_BURST_DURATION = 50  # ms
ibi_merge_factor = 0

# If True: MIN_BURST_DURATION is applied at ALL stages (unit, region, and network).
# If False (default): it is applied only at the final NETWORK burst level (and in
# region_exclusion's network spans), leaving unit and region bursts unfiltered
# by duration. For most analyses you probably want this False so that only
# short network bursts are removed.
APPLY_MIN_BURST_DURATION_ALL_STAGES = False

# Filtering
FR_MIN_HZ = 0.8
SNR_MIN = 1.2
SNR_MAX = 25

# ============================================================
# RANK SURPRISE PARAMETERS
# ============================================================
# Multi-stage RS baseline (region/network): offset-based independence null.
# When enabled, region- and network-level RS compare the observed compiled onset train
# against a reference train built by applying a random constant offset per source train
# (unit/channel for region; region for network) and merging them. This breaks cross-source
# synchrony while preserving within-source timing statistics.
RS_OFFSET_NULL_ENABLE: bool = True
# Seed for reproducible random offsets (same seed => identical offsets and results).
RS_OFFSET_NULL_SEED: int = 123
# Max absolute offset magnitude (ms). Offsets are sampled uniformly in [0, RS_OFFSET_NULL_MAX_OFFSET_MS).
# If None, defaults to the estimated recording duration for that side (from latest burst end time).
RS_OFFSET_NULL_MAX_OFFSET_MS: float | None = 1000.0

# Stage 1 WIN-SHUFF null (Stella et al., eNeuro 2022): optional single-surrogate
# reference ISI pool from the same spike train (breaks fine-scale burst structure).
# RS_WIN_SHUFF_WINDOW_MS = shuffle window Δ_ws; RS_WIN_SHUFF_BIN_MS = inner bin b (must divide window evenly).
RS_WIN_SHUFF_STAGE1_ENABLE: bool = True
RS_WIN_SHUFF_WINDOW_MS: float = 200.0
RS_WIN_SHUFF_BIN_MS: float = 10.0
RS_WIN_SHUFF_SEED: int = 456

# Stage 1: unit/cluster bursts  
RS_Limit_stage1 = None
RS_Percentile_Limit_stage1 = 75
RS_alpha_percentage_stage1 = 0.08
RS_alpha_stage1 = -np.log(RS_alpha_percentage_stage1)

# Region-level
RS_Limit_region = None
RS_Percentile_Limit_region = 75
RS_alpha_percentage_region = 0.05
RS_alpha_region = -np.log(RS_alpha_percentage_region)

# Network-level
RS_Limit_network = None
RS_Percentile_Limit_network = 75
RS_alpha_percentage_network = 0.03
RS_alpha_network = -np.log(RS_alpha_percentage_network)

# Channel filtering for region/network bursts
MIN_UNIQUE_CHANNELS_REGION = 1
MIN_UNIQUE_CHANNELS_NETWORK = 0

RS_ALPHA_SETS = [ (0.05,    0.03,    0.02),(0.03,    0.02,    0.02)] # (stage1,  region,  network)
# RS_ALPHA_SETS = [ (0.03,    0.02,    0.02)] # (stage1,  region,  network)
# Optional sweep for Stage-1 WIN-SHUFF parameters (window_ms, bin_ms).
# These are iterated similarly to RS_ALPHA_SETS in main.py.
WIN_SHUFF_PARAM_SETS = [(200.0, 10.0),(500.0, 25.0)]  # (window_ms, bin_ms)

# ============================================================
# ALPHA MEAN-ISI PARAMETERS
# ============================================================
ALPHA_MAXISI = 0.2
MIN_MAXISI_MS = 5
MAX_MAXISI_MS = 100

# ============================================================
# CO-ACTIVITY
# ============================================================
coactivity_bins_s = [0.4]
OVERLAP_FRACTION = 0.2

# ============================================================
# PLOTTING - RASTER
# ============================================================
RASTER_OPACITY_BASE = 0.5
RASTER_OPACITY_BURST = 0.5
RASTER_DOT_SIZE = 3
RASTER_ROW_SPACING = 1
RASTER_PX_PER_ROW = 2
RASTER_MIN_HEIGHT = 700

# For Kilosort combined rasters: y-axis depth tick spacing (µm). Used to add readable depth ticks.
KILOSORT_DEPTH_TICK_UM = 500.0

# Raster layout for Kilosort:
# - False (default): units from all shanks are mingled and globally sorted by depth; y-axis shows depth ticks.
# - True: units are grouped into shank blocks, each block sorted by depth (legacy look).
KILOSORT_RASTER_GROUP_BY_SHANK = True
SUBPLOT_VSPACING = 0.04
COMMON_MARGINS = dict(t=70, r=240, l=110, b=60)
LEGEND_MARKER_SIZE = 10

# ============================================================
# PLOTTING - CO-ACTIVITY(FR) PANELS
# ============================================================
SHOW_SUMMED_COACTIVITY_PANEL = True
SHOW_REGION_COACTIVITY_PANEL = True
COACTIVITY_DOT_SIZE = 2
BURST_CHANNELS_COLOR = "rgb(139,69,19)"
CONNECT_LINES = True
SMOOTH_PANEL_SEC = 1
REGIONAL_LINE_WIDTH = 1.6
REGIONAL_OUTLINE_WIDTH = 0
REGIONAL_OUTLINE_COLOR = "rgba(255,255,255,0.85)"
REGIONAL_OPACITY = 0.85
PANEL_FIXED_PX = 250

# ============================================================
# PLOTTING - ISI
# ============================================================
ISI_HIST_MAX_MS = 10000
ISI_BAR_SHOW_STD = False

# ============================================================
# REGION COLORS & DISPLAY NAMES
# ============================================================
# If True, strip trailing digits from electrode head so GPi1/GPi2/GPi3 → GPi (one region).
# If False, keep them separate (GPi1, GPi2, GPi3 as distinct regions). Toggle as needed.
COMBINE_NUMBERED_REGIONS: bool = False

REGION_COLORS: dict[str, str] = {
    "GPi": "rgb(60, 110, 230)",
    "GPi1": "rgb(60, 110, 230)",
    "GPi2": "rgb(60, 110, 230)",
    "GPi3": "rgb(60, 110, 230)",
    "Vo": "rgb(255, 140, 0)",
    "VIMPPN": "rgb(0, 160, 0)",
    "VA": "rgb(253, 153, 255)",
    "NA": "rgb(0, 190, 190)",
    "VoSTNSNr": "rgb(255, 140, 0)",
    "ANT": "rgb(255, 140, 0)",
    "CMCL": "rgb(253, 153, 255)",
    "VOSTN": "rgb(255, 140, 0)",
    "VIM": "rgb(0, 160, 0)",
}

# Used when KILOSORT_COLOR_BY_SHANK and electrode names contain ``shankN`` (see infer_shank / infer_region).
SHANK_COLORS: dict[str, str] = {
    "shank0": "rgb(60, 110, 230)",
    "shank1": "rgb(255, 140, 0)",
    "shank2": "rgb(0, 160, 0)",
    "shank3": "rgb(253, 153, 255)",
    "shank4": "rgb(0, 190, 190)",
    "shank5": "rgb(180, 120, 40)",
    "shank6": "rgb(100, 100, 255)",
    "shank7": "rgb(255, 80, 80)",
}

REGION_DISPLAY: dict[str, str] = {
    "GPi": "GPi",
    "GPi1": "GPi 1",
    "GPi2": "GPi 2",
    "GPi3": "GPi 3",
    "Vo": "Vo",
    "VOSTN": "VO/STN",
    "VoSTNSNr": "VO/STN/SNr",
    "CMCL": "CM/CL",
    "VIMPPN": "VIM/PPN",
    "VA": "VA",
    "NA": "NA",
    "VIM": "VIM",
    "ANT": "ANT",
    "shank0": "Shank 0",
    "shank1": "Shank 1",
    "shank2": "Shank 2",
    "shank3": "Shank 3",
    "shank4": "Shank 4",
    "shank5": "Shank 5",
    "shank6": "Shank 6",
    "shank7": "Shank 7",
}