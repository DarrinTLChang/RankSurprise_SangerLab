from __future__ import annotations
from pathlib import Path
import numpy as np

# ============================================================
# DATASETS
# ============================================================
SpikeTime_Mat_File = [
     #511 on the fence


    # "patient_data/s432/spikeTime_p2.mat",
    # "patient_data/s432/spikeTime_p7.mat",
    # "patient_data/s432/spikeTime_p9.mat",

    # # "patient_data/s508/spikeTime_p1.mat",
    # # "patient_data/s508/spikeTime_p2.mat",
    # # "patient_data/s508/spikeTime_p5.mat",
    # # "patient_data/s508/spikeTime_p6.mat",
    # # "patient_data/s508/spikeTime_p7.mat",
    # # "patient_data/s508/spikeTime_p8.mat",
    # # "patient_data/s508/spikeTime_p9.mat",
    # # "patient_data/s508/spikeTime_p10.mat",

    # # "patient_data/s509/spikeTime_p2.mat",
    # # "patient_data/s509/spikeTime_p3.mat",
    # # "patient_data/s509/spikeTime_p4.mat",
    # # "patient_data/s509/spikeTime_p5.mat",
    # # "patient_data/s509/spikeTime_p6.mat",
    # # "patient_data/s509/spikeTime_p7.mat",
    # # "patient_data/s509/spikeTime_p8.mat",
    # # "patient_data/s509/spikeTime_p9.mat",

    # # "patient_data/s510/spikeTime_p2.mat",
    # # "patient_data/s510/spikeTime_p3.mat",
    # # "patient_data/s510/spikeTime_p4.mat",
    # # "patient_data/s510/spikeTime_p5.mat",
    # # "patient_data/s510/spikeTime_p10.mat",
    # # "patient_data/s510/spikeTime_p11.mat",
    # # "patient_data/s510/spikeTime_p12.mat",
    # # "patient_data/s510/spikeTime_p13.mat",
    # # "patient_data/s510/spikeTime_p14.mat",
    # # "patient_data/s510/spikeTime_p15.mat",

    # # "patient_data/s511/spikeTime_p1.mat",
    # # "patient_data/s511/spikeTime_p2.mat",
    # # "patient_data/s511/spikeTime_p3.mat",
    # # "patient_data/s511/spikeTime_p4.mat",
    # # "patient_data/s511/spikeTime_p5.mat",
    # # "patient_data/s511/spikeTime_p6.mat",

    # "patient_data/s512/spikeTime_p1.mat",
    # "patient_data/s512/spikeTime_p2.mat",
    # "patient_data/s512/spikeTime_p3.mat",
    # "patient_data/s512/spikeTime_p4.mat",
    # "patient_data/s512/spikeTime_p9.mat",
    # "patient_data/s512/spikeTime_p10.mat",
    # "patient_data/s512/spikeTime_p11.mat",
    # "patient_data/s512/spikeTime_p12.mat",

    # "patient_data/s515/spikeTime_p1.mat",
    # "patient_data/s515/spikeTime_p2.mat",
    # "patient_data/s515/spikeTime_p3.mat",
    # "patient_data/s515/spikeTime_p4.mat",

    # # "patient_data/s516/spikeTime_p2.mat",
    # # "patient_data/s516/spikeTime_p3.mat",
    # # "patient_data/s516/spikeTime_p4.mat",
    # # "patient_data/s516/spikeTime_p5.mat",
    # # "patient_data/s516/spikeTime_p6.mat",
    # # "patient_data/s516/spikeTime_p7.mat",
    # # "patient_data/s516/spikeTime_p8.mat",
    # # "patient_data/s516/spikeTime_p9.mat",

    # "patient_data/s517/spikeTime_p9.mat",
    # "patient_data/s517/spikeTime_p10.mat",
    # "patient_data/s517/spikeTime_p11.mat",

    # "patient_data/s519/spikeTime_p1.mat",
    # "patient_data/s519/spikeTime_p2.mat",

    # "patient_data/s520/spikeTime_p10.mat", #
    # "patient_data/s520/spikeTime_p12.mat", #
    # "patient_data/s520/spikeTime_p13.mat", #

    # "patient_data/s521/spikeTime_p1.mat",
    # "patient_data/s521/spikeTime_p2.mat",
    # "patient_data/s521/spikeTime_p3.mat",
    # "patient_data/s521/spikeTime_p4.mat",
    # "patient_data/s521/spikeTime_p10.mat",
    # "patient_data/s521/spikeTime_p11.mat",
    # "patient_data/s521/spikeTime_p12.mat",
    # "patient_data/s521/spikeTime_p13.mat",

    # "patient_data/s522/spikeTime_p2.mat",
    # "patient_data/s522/spikeTime_p3.mat",
    # "patient_data/s522/spikeTime_p6.mat",
    # "patient_data/s522/spikeTime_p7.mat",##
    # "patient_data/s522/spikeTime_p13.mat",
    # "patient_data/s522/spikeTime_p16.mat",
    # "patient_data/s522/spikeTime_p17.mat",
    # "patient_data/s522/spikeTime_p5.mat",  #very long

    # "patient_data/s523/spikeTime_p1.mat",
    # "patient_data/s523/spikeTime_p3.mat",
    # "patient_data/s523/spikeTime_p7.mat",

    # "patient_data/s524/spikeTime_p1.mat",
    # "patient_data/s524/spikeTime_p2.mat",
    # "patient_data/s524/spikeTime_p3.mat",
    # "patient_data/s524/spikeTime_p4.mat",

    # "patient_data/s527/spikeTime_p1.mat",
    # "patient_data/s527/spikeTime_p2.mat",
    # "patient_data/s527/spikeTime_p3.mat",

    # "patient_data/s530/spikeTime_p1.mat",
    # "patient_data/s530/spikeTime_p2.mat",
    # "patient_data/s530/spikeTime_p3.mat",

    # "patient_data/s531/day1_test/spikeTime_p2.mat",
    # "patient_data/s531/day1_test/spikeTime_p3.mat",
    # 'patient_data/s531/day1_baseline/spikeTime_p3.mat',
    

    "patient_data/s531/day2_baseline/spikeTime_p2.mat",
    "patient_data/s531/day2_baseline/spikeTime_p3.mat",


    # "patient_data/s531/day4_baseline/spikeTime_p5.mat",


]

# Root for fast proxy Excel files (same tree used by compare_proxy_vs_network.py).
# Each patient has folders sXXX_YYYYMM/periodN with:
#   subject s508 • period 2_L_bursts_rates_proxies.xlsx
#   subject s508 • period 2_R_bursts_rates_proxies.xlsx
PROXY_ROOT = "/Volumes/D_Drive/rasters_all_with_fast_proxies(neo)"

# Root for network/region burst CSVs (outputs_RS_burst). Used by proxy lag, detection demo,
# onset-triggered, and slope scripts when loading burst data.
BURST_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst")

# Root for all proxy-vs-burst analysis outputs. Each script writes under a subdir:
#   proxy_lag_plots, proxy_detection_demo, onset_triggered_plots, proxy_slope_detection.
# Change this to redirect all proxy analysis outputs (e.g. to D_Drive).
PROXY_ANALYSIS_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/proxy_analysis")

# Run tag for proxy scripts (proxy_lag_analysis, proxy_detection_demo, onset_triggered_proxy,
# proxy_slope_detection). Must match a rankSurprise subdir under BURST_ROOT/patient/PeriodN/.
# Change here to switch burst parameter set without passing --run-tag each time.
PROXY_ANALYSIS_RUN_TAG = (
    "separateGPi__SNR=1.2__FR=0.8Hz__aClust=8%__limClust=75"
    "__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3"
    "__minDur=0ms__minCh=1__region__network"
)

# ============================================================
# PIPELINE TOGGLES
# ============================================================
presentation_mode = False
PLOT_RASTER = True
PLOT_FR = True
# Fixed y-axis max for FR panel (Hz). When set (e.g. 0.2), all periods use [0, FR_Y_MAX]
# so scale is comparable; when None, each plot uses data range.
FR_Y_MAX = 0.2

# High-level figure toggles
# - PLOT_SANGER_PRESENTATION_FIG: classic SangerLab figure with raster + EMG + FR
# - PLOT_CORRELATION_GRAPH: raster + proxy (from rasters_all_with_fast_proxies) + FR
# - COMPUTE_PROXY_VS_FR: correlate proxy with FR (hemi + region level), write CSV
PLOT_SANGER_PRESENTATION_FIG = True
PLOT_CORRELATION_GRAPH = True
COMPUTE_PROXY_VS_FR = True

pooling_toggle = False
plot_region_bar = True
disable_bursts = False
plot_network_bar = True


# If True: use fast proxy activity (from Excel proxies) in the bottom panel of
# the SangerLab presentation figure instead of EMG. When this is True, EMG is
# not loaded and the bottom panel shows hemi_proxy instead, with the same size
# and layout as the EMG panel.
PLOT_PROXY_PANEL = False
PLOT_EMG = False

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
MIN_BURST_DURATION = 0  # ms
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
SNR_MAX = 1000

# ============================================================
# RANK SURPRISE PARAMETERS
# ============================================================
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
    "VIMPPN": "rgb(0,160,0)",
    "VoSTNSNr": "rgb(255,140,0)",
    "ANT": "rgb(255,140,0)",
    "CMCL": "rgb(253,153,255)",
    "VOSTN": "rgb(255,140,0)",
    "VIM": "rgb(0,160,0)",
}

REGION_DISPLAY: dict[str, str] = {
    "GPi": "GPi",
    "GPi1": "GPi 1",
    "GPi2": "GPi 2",
    "GPi3": "GPi 3",
    "VOSTN": "VO/STN",
    "VoSTNSNr": "VO/STN/SNr",
    "CMCL": "CM/CL",
    "VIMPPN": "VIM/PPN",
    "VIM": "VIM",
    "ANT": "ANT",
}
