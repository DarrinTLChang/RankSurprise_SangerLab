from __future__ import annotations
import numpy as np

# ============================================================
# DATASETS
# ============================================================
SpikeTime_Mat_File = [

    # "patient_data/s432/spikeTime_p2.mat", 
    # "patient_data/s432/spikeTime_p7.mat",
    # "patient_data/s432/spikeTime_p9.mat", 

    # "patient_data/s508/spikeTime_p1.mat",
    # "patient_data/s508/spikeTime_p2.mat", 
    # "patient_data/s508/spikeTime_p5.mat", 
    # "patient_data/s508/spikeTime_p6.mat", 
    # "patient_data/s508/spikeTime_p7.mat", 
    # "patient_data/s508/spikeTime_p8.mat", 
    # "patient_data/s508/spikeTime_p9.mat", 
    # "patient_data/s508/spikeTime_p10.mat", 

    # "patient_data/s509/spikeTime_p2.mat", 
    # "patient_data/s509/spikeTime_p3.mat", 
    # "patient_data/s509/spikeTime_p4.mat", 
    # "patient_data/s509/spikeTime_p5.mat", 
    # "patient_data/s509/spikeTime_p6.mat", 
    # "patient_data/s509/spikeTime_p7.mat", 
    # "patient_data/s509/spikeTime_p8.mat", 
    # "patient_data/s509/spikeTime_p9.mat",


    # "patient_data/s510/spikeTime_p2.mat", 
    # "patient_data/s510/spikeTime_p3.mat", 
    # "patient_data/s510/spikeTime_p4.mat", 
    # "patient_data/s510/spikeTime_p5.mat", 
    # "patient_data/s510/spikeTime_p10.mat", 
    # "patient_data/s510/spikeTime_p11.mat", 
    # "patient_data/s510/spikeTime_p12.mat", 
    # "patient_data/s510/spikeTime_p13.mat", 
    # "patient_data/s510/spikeTime_p14.mat", 
    # "patient_data/s510/spikeTime_p15.mat", 
  
    # "patient_data/s511/spikeTime_p1.mat", 
    # "patient_data/s511/spikeTime_p2.mat", 
    # "patient_data/s511/spikeTime_p3.mat", 
    # "patient_data/s511/spikeTime_p4.mat", 
    # "patient_data/s511/spikeTime_p5.mat", 
    # "patient_data/s511/spikeTime_p6.mat", 
 
    # "patient_data/s512/spikeTime_p1.mat", 
    # "patient_data/s512/spikeTime_p2.mat", 
    # "patient_data/s512/spikeTime_p3.mat", 
    # "patient_data/s512/spikeTime_p4.mat", 
    # "patient_data/s512/spikeTime_p9.mat", 
    # "patient_data/s512/spikeTime_p10.mat", 
    # "patient_data/s512/spikeTime_p11.mat", 
    # "patient_data/s512/spikeTime_p12.mat", 

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
    # "patient_data/s522/spikeTime_p7.mat",
    # "patient_data/s522/spikeTime_p13.mat",
    # "patient_data/s522/spikeTime_p16.mat",
    # "patient_data/s522/spikeTime_p17.mat",
    # "patient_data/s522/spikeTime_p5.mat",  #very long

    # "patient_data/s523/spikeTime_p1.mat",
    # "patient_data/s523/spikeTime_p3.mat",

    # "patient_data/s527/spikeTime_p1.mat",
    # "patient_data/s527/spikeTime_p2.mat",
    # "patient_data/s527/spikeTime_p3.mat",

    # "patient_data/s530/spikeTime_p1.mat",
    "patient_data/s530/spikeTime_p2.mat",
    # "patient_data/s530/spikeTime_p3.mat",
]

# ============================================================
# PIPELINE TOGGLES
# ============================================================
presentation_mode = False

PLOT_RASTER = True
PLOT_EMG = True
PLOT_FR = True

pooling_toggle = False
plot_region_bar = True
disable_bursts = False
plot_network_bar = True

# Method selection (exactly one must be True)
maxisi_toggle = False
alpha_meanisi_toggle = False
rankSurprise_toggle = True
RS_region_burst_toggle = True
plot_network_bursts = False
RS_NETWORK_ONSETS_TOGGLE = True
RUN_REGION_EXCLUSION_STUDY = True  # Leave-K-out span overlap vs GT (reads existing CSVs)

# ============================================================
# MAXISI PARAMETERS
# ============================================================
maxISI_thresholds = [0]

# ISI detection
MIN_SPIKES_IN_BURST = 3
MIN_BURST_DURATION = 50  # ms
ibi_merge_factor = 0

# Filtering
FR_MIN_HZ = 0.8
SNR_MIN = 1.2

# ============================================================
# RANK SURPRISE PARAMETERS
# ============================================================
# Stage 1: unit/cluster bursts
RS_Limit_stage1 = None
RS_Percentile_Limit_stage1 = 75
RS_alpha_percentage_stage1 = 0.03
RS_alpha_stage1 = -np.log(RS_alpha_percentage_stage1)

# Region-level
RS_Limit_region = None
RS_Percentile_Limit_region = 75
RS_alpha_percentage_region = 0.02
RS_alpha_region = -np.log(RS_alpha_percentage_region)

# Network-level
RS_Limit_network = None
RS_Percentile_Limit_network = 75
RS_alpha_percentage_network = 0.01
RS_alpha_network = -np.log(RS_alpha_percentage_network)

# Channel filtering for region/network bursts
MIN_UNIQUE_CHANNELS_REGION = 1
MIN_UNIQUE_CHANNELS_NETWORK = 1

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
# If True, strip trailing digits from electrode head so GPi1/GPi2 → GPi (one region).
# If False, keep them separate (GPi1, GPi2 as distinct regions). Toggle as needed.
COMBINE_NUMBERED_REGIONS: bool = True

REGION_COLORS: dict[str, str] = {
    "GPi": "rgb(60, 110, 230)",
    "GPi1": "rgb(60, 110, 230)",
    "GPi2": "rgb(60, 110, 230)",
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
    "VOSTN": "VO/STN",
    "VoSTNSNr": "VO/STN/SNr",
    "CMCL": "CM/CL",
    "VIMPPN": "VIM/PPN",
    "VIM": "VIM",
    "ANT": "ANT",
}
