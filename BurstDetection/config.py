from __future__ import annotations
from pathlib import Path
import numpy as np

# Central configuration for the burst detection pipeline.
# Keep variable names stable: many modules use `from config import *`.

# ============================================================
# DATASETS
# ============================================================
# Keep this list as the authoritative dataset registry for `main.py`.
# You can run subsets via CLI (`--only-datasets`, `--only-dataset-contains`, `--only-dataset-kind`).
SpikeTime_Mat_File = [
    # "patient_data/burst_pause_example/spikeTime_p1.mat",


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
    "patient_data/s522/spikeTime_p7.mat",
    "patient_data/s522/spikeTime_p13.mat",
    "patient_data/s522/spikeTime_p16.mat",
    "patient_data/s522/spikeTime_p17.mat",
    "patient_data/s522/spikeTime_p5.mat",

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

    "patient_data/burstpaper_test/spikeTime_p1.mat",
    "patient_data/burstpaper_test/spikeTime_p2.mat",
    "patient_data/burstpaper_test/spikeTime_p3.mat",
    "patient_data/burstpaper_test/spikeTime_p4.mat",
    "patient_data/burstpaper_test/spikeTime_p5.mat",
    "patient_data/burstpaper_test/spikeTime_p6.mat",

    "patient_data/burstpaper_test/burst_pause/spikeTime_p1.mat",
    "patient_data/burstpaper_test/drift/spikeTime_p1.mat",
    "patient_data/burstpaper_test/drift/spikeTime_p2.mat",
    "patient_data/burstpaper_test/drift/spikeTime_p3.mat",


    "patient_data/s533/awake_no_stim/spikeTime_p1.mat",
    "patient_data/s533/sleep/spikeTime_p2.mat",

    # Examples / presets (keep commented; enable as needed):
    # "patient_data/s531/day1_baseline/spikeTime_p2.mat",

    r"kilosort:H:\rat data\m360\shank0\imec0\kilosort4",
    r"kilosort:H:\rat data\m360\shank1\imec0\kilosort4",
    r"kilosort:H:\rat data\m360\shank2\imec0\kilosort4",
    r"kilosort:H:\rat data\m360\shank3\imec0\kilosort4",

    r"kilosort:H:\rat data\m360\shank0\imec1\kilosort4",
    r"kilosort:H:\rat data\m360\shank1\imec1\kilosort4",
    r"kilosort:H:\rat data\m360\shank2\imec1\kilosort4",
    r"kilosort:H:\rat data\m360\shank3\imec1\kilosort4",

    r"kilosort:H:\rat data\m360\shank0\imec2\kilosort4",
    r"kilosort:H:\rat data\m360\shank1\imec2\kilosort4",
    r"kilosort:H:\rat data\m360\shank2\imec2\kilosort4",
    r"kilosort:H:\rat data\m360\shank3\imec2\kilosort4",


    r"kilosort:H:\rat data\m361\shank0\imec0\kilosort4",
    r"kilosort:H:\rat data\m361\shank1\imec0\kilosort4",
    r"kilosort:H:\rat data\m361\shank2\imec0\kilosort4",
    r"kilosort:H:\rat data\m361\shank3\imec0\kilosort4",

    r"kilosort:H:\rat data\m361\shank0\imec1\kilosort4",
    r"kilosort:H:\rat data\m361\shank1\imec1\kilosort4",
    r"kilosort:H:\rat data\m361\shank2\imec1\kilosort4",
    r"kilosort:H:\rat data\m361\shank3\imec1\kilosort4",

    r"kilosort:H:\rat data\m361\shank0\imec2\kilosort4",
    r"kilosort:H:\rat data\m361\shank1\imec2\kilosort4",
    r"kilosort:H:\rat data\m361\shank2\imec2\kilosort4",
    r"kilosort:H:\rat data\m361\shank3\imec2\kilosort4",

    r"kilosortset:H:\rat data\m360\shank0\imec0\kilosort4;H:\rat data\m360\shank1\imec0\kilosort4;H:\rat data\m360\shank2\imec0\kilosort4;H:\rat data\m360\shank3\imec0\kilosort4",
    r"kilosortset:H:\rat data\m360\shank0\imec1\kilosort4;H:\rat data\m360\shank1\imec1\kilosort4;H:\rat data\m360\shank2\imec1\kilosort4;H:\rat data\m360\shank3\imec1\kilosort4",
    r"kilosortset:H:\rat data\m360\shank0\imec2\kilosort4;H:\rat data\m360\shank1\imec2\kilosort4;H:\rat data\m360\shank2\imec2\kilosort4;H:\rat data\m360\shank3\imec2\kilosort4",
   
    r"kilosortset:H:\rat data\m361\shank0\imec0\kilosort4;H:\rat data\m361\shank1\imec0\kilosort4;H:\rat data\m361\shank2\imec0\kilosort4;H:\rat data\m361\shank3\imec0\kilosort4",
    r"kilosortset:H:\rat data\m361\shank0\imec1\kilosort4;H:\rat data\m361\shank1\imec1\kilosort4;H:\rat data\m361\shank2\imec1\kilosort4;H:\rat data\m361\shank3\imec1\kilosort4",
    r"kilosortset:H:\rat data\m361\shank0\imec2\kilosort4;H:\rat data\m361\shank1\imec2\kilosort4;H:\rat data\m361\shank2\imec2\kilosort4;H:\rat data\m361\shank3\imec2\kilosort4",


    # "patient_data/s531/day1_test/spikeTime_p2.mat",


    # "patient_data/s531/day1_test/spikeTime_p2.mat",
    # "patient_data/s531/day1_test/spikeTime_p3.mat",
    # 'patient_data/s531/day1_baseline/spikeTime_p3.mat',
    

    # "patient_data/s531/day2_baseline/spikeTime_p2.mat",
    # "patient_data/s531/day2_baseline/spikeTime_p3.mat",


    # "patient_data/s531/day4_baseline/spikeTime_p5.mat",


    #GNAO1####
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec0",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec0",
    r"kilosortset:H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec0;H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec1;H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec2;H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec3",
    r"kilosortset:H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec0;H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec1;H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec2;H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec3",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec1",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec1",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec2",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec2",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank13_g0\M422_20250509shank13_g0_imec3",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250509shank24_g0\M422_20250509shank24_g0_imec3",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec0",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec0",
    r"kilosortset:H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec0;H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec1;H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec2;H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec3",
    r"kilosortset:H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec0;H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec1;H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec2;H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec3",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec1",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec1",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec2",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec2",

    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank13_g0\M422_20250513shank13_g0_imec3",
    r"kilosort:H:\Mouse\M422_GNAO1\M422_20250513shank24_g0\M422_20250513shank24_g0_imec3",
    
    #control mice 401####
    r"kilosort:H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec0",
    r"kilosort:H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec0",
    r"kilosortset:H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec0;H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec1;H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec2;H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec3",
    r"kilosortset:H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec0;H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec1;H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec2;H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec3",

    r"kilosort:H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec1",
    r"kilosort:H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec1",

    r"kilosort:H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec2",
    r"kilosort:H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec2",

    r"kilosort:H:\Mouse\M401_WT\M401_20240911S13_g0\M401_20240911S13_g0_imec3",
    r"kilosort:H:\Mouse\M401_WT\M401_20240911S24_g0\M401_20240911S24_g0_imec3",

    r"kilosort:H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec0",
    r"kilosort:H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec0",
    r"kilosortset:H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec0;H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec1;H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec2;H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec3",
    r"kilosortset:H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec0;H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec1;H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec2;H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec3",

    r"kilosort:H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec1",
    r"kilosort:H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec1",

    r"kilosort:H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec2",
    r"kilosort:H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec2",

    r"kilosort:H:\Mouse\M401_WT\M401_20240917S13_g0\M401_20240917S13_g0_imec3",
    r"kilosort:H:\Mouse\M401_WT\M401_20240917S24_g0\M401_20240917S24_g0_imec3",

    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec0",
    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec0",
    r"kilosortset:H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec0;H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec1;H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec2;H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec3",
    r"kilosortset:H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec0;H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec1;H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec2;H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec3",

    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec1",
    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec1",

    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec2",
    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec2",

    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank13_g0\M415_20250226Shank13_g0_imec3",
    r"kilosort:H:\Mouse\M415_WT\M415_20250226Shank24_g0\M415_20250226Shank24_g0_imec3",

    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec0",
    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec0",
    r"kilosortset:H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec0;H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec1;H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec2;H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec3",
    r"kilosortset:H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec0;H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec1;H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec2;H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec3",

    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec1",
    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec1",

    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec2",
    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec2",

    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank13_g0\M415_20250301Shank13_g0_imec3",
    r"kilosort:H:\Mouse\M415_WT\M415_20250301Shank24_g0\M415_20250301Shank24_g0_imec3",

    r"kilosortset:H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec0;H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec1;H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec2",
    r"kilosortset:H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec0;H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec1;H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec2",
    r"kilosortset:H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec0;H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec1;H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec2",
    r"kilosortset:H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec0;H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec1;H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec2",
    
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec0",
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec0",
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec1",
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec1",
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S13_g0\M402_20240828S13_g0_imec2",
    r"kilosort:H:\Mouse\M402_WT\M402_20240828S24_g0\M402_20240828S24_g0_imec2",
    
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec0",
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec0",
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec1",
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec1",
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S13_g0\M402_20240830S13_g0_imec2",
    r"kilosort:H:\Mouse\M402_WT\M402_20240830S24_g0\M402_20240830S24_g0_imec2",

]

# ============================================================
# PROXY INPUT CONFIGURATION
# ============================================================
# Single CSV with bilateral proxy (main.py when PLOT_CORRELATION_GRAPH / COMPUTE_PROXY_VS_FR / PLOT_PROXY_PANEL).
# Path may be absolute or relative to the repository root. Time column is in seconds.

PROXY_CSV = r"H:\s531_binary\period2_baseline\offline\hemisphere_neo_binned.csv"  # e.g. r"H:\data\session_proxies.csv"
PROXY_TIME_COL = "time_s"
PROXY_LEFT_COL = "hemisphere_L_median_proxy"
PROXY_RIGHT_COL = "hemisphere_R_median_proxy"

# ============================================================
# KILOSORT CONTROLS
# ============================================================
# Kilosort 4 (see pipeline/kilosort_loader.py). Used when a dataset entry is ``kilosort:<path/to/kilosort4>``.
# good_only: keep only clusters with KSLabel ``good`` in cluster_group.tsv (False = include MUA/noise too).
# skip_pipeline_snr_filter: Kilosort units have NaN SNR here; True skips SNR_MIN/SNR_MAX in build_cache (FR_MIN_HZ still applies).
# max_duration_s: None = full recording; set e.g. 60.0 to use only spikes in the first 60 seconds
KILOSORT_GOOD_ONLY = True
KILOSORT_SKIP_PIPELINE_SNR_FILTER = True
KILOSORT_MAX_DURATION_S = 600  # e.g. 60.0 for first 60 s only
# For Kilosort synthetic names ``rat_*_shankN_*``: group/color by shank like regions (infer_region → shank0, …).
KILOSORT_COLOR_BY_SHANK = True

# Write a MATLAB spikeTime .mat alongside outputs for Kilosort datasets.
KILOSORT_WRITE_SPIKETIME_MAT = False

# Duplicate a copy of the main SangerLab HTML into one flat folder (gallery).
# The original output folder structure remains unchanged.
DUPLICATE_SANGER_HTML = True
DUPLICATE_SANGER_HTML_DIR = Path(r"H:\SangerLabBursts\sanger_html_gallery_seperated")

# ============================================================
# OUTPUT ROOTS (NEW LAYOUT)
# ============================================================
# All pipeline outputs are written under one root per method:
#   <METHOD_ROOT>\<patient>\<PeriodN>\<run_tag>\...
#
# Examples:
#   H:\SangerLabBursts_RS\s531\Period3\<run_tag>\...
#   H:\SangerLabBursts_maxISI\s531\Period3\<run_tag>\...
OUTPUT_ROOT_RS = Path(r"H:\SangerLabBursts_RS")
OUTPUT_ROOT_RS_MOUSE = Path(r"H:\mouse_RS")
OUTPUT_ROOT_MAXISI = Path(r"H:\SangerLabBursts_maxISI")
OUTPUT_ROOT_ALPHA_MEANISI = Path(r"H:\SangerLabBursts_alphaMeanISI")

# ============================================================
# PIPELINE TOGGLES
# ============================================================
presentation_mode = False
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

# ============================================================
# METHOD SELECTION
# ============================================================
# Exactly one detector family should be active.
maxisi_toggle = False
alpha_meanisi_toggle = False
rankSurprise_toggle = True
RS_region_burst_toggle = True
plot_network_bursts = False
RS_NETWORK_ONSETS_TOGGLE = True
# If True: network burst span = onset window start + median(burst lengths). Outputs go to *_onset_length folders.
NETWORK_SPAN_ONSET_PLUS_LENGTH = True
# If True: region burst span = onset window start + median(burst lengths). Same norm for raster bars and CSVs.
REGION_SPAN_ONSET_PLUS_LENGTH = True

# ============================================================
# SHARED DETECTION CONSTRAINTS
# ============================================================
# Shared across burst detectors unless a method-specific override is used.
MIN_SPIKES_IN_BURST = 3
MIN_BURST_DURATION = 20  # ms
# Optional upper bound for burst duration (ms). When set (not None), bursts/windows
# longer than this are dropped at the stage(s) where duration constraints apply.
MAX_BURST_DURATION = 10000
ibi_merge_factor = 0

# If True: MIN_BURST_DURATION is applied at ALL stages (unit, region, and network).
# If False (default): it is applied only at the final NETWORK burst level (and in
# saved network spans), leaving unit and region bursts unfiltered
# by duration. For most analyses you probably want this False so that only
# short network bursts are removed.
APPLY_MIN_BURST_DURATION_ALL_STAGES = False

# Unit quality filtering
FR_MIN_HZ = 0
SNR_MIN = 0.8
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
RS_OFFSET_NULL_MAX_OFFSET_MS: float | None = 10000.0

# Stage 1 core (unit/cluster bursts)
RS_Limit_stage1 = None
RS_Percentile_Limit_stage1 = 75
RS_alpha_percentage_stage1 = 0.08
RS_alpha_stage1 = -np.log(RS_alpha_percentage_stage1)

# Stage 1 WIN-SHUFF null (Stella et al., eNeuro 2022)
# Optional single-surrogate reference ISI pool from the same spike train.
# RS_WIN_SHUFF_WINDOW_MS = shuffle window Δ_ws; RS_WIN_SHUFF_BIN_MS = inner bin b.
RS_WIN_SHUFF_STAGE1_ENABLE: bool = True
RS_WIN_SHUFF_WINDOW_MS: float = 200.0
RS_WIN_SHUFF_BIN_MS: float = 10.0
RS_WIN_SHUFF_SEED: int = RS_OFFSET_NULL_SEED
# Auto sizing from recording duration:
#   window_ms = AUTO_WINDOW_FRACTION_RECORDING * recording_ms
#   bin_ms    = AUTO_BIN_FRACTION_OF_WINDOW   * window_ms
# then clipped to min/max bounds and snapped to an integer bin count.
RS_WIN_SHUFF_AUTO_FROM_RECORDING: bool = True
RS_WIN_SHUFF_AUTO_WINDOW_FRACTION_RECORDING: float = 0.10
RS_WIN_SHUFF_AUTO_BIN_FRACTION_OF_WINDOW: float = 0.10
RS_WIN_SHUFF_AUTO_WINDOW_MIN_MS: float = 50.0
RS_WIN_SHUFF_AUTO_WINDOW_MAX_MS: float = 10000.0
RS_WIN_SHUFF_AUTO_BIN_MIN_MS: float = 5.0
RS_WIN_SHUFF_AUTO_BIN_MAX_MS: float = 1000.0

# Stage 1 local segmentation (Problem B: slow firing-rate drift)
# - nonoverlap: disjoint chunks [0,L), [L,2L), ...
# - sliding: fixed chunk length L with stride L*(1-overlap_fraction)
# - custom: explicit non-overlapping windows from RS_STAGE1_CUSTOM_WINDOWS_S (seconds)
RS_STAGE1_LOCAL_SEGMENT_ENABLE: bool = True
# RS_STAGE1_SEGMENT_MODE: str = "nonoverlap"  # "nonoverlap" | "sliding" | "custom"
RS_STAGE1_SEGMENT_MODE: str = "sliding"  # "nonoverlap" | "sliding" | "custom"
# RS_STAGE1_SEGMENT_MODE: str = "custom"  # "nonoverlap" | "sliding" | "custom"

RS_STAGE1_SEGMENT_LEN_S: float = 5.0
RS_STAGE1_SEGMENT_OVERLAP_FRACTION: float = 0.4
RS_STAGE1_SEGMENT_MIN_SPIKES: int = 3
RS_STAGE1_POSTHOC_MERGE_ENABLE: bool = True
RS_STAGE1_POSTHOC_MERGE_GAP_MS: float = 20.0
RS_STAGE1_POSTHOC_DEDUP_IOU_MIN: float = 0.8
# Interaction notes:
# - "custom" mode uses only RS_STAGE1_CUSTOM_WINDOWS_S.
# - overlap + dedup settings are relevant in "sliding" mode.
# - posthoc merge stitches neighboring detections split by chunk boundaries.
# Custom windows for Stage-1 segmentation (seconds, absolute recording time).
# Used only when RS_STAGE1_SEGMENT_MODE == "custom".
RS_STAGE1_CUSTOM_WINDOWS_S: list[tuple[float, float]] = []

# Stage 2: region-level
RS_Limit_region = None
RS_Percentile_Limit_region = 75
RS_alpha_percentage_region = 0.05
RS_alpha_region = -np.log(RS_alpha_percentage_region)

# Stage 3: network-level
RS_Limit_network = None
RS_Percentile_Limit_network = 75
RS_alpha_percentage_network = 0.03
RS_alpha_network = -np.log(RS_alpha_percentage_network)

# Participation filters for region/network bursts
MIN_UNIQUE_CHANNELS_REGION = 1
MIN_UNIQUE_CHANNELS_NETWORK = 1
# Region-level participation requirement for network bursts.
# 1 preserves current behavior; >=2 enforces cross-region network events.
MIN_UNIQUE_REGIONS_NETWORK = 2

# Parameter sweeps
RS_ALPHA_SETS = [(0.08,    0.06,    0.03)] # (stage1, region, network)
# Optional sweep for Stage-1 WIN-SHUFF parameters (window_ms, bin_ms).
# These are iterated similarly to RS_ALPHA_SETS in main.py when
# RS_WIN_SHUFF_AUTO_FROM_RECORDING is False (fixed-size mode).
WIN_SHUFF_PARAM_SETS = []  # (window_ms, bin_ms)

# ============================================================
# ALTERNATIVE DETECTOR PARAMETERS
# ============================================================
# maxISI detector
maxISI_thresholds = [0]

# alpha-mean-ISI detector
ALPHA_MAXISI = 0.2
MIN_MAXISI_MS = 5
MAX_MAXISI_MS = 100

# ============================================================
# FIRING RATE
# ============================================================
firing_rate_bins_s = [0.4]
FIRING_RATE_OVERLAP_FRACTION = 0.2

# ============================================================
# PLOTTING - RASTER
# ============================================================
RASTER_OPACITY_BASE = 0.5
RASTER_OPACITY_BURST = 0.5
RASTER_DOT_SIZE = 3
RASTER_ROW_SPACING = 1

# For Kilosort combined rasters: y-axis depth tick spacing (µm). Used to add readable depth ticks.
KILOSORT_DEPTH_TICK_UM = 500.0

# Raster layout for Kilosort:
# - False (default): units from all shanks are mingled and globally sorted by depth; y-axis shows depth ticks.
# - True: units are grouped into shank blocks, each block sorted by depth (legacy look).
KILOSORT_RASTER_GROUP_BY_SHANK = True
COMMON_MARGINS = dict(t=70, r=240, l=110, b=60)

# ============================================================
# PLOTTING - FIRING-RATE PANELS
# ============================================================
COACTIVITY_DOT_SIZE = 2
SMOOTH_PANEL_SEC = 1

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
    "shank13": "rgb(60, 110, 230)",
    "shank24": "rgb(255, 140, 0)",
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
    "shank13": "Shank 13",
    "shank24": "Shank 24",
}