from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


BURST_DETECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BURST_DETECTION_ROOT))

import pipeline.detection as detection
from pipeline.intervals import (
    merge_windows,
    prepare_window_index,
    slice_sorted_inclusive,
    time_in_window_index,
)
from pipeline.numeric import moving_average


class SlidingWindowRegressionTests(unittest.TestCase):
    def test_inclusive_slice_matches_boolean_reference(self):
        spikes = np.asarray([0.0, 3000.0, 5000.0, 5000.1, 8000.0])
        actual = slice_sorted_inclusive(spikes, 3000.0, 5000.0)
        expected = spikes[(spikes >= 3000.0) & (spikes <= 5000.0)]
        np.testing.assert_array_equal(actual, expected)

    def test_changing_rate_fixed_seed_golden_output(self):
        spikes = np.asarray([
            0, 900, 1800, 2700, 3600, 4500, 4800, 4820, 4840,
            6000, 6900, 7800, 8100, 8120, 8140, 8160, 9900, 10800, 11700,
        ], dtype=float)
        settings = {
            "RS_STAGE1_LOCAL_SEGMENT_ENABLE": True,
            "RS_STAGE1_SEGMENT_MODE": "sliding",
            "RS_STAGE1_SEGMENT_LEN_S": 5.0,
            "RS_STAGE1_SEGMENT_OVERLAP_FRACTION": 0.4,
            "RS_STAGE1_SEGMENT_MIN_SPIKES": 3,
            "RS_STAGE1_POSTHOC_MERGE_ENABLE": True,
            "RS_STAGE1_POSTHOC_MERGE_GAP_MS": 20.0,
            "RS_STAGE1_POSTHOC_DEDUP_IOU_MIN": 0.8,
            "RS_WIN_SHUFF_STAGE1_ENABLE": True,
            "RS_WIN_SHUFF_WINDOW_MS": 200.0,
            "RS_WIN_SHUFF_BIN_MS": 10.0,
            "RS_WIN_SHUFF_SEED": 123,
            "RS_Limit_stage1": None,
            "RS_alpha_stage1": -np.log(0.08),
            "RS_Percentile_Limit_stage1": 75,
            "MIN_SPIKES_IN_BURST": 3,
            "APPLY_MIN_BURST_DURATION_ALL_STAGES": False,
            "MIN_BURST_DURATION": 20,
        }
        with patch.multiple(detection, **settings):
            windows = detection._stage1_chunk_windows_ms(spikes)
            rows = detection._detect_stage1_unit_bursts(
                spikes, "microGPi1_L_1_CommonFiltered", 0,
            )

        self.assertEqual(
            windows,
            [(0.0, 5000.0), (3000.0, 8000.0), (6000.0, 11000.0), (9000.0, 11700.0)],
        )
        payload = json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=True,
        ).encode()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "e356945c1ffed7b76f70d080d2eef7057884e8ee0a6e4963282fd467701f0b11",
        )


class SharedHelperRegressionTests(unittest.TestCase):
    def test_window_index_matches_linear_membership(self):
        windows = [(8.0, 10.0), (1.0, 4.0), (3.0, 6.0)]
        index = prepare_window_index(windows)
        for time_ms in [0.9, 1.0, 4.0, 5.0, 6.0, 6.1, 8.0, 10.0, 10.1]:
            expected = any(start <= time_ms <= end for start, end in windows)
            self.assertEqual(time_in_window_index(time_ms, index), expected)

    def test_merge_windows_preserves_touching_behavior(self):
        self.assertEqual(
            merge_windows([(3.0, 4.0), (1.0, 2.0), (2.0, 3.5), (7.0, 8.0)]),
            [(1.0, 4.0), (7.0, 8.0)],
        )

    def test_shared_moving_average_matches_previous_formula(self):
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        expected = np.convolve(values, np.ones(3) / 3.0, mode="same")
        np.testing.assert_array_equal(moving_average(values, 3), expected)


if __name__ == "__main__":
    unittest.main()
