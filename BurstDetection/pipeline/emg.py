# emg.py
from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from scipy import signal
import scipy.io as spio
from datetime import timedelta


def collapse_emg_traces(
    emg_traces: list[tuple[str, np.ndarray]],
    mode: str = "sum",
    label: str = "EMG summed",
) -> list[tuple[str, np.ndarray]]:
    """Convert multiple EMG traces into a single trace."""
    if emg_traces is None or len(emg_traces) == 0:
        return []

    Y = np.vstack([np.asarray(y, dtype=float) for _, y in emg_traces])
    Y = np.where(np.isfinite(Y), Y, 0.0)

    if mode == "sum":
        y = np.sum(Y, axis=0)
    elif mode == "mean":
        y = np.mean(Y, axis=0)
    elif mode == "rms":
        y = np.sqrt(np.mean(Y ** 2, axis=0))
    else:
        raise ValueError(f"Unknown collapse mode: {mode}")

    return [(label, y)]


def derive_emg_notes_from_spiketime(spiketime_path: str) -> tuple[str, str]:
    """
    spikeTime file: patient_data/s530/spikeTime_P2.mat
    returns:
      patient_data/s530/EMG_P2.mat
      patient_data/s530/Notes_P2.txt
    """
    p = Path(spiketime_path)
    patient_dir = p.parent

    m = re.search(r"_[pP](\d+)$", p.stem)
    if not m:
        raise ValueError(f"Cannot parse period from spike file: {spiketime_path}")

    per = m.group(1)
    emg = str(patient_dir / f"EMG_P{per}.mat")
    notes = str(patient_dir / f"Notes_P{per}.txt")
    return emg, notes


def build_emg_traces(indices, tag, emg_processed, selected_channel_names, mask):
    emg_traces = []
    pos_counter = 0

    for ichannel in indices:
        y = emg_processed[ichannel, :] + pos_counter - 10
        emg_traces.append(
            (f"EMG {selected_channel_names[ichannel]}", y[mask])
        )
        pos_counter += 1

    return emg_traces


def load_emg_raw(emg_mat_file: str):
    lib2 = spio.loadmat(emg_mat_file)
    emg_fs = lib2["sfEmg"][0][0]
    emg_data = lib2["emgDelsys"].copy()
    emg_segment_length = emg_data.shape[1] / emg_fs
    return emg_fs, emg_data, emg_segment_length


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, "same")
    return sma


def preprocess_emg_exact(emg_data: np.ndarray, fs_for_filter: float):
    gradient = False
    highpass_filter = False
    active_rest = False

    filt_order = 4
    cut_off_frq = 0.1
    sos = signal.butter(filt_order, cut_off_frq, "hp", fs=fs_for_filter, output="sos")

    moving_average_win_size = 100

    selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    selected_channel_names = [
        "R Bicep", "R Tricep", "R Flex", "R Extend",
        "L Bi", "L Tri", "L Flex", "L Extend",
        "R Quad", "R Ham", "R Tib", "R Gastroc",
        "L Quad", "L Ham", "L Tib", "L Gastroc"
    ]

    emg_processed = np.zeros((len(selected_channels), emg_data.shape[1]))
    for ichannel in range(len(selected_channels)):
        emg_processed[ichannel, :] = movingaverage(
            np.abs(emg_data[selected_channels[ichannel], :]),
            moving_average_win_size
        )

        if gradient:
            emg_processed[ichannel, :] = np.gradient(emg_processed[ichannel, :])
        elif highpass_filter:
            emg_processed[ichannel, :] = signal.sosfilt(sos, emg_processed[ichannel, :])

        emg_processed[ichannel, :] = emg_processed[ichannel, :] / np.max(emg_processed[ichannel, :])

        if active_rest:
            emg_processed[ichannel, :] = np.double(emg_processed[ichannel, :] > 0.1)

    return emg_processed, selected_channel_names


def parse_notes_events(notes_txt_file: str):
    """Parse event times from Notes file. Returns (event_times_sec, event_names)."""
    with open(notes_txt_file, "r") as f:
        lines = f.readlines()

    ref_time = 0
    event_times = []
    event_names = []

    for line in lines:
        if "Start" in line:
            new_result = re.findall("[0-9]+:[0-9]+:[0-9]+[pma]+", line)
            times = new_result[0].split(":")
            hours = int(re.findall("[0-9]+", times[0])[0])
            if "pm" in times[2] and hours != 12:
                hours = hours + 12
            minutes = int(re.findall("[0-9]+", times[1])[0])
            seconds = int(re.findall("[0-9]+", times[2])[0])
            ref_time = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        if "Note-" in line:
            new_result = re.findall("[0-9]+:[0-9]+:[0-9]+[pma]+", line)
            times = new_result[0].split(":")
            hours = int(re.findall("[0-9]+", times[0])[0])
            if "pm" in times[2] and hours != 12:
                hours = hours + 12
            minutes = int(re.findall("[0-9]+", times[1])[0])
            seconds = int(re.findall("[0-9]+", times[2])[0])
            event_time = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            event_name = line[line.find("[") + 1:line.find("]")]
            event_times.append((event_time - ref_time).total_seconds())
            event_names.append(event_name)

    return event_times, event_names


def make_emg_timebase(emg_segment_length: float, n_samples: int):
    return np.linspace(0, emg_segment_length, n_samples)
