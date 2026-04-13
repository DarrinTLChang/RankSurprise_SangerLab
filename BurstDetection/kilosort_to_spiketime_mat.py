#!/usr/bin/env python3
"""
Convert a Kilosort 4 output folder to spikeTime_p*.mat for use with any tool that expects
the SangerLab MATLAB layout.

Example:
  python kilosort_to_spiketime_mat.py --ks-dir "G:/rat data/m360/shank0/imec0/kilosort4" --out m360_shank0_imec0_spikeTime.mat
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.kilosort_loader import load_kilosort_dir, save_spiketime_mat


def main() -> None:
    p = argparse.ArgumentParser(description="Kilosort 4 folder → spikeTime .mat")
    p.add_argument("--ks-dir", type=Path, required=True, help="Path to kilosort4 output directory")
    p.add_argument("--out", type=Path, required=True, help="Output .mat path")
    p.add_argument("--include-mua", action="store_true", help="Set good_only=False (include non-good KSLabel clusters)")
    p.add_argument(
        "--max-duration-s",
        type=float,
        default=None,
        metavar="SEC",
        help="Keep only spikes in [0, SEC] seconds; default full recording",
    )
    args = p.parse_args()

    spike_struct, meta = load_kilosort_dir(
        args.ks_dir,
        good_only=not args.include_mua,
        max_duration_s=args.max_duration_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_spiketime_mat(args.out, spike_struct)
    print(f"Wrote {args.out}  (n_units={meta['n_channels']}, record_len_s={meta['record_len_s']:.3f})")


if __name__ == "__main__":
    main()
