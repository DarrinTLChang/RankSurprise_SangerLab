#!/usr/bin/env python3
"""
Copy only the separateGPi__... run_tag folders from outputs_RS_burst_2 into outputs_RS_burst,
renaming each folder to include __minDur=0ms and __minCh=1 so it doesn't overwrite existing runs.
Skips the non-separateGPi folder (e.g. SNR=1.2__FR=... without separateGPi).

Run from repo root or BurstDetection:
  python scripts/copy_RS_burst_2_separateGPi_into_RS_burst.py

Edit SOURCE_ROOT and DEST_ROOT below if needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Same layout as compare_proxy_vs_network.py
SOURCE_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst_2")
DEST_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst")
# Only copy run_tag folders whose name starts with this
RUN_TAG_PREFIX = "separateGPi__"
# Insert this before __region__network in the folder name so the copy has a distinct name
SUFFIX_BEFORE_REGION = "__minDur=0ms__minCh=1"


def run_tag_to_dest_name(run_tag: str) -> str:
    """Append __minDur=0ms and __minCh=1 before __region__network so the folder name is unique and descriptive."""
    if "__region__network" not in run_tag:
        return run_tag + SUFFIX_BEFORE_REGION
    if SUFFIX_BEFORE_REGION in run_tag:
        return run_tag
    return run_tag.replace("__region__network", SUFFIX_BEFORE_REGION + "__region__network")


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        print(f"Source not found: {SOURCE_ROOT}")
        return
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Find all .../rankSurprise/<run_tag> under source where run_tag starts with separateGPi__
    n_copied = 0
    for rank_surprise_dir in SOURCE_ROOT.rglob("rankSurprise"):
        if not rank_surprise_dir.is_dir():
            continue
        # rank_surprise_dir = .../patient/PeriodN/rankSurprise
        for run_tag_dir in rank_surprise_dir.iterdir():
            if not run_tag_dir.is_dir():
                continue
            run_tag = run_tag_dir.name
            if not run_tag.startswith(RUN_TAG_PREFIX):
                continue
            dest_name = run_tag_to_dest_name(run_tag)
            rel = run_tag_dir.relative_to(SOURCE_ROOT)
            dest_run_dir = DEST_ROOT / rel.parent / dest_name
            if dest_run_dir.exists():
                print(f"  Skip (exists): {dest_run_dir}")
                continue
            dest_run_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_tag_dir, dest_run_dir)
            n_copied += 1
            print(f"  Copied: {rel} -> {dest_run_dir.relative_to(DEST_ROOT)}")

    print(f"Done. Copied {n_copied} run_tag folder(s) into {DEST_ROOT}")


if __name__ == "__main__":
    main()
