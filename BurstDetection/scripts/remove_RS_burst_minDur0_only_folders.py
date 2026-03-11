#!/usr/bin/env python3
"""
Remove run_tag folders under outputs_RS_burst that have __minDur=0ms but NOT __minCh=1.
These are the copies made before we added minCh=1 to the copy script. Keeps folders that
have __minDur=0ms__minCh=1__region__network.

Run from repo root or BurstDetection:
  python scripts/remove_RS_burst_minDur0_only_folders.py

Edit DEST_ROOT below if needed. Run with --dry-run to only print what would be removed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEST_ROOT = Path("/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst")


def rmtree_robust(path: Path) -> None:
    """Remove directory tree; ignore FileNotFoundError (e.g. stale ._ files on external drives)."""
    def onerror(func, path, exc_info):
        if exc_info[0] is not FileNotFoundError:
            raise exc_info[1]
    shutil.rmtree(path, onerror=onerror)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove __minDur=0ms (no minCh=1) run_tag folders from outputs_RS_burst.")
    parser.add_argument("--dry-run", action="store_true", help="Only print paths that would be removed.")
    args = parser.parse_args()

    if not DEST_ROOT.is_dir():
        print(f"Not found: {DEST_ROOT}")
        return

    removed = 0
    for rank_surprise_dir in DEST_ROOT.rglob("rankSurprise"):
        if not rank_surprise_dir.is_dir():
            continue
        for run_tag_dir in rank_surprise_dir.iterdir():
            if not run_tag_dir.is_dir():
                continue
            name = run_tag_dir.name
            # Remove only: has __minDur=0ms and does NOT have minCh anywhere (so the old copy; keep minCh=1, minCh=5, etc.)
            if "__minDur=0ms" in name and "minCh" not in name:
                if args.dry_run:
                    print(f"Would remove: {run_tag_dir.relative_to(DEST_ROOT)}")
                else:
                    rmtree_robust(run_tag_dir)
                    print(f"Removed: {run_tag_dir.relative_to(DEST_ROOT)}")
                removed += 1

    if args.dry_run:
        print(f"Dry run: {removed} folder(s) would be removed.")
    else:
        print(f"Done. Removed {removed} folder(s).")


if __name__ == "__main__":
    main()
