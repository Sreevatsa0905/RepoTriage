#!/usr/bin/env python3
"""
Download the official NLBSE 2024 Issue Report Classification dataset.

Source: https://github.com/nlbse2024/issue-report-classification
Files:  data/issues_train.csv, data/issues_test.csv

Downloads to: data/external/nlbse2024/
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.config import (
    DATASET_URL_TRAIN,
    DATASET_URL_TEST,
    EXTERNAL_DIR,
    TRAIN_CSV,
    TEST_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file from a URL to a local path, with fallback to temp clone."""
    if dest.exists() and not force:
        logger.info("Already exists (skipping): %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if we already have it in temp clone
    temp_clone = Path(r"C:\Users\sreek\AppData\Local\Temp\nlbse2024\data") / dest.name
    if temp_clone.exists():
        logger.info("Found in local temp clone: %s -> copying to %s", temp_clone, dest)
        shutil.copy2(temp_clone, dest)
        size_kb = dest.stat().st_size / 1024
        logger.info("  Copied %s (%.1f KB)", dest.name, size_kb)
        return dest

    logger.info("Downloading %s → %s", url, dest)
    req = Request(url, headers={"User-Agent": "RepoTriage/0.1"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (HTTPError, URLError) as e:
        logger.error("Download failed: %s", e)
        raise

    dest.write_bytes(data)
    size_kb = len(data) / 1024
    md5 = hashlib.md5(data).hexdigest()
    logger.info("  Saved %s (%.1f KB, MD5: %s)", dest.name, size_kb, md5)
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download NLBSE 2024 dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    print("=" * 60)
    print("  NLBSE 2024 Issue Report Classification — Dataset Download")
    print("=" * 60)

    download_file(DATASET_URL_TRAIN, TRAIN_CSV, force=args.force)
    download_file(DATASET_URL_TEST, TEST_CSV, force=args.force)

    # Quick validation
    for path, label in [(TRAIN_CSV, "Train"), (TEST_CSV, "Test")]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  {label}: {path.name} ({size_kb:.1f} KB)")
        else:
            print(f"  {label}: MISSING")

    print("\nDone. Files saved to:", EXTERNAL_DIR)


if __name__ == "__main__":
    main()
