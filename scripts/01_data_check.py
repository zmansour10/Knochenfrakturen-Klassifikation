"""
01_data_check.py

Scans the bone fracture dataset and produces 4 CSV files:
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Config

DATASET_DIR = Path("dataset")          
RESULTS_DIR = Path("results/data_check")
VALID_SPLITS = {"Train", "Test"}       
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# An image is flagged as a size outlier if its width or height is more than
# this many standard deviations from the mean across all images.
SIZE_OUTLIER_STD = 3.0

# Helpers

def file_md5(path: Path, chunk: int = 8192) -> str | None:
    """Return MD5 hex digest of a file, or None if the file is unreadable."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return None

def try_open(path: Path) -> tuple[bool, int | None, int | None]:
    """
    Try to open an image with Pillow.
    Returns (readable, width, height).
    """
    try:
        with Image.open(path) as img:
            img.verify()           

        with Image.open(path) as img:
            w, h = img.size
        return True, w, h
    except (UnidentifiedImageError, Exception):
        return False, None, None

# Walk the dataset and build the raw inventory

def build_inventory(dataset_dir: Path) -> pd.DataFrame:
    print("Scanning dataset …")
    rows = []

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name

        for split_dir in sorted(class_dir.iterdir()):
            if not split_dir.is_dir():
                continue

            split_name = split_dir.name
            if split_name not in VALID_SPLITS:
                print(f"  [WARN] Unexpected split folder: {split_dir}")

            for img_path in sorted(split_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue

                readable, w, h = try_open(img_path)
                file_size_kb = round(img_path.stat().st_size / 1024, 2) if img_path.exists() else None
                md5 = file_md5(img_path)

                rows.append(
                    {
                        "filepath": str(img_path),
                        "class": class_name,
                        "split": split_name,
                        "width": w,
                        "height": h,
                        "readable": readable,
                        "file_size_kb": file_size_kb,
                        "md5": md5,
                        "is_duplicate": False,   
                        "is_size_outlier": False,
                    }
                )

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} image files across {len(class_dirs)} classes.")
    return df

# Mark duplicates (same MD5 hash)

def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the first occurrence; mark the rest as duplicates
    md5_seen: dict[str, str] = {}
    dup_flags = []

    for _, row in df.iterrows():
        md5 = row["md5"]
        if md5 is None or not row["readable"]:
            dup_flags.append(False)
            continue
        if md5 in md5_seen:
            dup_flags.append(True)
        else:
            md5_seen[md5] = row["filepath"]
            dup_flags.append(False)

    df["is_duplicate"] = dup_flags
    n_dup = sum(dup_flags)
    print(f"  Duplicates found: {n_dup}")
    return df

# Mark size outliers

def mark_size_outliers(df: pd.DataFrame, std_threshold: float = SIZE_OUTLIER_STD) -> pd.DataFrame:
    readable = df[df["readable"] & ~df["is_duplicate"]].copy()

    if readable.empty:
        df["is_size_outlier"] = False
        return df

    w_mean, w_std = readable["width"].mean(), readable["width"].std()
    h_mean, h_std = readable["height"].mean(), readable["height"].std()

    def is_outlier(row):
        if not row["readable"]:
            return False
        if w_std > 0 and abs(row["width"] - w_mean) > std_threshold * w_std:
            return True
        if h_std > 0 and abs(row["height"] - h_mean) > std_threshold * h_std:
            return True
        return False

    df["is_size_outlier"] = df.apply(is_outlier, axis=1)
    n_outlier = df["is_size_outlier"].sum()
    print(f"  Size outliers (>{std_threshold}σ from mean {w_mean:.0f}×{h_mean:.0f}): {n_outlier}")
    return df

# Build the 4 output CSVs

def make_image_inventory(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "filepath", "class", "split",
        "width", "height", "readable",
        "is_duplicate", "is_size_outlier",
        "file_size_kb", "md5",
    ]
    return df[cols].copy()

def make_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    # Count only readable, non-duplicate images
    clean = df[df["readable"] & ~df["is_duplicate"]]

    rows = []
    for class_name in sorted(clean["class"].unique()):
        cls_df = clean[clean["class"] == class_name]
        train_count = int((cls_df["split"] == "Train").sum())
        test_count  = int((cls_df["split"] == "Test").sum())
        total       = train_count + test_count
        rows.append(
            {
                "class": class_name,
                "train_count": train_count,
                "test_count": test_count,
                "total": total,
                "train_pct": round(100 * train_count / total, 1) if total else 0,
                "test_pct": round(100 * test_count / total, 1) if total else 0,
            }
        )

    dist_df = pd.DataFrame(rows)

    # Add a TOTAL row at the bottom
    total_row = pd.DataFrame(
        [
            {
                "class": "TOTAL",
                "train_count": dist_df["train_count"].sum(),
                "test_count": dist_df["test_count"].sum(),
                "total": dist_df["total"].sum(),
                "train_pct": round(100 * dist_df["train_count"].sum() / dist_df["total"].sum(), 1),
                "test_pct": round(100 * dist_df["test_count"].sum() / dist_df["total"].sum(), 1),
            }
        ]
    )
    dist_df = pd.concat([dist_df, total_row], ignore_index=True)
    return dist_df

def make_flagged(df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for _, row in df.iterrows():
        reasons = []
        if not row["readable"]:
            reasons.append("corrupted")
        if row["is_duplicate"]:
            reasons.append("duplicate")
        if row["is_size_outlier"]:
            reasons.append("size_outlier")
        if reasons:
            flags.append(
                {
                    "filepath": row["filepath"],
                    "class": row["class"],
                    "split": row["split"],
                    "reason": ", ".join(reasons),
                    "width": row["width"],
                    "height": row["height"],
                    "file_size_kb": row["file_size_kb"],
                }
            )
    return pd.DataFrame(flags)

def make_summary(df: pd.DataFrame, dist_df: pd.DataFrame, flagged_df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["readable"] & ~df["is_duplicate"]]
    total_clean = len(clean)

    class_totals = dist_df[dist_df["class"] != "TOTAL"]["total"]
    is_balanced = bool(class_totals.max() <= 2 * class_totals.min()) if len(class_totals) > 0 else True

    summary = {
        "total_files_found": len(df),
        "num_classes": df["class"].nunique(),
        "num_splits": df["split"].nunique(),
        "num_readable": int(df["readable"].sum()),
        "num_corrupted": int((~df["readable"]).sum()),
        "num_duplicates": int(df["is_duplicate"].sum()),
        "num_size_outliers": int(df["is_size_outlier"].sum()),
        "total_flagged": len(flagged_df),
        "total_clean_images": total_clean,
        "mean_width": round(df[df["readable"]]["width"].mean(), 1),
        "mean_height": round(df[df["readable"]]["height"].mean(), 1),
        "min_width": int(df[df["readable"]]["width"].min()) if df["readable"].any() else None,
        "max_width": int(df[df["readable"]]["width"].max()) if df["readable"].any() else None,
        "min_height": int(df[df["readable"]]["height"].min()) if df["readable"].any() else None,
        "max_height": int(df[df["readable"]]["height"].max()) if df["readable"].any() else None,
        "is_balanced": is_balanced,
        "smallest_class_total": int(class_totals.min()) if len(class_totals) else None,
        "largest_class_total": int(class_totals.max()) if len(class_totals) else None,
    }

    return pd.DataFrame([summary])


# %% Main ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw nventory
    df = build_inventory(DATASET_DIR)

    df = mark_duplicates(df)
    df = mark_size_outliers(df)

    # Output tables
    inventory_df = make_image_inventory(df)
    dist_df      = make_class_distribution(df)
    flagged_df   = make_flagged(df)
    summary_df   = make_summary(df, dist_df, flagged_df)

    # Save CSVs
    inventory_path = RESULTS_DIR / "image_inventory.csv"
    dist_path      = RESULTS_DIR / "class_distribution.csv"
    flagged_path   = RESULTS_DIR / "flagged_images.csv"
    summary_path   = RESULTS_DIR / "summary.csv"

    inventory_df.to_csv(inventory_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    flagged_df.to_csv(flagged_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Summary 
    print("\n" + "=" * 30)
    print("DATA CHECK SUMMARY")
    print("=" * 30)
    s = summary_df.iloc[0]
    print(f"  Total files found   : {s['total_files_found']}")
    print(f"  Classes             : {s['num_classes']}")
    print(f"  Readable            : {s['num_readable']}")
    print(f"  Clean images        : {s['total_clean_images']}")
    print(f"  Image sizes (WxH)   : {s['min_width']}–{s['max_width']} × {s['min_height']}–{s['max_height']} px")

    print()
    print("Class distribution (clean images):")
    print(dist_df.to_string(index=False))
    print()
    print(f"CSVs written to: {RESULTS_DIR}/")
    print(f"  {inventory_path.name}")
    print(f"  {dist_path.name}")
    print(f"  {flagged_path.name}")
    print(f"  {summary_path.name}")
    print("=" * 30)

if __name__ == "__main__":
    main()