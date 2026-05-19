"""
03_make_split.py

Creates a reproducible train/validation/test split for the bone fracture dataset.
"""
# %% Imports

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# %% Config

INVENTORY_CSV = Path("results/data_check/image_inventory.csv")
RESULTS_DIR = Path("results/splits")

OUTPUT_SPLIT_CSV = RESULTS_DIR / "dataset_split.csv"
OUTPUT_DISTRIBUTION_CSV = RESULTS_DIR / "split_distribution.csv"

VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

REMOVE_SIZE_OUTLIERS = True


# %% Helpers

def load_clean_inventory() -> pd.DataFrame:
    """
    Load the image inventory and keep only usable images.
    """
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INVENTORY_CSV}. "
            "Run scripts/01_data_check.py first."
        )

    df = pd.read_csv(INVENTORY_CSV)

    clean_df = df[
        (df["readable"] == True)
        & (df["is_duplicate"] == False)
    ].copy()

    if REMOVE_SIZE_OUTLIERS:
        clean_df = clean_df[clean_df["is_size_outlier"] == False].copy()

    clean_df = clean_df.reset_index(drop=True)

    print(f"Images in inventory       : {len(df)}")
    print(f"Images after cleaning     : {len(clean_df)}")
    print(f"Size outliers removed     : {REMOVE_SIZE_OUTLIERS}")

    return clean_df

def create_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create train/val/test split.
    """
    train_original = df[df["split"] == "Train"].copy()
    test_original = df[df["split"] == "Test"].copy()

    if train_original.empty:
        raise ValueError("No images found with split == 'Train'.")

    if test_original.empty:
        raise ValueError("No images found with split == 'Test'.")

    train_df, val_df = train_test_split(
        train_original,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_original["class"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_original.copy()

    train_df["final_split"] = "train"
    val_df["final_split"] = "val"
    test_df["final_split"] = "test"

    split_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    split_df = split_df[
        [
            "filepath",
            "class",
            "split",
            "final_split",
            "width",
            "height",
            "readable",
            "is_duplicate",
            "is_size_outlier",
            "file_size_kb",
            "md5",
        ]
    ].copy()

    split_df = split_df.sort_values(
        by=["final_split", "class", "filepath"]
    ).reset_index(drop=True)

    return split_df

def create_distribution_table(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create class distribution table for train/val/test.
    """
    dist = pd.crosstab(
        split_df["class"],
        split_df["final_split"]
    )

    for col in ["train", "val", "test"]:
        if col not in dist.columns:
            dist[col] = 0

    dist = dist[["train", "val", "test"]]
    dist["total"] = dist.sum(axis=1)

    total_row = pd.DataFrame(
        {
            "train": [dist["train"].sum()],
            "val": [dist["val"].sum()],
            "test": [dist["test"].sum()],
            "total": [dist["total"].sum()],
        },
        index=["TOTAL"]
    )

    dist = pd.concat([dist, total_row])

    dist = dist.reset_index().rename(columns={"index": "class"})

    return dist

# %% Main ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = load_clean_inventory()
    split_df = create_split(clean_df)
    distribution_df = create_distribution_table(split_df)

    split_df.to_csv(OUTPUT_SPLIT_CSV, index=False)
    distribution_df.to_csv(OUTPUT_DISTRIBUTION_CSV, index=False)

    print("\n" + "=" * 30)
    print(f"Files written to:")
    print(f"  {OUTPUT_SPLIT_CSV}")
    print(f"  {OUTPUT_DISTRIBUTION_CSV}")
    print("=" * 30)

if __name__ == "__main__":
    main()