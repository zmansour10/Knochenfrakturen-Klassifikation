"""
05_dataset.py

Creates Keras tf.data.Dataset objects from the processed image folders.
"""

# %% Imports 
from pathlib import Path

import pandas as pd
import tensorflow as tf

# %% Config

#PROCESSED_DATASET_DIR = Path("dataset_processed")
PROCESSED_DATASET_DIR = Path("dataset_processed_gray")
RESULTS_DIR = Path("results/dataset")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

TRAIN_DIR = PROCESSED_DATASET_DIR / "train"
VAL_DIR = PROCESSED_DATASET_DIR / "val"
TEST_DIR = PROCESSED_DATASET_DIR / "test"

# %% Dataset loading

def load_keras_dataset(
    directory: Path,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    seed: int = SEED,
) -> tf.data.Dataset:
    """
    Load a dataset from a folder structure using Keras.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
    )

    return dataset

def normalize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Normalize image pixels from [0, 255] to [0, 1].
    """
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    dataset = dataset.map(
        lambda images, labels: (normalization_layer(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset

def optimize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Improve training performance using caching and prefetching.
    """
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def get_datasets(
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Main function used by training scripts.

    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    raw_train_ds = load_keras_dataset(
        TRAIN_DIR,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
    )

    raw_val_ds = load_keras_dataset(
        VAL_DIR,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
    )

    raw_test_ds = load_keras_dataset(
        TEST_DIR,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
    )

    class_names = raw_train_ds.class_names

    train_ds = normalize_dataset(raw_train_ds)
    val_ds = normalize_dataset(raw_val_ds)
    test_ds = normalize_dataset(raw_test_ds)

    train_ds = optimize_dataset(train_ds)
    val_ds = optimize_dataset(val_ds)
    test_ds = optimize_dataset(test_ds)

    return train_ds, val_ds, test_ds, class_names

# Summary helpers

def count_images_per_split() -> pd.DataFrame:
    """
    Count image files per class and split.
    """
    rows = []

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("val", VAL_DIR),
        ("test", TEST_DIR),
    ]:
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            image_files = list(class_dir.glob("*.png"))

            rows.append(
                {
                    "split": split_name,
                    "class": class_dir.name,
                    "count": len(image_files),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    pivot = df.pivot_table(
        index="class",
        columns="split",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["class", "train", "val", "test"]]
    pivot["total"] = pivot[["train", "val", "test"]].sum(axis=1)

    total_row = pd.DataFrame(
        [
            {
                "class": "TOTAL",
                "train": pivot["train"].sum(),
                "val": pivot["val"].sum(),
                "test": pivot["test"].sum(),
                "total": pivot["total"].sum(),
            }
        ]
    )

    pivot = pd.concat([pivot, total_row], ignore_index=True)

    return pivot

def save_dataset_summary(class_names: list[str]) -> None:
    """
    Save class names and dataset distribution.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names_path = RESULTS_DIR / "class_names.txt"
    with open(class_names_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    summary_df = count_images_per_split()
    summary_path = RESULTS_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDataset distribution:")
    print(summary_df.to_string(index=False))

    print("\nFiles written:")
    print(f"  {class_names_path}")
    print(f"  {summary_path}")



# %% Main ------------------------------------------------------------

def main():
    print("Loading Keras datasets...")

    train_ds, val_ds, test_ds, class_names = get_datasets()

    print("\n" + "=" * 30)
    print("KERAS DATASET SUMMARY")
    print("=" * 30)
    print(f"Image size : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Classes    : {len(class_names)}")
    print(f"Class names:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")

    save_dataset_summary(class_names)

    print("=" * 30)

if __name__ == "__main__":
    main()