"""
04_preprocess_grayscale.py

Creates a grayscale-standardized processed dataset.
"""
# %% Imports
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageOps

# %% Config

SPLIT_CSV = Path("results/splits/dataset_split.csv")

OUTPUT_DIR = Path("dataset_processed_gray")
#OUTPUT_DIR = Path("dataset_processed_pseudocolor_jet")
RESULTS_DIR = Path("results/splits")

IMAGE_SIZE = (224, 224)
OUTPUT_FORMAT = "PNG"

CLEAR_OUTPUT_DIR = True

# %% Helpers

def safe_class_name(class_name: str) -> str:
    """
    Make class names folder-safe.
    """
    return (
        str(class_name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def prepare_output_dir() -> None:
    """
    Create a clean output directory.
    """
    if CLEAR_OUTPUT_DIR and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_output_path(row: pd.Series, index: int) -> Path:
    """
    Build output path for processed image.
    """
    final_split = row["final_split"]
    class_name = safe_class_name(row["class"])

    original_path = Path(row["filepath"])
    original_stem = original_path.stem

    filename = f"{index:05d}_{original_stem}.png"

    return OUTPUT_DIR / final_split / class_name / filename

def preprocess_image_grayscale(input_path: Path, output_path: Path) -> bool:
    """
    Load image, apply EXIF correction, convert to grayscale,
    resize, convert back to RGB, and save.

    The final image has 3 channels, but all channels contain
    the same grayscale information.
    """
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)

            # Remove artificial color tint.
            img = img.convert("L")

            # Resize to fixed CNN input size.
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Convert back to RGB for input_shape=(224,224,3).
            img = img.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format=OUTPUT_FORMAT)

        return True

    except Exception as e:
        print(f"[ERROR] Could not process {input_path}: {e}")
        return False

def preprocess_image_pseudocolor(
    input_path: Path,
    output_path: Path,
    colormap: str = "hot",
) -> bool:
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)

            # Convert to grayscale
            img = img.convert("L")
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Apply pseudocolor map: grayscale → RGBA float → RGB uint8
            gray_array = np.array(img, dtype=np.float32) / 255.0
            cmap = cm.get_cmap(colormap)
            colored = cmap(gray_array)              # shape (H, W, 4), float [0,1]
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

            result = Image.fromarray(colored_rgb, mode="RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path, format=OUTPUT_FORMAT)

        return True

    except Exception as e:
        print(f"[ERROR] Could not process {input_path}: {e}")
        return False

def create_distribution(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create train/val/test distribution table.
    """
    valid_df = processed_df[processed_df["success"] == True]

    distribution = pd.crosstab(
        valid_df["class"],
        valid_df["final_split"],
    )

    for col in ["train", "val", "test"]:
        if col not in distribution.columns:
            distribution[col] = 0

    distribution = distribution[["train", "val", "test"]]
    distribution["total"] = distribution.sum(axis=1)

    total_row = pd.DataFrame(
        {
            "train": [distribution["train"].sum()],
            "val": [distribution["val"].sum()],
            "test": [distribution["test"].sum()],
            "total": [distribution["total"].sum()],
        },
        index=["TOTAL"],
    )

    distribution = pd.concat([distribution, total_row])
    distribution = distribution.reset_index().rename(columns={"index": "class"})

    return distribution

# %% Main ------------------------------------------------------------

def main():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {SPLIT_CSV}. "
            "Run 03_make_split.py first."
        )

    prepare_output_dir()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(SPLIT_CSV)

    rows = []
    success_count = 0
    failed_count = 0

    print("=" * 30)
    print("Grayscale preprocessing")
    print("=" * 30)
    print(f"Input split CSV      : {SPLIT_CSV}")
    print(f"Output dataset       : {OUTPUT_DIR}")
    print(f"Target image size    : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Images to process    : {len(split_df)}")
    print()

    for index, row in split_df.iterrows():
        if index % 100 == 0:
            print(f"Processing image {index + 1}/{len(split_df)}")

        input_path = Path(row["filepath"])
        output_path = build_output_path(row, index)
        #success = preprocess_image_pseudocolor(input_path, output_path, "jet")
        success = preprocess_image_grayscale(input_path, output_path)

        if success:
            success_count += 1
        else:
            failed_count += 1

        rows.append(
            {
                "original_filepath": str(input_path),
                "processed_filepath": str(output_path),
                "class": row["class"],
                "original_split": row["split"],
                "final_split": row["final_split"],
                "success": success,
                "preprocessing": "grayscale_to_rgb_224x224",
            }
        )

    processed_df = pd.DataFrame(rows)
    processed_csv = RESULTS_DIR / "processed_dataset_gray.csv"
    processed_df.to_csv(processed_csv, index=False)

    distribution = create_distribution(processed_df)
    distribution_csv = RESULTS_DIR / "processed_distribution_gray.csv"
    distribution.to_csv(distribution_csv, index=False)

    print("\n" + "=" * 30)
    print("GRAYSCALE PREPROCESSING SUMMARY")
    print("=" * 30)
    print(f"Images processed successfully : {success_count}")
    print(f"Images failed                 : {failed_count}")
    print()
    print("Distribution:")
    print(distribution.to_string(index=False))
    print()
    print("Files written:")
    print(f"  {processed_csv}")
    print(f"  {distribution_csv}")
    print(f"  {OUTPUT_DIR}/")
    print("=" * 30)

if __name__ == "__main__":
    main()
