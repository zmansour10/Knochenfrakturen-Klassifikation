# %% Imports
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image, ImageOps

# %% Config

SPLIT_CSV = Path("results/splits/dataset_split.csv")
OUTPUT_DIR = Path("dataset_processed")

IMAGE_SIZE = (224, 224)
OUTPUT_FORMAT = "PNG"

CLEAR_OUTPUT_DIR = True

# %% Helpers

def safe_class_name(class_name: str) -> str:
    """
    Keep class names folder-safe.
    """
    return (
        str(class_name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def preprocess_image(input_path: Path, output_path: Path) -> bool:
    """
    Load, convert, resize, and save one image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with Image.open(input_path) as img:
            # Correct image orientation if EXIF metadata exists.
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            # Resize all images to same shape.
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format=OUTPUT_FORMAT)

        return True

    except Exception as e:
        print(f"[ERROR] Could not process {input_path}: {e}")
        return False

def prepare_output_dir() -> None:
    """
    Prepare clean output directory.
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

# %% Main ------------------------------------------------------------

def main():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {SPLIT_CSV}. "
            "Run scripts/03_make_split.py first."
        )

    prepare_output_dir()

    split_df = pd.read_csv(SPLIT_CSV)

    rows = []
    success_count = 0
    failed_count = 0

    print(f"Loaded split file with {len(split_df)} images.")
    print(f"Writing processed dataset to: {OUTPUT_DIR}")

    for index, row in split_df.iterrows():
        if index % 100 == 0:
            print(f"  Processing image {index + 1}/{len(split_df)}")

        input_path = Path(row["filepath"])
        output_path = build_output_path(row, index)

        success = preprocess_image(input_path, output_path)

        if success:
            success_count += 1
        else:
            failed_count += 1

        rows.append(
            {
                "original_filepath": str(input_path),
                "processed_filepath": str(output_path),
                "class": row["class"],
                "final_split": row["final_split"],
                "success": success,
            }
        )

    processed_df = pd.DataFrame(rows)

    processed_csv = Path("results/splits/processed_dataset.csv")
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_csv, index=False)

    distribution = pd.crosstab(
        processed_df[processed_df["success"] == True]["class"],
        processed_df[processed_df["success"] == True]["final_split"]
    )

    for col in ["train", "val", "test"]:
        if col not in distribution.columns:
            distribution[col] = 0

    distribution = distribution[["train", "val", "test"]]
    distribution["total"] = distribution.sum(axis=1)

    distribution_csv = Path("results/splits/processed_distribution.csv")
    distribution.to_csv(distribution_csv)

    print("\n" + "=" * 30)
    print("PREPROCESSING SUMMARY")
    print("=" * 30)
    print(f"Images processed successfully : {success_count}")
    print(f"Images failed                 : {failed_count}")
    print()
    print("Processed data distribution:")
    print(distribution.to_string())
    print()
    print("Files written:")
    print(f"  {processed_csv}")
    print(f"  {distribution_csv}")
    print(f"  {OUTPUT_DIR}/")
    print("=" * 30)

if __name__ == "__main__":
    main()