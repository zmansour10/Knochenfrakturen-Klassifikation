
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from m1lib.m2_transfer_wrapper import TransferConfig, run_transfer_experiment


def main():
    cfg = TransferConfig(
        backbone="inception_v3",

        image_size=(299, 299),
        input_shape=(299, 299, 3),

        # Phase 1 
        phase1_epochs=40,
        phase1_lr=3e-4,

        # Phase 2 
        phase2_epochs=40,
        phase2_lr=5e-5,
        unfreeze_layers=50,


        pooling="gap", # "flatten",# 
        dense_units=512,
        dropout_rate=0.4,

        dataset_dir="dataset_processed_gray",
        batch_size=32,
        seed=42,

        use_augmentation=True,

        early_stopping_patience=10,
        reduce_lr_patience=5,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-7,
    )

    summary = run_transfer_experiment(
        model_name="tl_inception_v3",
        cfg=cfg,
    )

if __name__ == "__main__":
    main()