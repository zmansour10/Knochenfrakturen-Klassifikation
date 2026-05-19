"""
09_train_cnn05_deeper_gap_augmentation.py

CNN-05: CNN-03 Deeper GAP architecture + mild data augmentation.
"""

from pathlib import Path
import sys

# Allow imports from project root when running this file from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tensorflow import keras
from tensorflow.keras import layers

from m1lib.training_wrapper import (
    ExperimentConfig,
    conv_block,
    dense_block,
    mild_data_augmentation,
    run_experiment,
)

def build_cnn05_deeper_gap_augmentation(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-05: CNN-03 Deeper GAP with mild data augmentation.

    Difference to CNN-03:
        - adds mild augmentation layers after the input
        - no Batch Normalization

    Augmentation:
        - RandomRotation(0.05)
        - RandomZoom(0.10)
        - RandomContrast(0.10)

    Architecture:
        Input 224x224x3

        Mild data augmentation

        Block 1:
            Conv2D(32)
            Conv2D(32)
            MaxPooling

        Block 2:
            Conv2D(64)
            Conv2D(64)
            MaxPooling

        Block 3:
            Conv2D(128)
            Conv2D(128)
            MaxPooling

        Block 4:
            Conv2D(256)
            MaxPooling

        GlobalAveragePooling2D

        Dense(512)
        Dropout(0.5)
        Dense(128)
        Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = mild_data_augmentation(name="mild_augmentation")(inputs)

    # 2 conv layers
    x = conv_block(
        x,
        filters=32,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block1",
    )

    # 2 conv layers
    x = conv_block(
        x,
        filters=64,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block2",
    )

    # 2 conv layers
    x = conv_block(
        x,
        filters=128,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block3",
    )

    # 1 conv layer
    x = conv_block(
        x,
        filters=256,
        conv_layers=1,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block4",
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = dense_block(x, units=512, dropout_rate=0.5, block_name="fc1")
    x = dense_block(x, units=128, dropout_rate=None, block_name="fc2")

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_05_Deeper_GAP_Augmentation",
    )

def main():
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_dir=Path("dataset_processed"),
        results_dir=Path("results/experiments"),

        image_size=(224, 224),
        input_shape=(224, 224, 3),

        batch_size=32,
        epochs=50,
        learning_rate=1e-4,
        loss="sparse_categorical_crossentropy",
        seed=42,

        early_stopping_patience=8,
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-6,

        cache_dataset=True,
    )

    run_experiment(
        model_name="CNN_05_Deeper_GAP_Augmentation",
        build_model_fn=build_cnn05_deeper_gap_augmentation,
        config=config,
        architecture_notes=(
            "Based on CNN_03_Deeper_GAP, but adds data augmentation "
            "layers after the input: RandomRotation(0.05), RandomZoom(0.10), "
            "and RandomContrast(0.10)."
        ),
    )

if __name__ == "__main__":
    main()
