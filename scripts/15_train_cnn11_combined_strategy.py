"""
15_train_cnn11_combined_strategy.py

CNN-11 
"""
# %% Imports

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tensorflow import keras
from tensorflow.keras import layers

from m1lib.training_wrapper import (
    ExperimentConfig,
    conv_block,
    dense_block,
    strong_data_augmentation,
    run_experiment,
)

# %% Model architecture

def build_cnn11_combined_strategy(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-11: Combined strategy.

    Architecture:
        Input 224x224x3

        Moderate augmentation:
            RandomTranslation(0.05)
            RandomRotation(0.08)
            RandomZoom(0.10)
            RandomContrast(0.15)

        Block 1:
            Conv2D(32, HeNormal) + BatchNorm + ReLU
            Conv2D(32, HeNormal) + BatchNorm + ReLU
            MaxPooling

        Block 2:
            Conv2D(64, HeNormal) + BatchNorm + ReLU
            Conv2D(64, HeNormal) + BatchNorm + ReLU
            MaxPooling

        Block 3:
            Conv2D(128, HeNormal) + BatchNorm + ReLU
            Conv2D(128, HeNormal) + BatchNorm + ReLU
            MaxPooling

        Block 4:
            Conv2D(256, HeNormal) + BatchNorm + ReLU
            MaxPooling

        GlobalAveragePooling2D

        Dense(512, HeNormal)
        Dropout(0.5)
        Dense(128, HeNormal)
        Dropout(0.3)
        Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = strong_data_augmentation(name="moderate_augmentation")(inputs)

    x = conv_block(
        x,
        filters=32,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=True,
        use_pooling=True,
        block_name="block1",
        kernel_initializer="he_normal",
    )

    x = conv_block(
        x,
        filters=64,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=True,
        use_pooling=True,
        block_name="block2",
        kernel_initializer="he_normal",
    )

    x = conv_block(
        x,
        filters=128,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=True,
        use_pooling=True,
        block_name="block3",
        kernel_initializer="he_normal",
    )

    x = conv_block(
        x,
        filters=256,
        conv_layers=1,
        kernel_size=3,
        use_batch_norm=True,
        use_pooling=True,
        block_name="block4",
        kernel_initializer="he_normal",
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = dense_block(
        x,
        units=512,
        dropout_rate=0.5,
        block_name="fc1",
        kernel_initializer="he_normal",
    )

    x = dense_block(
        x,
        units=128,
        dropout_rate=0.3,
        block_name="fc2",
        kernel_initializer="he_normal",
    )

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_11_Combined_AugBN_Gray",
    )

# %% Main

def main():
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_dir=Path("dataset_processed_gray"),

        results_dir=Path("results/experiments"),

        image_size=(224, 224),
        input_shape=(224, 224, 3),

        batch_size=16,

        epochs=60,
        learning_rate=3e-4,
        loss="sparse_categorical_crossentropy",
        seed=42,

        early_stopping_patience=10,
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-6,

        cache_dataset=True,

        label_smoothing=0.1,

        use_cosine_decay=True,
        cosine_alpha=0.1,
    )

    run_experiment(
        model_name="CNN_11_Combined_AugBN_Gray",
        build_model_fn=build_cnn11_combined_strategy,
        config=config,
        architecture_notes=(
            "Additional experiment based on the best grayscale models. "
            "Architectural changes: moderate augmentation layers and BatchNorm "
            "after each convolution layer."
        ),
    )

if __name__ == "__main__":
    main()
