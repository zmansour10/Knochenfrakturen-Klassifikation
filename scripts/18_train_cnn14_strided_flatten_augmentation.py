"""
18_train_cnn14_strided_flatten_augmentation.py

CNN-14: CNN-12 + data augmentation.
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
    run_experiment,
)

# %% Augmentation

def moderate_xray_augmentation(name: str = "moderate_xray_augmentation") -> keras.Sequential:
    """
    Moderate augmentation for X-ray images.
    """
    return keras.Sequential(
        [
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                name="aug_translation",
            ),
            layers.RandomRotation(
                factor=0.05,
                name="aug_rotation",
            ),
            layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                name="aug_zoom",
            ),
            layers.RandomContrast(
                factor=0.10,
                name="aug_contrast",
            ),
        ],
        name=name,
    )


# %% Layers

def strided_conv_block(
    x,
    filters: int,
    stride: int = 2,
    dropout_rate: float = 0.2,
    block_name: str = "strided_block",
):
    """
    Strided convolution block.
    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        activation=None,
        kernel_initializer="he_normal",
        name=f"{block_name}_conv",
    )(x)

    x = layers.BatchNormalization(
        momentum=0.9,
        name=f"{block_name}_bn",
    )(x)

    x = layers.LeakyReLU(
        negative_slope=0.2,
        name=f"{block_name}_leakyrelu",
    )(x)

    x = layers.Dropout(
        rate=dropout_rate,
        name=f"{block_name}_dropout",
    )(x)

    return x

def dense_bn_leaky_block(
    x,
    units: int,
    dropout_rate: float = 0.3,
    block_name: str = "dense_block",
):
    """
    Dense + BatchNorm + LeakyReLU + Dropout block.
    """
    x = layers.Dense(
        units=units,
        kernel_initializer="he_normal",
        name=f"{block_name}_dense",
    )(x)

    x = layers.BatchNormalization(
        momentum=0.9,
        name=f"{block_name}_bn",
    )(x)

    x = layers.LeakyReLU(
        negative_slope=0.2,
        name=f"{block_name}_leakyrelu",
    )(x)

    x = layers.Dropout(
        rate=dropout_rate,
        name=f"{block_name}_dropout",
    )(x)

    return x

# %% Model Architecture

def build_cnn14_strided_flatten_augmentation(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-14: CNN-12 with moderate augmentation.

    Difference to CNN-12:
        - adds augmentation layers at the input
        - keeps the same strided-convolution + Flatten architecture
        - keeps the same dense classifier size
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = moderate_xray_augmentation(name="moderate_xray_augmentation")(inputs)

    x = strided_conv_block(
        x,
        filters=32,
        stride=2,
        dropout_rate=0.2,
        block_name="block1",
    )

    x = strided_conv_block(
        x,
        filters=64,
        stride=2,
        dropout_rate=0.2,
        block_name="block2",
    )

    x = strided_conv_block(
        x,
        filters=128,
        stride=2,
        dropout_rate=0.2,
        block_name="block3",
    )

    x = strided_conv_block(
        x,
        filters=256,
        stride=2,
        dropout_rate=0.2,
        block_name="block4",
    )

    x = strided_conv_block(
        x,
        filters=256,
        stride=1,
        dropout_rate=0.2,
        block_name="block5",
    )

    x = layers.Flatten(name="flatten")(x)

    x = dense_bn_leaky_block(
        x,
        units=256,
        dropout_rate=0.3,
        block_name="fc1",
    )

    x = dense_bn_leaky_block(
        x,
        units=128,
        dropout_rate=0.3,
        block_name="fc2",
    )

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_14_StridedConv_Flatten_Augmentation",
    )


# %% Main

def main():
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,
        dataset_dir=Path("dataset_processed_gray"),

        results_dir=Path("results/experiments"),

        image_size=(224, 224),
        input_shape=(224, 224, 3),

        batch_size=32,
        epochs=30,

        learning_rate=5e-4,

        loss="sparse_categorical_crossentropy",
        seed=42,

        early_stopping_patience=8,
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-6,

        cache_dataset=True,
    )

    run_experiment(
        model_name="CNN_14_StridedConv_Flatten_Augmentation",
        build_model_fn=build_cnn14_strided_flatten_augmentation,
        config=config,
        architecture_notes=(
            "Keeps the same strided Conv2D + BatchNorm + LeakyReLU + Dropout + Flatten "
            "architecture, but adds moderate augmentation at the input: "
            "RandomTranslation(0.05), RandomRotation(0.05), RandomZoom(0.10), "
            "and RandomContrast(0.10)."
        ),
    )

if __name__ == "__main__":
    main()
