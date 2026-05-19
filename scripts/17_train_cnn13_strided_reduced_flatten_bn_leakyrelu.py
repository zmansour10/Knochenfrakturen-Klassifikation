"""
17_train_cnn13_strided_reduced_flatten_bn_leakyrelu.py

CNN-13: Strided Conv2D + reduced Flatten head + BatchNorm + LeakyReLU.
"""

# %% imports

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

# %% helper functions

def strided_conv_block(
    x,
    filters: int,
    stride: int = 2,
    dropout_rate: float = 0.25,
    block_name: str = "strided_block",
):
    """
    Conv2D downsampling block with BatchNorm, LeakyReLU, and Dropout.
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
    dropout_rate: float = 0.4,
    block_name: str = "dense_block",
):
    """
    Dense classifier block with BatchNorm, LeakyReLU, and Dropout.
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

def build_cnn13_strided_reduced_flatten_bn_leakyrelu(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-13: reduced Flatten-head version of CNN-12.

    Difference to CNN-12:
        - adds MaxPooling2D before Flatten
        - reduces Dense(256)->Dense(128)
        - reduces Dense(128)->Dense(64)
        - increases classifier dropout to 0.4
        - lowers learning rate in config

    Requirement:
        - 5 Conv2D layers
        - 2 hidden fully connected layers
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = strided_conv_block(
        inputs,
        filters=32,
        stride=2,
        dropout_rate=0.20,
        block_name="block1",
    )

    x = strided_conv_block(
        x,
        filters=64,
        stride=2,
        dropout_rate=0.20,
        block_name="block2",
    )

    x = strided_conv_block(
        x,
        filters=128,
        stride=2,
        dropout_rate=0.25,
        block_name="block3",
    )

    x = strided_conv_block(
        x,
        filters=256,
        stride=2,
        dropout_rate=0.25,
        block_name="block4",
    )


    x = strided_conv_block(
        x,
        filters=256,
        stride=1,
        dropout_rate=0.25,
        block_name="block5",
    )

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pre_flatten_pool",
    )(x)

    x = layers.Flatten(name="flatten")(x)

    x = dense_bn_leaky_block(
        x,
        units=128,
        dropout_rate=0.40,
        block_name="fc1",
    )

    x = dense_bn_leaky_block(
        x,
        units=64,
        dropout_rate=0.40,
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
        name="CNN_13_StridedConv_ReducedFlatten_BN_LeakyReLU",
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
        epochs=40,

        # Lower than CNN-12's 5e-4 because CNN-12 overfit quickly and validation loss became unstable.
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
        model_name="CNN_13_StridedConv_ReducedFlatten_BN_LeakyReLU",
        build_model_fn=build_cnn13_strided_reduced_flatten_bn_leakyrelu,
        config=config,
        architecture_notes=(
            "Improved version of CNN_12. Keeps strided Conv2D, BatchNorm, "
            "LeakyReLU, Dropout, and Flatten, but adds MaxPooling2D before "
            "Flatten to reduce the flattened feature vector from 50,176 to "
            "12,544 features. The classifier head is reduced to Dense(128) "
            "and Dense(64) with Dropout(0.4). This tests whether CNN_12's "
            "performance can be preserved while reducing overfitting."
        ),
    )

if __name__ == "__main__":
    main()
