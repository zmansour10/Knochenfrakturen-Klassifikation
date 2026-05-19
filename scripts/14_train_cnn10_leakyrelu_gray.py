"""
14_train_cnn10_leakyrelu_gray.py

CNN-10: CNN-06 architecture with LeakyReLU instead of ReLU.
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
    dense_block,
    run_experiment,
)

# %% Model architecture

def conv_block_leaky(
    x,
    filters: int,
    conv_layers: int = 1,
    kernel_size: int = 3,
    alpha: float = 0.1,
    use_pooling: bool = True,
    block_name: str = "conv_block",
):
    """
    Convolution block using LeakyReLU instead of ReLU.
    """
    for i in range(conv_layers):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            activation=None,
            name=f"{block_name}_conv{i + 1}",
        )(x)

        x = layers.LeakyReLU(
            negative_slope=alpha,
            name=f"{block_name}_leakyrelu{i + 1}",
        )(x)

    if use_pooling:
        x = layers.MaxPooling2D(
            pool_size=(2, 2),
            name=f"{block_name}_pool",
        )(x)

    return x

def build_cnn10_leakyrelu_gray(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-10: CNN-06 with LeakyReLU activations.

    Difference to CNN-06:
        - ReLU replaced with LeakyReLU(alpha=0.1)
        - same grayscale-standardized images
        - same stronger dropout
        - same GlobalAveragePooling2D

    Architecture:
        Input 224x224x3

        Block 1:
            Conv2D(32) + LeakyReLU
            Conv2D(32) + LeakyReLU
            MaxPooling

        Block 2:
            Conv2D(64) + LeakyReLU
            Conv2D(64) + LeakyReLU
            MaxPooling

        Block 3:
            Conv2D(128) + LeakyReLU
            Conv2D(128) + LeakyReLU
            MaxPooling

        Block 4:
            Conv2D(256) + LeakyReLU
            MaxPooling

        GlobalAveragePooling2D

        Dense(512)
        Dropout(0.6)
        Dense(128)
        Dropout(0.3)
        Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # 2 conv layers
    x = conv_block_leaky(
        inputs,
        filters=32,
        conv_layers=2,
        kernel_size=3,
        alpha=0.1,
        use_pooling=True,
        block_name="block1",
    )

    # 2 conv layers
    x = conv_block_leaky(
        x,
        filters=64,
        conv_layers=2,
        kernel_size=3,
        alpha=0.1,
        use_pooling=True,
        block_name="block2",
    )

    # 2 conv layers
    x = conv_block_leaky(
        x,
        filters=128,
        conv_layers=2,
        kernel_size=3,
        alpha=0.1,
        use_pooling=True,
        block_name="block3",
    )

    # 1 conv layer
    x = conv_block_leaky(
        x,
        filters=256,
        conv_layers=1,
        kernel_size=3,
        alpha=0.1,
        use_pooling=True,
        block_name="block4",
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = dense_block(x, units=512, dropout_rate=0.6, block_name="fc1")
    x = dense_block(x, units=128, dropout_rate=0.3, block_name="fc2")

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_10_LeakyReLU_Gray",
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
        model_name="CNN_10_LeakyReLU_Gray",
        build_model_fn=build_cnn10_leakyrelu_gray,
        config=config,
        architecture_notes=(
            "Based on CNN_06_Deeper_GAP_StrongerDropout_Gray, but replaces "
            "all standard ReLU activations after convolutional layers with "
            "LeakyReLU(negative_slope=0.1). The model keeps grayscale-standardized "
            "images, GlobalAveragePooling2D, and stronger dropout: Dropout(0.6) "
            "after Dense(512) and Dropout(0.3) after Dense(128)."
        ),
    )

if __name__ == "__main__":
    main()
