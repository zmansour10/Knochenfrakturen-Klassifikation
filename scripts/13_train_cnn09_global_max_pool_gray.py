"""
13_train_cnn09_global_max_pool_gray.py

CNN-09: Deeper grayscale CNN using GlobalMaxPooling2D 
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
    run_experiment,
)

# %% Model architecture

def build_cnn09_global_max_pool_gray(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-09: Deeper grayscale CNN with GlobalMaxPooling2D.

    Difference to CNN-06:
        - replaces GlobalAveragePooling2D with GlobalMaxPooling2D
        - uses grayscale-standardized dataset
        - stronger dropout

    Architecture:
        Input 224x224x3

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

        GlobalMaxPooling2D

        Dense(512)
        Dropout(0.6)
        Dense(128)
        Dropout(0.3)
        Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # 2 conv layers
    x = conv_block(
        inputs,
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

    # Main architectural change compared with CNN-06
    x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)

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
        name="CNN_09_GlobalMaxPool_Gray",
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
        model_name="CNN_09_GlobalMaxPool_Gray",
        build_model_fn=build_cnn09_global_max_pool_gray,
        config=config,
        architecture_notes=(
            "Based on CNN_06_Deeper_GAP_StrongerDropout_Gray, but replaces "
            "GlobalAveragePooling2D with GlobalMaxPooling2D."
    )

if __name__ == "__main__":
    main()

