"""
11_train_cnn07_wider_gap_stronger_dropout_gray.py

CNN-07: Deeper GAP architecture with stronger dropout and wider filters,
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

def build_cnn07_wider_gap_stronger_dropout(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-07: Wider Deeper GAP with stronger dropout.

    Difference to CNN-06:
        - wider convolution blocks
        - filters increased from 32/64/128/256 to 64/128/256/512
        - uses grayscale-standardized images
        - keeps stronger dropout

    Architecture:
        Input 224x224x3

        Block 1:
            Conv2D(64)
            Conv2D(64)
            MaxPooling

        Block 2:
            Conv2D(128)
            Conv2D(128)
            MaxPooling

        Block 3:
            Conv2D(256)
            Conv2D(256)
            MaxPooling

        Block 4:
            Conv2D(512)
            MaxPooling

        GlobalAveragePooling2D

        Dense(512)
        Dropout(0.6)
        Dense(128)
        Dropout(0.3)
        Dense(num_classes, softmax)

    Requirement:
        - 7 Conv2D layers
        - 2 hidden fully connected layers
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # 2 conv layers
    x = conv_block(
        inputs,
        filters=64,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block1",
    )

    # 2 conv layers
    x = conv_block(
        x,
        filters=128,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block2",
    )

    # 2 conv layers
    x = conv_block(
        x,
        filters=256,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block3",
    )

    # 1 conv layer
    x = conv_block(
        x,
        filters=512,
        conv_layers=1,
        kernel_size=3,
        use_batch_norm=False,
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
        name="CNN_07_Wider_GAP_StrongerDropout_Gray",
    )

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
        model_name="CNN_07_Wider_GAP_StrongerDropout_Gray",
        build_model_fn=build_cnn07_wider_gap_stronger_dropout,
        config=config,
        architecture_notes=(
            "Based on CNN_06_Deeper_GAP_StrongerDropout_Gray, but increases "
            "the convolutional filter widths from 32/64/128/256 to "
            "64/128/256/512. The model keeps grayscale-standardized images "
            "and stronger dropout: Dropout(0.6) after Dense(512) and "
            "Dropout(0.3) after Dense(128)."
        ),
    )

if __name__ == "__main__":
    main()
