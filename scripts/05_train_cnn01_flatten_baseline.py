"""
05_train_cnn01_flatten_baseline.py

CNN-01: Flatten baseline architecture.
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

def build_cnn01_flatten_baseline(
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    CNN-01: Flatten baseline.

    CNN baseline with Flatten before the dense classifier.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # 2 convolutional layers
    x = conv_block(
        inputs,
        filters=32,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block1",
    )

    # 2 convolutional layers
    x = conv_block(
        x,
        filters=64,
        conv_layers=2,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block2",
    )

    # 1 convolutional layer
    x = conv_block(
        x,
        filters=128,
        conv_layers=1,
        kernel_size=3,
        use_batch_norm=False,
        use_pooling=True,
        block_name="block3",
    )


    x = layers.Flatten(name="flatten")(x)

    x = dense_block(x, units=256, dropout_rate=0.5, block_name="fc1")
    x = dense_block(x, units=128, dropout_rate=None, block_name="fc2")

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_softmax",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_01_Flatten_Baseline",
    )

def main():
    config = ExperimentConfig(
        project_root=PROJECT_ROOT,

        dataset_dir=Path("dataset_processed"),

        results_dir=Path("results/experiments"),

        image_size=(224, 224),
        input_shape=(224, 224, 3),

        batch_size=32,
        epochs=25,
        learning_rate=1e-4,
        loss="sparse_categorical_crossentropy",
        seed=42,

        early_stopping_patience=5,
        reduce_lr_patience=3,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-6,

        cache_dataset=True,
    )

    run_experiment(
        model_name="CNN_01_Flatten_Baseline",
        build_model_fn=build_cnn01_flatten_baseline,
        config=config,
        architecture_notes=(
            "Initial baseline architecture using Flatten before the dense classifier."
        ),
    )

if __name__ == "__main__":
    main()
