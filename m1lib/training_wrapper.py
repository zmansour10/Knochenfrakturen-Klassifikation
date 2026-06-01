from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Configuration
@dataclass
class ExperimentConfig:
    project_root: Path = Path(".")
    dataset_dir: Path = Path("dataset_processed")
    results_dir: Path = Path("results/experiments")

    image_size: tuple[int, int] = (224, 224)
    input_shape: tuple[int, int, int] = (224, 224, 3)

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    loss: str = "sparse_categorical_crossentropy"
    seed: int = 42

    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    cache_dataset: bool = True

    label_smoothing: float = 0.0
    use_cosine_decay: bool = False
    cosine_alpha: float = 0.1

# Reproducibility
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



# Dataset handling
def load_raw_dataset(
    directory: Path,
    config: ExperimentConfig,
    shuffle: bool,
) -> tf.data.Dataset:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=shuffle,
        seed=config.seed,
    )

def normalize_and_optimize(
    dataset: tf.data.Dataset,
    config: ExperimentConfig,
    num_classes: int | None = None,
) -> tf.data.Dataset:
    normalization_layer = layers.Rescaling(1.0 / 255)

    dataset = dataset.map(
        lambda images, labels: (normalization_layer(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # For label smoothing, Keras CategoricalCrossentropy needs one-hot labels.
    if config.label_smoothing > 0:
        if num_classes is None:
            raise ValueError("num_classes must be provided when label_smoothing > 0.")

        dataset = dataset.map(
            lambda images, labels: (
                images,
                tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if config.cache_dataset:
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_datasets(
    config: ExperimentConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    dataset_root = config.project_root / config.dataset_dir

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    raw_train_ds = load_raw_dataset(train_dir, config, shuffle=True)
    raw_val_ds = load_raw_dataset(val_dir, config, shuffle=False)
    raw_test_ds = load_raw_dataset(test_dir, config, shuffle=False)

    class_names = raw_train_ds.class_names
    num_classes = len(class_names)

    train_ds = normalize_and_optimize(raw_train_ds, config, num_classes=num_classes)
    val_ds = normalize_and_optimize(raw_val_ds, config, num_classes=num_classes)
    test_ds = normalize_and_optimize(raw_test_ds, config, num_classes=num_classes)

    return train_ds, val_ds, test_ds, class_names


# Reusable model blocks
def conv_block(
    x,
    filters: int,
    conv_layers: int = 1,
    kernel_size: int = 3,
    use_batch_norm: bool = False,
    use_pooling: bool = True,
    pool_size: tuple[int, int] = (2, 2),
    block_name: str = "conv_block",
    kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
):
    """
    convolution block.
    """
    for i in range(conv_layers):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            activation=None,
            kernel_initializer=kernel_initializer,
            name=f"{block_name}_conv{i + 1}",
        )(x)

        if use_batch_norm:
            x = layers.BatchNormalization(name=f"{block_name}_bn{i + 1}")(x)

        x = layers.ReLU(name=f"{block_name}_relu{i + 1}")(x)

    if use_pooling:
        x = layers.MaxPooling2D(
            pool_size=pool_size,
            name=f"{block_name}_pool",
        )(x)

    return x

def dense_block(
    x,
    units: int,
    dropout_rate: float | None = None,
    block_name: str = "dense_block",
    kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
):
    x = layers.Dense(
        units,
        activation="relu",
        kernel_initializer=kernel_initializer,
        name=f"{block_name}_dense",
    )(x)

    if dropout_rate is not None and dropout_rate > 0:
        x = layers.Dropout(
            dropout_rate,
            name=f"{block_name}_dropout",
        )(x)

    return x

def mild_data_augmentation(name: str = "mild_data_augmentation") -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name=name,
    )

def strong_data_augmentation(name: str = "strong_data_augmentation") -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.15),
        ],
        name=name,
    )


# Model checks and saving
def count_layer_types(model: keras.Model) -> dict[str, int]:
    counts = {
        "Conv2D": 0,
        "Dense": 0,
        "MaxPooling2D": 0,
        "Dropout": 0,
        "BatchNormalization": 0,
        "Flatten": 0,
        "GlobalAveragePooling2D": 0,
        "GlobalMaxPooling2D": 0,
        "LeakyReLU": 0,
        "ReLU": 0,
        "RandomRotation": 0,
        "RandomZoom": 0,
        "RandomContrast": 0,
        "RandomTranslation": 0,
    }

    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type in counts:
            counts[layer_type] += 1

        # Count augmentation layers inside nested Sequential models.
        if isinstance(layer, keras.Sequential):
            for sublayer in layer.layers:
                sublayer_type = sublayer.__class__.__name__
                if sublayer_type in counts:
                    counts[sublayer_type] += 1

    return counts

def check_m1_requirements(model: keras.Model) -> dict[str, int]:
    counts = count_layer_types(model)

    num_conv = counts["Conv2D"]
    num_dense_total = counts["Dense"]
    num_hidden_fc = num_dense_total - 1

    if num_conv < 5:
        raise ValueError(
            f"M1 requirement failed: model has only {num_conv} Conv2D layers."
        )

    if num_hidden_fc < 2:
        raise ValueError(
            f"M1 requirement failed: model has only {num_hidden_fc} hidden fully connected layers."
        )

    return {
        **counts,
        "hidden_fc_layers": num_hidden_fc,
        "total_params": int(model.count_params()),
    }

def save_model_summary(model: keras.Model, output_path: Path) -> None:
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_json(data: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Compile, callbacks, evaluation
def compile_model(
    model: keras.Model,
    config: ExperimentConfig,
    steps_per_epoch: int | None = None,
) -> keras.Model:
    if config.use_cosine_decay:
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required when use_cosine_decay=True.")

        decay_steps = int(steps_per_epoch * config.epochs)

        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=decay_steps,
            alpha=config.cosine_alpha,
        )
    else:
        learning_rate = config.learning_rate

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    if config.label_smoothing > 0:
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=config.label_smoothing
        )
    else:
        loss_fn = config.loss

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    return model

def build_callbacks(output_dir: Path, config: ExperimentConfig) -> list[keras.callbacks.Callback]:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    if not config.use_cosine_decay:
        callbacks.insert(
            1,
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
                verbose=1,
            ),
        )

    return callbacks

def dataset_to_numpy_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels = []

    for _, batch_labels in dataset:
        batch_np = batch_labels.numpy()

        if batch_np.ndim > 1:
            batch_np = np.argmax(batch_np, axis=1)

        labels.extend(batch_np)

    return np.asarray(labels)

def predict_classes(model: keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    probabilities = model.predict(dataset)
    return np.argmax(probabilities, axis=1)

def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> dict:
    y_true = dataset_to_numpy_labels(test_ds)
    y_pred = predict_classes(model, test_ds)

    test_loss, test_accuracy_keras = model.evaluate(test_ds, verbose=0)
    test_accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy_keras": float(test_accuracy_keras),
        "test_accuracy_sklearn": float(test_accuracy),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    return metrics

def get_dataset_cardinality(dataset: tf.data.Dataset) -> int:
    cardinality = tf.data.experimental.cardinality(dataset).numpy()

    if cardinality < 0:
        raise ValueError(
            "Could not determine dataset cardinality. "
            "Disable cosine decay or provide a dataset with known cardinality."
        )

    return int(cardinality)


# --------------------
# Main experiment runner

def run_experiment(
    model_name: str,
    build_model_fn,
    config: ExperimentConfig | None = None,
    architecture_notes: str = "",
) -> dict:

    if config is None:
        config = ExperimentConfig()

    set_seeds(config.seed)

    output_dir = config.project_root / config.results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Experiment: {model_name}")
    print("=" * 70)

    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets(config)
    num_classes = len(class_names)
    steps_per_epoch = get_dataset_cardinality(train_ds)

    with open(output_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"Classes: {num_classes}")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    print("\nBuilding model...")
    model = build_model_fn(config.input_shape, num_classes)
    model._name = model_name

    architecture_info = check_m1_requirements(model)
    compile_model(model, config, steps_per_epoch=steps_per_epoch)

    model.summary()
    save_model_summary(model, output_dir / "architecture_summary.txt")

    experiment_metadata = {
        "model_name": model_name,
        "architecture_notes": architecture_notes,
        "class_names": class_names,
        "config": {
            "image_size": list(config.image_size),
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "loss": config.loss,
            "seed": config.seed,
            "label_smoothing": config.label_smoothing,
            "use_cosine_decay": config.use_cosine_decay,
            "cosine_alpha": config.cosine_alpha,
        },
        "steps_per_epoch": steps_per_epoch,
        "architecture_info": architecture_info,
    }

    save_json(experiment_metadata, output_dir / "experiment_metadata.json")

    print("\nM1 architecture check passed:")
    print(f"  Conv2D layers          : {architecture_info['Conv2D']}")
    print(f"  Hidden FC layers       : {architecture_info['hidden_fc_layers']}")
    print(f"  Total parameters       : {architecture_info['total_params']:,}")

    print("\nTraining...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=build_callbacks(output_dir, config),
    )

    training_time_seconds = time.time() - start_time

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    print("\nSaving final model...")
    model.save(output_dir / "final_model.keras")

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_ds, class_names, output_dir)

    final_summary = {
        "model_name": model_name,
        "training_time_seconds": float(training_time_seconds),
        "epochs_ran": len(history.history["loss"]),
        **architecture_info,
        **metrics,
    }

    pd.DataFrame([final_summary]).to_csv(output_dir / "experiment_summary.csv", index=False)
    save_json(final_summary, output_dir / "experiment_summary.json")

    print("\nFinal summary:")
    for key, value in final_summary.items():
        print(f"  {key}: {value}")

    print("\nResults saved to:")
    print(f"  {output_dir}")
    print("=" * 70)

    return final_summary
