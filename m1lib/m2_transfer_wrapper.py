
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization

from m1lib.training_wrapper import (
    ExperimentConfig,
    build_callbacks,
    dataset_to_numpy_labels,
    evaluate_model,
    get_dataset_cardinality,
    get_datasets,
    predict_classes,
    save_json,
    save_model_summary,
    set_seeds,
)


# Transfer-Learning-Konfiguration

@dataclass
class TransferConfig:
    """Parameter speziell fuer Transfer Learning (erganzt ExperimentConfig)."""

    backbone: str = "inception_v3"

    # Phase 1 – Feature Extraction
    phase1_epochs: int = 20
    phase1_lr: float = 1e-3

    # Phase 2 – Fine-Tuning
    phase2_epochs: int = 30
    phase2_lr: float = 1e-5      
    unfreeze_layers: int = 20   

    # Klassifikationskopf
    pooling: str = "gap"          # "gap" (GlobalAveragePooling2D) oder "flatten"
    dense_units: int = 256
    dropout_rate: float = 0.5

    image_size: tuple[int, int] = (224, 224)
    input_shape: tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    seed: int = 42
    dataset_dir: str = "dataset_processed"
    results_dir: str = "results/m2_experiments"
    project_root: str = "."

    # Early-Stopping 
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7

    # Daten-Augmentation
    use_augmentation: bool = False

    # Datensatz-Caching
    cache_dataset: bool = True


# Backbone laden
BACKBONE_MAP = {
    "inception_v3":   keras.applications.InceptionV3,
}

@keras.utils.register_keras_serializable(name="preprocess_inception_v3")
def preprocess_inception_v3(x):
    return keras.applications.inception_v3.preprocess_input(x)

PREPROCESS_MAP = {
    "inception_v3":   preprocess_inception_v3,
}


def _get_backbone(name: str, input_shape: tuple):
    name = name.lower()
    if name not in BACKBONE_MAP:
        raise ValueError(f"Unbekannter Backbone: {name}. Erlaubt: {list(BACKBONE_MAP)}")
    return BACKBONE_MAP[name](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )


# Modell bauen
def build_transfer_model(
    input_shape: tuple,
    num_classes: int,
    cfg: TransferConfig,
) -> keras.Model:
    """
    Baut ein Transfer-Learning-Modell:
      Input → Backbone (eingefroren) → GlobalAveragePooling → Dense → Dropout → Output

    Rueckgabe: Modell mit eingefrorenem Backbone (bereit fuer Phase 1).
    """
    backbone = _get_backbone(cfg.backbone, input_shape)

    # Backbone komplett einfrieren (Phase 1)
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")

    x = inputs

    # Daten-Augmentation 
    if cfg.use_augmentation:
        augmentation = keras.Sequential(
            [
                #layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.03),
                #layers.RandomZoom(0.10),
                layers.RandomContrast(0.10),
                layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            ],
            name="data_augmentation",
        )
        x = augmentation(x)

    # Backbone-spezifische Vorverarbeitung (skaliert von [0,1] auf Backbone-Range)
    x = layers.Rescaling(255.0, name="rescale_to_raw")(x)

    preprocess_fn = PREPROCESS_MAP[cfg.backbone.lower()]
    x = layers.Lambda(
        preprocess_fn,
        name="backbone_preprocess",
    )(x)

    x = backbone(x, training=False)

    # Klassifikationskopf
    if cfg.pooling == "flatten":
        x = layers.Flatten(name="flatten")(x)
    else:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        
    x = layers.Dense(cfg.dense_units, activation="relu", name="fc_head")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs)
    return model


def unfreeze_backbone(model: keras.Model, cfg: TransferConfig) -> keras.Model:
    """
    Taut die letzten `unfreeze_layers` Layer des Backbones fuer Fine-Tuning auf.
    Alle frueheren Layer bleiben eingefroren (generische Low-Level-Features).
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential):
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Kein Backbone-Teilmodell gefunden.")

    backbone.trainable = True

    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    for layer in backbone.layers[: -cfg.unfreeze_layers]:
        layer.trainable = False

    n_trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Backbone: {n_trainable}/{len(backbone.layers)} Layer trainierbar")
    return model


def _transfer_config_to_experiment_config(cfg: TransferConfig) -> ExperimentConfig:
    return ExperimentConfig(
        project_root=Path(cfg.project_root),
        dataset_dir=Path(cfg.dataset_dir),
        results_dir=Path(cfg.results_dir),
        image_size=cfg.image_size,
        input_shape=cfg.input_shape,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        cache_dataset=cfg.cache_dataset,
        # Folgende Felder nicht fuer Datenladen relevant
        epochs=cfg.phase1_epochs + cfg.phase2_epochs,
        learning_rate=cfg.phase1_lr,
        early_stopping_patience=cfg.early_stopping_patience,
        reduce_lr_patience=cfg.reduce_lr_patience,
        reduce_lr_factor=cfg.reduce_lr_factor,
        min_learning_rate=cfg.min_learning_rate,
    )


# Hauptexperiment
def run_transfer_experiment(
    model_name: str,
    cfg: TransferConfig,
) -> dict:
    set_seeds(cfg.seed)

    results_root = Path(cfg.project_root) / cfg.results_dir / model_name
    results_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Transfer-Learning-Experiment: {model_name}")
    print(f"  Backbone  : {cfg.backbone.upper()}")
    print(f"  Datensatz : {cfg.dataset_dir}")
    print("=" * 70)

    # --- Daten ---
    exp_cfg = _transfer_config_to_experiment_config(cfg)
    train_ds, val_ds, test_ds, class_names = get_datasets(exp_cfg)
    print("shape of train_ds", train_ds.element_spec)
    num_classes = len(class_names)

    with open(results_root / "class_names.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"\nKlassen ({num_classes}): {class_names}")

    # --- Modell ---
    print("\nBaue Modell ...")
    model = build_transfer_model(cfg.input_shape, num_classes, cfg)
    model._name = model_name

    total_params = int(model.count_params())
    backbone_params = sum(
        l.count_params()
        for l in model.layers
        if isinstance(l, keras.Model)
    )
    print(f"  Gesamt-Parameter : {total_params:,}")
    print(f"  Backbone-Parameter: {backbone_params:,}")
    print(f"  Kopf-Parameter    : {total_params - backbone_params:,}")

    save_model_summary(model, results_root / "architecture_summary.txt")

    # Metadaten speichern
    metadata = {
        "model_name": model_name,
        "backbone": cfg.backbone,
        "class_names": class_names,
        "total_params": total_params,
        "backbone_params": backbone_params,
        "config": {
            "phase1_epochs": cfg.phase1_epochs,
            "phase1_lr": cfg.phase1_lr,
            "phase2_epochs": cfg.phase2_epochs,
            "phase2_lr": cfg.phase2_lr,
            "unfreeze_layers": cfg.unfreeze_layers,
            "dense_units": cfg.dense_units,
            "dropout_rate": cfg.dropout_rate,
            "dataset_dir": cfg.dataset_dir,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
    }
    save_json(metadata, results_root / "experiment_metadata.json")

    # Phase 1 – Feature Extraction
    print("\n--- Phase 1: Feature Extraction (Backbone eingefroren) ---")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.phase1_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p1 = _build_transfer_callbacks(
        results_root / "phase1_best.keras",
        cfg,
        monitor="val_loss",
    )

    start_p1 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase1_epochs,
        callbacks=callbacks_p1,
        verbose=1,
    )
    time_p1 = time.time() - start_p1

    # Bestes Modell aus Phase 1 laden
    if (results_root / "phase1_best.keras").exists():
        model = keras.models.load_model(results_root / "phase1_best.keras")

    _save_history(history_p1, results_root / "history_phase1.csv")
    print(f"  Phase 1 abgeschlossen ({time_p1:.0f}s, {len(history_p1.history['loss'])} Epochen)")

    # Phase 2 – Fine-Tuning
    print(f"\n--- Phase 2: Fine-Tuning (letzte {cfg.unfreeze_layers} Backbone-Layer) ---")

    model = unfreeze_backbone(model, cfg)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.phase2_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = _build_transfer_callbacks(
        results_root / "phase2_best.keras",
        cfg,
        monitor="val_loss",
    )

    start_p2 = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase2_epochs,
        callbacks=callbacks_p2,
        verbose=1,
    )
    time_p2 = time.time() - start_p2

    # Bestes Modell aus Phase 2 laden
    if (results_root / "phase2_best.keras").exists():
        model = keras.models.load_model(results_root / "phase2_best.keras")

    _save_history(history_p2, results_root / "history_phase2.csv")
    print(f"  Phase 2 abgeschlossen ({time_p2:.0f}s, {len(history_p2.history['loss'])} Epochen)")

    _merge_histories(history_p1, history_p2, results_root / "history_combined.csv")

    model.save(results_root / "final_model.keras")

    # Evaluation auf Test-Set
    print("\nEvaluation auf Test-Set ...")
    metrics = evaluate_model(model, test_ds, class_names, results_root)

    total_time = time_p1 + time_p2
    epochs_ran = len(history_p1.history["loss"]) + len(history_p2.history["loss"])

    summary = {
        "model_name": model_name,
        "backbone": cfg.backbone,
        "total_params": total_params,
        "backbone_params": backbone_params,
        "training_time_seconds": float(total_time),
        "phase1_epochs_ran": len(history_p1.history["loss"]),
        "phase2_epochs_ran": len(history_p2.history["loss"]),
        "epochs_ran": epochs_ran,
        **metrics,
    }

    pd.DataFrame([summary]).to_csv(results_root / "experiment_summary.csv", index=False)
    save_json(summary, results_root / "experiment_summary.json")

    return summary


def _build_transfer_callbacks(
    checkpoint_path: Path,
    cfg: TransferConfig,
    monitor: str = "val_loss",
) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_learning_rate,
            verbose=1,
        ),
    ]


def _save_history(history, path: Path) -> None:
    pd.DataFrame(history.history).to_csv(path, index=False)


def _merge_histories(h1, h2, path: Path) -> None:
    df1 = pd.DataFrame(h1.history)
    df2 = pd.DataFrame(h2.history)
    df1["phase"] = 1
    df2["phase"] = 2
    combined = pd.concat([df1, df2], ignore_index=True)
    combined["epoch"] = range(1, len(combined) + 1)
    combined.to_csv(path, index=False)
