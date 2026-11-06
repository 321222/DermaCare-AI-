"""Train and compare a custom CNN and MobileNetV2 on the local image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
IMAGE_SIZE = (224, 224)
SEED = 42


def collect_dataset(root: Path) -> tuple[list[str], list[int]]:
    files: list[str] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset class directory: {class_dir}")
        class_files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        files.extend(str(path) for path in class_files)
        labels.extend([class_index] * len(class_files))
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(counts < 7):
        raise ValueError(f"Need at least 7 images per class for a stratified 70/15/15 split; found {dict(zip(CLASS_NAMES, counts))}")
    return files, labels


def split_dataset(paths: list[str], labels: list[int]) -> tuple[Any, ...]:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    contents = tf.io.read_file(path)
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def make_dataset(paths: list[str], labels: list[int], training: bool = False) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset


def augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def custom_cnn(class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1 / 255.0)(x)
    for filters, dropout in ((32, 0.15), (64, 0.20), (128, 0.25), (256, 0.30)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def mobilenet_model(class_count: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    augment = augmentation()(inputs)
    preprocess = tf.keras.layers.Rescaling(
        scale=1 / 127.5, offset=-1, name="mobilenet_preprocess"
    )(augment)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    features = preprocess
    for layer in base.layers[1:]:
        features = layer(features)
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v2"), base


def callbacks(checkpoint: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def plot_history(histories: dict[str, tf.keras.callbacks.History], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        for metric, axis in (("accuracy", axes[0]), ("loss", axes[1])):
            values = history.history.get(metric, [])
            if values:
                axis.plot(values, label=f"{name} train")
            validation = history.history.get(f"val_{metric}", [])
            if validation:
                axis.plot(validation, linestyle="--", label=f"{name} validation")
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, labels: list[int],
                   output_dir: Path, prefix: str) -> dict[str, Any]:
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=range(len(CLASS_NAMES)))
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
             xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted", ylabel="Actual", title=f"{prefix} confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    binary = label_binarize(labels, classes=range(len(CLASS_NAMES)))
    fig, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(CLASS_NAMES):
        if len(np.unique(binary[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{prefix} one-vs-rest ROC")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)
    (output_dir / f"{prefix}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }


def train(data_dir: Path, output_dir: Path, epochs: int, fine_tune_epochs: int) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = collect_dataset(data_dir)
    tr_p, tr_y, va_p, va_y, te_p, te_y = split_dataset(paths, labels)
    train_ds = make_dataset(tr_p, tr_y, training=True)
    val_ds = make_dataset(va_p, va_y)
    test_ds = make_dataset(te_p, te_y)
    class_weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=np.asarray(tr_y))
    weights = {index: float(weight) for index, weight in enumerate(class_weights)}
    histories: dict[str, tf.keras.callbacks.History] = {}
    candidates: dict[str, tf.keras.Model] = {}

    cnn_path = output_dir / "custom_cnn_best.keras"
    cnn = custom_cnn(len(CLASS_NAMES))
    cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    histories["custom_cnn"] = cnn.fit(
        train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
        callbacks=callbacks(cnn_path), verbose=1
    )
    candidates["custom_cnn"] = tf.keras.models.load_model(cnn_path)

    transfer_path = output_dir / "mobilenet_v2_best.keras"
    transfer, base = mobilenet_model(len(CLASS_NAMES))
    transfer.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_frozen = transfer.fit(
        train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
        callbacks=callbacks(transfer_path), verbose=1
    )
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    transfer.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_tuned = transfer.fit(
        train_ds, validation_data=val_ds, epochs=fine_tune_epochs, class_weight=weights,
        callbacks=callbacks(transfer_path), verbose=1
    )
    history_frozen.history = {
        key: history_frozen.history.get(key, []) + history_tuned.history.get(key, [])
        for key in set(history_frozen.history) | set(history_tuned.history)
    }
    histories["mobilenet_v2"] = history_frozen
    candidates["mobilenet_v2"] = tf.keras.models.load_model(transfer_path)

    plot_history(histories, output_dir)
    metrics: dict[str, Any] = {}
    for name, candidate in candidates.items():
        metrics[name] = evaluate_model(candidate, val_ds, va_y, output_dir, f"{name}_validation")
    best_name = max(metrics, key=lambda name: (metrics[name]["accuracy"], metrics[name]["macro_f1"]))
    best_model = candidates[best_name]
    test_metrics = evaluate_model(best_model, test_ds, te_y, output_dir, "test")
    best_model.save(output_dir / "skin_disease_cnn.keras")
    (output_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    comparison = {
        "selected_model": best_name,
        "validation": {name: {"accuracy": data["accuracy"], "macro_f1": data["macro_f1"]}
                       for name, data in metrics.items()},
        "held_out_test": {"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"]},
        "test_metrics_are_not_clinical_validation": True,
    }
    (output_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "validation_accuracy", "validation_macro_f1", "selected"])
        for name, result in metrics.items():
            writer.writerow([
                name,
                f"{result['accuracy']:.6f}",
                f"{result['macro_f1']:.6f}",
                name == best_name,
            ])
    print("Model comparison (validation):")
    print(f"{'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'Selected':>10}")
    for name, result in metrics.items():
        print(f"{name:<18} {result['accuracy']:>10.4f} {result['macro_f1']:>10.4f} {str(name == best_name):>10}")
    print(json.dumps(comparison, indent=2))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.fine_tune_epochs)


if __name__ == "__main__":
    main()
