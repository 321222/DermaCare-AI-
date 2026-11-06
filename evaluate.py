"""Evaluate the trained classifier against a labeled dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)


def evaluate(model_path: Path, dataset_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads((model_path.parent / "class_names.json").read_text(encoding="utf-8"))
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        class_names=names,
        image_size=IMAGE_SIZE,
        batch_size=32,
        shuffle=False,
        label_mode="int",
    )
    model = tf.keras.models.load_model(model_path)
    labels = np.concatenate([y.numpy() for _, y in dataset])
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, target_names=names, output_dict=True, zero_division=0)
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    matrix = confusion_matrix(labels, predicted, labels=range(len(names)))
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
             yticklabels=names, xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    binary_labels = label_binarize(labels, classes=range(len(names)))
    fig, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(names):
        if len(np.unique(binary_labels[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_labels[:, index], probabilities[:, index])
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)
    return {"report": report, "confusion_matrix": matrix.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("skin_disease_cnn.keras"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    args = parser.parse_args()
    results = evaluate(args.model, args.dataset, args.output)
    print(json.dumps(results["report"], indent=2))


if __name__ == "__main__":
    main()
