import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES


@torch.no_grad()
def predict_loader(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def evaluate_accuracy(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_classification_report(
    y_true,
    y_pred,
    output_path: Path,
) -> dict:
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    output_path.write_text(report_text, encoding="utf-8")
    return report_dict


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def save_training_curves(history: dict, output_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["test_loss"], label="Test loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["test_accuracy"], label="Test accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
