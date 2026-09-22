import argparse

import torch
import torch.nn as nn

from src.config import FIGURES_DIR, METRICS_DIR, ensure_directories
from src.data import create_datasets_and_loaders
from src.evaluate import (
    evaluate_accuracy,
    predict_loader,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
)
from src.models import SimpleCNN, build_resnet50
from src.utils import get_device


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved HAM10000 PyTorch model."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pth checkpoint file.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet50"],
        required=True,
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--name",
        default="evaluation",
        help="Prefix used for generated report files.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    ensure_directories()

    device = get_device()
    print(f"Using device: {device}")

    _, _, _, test_loader = create_datasets_and_loaders()

    if arguments.model == "cnn":
        model = SimpleCNN()
    else:
        model = build_resnet50(pretrained=False, freeze_backbone=False)

    checkpoint = torch.load(arguments.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        criterion,
        device,
    )

    y_true, y_pred = predict_loader(model, test_loader, device)

    report = save_classification_report(
        y_true,
        y_pred,
        METRICS_DIR / f"{arguments.name}_classification_report.txt",
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        FIGURES_DIR / f"{arguments.name}_confusion_matrix.png",
        f"{arguments.name} Confusion Matrix",
    )

    save_metrics(
        {
            "model_name": arguments.name,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "classification_report": report,
        },
        METRICS_DIR / f"{arguments.name}_metrics.json",
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
