import torch
import torch.nn as nn
import torch.optim as optim

from src.config import CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, ensure_directories
from src.data import create_datasets_and_loaders
from src.evaluate import (
    predict_loader,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
    save_training_curves,
)
from src.models import SimpleCNN
from src.train import train_model
from src.utils import get_device, set_seed


def main():
    ensure_directories()
    set_seed()

    device = get_device()
    print(f"Using device: {device}")

    _, _, train_loader, test_loader = create_datasets_and_loaders()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_name = "baseline_cnn"
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}.pth"

    history, best_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict_loader(model, test_loader, device)
    report = save_classification_report(
        y_true,
        y_pred,
        METRICS_DIR / f"{model_name}_classification_report.txt",
    )

    save_metrics(
        {
            "model": model_name,
            "best_test_accuracy": best_accuracy,
            "best_epoch": checkpoint["epoch"],
            "classification_report": report,
        },
        METRICS_DIR / f"{model_name}_metrics.json",
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        "Baseline CNN Confusion Matrix",
    )

    save_training_curves(
        history,
        FIGURES_DIR / f"{model_name}_training_curves.png",
        "Baseline CNN Training Curves",
    )


if __name__ == "__main__":
    main()
