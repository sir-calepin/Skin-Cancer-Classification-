import torch
import torch.nn as nn
import torch.optim as optim

from src.config import FINAL_MODELS_DIR, FIGURES_DIR, METRICS_DIR, ensure_directories
from src.data import create_datasets_and_loaders
from src.evaluate import (
    predict_loader,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
    save_training_curves,
)
from src.models import build_resnet50
from src.train import train_model
from src.utils import get_device, set_seed


def main():
    ensure_directories()
    set_seed()

    device = get_device()
    print(f"Using device: {device}")

    _, _, train_loader, test_loader = create_datasets_and_loaders()

    model = build_resnet50(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=0.001,
    )

    model_name = "resnet50_final"
    checkpoint_path = FINAL_MODELS_DIR / f"{model_name}.pth"

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
        "Final ResNet50 Confusion Matrix",
    )

    save_training_curves(
        history,
        FIGURES_DIR / f"{model_name}_training_curves.png",
        "Final ResNet50 Training Curves",
    )

    print(f"\nBest test accuracy: {best_accuracy:.4f}")
    print(f"Saved final model: {checkpoint_path}")


if __name__ == "__main__":
    main()
