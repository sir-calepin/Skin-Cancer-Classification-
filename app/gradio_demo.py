from pathlib import Path
import sys

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_LABELS,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.utils import get_device

MODEL_PATH = PROJECT_ROOT / "models" / "final" / "resnet50_final.pth"

device = get_device()

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def build_model():
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run `python scripts/train_resnet50_final.py` first."
        )

    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()


def predict_image(image: Image.Image):
    if image is None:
        return {}

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    predictions = {
        f"{class_name} — {CLASS_LABELS[class_name]}": float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }

    return predictions


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a dermatoscopic image"),
    outputs=gr.Label(num_top_classes=3, label="Top predictions"),
    title="HAM10000 Skin Lesion Classifier",
    description=(
        "Educational demo using the final ResNet50 transfer-learning model. "
        "This tool is not clinically validated and must not be used for diagnosis."
    ),
)


if __name__ == "__main__":
    demo.launch()
