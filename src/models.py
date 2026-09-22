import torch.nn as nn
from torchvision.models import ResNet50Weights, resnet50


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features)


def build_resnet50(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    weights = ResNet50Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    return model
