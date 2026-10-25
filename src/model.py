import torch.nn as nn
from torchvision import models

from src.dataset import CLASS_NAMES


def build_model(num_classes=len(CLASS_NAMES), pretrained=True):
    """ResNet18 with a new final layer for our ECG classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the 1000-class ImageNet head with a head for our classes
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    return model
