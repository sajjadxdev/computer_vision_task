import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class BaselineCNN(nn.Module):
    """
    Model 1: Custom Baseline Convolutional Neural Network.
    
    Architecture:
    - 3 Conv-BatchNorm-ReLU-MaxPool Blocks
    - Global Average Pooling / Adaptive Avg Pool
    - Fully Connected Classifier with Dropout
    
    Designed to serve as a lightweight, fast-training baseline.
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3, dropout: float = 0.3):
        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def build_baseline_model(config: Dict[str, Any]) -> nn.Module:
    num_classes = config["data"]["num_classes"]
    dropout = config["models"]["baseline_cnn"].get("dropout", 0.3)
    return BaselineCNN(num_classes=num_classes, dropout=dropout)
