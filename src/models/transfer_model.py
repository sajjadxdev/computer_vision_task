import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("TransferModel")


class TransferClassifier(nn.Module):
    """
    Model 2: Stronger Transfer Learning Model using MobileNetV2 (or ResNet18).
    
    Why MobileNetV2?
    - Pretrained on ImageNet (rich low-level edge/texture features transferrable to fire/smoke)
    - Exceptionally lightweight (~3.5M parameters)
    - Ultra-fast CPU inference latency (<15ms per image)
    - Ideal for edge/laptop/cloud free-tier deployments under strict compute constraints
    """
    def __init__(
        self,
        architecture: str = "mobilenet_v2",
        num_classes: int = 2,
        pretrained: bool = True,
        unfreeze_last_n_layers: int = 2,
        dropout: float = 0.4
    ):
        super(TransferClassifier, self).__init__()
        self.architecture = architecture.lower()

        if self.architecture == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)

            # Freeze early feature layers to preserve low-level edge representation
            for param in base_model.features.parameters():
                param.requires_grad = False

            # Optionally unfreeze the last N blocks for fine-tuning
            if unfreeze_last_n_layers > 0:
                for block in list(base_model.features.children())[-unfreeze_last_n_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True

            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            self.model = base_model

        elif self.architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)

            for param in base_model.parameters():
                param.requires_grad = False

            # Unfreeze layer4 for fine tuning
            if unfreeze_last_n_layers > 0:
                for param in base_model.layer4.parameters():
                    param.requires_grad = True

            in_features = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
            self.model = base_model
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_transfer_model(config: Dict[str, Any]) -> nn.Module:
    transfer_cfg = config["models"]["transfer_learning"]
    num_classes = config["data"]["num_classes"]

    model = TransferClassifier(
        architecture=transfer_cfg.get("architecture", "mobilenet_v2"),
        num_classes=num_classes,
        pretrained=transfer_cfg.get("pretrained", True),
        unfreeze_last_n_layers=transfer_cfg.get("unfreeze_last_n_layers", 2),
        dropout=transfer_cfg.get("dropout", 0.4)
    )
    logger.info(f"Built Transfer Learning model ({transfer_cfg.get('architecture')}) - Pretrained: True")
    return model
