"""MobileNetV3-Small com head substituído para fine-tuning."""

import torch
from torch import nn
from torchvision import models


class MobileNetTransfer(nn.Module):
    """MobileNetV3-Small pré-treinado com classifier re-treinável."""

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # substitui apenas a última camada linear
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
