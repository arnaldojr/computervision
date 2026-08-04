"""Carrega MobileNetV3-Small pré-treinado no ImageNet (sem necessidade de treino)."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models


# Rótulos das 1000 classes do ImageNet — carregados do arquivo de labels
_LABELS_PATH = Path(__file__).resolve().parent / "artifacts" / "imagenet_classes.txt"


def _load_classes() -> list[str]:
    if _LABELS_PATH.exists():
        return [line.strip() for line in _LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    # fallback: índices numéricos se o arquivo não existir
    return [str(i) for i in range(1000)]


class ModeloPreTreinado(nn.Module):
    """Wrapper do MobileNetV3-Small com pesos ImageNet prontos para inferência."""

    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        # weights="IMAGENET1K_V1" faz download automático na primeira execução
        self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.backbone.eval()
        self.backbone.to(self.device)
        self.classes = _load_classes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
