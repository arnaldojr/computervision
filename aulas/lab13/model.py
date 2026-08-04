"""DeiT-Tiny fine-tuned para classificação de imagens via HuggingFace Transformers."""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "deit_cifar10"
_PRETRAINED_ID = "facebook/deit-tiny-patch16-224"


class DeiTClassifier(nn.Module):
    """Wrapper do DeiT-Tiny com processor integrado."""

    def __init__(self, num_classes: int = 10, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        # carrega do artefato local se disponível, senão do HuggingFace Hub
        source = str(_ARTIFACTS_DIR) if _ARTIFACTS_DIR.exists() else _PRETRAINED_ID

        self.processor = AutoImageProcessor.from_pretrained(source)
        self.model = AutoModelForImageClassification.from_pretrained(
            source,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()
        self.model.to(self.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
