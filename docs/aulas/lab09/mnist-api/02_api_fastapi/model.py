import torch
from torch import nn


class RedeNeuralSimples(nn.Module):
    """
    Mesma arquitetura usada no notebook.

    Entrada esperada:
        Tensor no formato [batch, 1, 28, 28]

    Saída:
        Tensor no formato [batch, 10], contendo os logits das classes 0 a 9.
    """

    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.rede = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.rede(x)
        return logits
