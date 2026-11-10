"""Implementacja perceptronu wielowarstwowego (MLP) w PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from .config import ACTIVATIONS, Config

INPUT_DIM = 28 * 28
NUM_CLASSES = 10


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Nieznana aktywacja: {name!r}. Dostępne: {ACTIVATIONS}")


class MLP(nn.Module):
    """W pełni połączona sieć dla spłaszczonych obrazów 28x28.

    Każdy blok ukryty to Linear -> (opcjonalnie BatchNorm) -> aktywacja ->
    (opcjonalnie Dropout). Ostatnia warstwa zwraca logity dla 10 klas.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_sizes: tuple[int, ...] = (256, 128),
        num_classes: int = NUM_CLASSES,
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("MLP wymaga co najmniej jednej warstwy ukrytej")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Nieznana aktywacja: {activation!r}")

        self.activation_name = activation
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)

        layers: list[nn.Module] = [nn.Flatten()]
        previous = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = size
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if self.activation_name in ("relu", "leaky_relu", "gelu"):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: Config, num_classes: int = NUM_CLASSES) -> MLP:
    return MLP(
        input_dim=INPUT_DIM,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=num_classes,
        activation=cfg.activation,
        dropout=cfg.dropout,
        batchnorm=cfg.batchnorm,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_summary(model: MLP) -> str:
    hidden = " -> ".join(str(size) for size in model.hidden_sizes)
    return (
        f"MLP: 784 -> {hidden} -> 10 | aktywacja={model.activation_name} | "
        f"parametry={count_parameters(model):,}"
    ).replace(",", " ")
