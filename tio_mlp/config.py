"""Konfiguracja pojedynczego uruchomienia.

Wszystkie hiperparametry są w jednym, niemutowalnym dataclassie. Dzięki temu
eksperymenty mogą tworzyć warianty przez ``cfg.replace(...)``, a konfiguracja
jest zapisywana do JSON razem z wynikami.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh", "sigmoid")
OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULERS = ("none", "plateau", "cosine")
AUGMENTATIONS = ("none", "affine", "rotation")


@dataclass(frozen=True)
class Config:
    # --- dane ---
    data_dir: str = "data"
    output_dir: str = "runs"
    val_size: int = 5000
    batch_size: int = 64
    num_workers: int = 0
    augment: str = "none"
    train_subset: int = 0  # 0 = cały zbiór; >0 tylko do szybkich testów
    test_subset: int = 0

    # --- model ---
    hidden_sizes: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dropout: float = 0.0
    batchnorm: bool = False

    # --- optymalizacja ---
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "none"
    label_smoothing: float = 0.0
    grad_clip: float = 0.0  # 0 = wyłączone

    # --- uczenie ---
    epochs: int = 10
    patience: int = 0  # 0 = bez wczesnego zatrzymywania
    seed: int = 42
    split_seed: int = 42  # osobne ziarno: podział danych jest stały w eksperymentach
    device: str = "auto"

    # --- uruchomienie ---
    name: str = ""
    quiet: bool = False
    download: bool = True

    def validate(self) -> None:
        errors: list[str] = []
        if not self.hidden_sizes:
            errors.append("hidden_sizes nie może być puste")
        if any(h <= 0 for h in self.hidden_sizes):
            errors.append("każda warstwa ukryta musi mieć dodatnią liczbę neuronów")
        if self.activation not in ACTIVATIONS:
            errors.append(f"activation musi być jednym z {ACTIVATIONS}")
        if self.optimizer not in OPTIMIZERS:
            errors.append(f"optimizer musi być jednym z {OPTIMIZERS}")
        if self.scheduler not in SCHEDULERS:
            errors.append(f"scheduler musi być jednym z {SCHEDULERS}")
        if self.augment not in AUGMENTATIONS:
            errors.append(f"augment musi być jednym z {AUGMENTATIONS}")
        if not 0.0 <= self.dropout < 1.0:
            errors.append("dropout musi należeć do [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            errors.append("label_smoothing musi należeć do [0, 1)")
        if self.lr <= 0:
            errors.append("lr musi być dodatnie")
        if self.weight_decay < 0:
            errors.append("weight_decay nie może być ujemne")
        if self.momentum < 0:
            errors.append("momentum nie może być ujemne")
        if self.batch_size <= 0:
            errors.append("batch_size musi być dodatnie")
        if self.epochs <= 0:
            errors.append("epochs musi być dodatnie")
        if not 0 < self.val_size < 60000:
            errors.append("val_size musi należeć do (0, 60000)")
        if self.num_workers < 0:
            errors.append("num_workers nie może być ujemne")
        if self.grad_clip < 0:
            errors.append("grad_clip nie może być ujemne")
        if self.patience < 0:
            errors.append("patience nie może być ujemne")
        if self.train_subset < 0 or self.test_subset < 0:
            errors.append("rozmiary podzbiorów nie mogą być ujemne")
        if errors:
            raise ValueError("Niepoprawna konfiguracja: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        allowed = {field.name for field in fields(cls)}
        data = {key: value for key, value in values.items() if key in allowed}
        if "hidden_sizes" in data:
            data["hidden_sizes"] = tuple(int(value) for value in data["hidden_sizes"])
        return cls(**data)

    def replace(self, **changes: Any) -> Config:
        return replace(self, **changes)

    def default_name(self) -> str:
        hidden = "-".join(str(size) for size in self.hidden_sizes)
        parts = [
            f"mlp{hidden}",
            self.activation,
            self.optimizer,
            f"lr{self.lr:g}",
            f"bs{self.batch_size}",
        ]
        if self.dropout > 0:
            parts.append(f"do{self.dropout:g}")
        if self.batchnorm:
            parts.append("bn")
        if self.augment != "none":
            parts.append(self.augment)
        return "_".join(parts)

    def describe(self) -> str:
        hidden = " -> ".join(str(size) for size in self.hidden_sizes)
        return (
            f"MLP 784 -> {hidden} -> 10 | aktywacja={self.activation}, "
            f"dropout={self.dropout:g}, batchnorm={self.batchnorm} | "
            f"optymalizator={self.optimizer}, lr={self.lr:g}, "
            f"weight_decay={self.weight_decay:g}, scheduler={self.scheduler} | "
            f"epochs={self.epochs}, batch_size={self.batch_size}, augment={self.augment}"
        )
