"""Dane MNIST: pobranie, podział, normalizacja i augmentacja.

Podział jest stratyfikowany: z oficjalnego zbioru treningowego (60 000)
wydzielamy zbiór walidacyjny, a oficjalny zbiór testowy (10 000) zostaje
nietknięty do końcowej oceny. Statystyki normalizacji liczymy wyłącznie na
części treningowej (bez wycieku z walidacji/testu).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import Config
from .utils import save_json


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    mean: list[float]
    std: list[float]
    sizes: dict[str, int]
    class_counts: dict[str, list[int]]
    validation: dict[str, dict]

    def summary(self) -> dict:
        return {
            "sizes": self.sizes,
            "class_names": self.class_names,
            "class_counts": self.class_counts,
            "normalization": {"mean": self.mean, "std": self.std},
            "validation": self.validation,
        }


def _validate_raw_dataset(dataset: datasets.MNIST, name: str) -> dict:
    """Podstawowa kontrola jakości: kształt, zakres pikseli i etykiet."""
    data = dataset.data
    targets = dataset.targets
    return {
        "name": name,
        "samples": int(data.shape[0]),
        "image_shape": list(data.shape[1:]),
        "pixel_min": int(data.min()),
        "pixel_max": int(data.max()),
        "label_min": int(targets.min()),
        "label_max": int(targets.max()),
        "missing_values": 0,  # MNIST jest kompletnym zbiorem uint8
    }


def _stratified_split(targets, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Dzieli indeksy po klasach, aby zachować proporcje etykiet."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    total = len(targets)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        class_val = round(len(indices) * val_size / total)
        class_val = min(max(class_val, 1), len(indices) - 1)
        val_parts.append(indices[:class_val])
        train_parts.append(indices[class_val:])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _limit_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=limit, replace=False)
    return np.sort(chosen)


def _compute_stats(raw_dataset: datasets.MNIST, indices: np.ndarray) -> tuple[list[float], list[float]]:
    images = raw_dataset.data[indices].to(torch.float32).div_(255.0)
    mean = images.mean(dim=(0, 1, 2)).reshape(-1).tolist()
    std = images.std(dim=(0, 1, 2), unbiased=False).reshape(-1).tolist()
    return mean, std


def _load_or_compute_stats(
    root: Path,
    raw_dataset: datasets.MNIST,
    train_indices: np.ndarray,
    cfg: Config,
) -> tuple[list[float], list[float]]:
    stats_path = root / "mnist_stats.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            if (
                payload.get("split_seed") == cfg.split_seed
                and payload.get("val_size") == cfg.val_size
                and len(payload.get("mean", [])) == 1
            ):
                return payload["mean"], payload["std"]
        except (json.JSONDecodeError, KeyError):
            pass

    mean, std = _compute_stats(raw_dataset, train_indices)
    save_json(
        stats_path,
        {
            "split_seed": cfg.split_seed,
            "val_size": cfg.val_size,
            "mean": mean,
            "std": std,
            "note": "statystyki policzone wyłącznie na części treningowej",
        },
    )
    return mean, std


def _build_transform(mean: list[float], std: list[float], augment: str, train: bool):
    operations: list = []
    if train and augment == "affine":
        operations.append(
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05))
        )
    elif train and augment == "rotation":
        operations.append(transforms.RandomRotation(degrees=10))
    operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(operations)


def _worker_init(seed: int):
    def initializer(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return initializer


def build_dataloaders(cfg: Config) -> DataBundle:
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    raw_train = datasets.MNIST(
        root=str(root),
        train=True,
        download=cfg.download,
        transform=transforms.ToTensor(),
    )
    raw_test = datasets.MNIST(
        root=str(root),
        train=False,
        download=cfg.download,
        transform=transforms.ToTensor(),
    )
    validation = {
        "train": _validate_raw_dataset(raw_train, "train"),
        "test": _validate_raw_dataset(raw_test, "test"),
    }

    train_indices, val_indices = _stratified_split(raw_train.targets, cfg.val_size, cfg.split_seed)
    mean, std = _load_or_compute_stats(root, raw_train, train_indices, cfg)

    train_indices = _limit_indices(train_indices, cfg.train_subset, cfg.split_seed)
    test_indices = _limit_indices(
        np.arange(len(raw_test)), cfg.test_subset, cfg.split_seed + 1
    )

    train_dataset = Subset(
        datasets.MNIST(
            root=str(root),
            train=True,
            download=cfg.download,
            transform=_build_transform(mean, std, cfg.augment, train=True),
        ),
        train_indices.tolist(),
    )
    val_dataset = Subset(
        datasets.MNIST(
            root=str(root),
            train=True,
            download=cfg.download,
            transform=_build_transform(mean, std, "none", train=False),
        ),
        val_indices.tolist(),
    )
    test_dataset = Subset(
        datasets.MNIST(
            root=str(root),
            train=False,
            download=cfg.download,
            transform=_build_transform(mean, std, "none", train=False),
        ),
        test_indices.tolist(),
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    pin_memory = cfg.device in ("cuda", "auto") and torch.cuda.is_available()
    common = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init(cfg.seed),
    }
    if cfg.num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    train_targets = np.asarray(raw_train.targets)[train_indices]
    val_targets = np.asarray(raw_train.targets)[val_indices]
    test_targets = np.asarray(raw_test.targets)[test_indices]
    class_counts = {
        "train": np.bincount(train_targets, minlength=10).astype(int).tolist(),
        "val": np.bincount(val_targets, minlength=10).astype(int).tolist(),
        "test": np.bincount(test_targets, minlength=10).astype(int).tolist(),
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=[str(index) for index in range(10)],
        mean=mean,
        std=std,
        sizes={
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        class_counts=class_counts,
        validation=validation,
    )
