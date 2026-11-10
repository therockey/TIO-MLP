"""Pętla uczenia, walidacja i budowa optymalizatora."""

from __future__ import annotations

import copy
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .config import Config
from .metrics import classification_metrics


@dataclass
class EvalResult:
    loss: float
    preds: np.ndarray
    targets: np.ndarray
    probs: np.ndarray


def build_criterion(cfg: Config) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.momentum > 0,
        )
    raise ValueError(f"Nieznany optymalizator: {cfg.optimizer!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01
        )
    return None


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(
        loader,
        desc="  trening",
        leave=False,
        disable=cfg.quiet or not sys.stderr.isatty(),
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
        if not cfg.quiet:
            progress.set_postfix(
                loss=f"{running_loss / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.4f}",
            )
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    if total == 0:
        return EvalResult(
            loss=0.0,
            preds=np.empty(0, dtype=np.int64),
            targets=np.empty(0, dtype=np.int64),
            probs=np.empty((0, 10), dtype=np.float32),
        )
    return EvalResult(
        loss=running_loss / total,
        preds=np.concatenate(all_preds),
        targets=np.concatenate(all_targets),
        probs=np.concatenate(all_probs),
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], dict[str, torch.Tensor] | None, int, float]:
    """Trenuje model, wybierając epokę o najlepszym walidacyjnym macro-F1."""
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        val_result = evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(
            val_result.targets, val_result.preds, num_classes=10
        )
        elapsed = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_result.loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "lr": float(learning_rate),
            "seconds": float(elapsed),
        }
        history.append(row)
        log(
            f"Epoka {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_result.loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"F1={val_metrics['macro_f1']:.4f} | {elapsed:.1f} s"
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["macro_f1"])
            else:
                scheduler.step()

        if val_metrics["macro_f1"] > best_val_f1 + 1e-6:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
                log(
                    f"Wczesne zatrzymanie: brak poprawy walidacyjnego F1 "
                    f"przez {cfg.patience} epok."
                )
                break

    return history, best_state, best_epoch, best_val_f1
