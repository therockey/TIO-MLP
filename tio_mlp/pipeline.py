"""Pojedyncze uruchomienie: dane -> model -> trening -> metryki -> artefakty."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import build_dataloaders
from .engine import build_criterion, evaluate, fit
from .metrics import classification_metrics
from .model import build_model, count_parameters, model_summary
from .reporting import (
    plot_confusion_matrix,
    plot_history,
    save_confusion_csv,
    save_history_csv,
    save_predictions_csv,
)
from .utils import format_seconds, resolve_device, save_json, set_seed, timestamp


def _make_logger(cfg: Config, log: Callable[[str], None] | None) -> Callable[[str], None]:
    if log is not None:
        return log
    if cfg.quiet:
        return lambda message: None
    return print


def run_training(
    cfg: Config,
    run_dir: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    cfg.validate()
    log = _make_logger(cfg, log)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    if run_dir is None:
        name = cfg.name or cfg.default_name()
        run_dir = Path(cfg.output_dir) / f"{name}_{timestamp()}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", cfg.to_dict())

    log("=" * 78)
    log(cfg.describe())
    log(f"Ziarno: {cfg.seed} | urządzenie: {device} | katalog: {run_dir}")

    bundle = build_dataloaders(cfg)
    save_json(run_dir / "data_summary.json", bundle.summary())
    train_check = bundle.validation["train"]
    test_check = bundle.validation["test"]
    log(
        f"Dane: train={bundle.sizes['train']}, val={bundle.sizes['val']}, "
        f"test={bundle.sizes['test']} | normalizacja mean={bundle.mean}, std={bundle.std}"
    )
    log(
        f"Walidacja danych: obrazy {train_check['image_shape']}, piksele "
        f"[{train_check['pixel_min']}, {train_check['pixel_max']}], etykiety "
        f"[{train_check['label_min']}, {train_check['label_max']}], braki=0 "
        f"(train={train_check['samples']}, test={test_check['samples']})"
    )

    model = build_model(cfg).to(device)
    log(model_summary(model))

    history, best_state, best_epoch, best_val_f1 = fit(
        model, bundle.train_loader, bundle.val_loader, cfg, device, log=log
    )
    if best_state is None:
        best_state = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        best_epoch = len(history)
        best_val_f1 = 0.0
    model.load_state_dict(best_state)

    criterion = build_criterion(cfg)
    val_result = evaluate(model, bundle.val_loader, criterion, device)
    test_result = evaluate(model, bundle.test_loader, criterion, device)
    val_metrics = classification_metrics(
        val_result.targets, val_result.preds, num_classes=10, class_names=bundle.class_names
    )
    test_metrics = classification_metrics(
        test_result.targets, test_result.preds, num_classes=10, class_names=bundle.class_names
    )

    save_history_csv(run_dir / "history.csv", history)
    save_json(
        run_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "epochs_run": len(history),
            "training_seconds": float(sum(row["seconds"] for row in history)),
            "model": {
                "hidden_sizes": list(cfg.hidden_sizes),
                "activation": cfg.activation,
                "dropout": cfg.dropout,
                "batchnorm": cfg.batchnorm,
                "parameters": count_parameters(model),
            },
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    save_confusion_csv(
        run_dir / "confusion_matrix.csv",
        test_metrics["confusion_matrix"],
        bundle.class_names,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        test_result.targets,
        test_result.preds,
        test_result.probs,
    )
    plot_history(history, run_dir / "learning_curves.png")
    plot_confusion_matrix(
        np.asarray(test_metrics["confusion_matrix"]),
        bundle.class_names,
        run_dir / "confusion_matrix.png",
    )
    torch.save(
        {
            "state_dict": best_state,
            "config": cfg.to_dict(),
            "class_names": bundle.class_names,
            "normalization": {"mean": bundle.mean, "std": bundle.std},
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
        },
        run_dir / "best_model.pt",
    )

    total_seconds = float(sum(row["seconds"] for row in history))
    log(
        f"Najlepsza epoka: {best_epoch} (walidacyjne macro-F1={best_val_f1:.4f}) | "
        f"czas treningu: {format_seconds(total_seconds)}"
    )
    log(
        f"TEST: accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['macro_precision']:.4f}, "
        f"recall={test_metrics['macro_recall']:.4f}, "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    log(f"Artefakty zapisano w: {run_dir}")

    return {
        "run_dir": str(run_dir),
        "config": cfg.to_dict(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "epochs_run": len(history),
        "train_seconds": total_seconds,
        "val_loss": float(val_result.loss),
        "test_loss": float(test_result.loss),
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "model_parameters": count_parameters(model),
    }
