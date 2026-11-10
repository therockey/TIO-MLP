"""Zestawy eksperymentów i agregacja wyników.

Każdy wariant zmienia dokładnie jeden element względem konfiguracji bazowej,
dzięki czemu można przypisać obserwowane różnice do konkretnego hiperparametru.
Każdy wariant jest powtarzany z różnymi ziarnami inicjalizacji i kolejności danych
(ziarno podziału danych pozostaje stałe).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import run_training
from .reporting import (
    plot_suite_comparison,
    save_table_csv,
    write_report,
)
from .utils import format_seconds, timestamp


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict


def _architecture() -> list[Variant]:
    return [
        Variant("64", {"hidden_sizes": (64,)}),
        Variant("128", {"hidden_sizes": (128,)}),
        Variant("256-128", {"hidden_sizes": (256, 128)}),
        Variant("512-256", {"hidden_sizes": (512, 256)}),
        Variant("256-128-64", {"hidden_sizes": (256, 128, 64)}),
    ]


def _activation() -> list[Variant]:
    return [
        Variant("relu", {"activation": "relu"}),
        Variant("leaky_relu", {"activation": "leaky_relu"}),
        Variant("gelu", {"activation": "gelu"}),
        Variant("tanh", {"activation": "tanh"}),
        Variant("sigmoid", {"activation": "sigmoid"}),
    ]


def _optimizer() -> list[Variant]:
    return [
        Variant("adam", {"optimizer": "adam", "lr": 1e-3}),
        Variant("adamw", {"optimizer": "adamw", "lr": 1e-3}),
        Variant(
            "sgd_momentum_lr0.05",
            {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9},
        ),
    ]


def _learning_rate() -> list[Variant]:
    return [
        Variant("lr1e-4", {"lr": 1e-4}),
        Variant("lr3e-4", {"lr": 3e-4}),
        Variant("lr1e-3", {"lr": 1e-3}),
        Variant("lr3e-3", {"lr": 3e-3}),
        Variant("lr1e-2", {"lr": 1e-2}),
    ]


def _batch_size() -> list[Variant]:
    return [
        Variant("bs32", {"batch_size": 32}),
        Variant("bs64", {"batch_size": 64}),
        Variant("bs128", {"batch_size": 128}),
        Variant("bs256", {"batch_size": 256}),
    ]


def _augmentation() -> list[Variant]:
    return [
        Variant("none", {"augment": "none"}),
        Variant("affine", {"augment": "affine"}),
        Variant("rotation", {"augment": "rotation"}),
    ]


def _dropout() -> list[Variant]:
    return [
        Variant("do0.0", {"dropout": 0.0}),
        Variant("do0.2", {"dropout": 0.2}),
        Variant("do0.5", {"dropout": 0.5}),
    ]


def _batchnorm() -> list[Variant]:
    return [
        Variant("off", {"batchnorm": False}),
        Variant("on", {"batchnorm": True}),
    ]


def _seeds() -> list[Variant]:
    return [Variant("baseline", {})]


SUITES: dict[str, Callable[[], list[Variant]]] = {
    "architecture": _architecture,
    "activation": _activation,
    "optimizer": _optimizer,
    "learning_rate": _learning_rate,
    "batch_size": _batch_size,
    "augmentation": _augmentation,
    "dropout": _dropout,
    "batchnorm": _batchnorm,
    "seeds": _seeds,
}
SUITE_NAMES = tuple(SUITES)
ALL_SUITES = tuple(name for name in SUITES if name != "seeds")

AGGREGATED_METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_loss",
    "val_accuracy",
    "val_macro_f1",
    "train_seconds",
    "best_epoch",
)


def _expand_suites(suites: list[str]) -> list[str]:
    expanded: list[str] = []
    for suite in suites:
        if suite == "all":
            expanded.extend(ALL_SUITES)
        elif suite in SUITES:
            expanded.append(suite)
        else:
            raise ValueError(f"Nieznany zestaw: {suite!r}. Dostępne: {SUITE_NAMES} + all")
    # usuń duplikaty, zachowując kolejność
    return list(dict.fromkeys(expanded))


def _run_row(suite: str, variant: Variant, seed: int, result: dict, cfg: Config) -> dict:
    test = result["test"]
    val = result["val"]
    return {
        "suite": suite,
        "variant": variant.name,
        "seed": seed,
        "hidden_sizes": "-".join(str(size) for size in cfg.hidden_sizes),
        "activation": cfg.activation,
        "dropout": cfg.dropout,
        "batchnorm": cfg.batchnorm,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "augment": cfg.augment,
        "scheduler": cfg.scheduler,
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "train_seconds": result["train_seconds"],
        "val_loss": result["val_loss"],
        "val_accuracy": val["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "test_loss": result["test_loss"],
        "test_accuracy": test["accuracy"],
        "test_macro_precision": test["macro_precision"],
        "test_macro_recall": test["macro_recall"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "run_dir": result["run_dir"],
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["variant"]), []).append(row)

    summary: list[dict] = []
    for (suite, variant), group in grouped.items():
        first = group[0]
        entry = {
            "suite": suite,
            "variant": variant,
            "runs": len(group),
            "hidden_sizes": first["hidden_sizes"],
            "activation": first["activation"],
            "dropout": first["dropout"],
            "batchnorm": first["batchnorm"],
            "optimizer": first["optimizer"],
            "lr": first["lr"],
            "batch_size": first["batch_size"],
            "augment": first["augment"],
        }
        for metric in AGGREGATED_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(entry)
    return summary


def _print_summary(summary: list[dict]) -> None:
    print("\nPodsumowanie (test, średnia ± odch. std.):")
    print(f"{'zestaw':<16} {'wariant':<18} {'accuracy':>20} {'macro-F1':>20}")
    print("-" * 78)
    for suite in dict.fromkeys(row["suite"] for row in summary):
        rows = sorted(
            [row for row in summary if row["suite"] == suite],
            key=lambda row: row["test_macro_f1_mean"],
            reverse=True,
        )
        for row in rows:
            print(
                f"{row['suite']:<16} {row['variant']:<18} "
                f"{row['test_accuracy_mean'] * 100:>7.2f}% ± {row['test_accuracy_std'] * 100:>4.2f} "
                f"{row['test_macro_f1_mean'] * 100:>7.2f}% ± {row['test_macro_f1_std'] * 100:>4.2f}"
            )


def run_suite(
    cfg: Config,
    suites: list[str],
    repeats: int = 3,
    experiment_name: str = "",
    log: Callable[[str], None] | None = None,
) -> Path:
    """Uruchamia wybrane zestawy eksperymentów i zapisuje zbiorczy raport."""
    cfg.validate()
    if repeats < 1:
        raise ValueError("Liczba powtórzeń musi być >= 1")
    suites = _expand_suites(suites)
    if log is None:
        log = (lambda message: None) if cfg.quiet else print

    base_name = experiment_name or "_".join(suites)
    root = Path(cfg.output_dir) / "experiments" / f"{base_name}_{timestamp()}"
    root.mkdir(parents=True, exist_ok=True)
    log(f"Katalog eksperymentów: {root}")
    log(f"Zestawy: {', '.join(suites)} | powtórzenia: {repeats}")

    rows: list[dict] = []
    total_runs = sum(len(SUITES[suite]()) for suite in suites) * repeats
    run_index = 0

    for suite in suites:
        for variant in SUITES[suite]():
            for repeat in range(repeats):
                run_index += 1
                seed = cfg.seed + repeat * 1000
                run_cfg = cfg.replace(
                    seed=seed,
                    name=f"{suite}__{variant.name}__seed{seed}",
                    **variant.overrides,
                )
                run_dir = root / suite / variant.name / f"seed_{seed}"
                log(
                    f"\n[{run_index}/{total_runs}] {suite} / {variant.name} "
                    f"| powtórzenie {repeat + 1}/{repeats} | seed={seed}"
                )
                result = run_training(run_cfg, run_dir=run_dir, log=log)
                rows.append(_run_row(suite, variant, seed, result, run_cfg))
                # zapis przyrostowy: częściowe wyniki przetrwają przerwanie długiej serii
                save_table_csv(root / "runs.csv", rows)

    save_table_csv(root / "runs.csv", rows)
    summary = _aggregate(rows)
    save_table_csv(root / "summary.csv", summary)

    for suite in suites:
        plot_suite_comparison(
            [row for row in summary if row["suite"] == suite],
            suite,
            root / f"comparison_{suite}.png",
        )

    write_report(root / "report.md", suites, summary, cfg, root, repeats)
    _print_summary(summary)
    log(f"\nŁączny czas treningu: {format_seconds(sum(row['train_seconds'] for row in rows))}")
    log(f"Raport: {root / 'report.md'}")
    return root
