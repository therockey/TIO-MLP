"""Definicja interfejsu wiersza poleceń (argparse)."""

from __future__ import annotations

import argparse
from dataclasses import fields

from .config import ACTIVATIONS, AUGMENTATIONS, OPTIMIZERS, SCHEDULERS, Config
from .experiments import SUITE_NAMES


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("dane")
    data.add_argument("--data-dir", default=Config.data_dir, help="katalog z MNIST")
    data.add_argument(
        "--val-size",
        type=int,
        default=Config.val_size,
        help="rozmiar zbioru walidacyjnego wydzielonego z części treningowej",
    )
    data.add_argument("--batch-size", type=int, default=Config.batch_size)
    data.add_argument("--num-workers", type=int, default=Config.num_workers)
    data.add_argument(
        "--augment",
        choices=AUGMENTATIONS,
        default=Config.augment,
        help="augmentacja tylko zbioru treningowego",
    )
    data.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="0 = pełny zbiór; >0 losowy podzbiór (tylko do szybkich testów)",
    )
    data.add_argument(
        "--test-subset",
        type=int,
        default=0,
        help="0 = pełny zbiór testowy; >0 losowy podzbiór (tylko do szybkich testów)",
    )
    data.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        default=True,
        help="nie pobieraj MNIST, jeśli brakuje plików",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=list(Config.hidden_sizes),
        help="liczba neuronów w kolejnych warstwach ukrytych, np. 256 128",
    )
    model.add_argument("--activation", choices=ACTIVATIONS, default=Config.activation)
    model.add_argument("--dropout", type=float, default=Config.dropout)
    model.add_argument(
        "--batchnorm",
        action="store_true",
        default=Config.batchnorm,
        help="włącz BatchNorm po każdej warstwie liniowej",
    )

    optimization = parser.add_argument_group("optymalizacja")
    optimization.add_argument("--optimizer", choices=OPTIMIZERS, default=Config.optimizer)
    optimization.add_argument("--lr", type=float, default=Config.lr)
    optimization.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    optimization.add_argument("--momentum", type=float, default=Config.momentum)
    optimization.add_argument("--scheduler", choices=SCHEDULERS, default=Config.scheduler)
    optimization.add_argument("--label-smoothing", type=float, default=Config.label_smoothing)
    optimization.add_argument("--grad-clip", type=float, default=Config.grad_clip)

    training = parser.add_argument_group("uczenie")
    training.add_argument("--epochs", type=int, default=Config.epochs)
    training.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help="wczesne zatrzymywanie po tylu epokach bez poprawy walidacyjnego F1; 0 = wyłączone",
    )
    training.add_argument("--seed", type=int, default=Config.seed)
    training.add_argument(
        "--split-seed",
        type=int,
        default=Config.split_seed,
        help="ziarno podziału train/val; stałe w eksperymentach",
    )
    training.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default=Config.device
    )
    training.add_argument("--output-dir", default=Config.output_dir)
    training.add_argument("--quiet", action="store_true", help="bez pasków postępu i logów")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="MLP dla MNIST (zadanie 4) — trening i eksperymenty porównawcze.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przykłady:\n"
            "  python run.py train --epochs 10\n"
            "  python run.py train --hidden-sizes 512 256 --dropout 0.2 --batchnorm\n"
            "  python run.py experiment --suite architecture activation --repeats 3\n"
            "  python run.py experiment --suite all --repeats 2 --epochs 15\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="pojedyncze uruchomienie treningu"
    )
    _add_common_arguments(train_parser)
    train_parser.add_argument(
        "--name", default="", help="nazwa katalogu wyników (domyślnie z konfiguracji)"
    )

    experiment_parser = subparsers.add_parser(
        "experiment", help="seria eksperymentów i raport porównawczy"
    )
    _add_common_arguments(experiment_parser)
    experiment_parser.add_argument(
        "--suite",
        nargs="+",
        choices=(*SUITE_NAMES, "all"),
        default=["architecture"],
        help="zestawy eksperymentów; 'all' uruchamia wszystkie poza 'seeds'",
    )
    experiment_parser.add_argument(
        "--repeats", type=int, default=3, help="liczba ziaren na każdy wariant"
    )
    experiment_parser.add_argument(
        "--experiment-name",
        default="",
        help="nazwa katalogu eksperymentów (domyślnie nazwy zestawów)",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    values = {
        field.name: getattr(args, field.name)
        for field in fields(Config)
        if hasattr(args, field.name)
    }
    return Config.from_dict(values)
