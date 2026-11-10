#!/usr/bin/env python3
"""Punkt wejściowy programu.

Przykłady:
    python run.py train --epochs 10
    python run.py train --hidden-sizes 512 256 --dropout 0.2 --batchnorm
    python run.py experiment --suite architecture activation --repeats 3
    python run.py experiment --suite all --repeats 2 --epochs 15
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tio_mlp.cli import build_parser, config_from_args
from tio_mlp.experiments import run_suite
from tio_mlp.pipeline import run_training


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    try:
        if args.command == "train":
            run_training(cfg)
        elif args.command == "experiment":
            root = run_suite(
                cfg,
                suites=args.suite,
                repeats=args.repeats,
                experiment_name=args.experiment_name,
            )
            print(f"\nWszystkie wyniki i raport: {root}")
        else:  # pragma: no cover - argparse wymusza poprawną komendę
            parser.error(f"Nieznana komenda: {args.command}")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Błąd: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
