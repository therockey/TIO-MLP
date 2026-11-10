"""Pomocnicze funkcje: ziarna losowości, urządzenie, zapis plików."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Ustawia ziarna we wszystkich używanych bibliotekach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Zamienia ``auto``/``cpu``/``cuda``/``cuda:N``/``mps`` na ``torch.device``."""
    device = device.strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Wybrano --device cuda, ale CUDA nie jest dostępna. "
                "Sprawdź, czy zainstalowano wersję PyTorch z CUDA: "
                "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
            )
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Niepoprawna nazwa urządzenia: {device!r}") from error
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Wybrano {device}, ale dostępne są tylko "
                f"{torch.cuda.device_count()} urządzenia CUDA."
            )
        return resolved

    if device == "mps":
        if not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Wybrano --device mps, ale MPS nie jest dostępny.")
        return torch.device("mps")

    if device != "cpu":
        raise RuntimeError(
            f"Nieznane urządzenie: {device!r}. Użyj auto, cpu, cuda, cuda:N lub mps."
        )
    return torch.device("cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(text: str) -> str:
    """Proste, bezpieczne nazwy katalogów (bez polskich znaków i spacji)."""
    replacements = {
        "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
        "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "run"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.0f} s"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Nie można zapisać do JSON obiektu {type(value)!r}")


def save_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
