"""Metryki klasyfikacji wieloklasowej.

Implementacja jest celowo niezależna od scikit-learn: macierz pomyłek oraz
precyzja, czułość i F1 są liczone bezpośrednio z definicji. Dla MNIST wszystkie
10 klas występuje w zbiorze testowym, ale kod obsługuje też klasy nieobecne
(wtedy ich metryki wynoszą 0, a średnia macro jest liczona po klasach z support > 0).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Zwraca macierz pomyłek, wiersze = prawdziwa klasa, kolumny = predykcja."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true i y_pred muszą mieć ten sam kształt")

    valid = (
        (y_true >= 0)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true[valid], y_pred[valid]), 1)
    return matrix


def classification_metrics(
    y_true,
    y_pred,
    num_classes: int = 10,
    class_names: list[str] | None = None,
) -> dict:
    """Liczy accuracy, precyzję, czułość, F1 (macro i ważoną) oraz macierz pomyłek."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true i y_pred muszą mieć ten sam kształt")

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    accuracy = float(tp.sum() / total) if total else 0.0

    present = support > 0
    macro_precision = float(precision[present].mean()) if present.any() else 0.0
    macro_recall = float(recall[present].mean()) if present.any() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0

    if total:
        weighted_precision = float(np.sum(precision * support) / total)
        weighted_recall = float(np.sum(recall * support) / total)
        weighted_f1 = float(np.sum(f1 * support) / total)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    names = class_names or [str(index) for index in range(num_classes)]
    per_class = {}
    for index, name in enumerate(names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "num_samples": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
