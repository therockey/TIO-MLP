"""Zapis artefaktów i wykresy (matplotlib w trybie bezokienkowym)."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


def save_table_csv(path: Path | str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(path: Path | str, history: Iterable[dict]) -> None:
    save_table_csv(path, history)


def save_confusion_csv(
    path: Path | str, matrix: list[list[int]] | np.ndarray, class_names: list[str]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def save_predictions_csv(
    path: Path | str,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confidence = probs.max(axis=1) if len(probs) else np.empty(0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "true", "pred", "confidence"])
        for index, (target, pred, conf) in enumerate(zip(targets, preds, confidence)):
            writer.writerow([index, int(target), int(pred), f"{float(conf):.6f}"])


def plot_history(history: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="trening")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="walidacja")
    axes[0].set_title("Strata")
    axes[0].set_xlabel("epoka")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="trening")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="walidacja")
    axes[1].set_title("Dokładność")
    axes[1].set_xlabel("epoka")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [row["val_macro_f1"] for row in history], marker="o", color="tab:green")
    axes[2].set_title("Walidacyjne macro-F1")
    axes[2].set_xlabel("epoka")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    path: Path | str,
    normalize: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    display = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    if not normalize:
        display = matrix

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    axis.set_title("Macierz pomyłek" + (" (znormalizowana)" if normalize else ""))

    threshold = display.max() * 0.55 if display.size else 0.0
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            axis.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format_pm(mean: float, std: float, percent: bool = True) -> str:
    if percent:
        return f"{mean * 100:.2f}% ± {std * 100:.2f} pp"
    return f"{mean:.4f} ± {std:.4f}"


def plot_suite_comparison(
    rows: list[dict],
    suite: str,
    path: Path | str,
    metric: str = "test_macro_f1",
) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row[f"{metric}_mean"], reverse=True)
    labels = [row["variant"] for row in ordered]
    means = np.array([row[f"{metric}_mean"] for row in ordered])
    stds = np.array([row[f"{metric}_std"] for row in ordered])

    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    axis.axhline(1.0 / 10.0, color="tab:red", linestyle="--", linewidth=1, label="losowo (10%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("macro-F1" if "f1" in metric else "dokładność")
    axis.set_title(f"Zestaw „{suite}”: {metric} na zbiorze testowym (średnia ± odch. std.)")
    axis.set_ylim(max(0.0, means.min() - 0.05), min(1.0, means.max() + 0.05))
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    for position, mean, std in zip(positions, means, stds):
        axis.text(position, mean + std + 0.004, f"{mean * 100:.2f}%", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _markdown_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Wariant | Konfiguracja | Accuracy (test) | Macro-F1 (test) | Czas / uruchomienie |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        configuration = (
            f"hidden={row['hidden_sizes']}, akt={row['activation']}, "
            f"opt={row['optimizer']}, lr={row['lr']:g}, bs={row['batch_size']}, "
            f"dropout={row['dropout']:g}, bn={row['batchnorm']}, aug={row['augment']}"
        )
        lines.append(
            "| {variant} | {configuration} | {accuracy} | {f1} | {seconds:.1f} s |".format(
                variant=row["variant"],
                configuration=configuration,
                accuracy=_format_pm(row["test_accuracy_mean"], row["test_accuracy_std"]),
                f1=_format_pm(row["test_macro_f1_mean"], row["test_macro_f1_std"]),
                seconds=row["train_seconds_mean"],
            )
        )
    return lines


def write_report(
    path: Path | str,
    suites: list[str],
    summary_rows: list[dict],
    cfg: Config,
    run_root: Path | str,
    repeats: int,
) -> None:
    """Tworzy raport Markdown z tabelami, wykresami i automatycznymi obserwacjami."""
    path = Path(path)
    run_root = Path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Raport z eksperymentów — MNIST MLP",
        "",
        "## Konfiguracja bazowa",
        "",
        f"- {cfg.describe()}",
        f"- podział: walidacja={cfg.val_size}, ziarno podziału={cfg.split_seed}",
        f"- liczba powtórzeń na wariant: {repeats}",
        "",
        "Model jest wybierany na podstawie walidacyjnego macro-F1, a wszystkie",
        "podane niżej metryki są liczone na **nietkniętym zbiorze testowym**.",
        "Wartości to średnia ± odchylenie standardowe po powtórzeniach.",
        "",
        "## Zestawy eksperymentów",
        "",
    ]

    for suite in suites:
        rows = [row for row in summary_rows if row["suite"] == suite]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["test_macro_f1_mean"], reverse=True)
        lines.append(f"### {suite}")
        lines.append("")
        lines.extend(_markdown_table(rows))
        lines.append("")
        plot_name = f"comparison_{suite}.png"
        if (run_root / plot_name).exists():
            lines.append(f"![Porównanie {suite}]({plot_name})")
            lines.append("")
        best = rows[0]
        worst = rows[-1]
        spread = (best["test_macro_f1_mean"] - worst["test_macro_f1_mean"]) * 100
        lines.append(
            f"Najlepszy wariant: **{best['variant']}** "
            f"(macro-F1 {_format_pm(best['test_macro_f1_mean'], best['test_macro_f1_std'])}). "
            f"Różnica macro-F1 między najlepszym i najgorszym wariantem: "
            f"**{spread:.2f} pp**."
        )
        lines.append("")

    if summary_rows:
        best_overall = max(summary_rows, key=lambda row: row["test_macro_f1_mean"])
        lines.extend(
            [
                "## Wnioski automatyczne",
                "",
                (
                    f"- Najlepszy wynik w całym badaniu: **{best_overall['suite']} / "
                    f"{best_overall['variant']}** — accuracy "
                    f"{_format_pm(best_overall['test_accuracy_mean'], best_overall['test_accuracy_std'])}, "
                    f"macro-F1 {_format_pm(best_overall['test_macro_f1_mean'], best_overall['test_macro_f1_std'])}."
                ),
                (
                    "- Różnice mniejsze niż około 0,2–0,3 pp należy traktować ostrożnie; "
                    "przy pojedynczym ziarnie mogą wynikać z wariancji inicjalizacji i kolejności danych."
                ),
                (
                    "- Pełne, szczegółowe metryki (precision/recall/F1 per klasa) oraz macierze "
                    "pomyłek znajdują się w katalogach poszczególnych uruchomień (`metrics.json`)."
                ),
                "",
                "## Pliki",
                "",
                "- `runs.csv` — wyniki każdego uruchomienia (każde ziarno osobno),",
                "- `summary.csv` — agregacja średnia ± odchylenie standardowe,",
                "- `comparison_*.png` — wykresy porównawcze,",
                "- `*/**/metrics.json` — metryki, macierz pomyłek i informacje o modelu.",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
