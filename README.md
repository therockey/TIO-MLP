# TIO-MLP — klasyfikacja cyfr MNIST

Implementacja zadania 4 z przedmiotu *Techniki inteligencji obliczeniowej*:
perceptron wielowarstwowy (MLP) w PyTorch do klasyfikacji cyfr MNIST.

Projekt jest przygotowany tak, aby w zadaniu 5 można było łatwo zwiększyć
głębokość sieci oraz włączyć `dropout` i `batch normalization` — te opcje są
już dostępne w CLI.

## Struktura projektu

```
run.py                  # punkt wejściowy, argparse i dwie komendy: train / experiment
tio_mlp/
  cli.py                # definicja argumentów wiersza poleceń
  config.py             # wszystkie hiperparametry + walidacja
  data.py               # MNIST, podział stratyfikowany, normalizacja, augmentacja
  model.py              # implementacja MLP w PyTorch
  engine.py             # pętla treningu, walidacja, optymalizator, scheduler
  metrics.py            # macierz pomyłek, accuracy, precision, recall, F1 (bez sklearn)
  pipeline.py           # pojedynczy trening: dane -> model -> metryki -> artefakty
  experiments.py        # zestawy eksperymentów, agregacja, raport
  reporting.py          # CSV, JSON, wykresy i raport Markdown
  utils.py              # ziarna, urządzenie, zapis JSON
scripts/
  download_data.sh      # pobieranie MNIST na Linux/macOS
  download_data.ps1     # pobieranie MNIST w PowerShell
tests/                  # szybkie testy jednostkowe metryk, modelu i konfiguracji
```

## Instalacja

Wymagany Python 3.10+ (projekt był testowany na Pythonie 3.14).

```bash
python -m venv .venv
source .venv/bin/activate

# CPU-only (zalecane, jeśli nie ma CUDA):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# pozostałe zależności
pip install -r requirements.txt
```

MNIST (ok. 11 MB) jest pobierany automatycznie do `data/` przy pierwszym
uruchomieniu.

## Pobieranie danych bez Pythona

Katalog `data/` jest w `.gitignore`, więc po sklonowaniu repo dane trzeba pobrać
osobno. Program robi to automatycznie, ale do przygotowania danych przed
instalacją Pythona (albo do pracy offline) służą proste skrypty:

```bash
# Linux / macOS
./scripts/download_data.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File .\scripts\download_data.ps1
```

Można też wskazać inny katalog:

```bash
./scripts/download_data.sh /tmp/mnist
powershell -ExecutionPolicy Bypass -File .\scripts\download_data.ps1 -DataDir C:\mnist
```

Skrypty pobierają oryginalne pliki IDX z mirrorów PyTorch, weryfikują sumy MD5
i rozpakowują je do `<katalog>/MNIST/raw` — dokładnie tam, gdzie szuka ich
`torchvision`. Są idempotentne: przy ponownym uruchomieniu sprawdzają istniejące
pliki zamiast pobierać je od nowa. Wymagania: `curl` lub `wget` oraz `gzip`
(Linux/macOS); PowerShell korzysta z wbudowanych mechanizmów systemu.

### Pochodzenie danych

`torchvision.datasets.MNIST` pobiera oryginalne pliki IDX z mirrora
`https://ossci-datasets.s3.amazonaws.com/mnist/` i weryfikuje ich sumy MD5.
W `data/MNIST/raw/` znajdują się więc te same obrazy i etykiety, które są
źródłem datasetu `oddrationale/mnist-in-csv` z Kaggle. CSV z Kaggle to tylko
inna forma zapisu: `train.csv` = 60 000 wierszy, `test.csv` = 10 000 wierszy,
kolumny `label` + 784 pikseli (0–255), a `pixel0 … pixel783` odpowiada
`image.flatten()` w kolejności wiersz-po-wierszu. Nasz kod dzieli oficjalne
60 000 na 55 000 treningowych + 5 000 walidacyjnych; oficjalne 10 000 testowych
pozostaje nietknięte. Normalizacja jest już preprocessingiem w `data.py`, a nie
częścią plików źródłowych.

## Szybki start

```bash
# jedno uruchomienie, domyślna architektura 784 -> 256 -> 128 -> 10
python run.py train --epochs 10

# szybszy test bez pełnego zbioru (do sprawdzenia, że wszystko działa)
python run.py train --epochs 2 --train-subset 4000 --val-size 1000 --test-subset 1000
```

Wyniki trafiają do `runs/<nazwa>_<data>/`. Najważniejsze pliki:

- `config.json` — pełna konfiguracja uruchomienia,
- `history.csv` — strata i metryki w każdej epoce,
- `metrics.json` — metryki walidacyjne i testowe, w tym per klasa,
- `confusion_matrix.csv` / `confusion_matrix.png` — macierz pomyłek,
- `learning_curves.png` — krzywe uczenia,
- `test_predictions.csv` — predykcje, prawdziwe etykiety i pewność modelu,
- `best_model.pt` — wagi najlepszej epoki (wybranej po walidacyjnym macro-F1).

## Eksperymenty

```bash
# porównanie architektur i funkcji aktywacji, 3 ziarna na wariant
python run.py experiment --suite architecture activation --repeats 3 --epochs 10

# pojedyncze zestawy
python run.py experiment --suite learning_rate --repeats 3
python run.py experiment --suite batch_size --repeats 3
python run.py experiment --suite augmentation --repeats 3

# wszystko naraz (długo!)
python run.py experiment --suite all --repeats 3 --epochs 15
```

Dostępne zestawy:

| Zestaw | Co porównuje |
|---|---|
| `architecture` | `64`, `128`, `256-128`, `512-256`, `256-128-64` |
| `activation` | `relu`, `leaky_relu`, `gelu`, `tanh`, `sigmoid` |
| `optimizer` | Adam, AdamW, SGD z momentum |
| `learning_rate` | `1e-4` … `1e-2` |
| `batch_size` | 32, 64, 128, 256 |
| `augmentation` | brak, `RandomAffine`, `RandomRotation` |
| `dropout` | 0.0, 0.2, 0.5 (przydatne w zadaniu 5) |
| `batchnorm` | bez / z BatchNorm (przydatne w zadaniu 5) |
| `seeds` | ten sam model na wielu ziarnach — ocena stabilności |
| `all` | wszystkie powyższe poza `seeds` |

Każdy wariant zmienia **jeden** element względem konfiguracji bazowej. Wyjątkiem
jest `optimizer / sgd_momentum_lr0.05`, gdzie oprócz optymalizatora ustawiono
`lr=0.05` — przy `lr=1e-3` SGD uczy się zbyt wolno, więc porównanie z Adamem
byłoby mylące. Dla każdego wariantu uruchamiane jest `--repeats` treningów z
różnymi ziarnami (`seed`, `seed + 1000`, …). Ziarno podziału danych
(`--split-seed`) jest stałe, więc wszystkie warianty widzą dokładnie ten sam
podział train/val/test.

W katalogu `runs/experiments/<nazwa>_<data>/` powstają:

- `runs.csv` — wynik każdego pojedynczego uruchomienia,
- `summary.csv` — średnia ± odchylenie standardowe dla każdego wariantu,
- `report.md` — raport z tabelami, wykresami i automatycznymi wnioskami,
- `comparison_<zestaw>.png` — wykres słupkowy z błędami,
- podkatalogi `/<zestaw>/<wariant>/seed_<seed>/` z pełnymi artefaktami.

## Przykładowy wynik

Na maszynie testowej (CPU, 6 rdzeni) domyślna architektura `784 -> 256 -> 128 -> 10`
po 5 epokach osiągnęła na zbiorze testowym accuracy **97.80%** i macro-F1
**97.79%** (około 10 s na epokę).

W eksperymencie `architecture` (2 ziarna, 5 epok) uzyskano m.in.:

| Architektura | Accuracy (test) | Macro-F1 (test) |
|---|---|---|
| `256-128` | 97.59% ± 0.21 pp | 97.57% ± 0.21 pp |
| `128` | 97.53% ± 0.05 pp | 97.51% ± 0.05 pp |
| `256-128-64` | 97.52% ± 0.09 pp | 97.50% ± 0.09 pp |
| `512-256` | 97.51% ± 0.12 pp | 97.49% ± 0.12 pp |
| `64` | 96.98% ± 0.00 pp | 96.96% ± 0.00 pp |

Widać malejące korzyści z rozbudowy sieci: pojedyncza warstwa 64 neuronów jest
wyraźnie słabsza, ale powyżej `256-128` różnice są niewielkie. Pełny trening
10–15 epok zwykle daje około 98% accuracy. Wyniki mogą się różnić o kilka
dziesiątych punktu procentowego w zależności od ziarna i sprzętu.

## Metodologia

- **Kontrola danych:** sprawdzany jest kształt obrazów, zakres pikseli (0–255),
  zakres etykiet (0–9) oraz brak braków danych; raport trafia do
  `data_summary.json`.
- **Podział:** oficjalne 60 000 obrazów treningowych dzielimy stratyfikowanie na
  trening i walidację (`--val-size`, domyślnie 5 000). Oficjalne 10 000 obrazów
  testowych pozostaje nietknięte do końcowej oceny.
- **Normalizacja:** średnia i odchylenie standardowe są liczone **tylko** na
  części treningowej i zapisywane w `data/mnist_stats.json`. Nie ma wycieku
  informacji z walidacji ani testu.
- **Augmentacja:** opcjonalna (`--augment affine|rotation`), stosowana wyłącznie
  do zbioru treningowego. Domyślnie wyłączona.
- **Model:** `Flatten -> Linear -> [BatchNorm] -> aktywacja -> [Dropout] -> ... -> Linear(10)`.
  Inicjalizacja Kaiming (ReLU/GELU) lub Xavier (tanh/sigmoid).
- **Uczenie:** `CrossEntropyLoss`, domyślnie Adam; opcjonalnie SGD/AdamW,
  scheduler `plateau`/`cosine`, early stopping po walidacyjnym macro-F1.
- **Wybór modelu:** najlepsza epoka według walidacyjnego macro-F1, a nie
  według zbioru testowego.
- **Metryki:** accuracy, precyzja, czułość i F1 (macro oraz ważone), macierz
  pomyłek, pełne metryki per klasa. Implementacja w `metrics.py` nie korzysta
  ze scikit-learn.

Zgodnie z ustaleniem prowadzącego laboratorium PyTorch jest używany do samej
implementacji MLP i automatycznego różniczkowania (zadania 4–5). Przygotowanie
danych, podział, metryki i analiza wyników są zaimplementowane w projekcie
samodzielnie, bez gotowych modeli ani pipeline'ów wysokiego poziomu.

## Reprodukcja i urządzenie

- `--seed` kontroluje inicjalizację wag i kolejność danych,
- `--split-seed` kontroluje podział danych,
- `--device auto|cpu|cuda|mps` wybiera urządzenie,
- na CPU z 6 rdzeniami jedna epoka dla `256-128` trwa zwykle kilka–kilkanaście
  sekund; pełny zestaw `all` z 3 powtórzeniami może zająć godzinę lub więcej.

## Testy

```bash
python -m unittest discover -s tests -v
```

## Zadanie 5

Do porównania MLP z głębszą siecią wystarczy zwiększyć `--hidden-sizes` oraz
włączyć `--dropout` i `--batchnorm`, np.:

```bash
python run.py train --hidden-sizes 512 256 128 64 --dropout 0.3 --batchnorm --epochs 20
python run.py experiment --suite dropout batchnorm --repeats 3 --epochs 20
```

Warto wtedy uruchomić również `--suite architecture` na tej samej konfiguracji
bazowej, aby porównać płytki i głęboki model w identycznych warunkach.
