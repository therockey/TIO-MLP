#!/usr/bin/env bash
# Pobiera oryginalne pliki MNIST (format IDX) do data/MNIST/raw.
# Użycie:
#   ./scripts/download_data.sh
#   ./scripts/download_data.sh /inny/katalog/data
set -euo pipefail

usage() {
    cat <<'EOF'
Uzycie: ./scripts/download_data.sh [KATALOG_DANYCH]

Bez argumentu dane trafiaja do <repo>/data.
Skrypt pobiera pliki IDX, weryfikuje MD5 i rozpakowuje je do
<KATALOG_DANYCH>/MNIST/raw, czyli dokladnie tam, gdzie szuka ich torchvision.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data}"
RAW_DIR="$DATA_DIR/MNIST/raw"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

MIRRORS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
    "http://yann.lecun.com/exdb/mnist"
)

md5_for() {
    case "$1" in
        train-images-idx3-ubyte.gz) echo "f68b3c2dcbeaaa9fbdd348bbdeb94873" ;;
        train-labels-idx1-ubyte.gz) echo "d53e105ee54ea40749a09fcbcd1e9432" ;;
        t10k-images-idx3-ubyte.gz)  echo "9fb629c4189551a2d022fa330f9573f3" ;;
        t10k-labels-idx1-ubyte.gz)  echo "ec29112dd5afa0611ce80d1b7f02629c" ;;
        *) echo "" ;;
    esac
}

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        echo "Blad: brak md5sum/md5 do weryfikacji plikow." >&2
        exit 1
    fi
}

download_file() {
    local file="$1"
    local target="$RAW_DIR/$file"
    local partial="$target.part"
    local url

    for mirror in "${MIRRORS[@]}"; do
        url="$mirror/$file"
        rm -f "$partial"
        echo "  probuje: $url"
        if command -v curl >/dev/null 2>&1; then
            if curl -fsSL --retry 3 --connect-timeout 20 -o "$partial" "$url"; then
                mv "$partial" "$target"
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q --tries=3 --timeout=20 -O "$partial" "$url"; then
                mv "$partial" "$target"
                return 0
            fi
        else
            echo "Blad: potrzebny jest curl albo wget." >&2
            exit 1
        fi
        rm -f "$partial"
    done
    return 1
}

if ! command -v gzip >/dev/null 2>&1; then
    echo "Blad: potrzebny jest program gzip do rozpakowania danych." >&2
    exit 1
fi

mkdir -p "$RAW_DIR"
echo "Katalog danych: $RAW_DIR"

for file in "${FILES[@]}"; do
    gz="$RAW_DIR/$file"
    out="${gz%.gz}"
    expected="$(md5_for "$file")"

    if [[ -f "$gz" ]] && [[ "$(md5_of "$gz")" == "$expected" ]]; then
        echo "[OK] $file"
    else
        echo "[POBIERANIE] $file"
        rm -f "$gz"
        if ! download_file "$file"; then
            echo "Blad: nie udalo sie pobrac $file z zadnego mirrora." >&2
            exit 1
        fi
        actual="$(md5_of "$gz")"
        if [[ "$actual" != "$expected" ]]; then
            echo "Blad: zly MD5 dla $file (oczekiwano $expected, jest $actual)." >&2
            exit 1
        fi
        echo "[OK] $file (MD5)"
    fi

    if [[ ! -f "$out" ]]; then
        echo "[ROZPAKOWANIE] $file"
        if ! gzip -dc "$gz" > "$out"; then
            rm -f "$out"
            echo "Blad: nie udalo sie rozpakowac $file." >&2
            exit 1
        fi
    fi
done

echo
echo "Gotowe. Pliki MNIST sa w: $RAW_DIR"
echo "Mozesz uruchomic: python run.py train --epochs 10"
