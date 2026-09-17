#!/usr/bin/env bash
# Severstal Steel Defect Detection veri setini Kaggle'dan indirir.
#
# On kosul: ~/.kaggle/kaggle.json (Kaggle hesap ayarlarindan API token) mevcut olmali
# ve o yarismaya katilim kurallari kabul edilmis olmali (kaggle.com/c/severstal-steel-defect-detection/rules).
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DATA_DIR"

kaggle competitions download -c severstal-steel-defect-detection -p "$DATA_DIR"
unzip -oq "$DATA_DIR/severstal-steel-defect-detection.zip" -d "$DATA_DIR"
rm "$DATA_DIR/severstal-steel-defect-detection.zip"

echo "Veri hazir: $DATA_DIR"
ls "$DATA_DIR"
