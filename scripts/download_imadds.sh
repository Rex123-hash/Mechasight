#!/usr/bin/env bash
# Download the official IMAD-DS archives from Zenodo record 12665499.
# Provenance (URLs, sizes, official MD5s) is recorded in
# reports/dataset_audit/download_provenance.json by scripts/verify_download.py.
set -euo pipefail
DEST="${1:-data/raw}"
BASE="https://zenodo.org/api/records/12665499/files"
mkdir -p "$DEST"
for f in BrushlessMotor.7z RoboticArm.7z; do
  echo "[$(date -u +%FT%TZ)] downloading $f"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$DEST/$f" "$BASE/$f/content"
  echo "[$(date -u +%FT%TZ)] done $f  ($(stat -c%s "$DEST/$f") bytes)"
done
