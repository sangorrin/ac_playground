#!/usr/bin/env bash
# Usage: bash resample_to_16k.sh [out_tmp LJS_accented_16K]
# This keeps GPU time just for VC; resampling is done locally.
set -euo pipefail
IN_DIR="${1:-out_tmp}"           # folder with WAVs 24k
OUT_DIR="${2:-LJS_accented_16K}" # output 16k mono
mkdir -p "$OUT_DIR"

for f in "$IN_DIR"/*.wav; do
  base="$(basename "$f")"
  ffmpeg -nostdin -y -i "$f" -ac 1 -ar 16000 "$OUT_DIR/$base"
done
