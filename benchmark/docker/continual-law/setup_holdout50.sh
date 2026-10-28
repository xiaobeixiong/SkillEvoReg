#!/usr/bin/env bash
set -euo pipefail
target="${1:?workspace required}"
mkdir -p "$target"
for file in controls.csv processing_records.csv nda-contract.txt merger_agreement.txt license_agreement.txt; do
  cp "/root/data/$file" "$target/$file"
done
