#!/usr/bin/env bash
set -euo pipefail
target="${1:?workspace required}"
mkdir -p "$target"
for file in transactions.csv lease_agreement.txt emissions_data.csv labor_requirements.csv employment_contract.txt system_config.json; do
  cp "/root/data/$file" "$target/$file"
done
