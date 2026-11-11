#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for v in 1 2; do
  python src/train.py --var "$v"
  python src/predict.py --var "$v"
done
python src/make_figures.py
