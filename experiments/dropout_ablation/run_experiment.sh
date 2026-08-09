#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e90ml-mpl}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pyml

python -u src/train.py -c experiments/dropout_ablation/config_dropout_off.yaml
python -u src/test.py -c experiments/dropout_ablation/config_dropout_off.yaml
python -u experiments/dropout_ablation/plot_comparison.py \
  --dropout data/output/ptep_demo_train_history.csv \
  --no-dropout data/output/ptep_demo_dropout0_train_history.csv \
  --f1-range 0.74 0.77 \
  --loss-range 0.50 0.55 \
  --output plots/train/ptep_demo_dropout_comparison.png
