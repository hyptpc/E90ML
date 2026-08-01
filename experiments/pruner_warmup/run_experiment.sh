#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${experiment_dir}/../.." && pwd)"

cd "${project_root}"

python src/tune.py -c "${experiment_dir}/config_warmup0_mac.yaml"
python src/tune.py -c "${experiment_dir}/config_warmup10_mac.yaml"
python "${experiment_dir}/compare_results.py" \
  --warmup0-db param/tune/v1_pruner_warmup0.db \
  --warmup0-study v1_pruner_warmup0 \
  --warmup10-db param/tune/v1_pruner_warmup10.db \
  --warmup10-study v1_pruner_warmup10 \
  --summary data/output/v1_pruner_warmup_comparison.csv \
  --paired-summary data/output/v1_pruner_warmup_paired_trials.csv \
  --output plots/tune/v1_pruner_warmup_comparison.png
