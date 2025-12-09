#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <user param>" >&2
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "activate your conda environment!" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$1"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "user param not found: $CONFIG_PATH" >&2
    exit 1
fi

CONFIG_PATH="$(python - <<'PY' "$CONFIG_PATH"
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
)"

python - <<'PY' "$SCRIPT_DIR" "$CONFIG_PATH"
import os
import sys
from textwrap import indent

import yaml

script_dir, config_path = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = yaml.safe_load(f) or {}

data_cfg = cfg.get("data", {})
tune_cfg = cfg.get("tuning", {})

files = data_cfg.get("files", {})
tree = data_cfg.get("tree_name")
features = data_cfg.get("feature_columns") or []

summary = [
    f"Config: {config_path}",
    f"Data files: {files if files else 'N/A'}",
    f"Tree: {tree}",
    f"Feature count: {len(features)}",
    f"Tuning fraction: {tune_cfg.get('fraction', 'N/A')}",
    f"Val split: {tune_cfg.get('val_split', 'N/A')}",
    f"Epochs: {tune_cfg.get('epochs', 'N/A')}",
    f"Trials: {tune_cfg.get('n_trials', 'N/A')}",
    f"Best params path: {tune_cfg.get('best_params_path', 'N/A')}",
    f"Trials CSV: {tune_cfg.get('study_summary_path', 'N/A')}",
]

print("Running hyperparameter tuning with config summary:")
print(indent("\n".join(summary), "  "))
PY

echo "Running hyperparameter tuning with config: $CONFIG_PATH"
python "$SCRIPT_DIR/src/tune.py" --config "$CONFIG_PATH"
