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
import sys
from pathlib import Path
from textwrap import indent

import yaml

script_dir, config_path = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = yaml.safe_load(f) or {}

data_cfg = cfg.get("data", {})
train_cfg = cfg.get("training", {})
tune_cfg = cfg.get("tuning", {})

def get(cfg, *keys):
    for k in keys:
        v = cfg.get(k)
        if v not in (None, ""):
            return v
    return None

files = data_cfg.get("files", {})
tree = data_cfg.get("tree_name")
label_col = data_cfg.get("label_column")
features = data_cfg.get("feature_columns") or []
best_params_file = get(train_cfg, "best_params_file", "best_params_path") or get(
    tune_cfg, "tune_params_file", "best_params_file", "best_params_path"
)
device_pref = train_cfg.get("device") or cfg.get("device")
plot_output = get(train_cfg, "plot_output_file", "plot_output_path", "plots_path", "plots_dir")
checkpoint_output = get(train_cfg, "checkpoint_file", "checkpoint_path")
model_output = get(train_cfg, "model_output_file", "model_output_path")
scaler_output = get(train_cfg, "scaler_output_file", "scaler_output_path")
metrics_output = get(train_cfg, "metrics_output_file", "metrics_output_path")
predictions_output = get(train_cfg, "predictions_output_file", "predictions_output_path")

if not checkpoint_output and model_output:
    checkpoint_output = model_output

summary = [
    f"Config: {config_path}",
    f"Data files: {files if files else 'N/A'}",
    f"Tree: {tree}",
    f"Label column: {label_col}",
    f"Feature count: {len(features)}",
    f"Training fraction: {train_cfg.get('fraction', 'N/A')}",
    f"Val split: {train_cfg.get('val_split', 'N/A')}",
    f"Epochs: {train_cfg.get('epochs', 'N/A')}",
    f"Patience: {train_cfg.get('patience', 'N/A')}",
    f"Best params file: {best_params_file or 'N/A'}",
    f"Checkpoint output: {checkpoint_output or model_output or 'N/A'}",
    f"Model output: {model_output or 'N/A'}",
    f"Scaler output: {scaler_output or 'N/A'}",
    f"Metrics output: {metrics_output or 'N/A'}",
    f"Predictions output: {predictions_output or 'N/A'}",
    f"Plot output: {plot_output or 'N/A'}",
]

if device_pref is not None:
    summary.append(f"Device preference: {device_pref}")

print("Running final training with config summary:")
print(indent("\n".join(summary), "  "))
PY

echo "Running final training with config: $CONFIG_PATH"
python "$SCRIPT_DIR/src/train.py" --config "$CONFIG_PATH"
