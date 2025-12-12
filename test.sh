#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <user param>" >&2
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

runtime_info="$(python - <<'PY' "$CONFIG_PATH"
import sys

import yaml

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = yaml.safe_load(f) or {}

runtime = cfg.get("runtime") or {}
conda_env = runtime.get("conda_env")
queue = runtime.get("queue") or runtime.get("lsf_queue")
job_name = runtime.get("job_name")

for item in (conda_env, queue, job_name):
    print(item or "")
PY
)"

mapfile -t runtime_fields <<<"$runtime_info"
CONDA_ENV="${runtime_fields[0]:-}"
QUEUE="${runtime_fields[1]:-}"
JOB_NAME_RAW="${runtime_fields[2]:-}"

if [[ -z "$CONDA_ENV" ]]; then
    echo "runtime.conda_env is required in the config: $CONFIG_PATH" >&2
    exit 1
fi

if [[ -z "$QUEUE" ]]; then
    echo "runtime.queue is required in the config: $CONFIG_PATH" >&2
    exit 1
fi

command -v bsub >/dev/null 2>&1 || { echo "bsub command not found in PATH." >&2; exit 1; }

CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
DEFAULT_JOB_NAME="e90ml_test_${CONFIG_BASENAME%.*}"
JOB_NAME="${JOB_NAME_RAW:-$DEFAULT_JOB_NAME}"

python - <<'PY' "$SCRIPT_DIR" "$CONFIG_PATH"
import sys
from textwrap import indent

import yaml

script_dir, config_path = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = yaml.safe_load(f) or {}

data_cfg = cfg.get("data", {})
train_cfg = cfg.get("training", {})
test_cfg = cfg.get("testing", {})
tune_cfg = cfg.get("tuning", {})
runtime_cfg = cfg.get("runtime", {})

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
model_output = get(train_cfg, "model_output_file", "model_output_path")
scaler_output = get(train_cfg, "scaler_output_file", "scaler_output_path")
test_fraction = test_cfg.get("fraction", 1.0)
batch_size = test_cfg.get("batch_size") or train_cfg.get("batch_size")
num_workers = test_cfg.get("num_workers") or train_cfg.get("num_workers")
metrics_output = get(test_cfg, "metrics_output_file", "metrics_output_path")
predictions_output = get(test_cfg, "predictions_output_file", "predictions_output_path")
device_pref = test_cfg.get("device") or train_cfg.get("device") or cfg.get("device")

summary = [
    f"Config: {config_path}",
    f"Data files: {files if files else 'N/A'}",
    f"Tree: {tree}",
    f"Label column: {label_col}",
    f"Feature count: {len(features)}",
    f"Test fraction: {test_fraction}",
    f"Batch size: {batch_size or 'N/A'}",
    f"Num workers: {num_workers or 'N/A'}",
    f"Best params file: {best_params_file or 'N/A'}",
    f"Model path: {model_output or 'N/A'}",
    f"Scaler path: {scaler_output or 'N/A'}",
    f"Metrics output: {metrics_output or 'N/A'}",
    f"Predictions output: {predictions_output or 'N/A'}",
    f"Conda env: {runtime_cfg.get('conda_env', 'N/A')}",
    f"LSF queue: {runtime_cfg.get('queue', 'N/A')}",
]

if device_pref is not None:
    summary.append(f"Device preference: {device_pref}")

print("Running evaluation with config summary:")
print(indent("\n".join(summary), "  "))
PY

echo "Submitting evaluation via bsub:"
echo "  queue: $QUEUE"
echo "  job:   $JOB_NAME"
echo "  conda: $CONDA_ENV"

job_cmd_template=$(cat <<'EOS'
set -euo pipefail

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    return
  fi

  for base in "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${base}/etc/profile.d/conda.sh"
      return
    fi
  done

  echo "conda command not found; ensure it is installed and initialized." >&2
  return 1
}

load_conda || exit 1
conda activate __CONDA_ENV__

cd __SCRIPT_DIR__
python __PY_FILE__ --config __CONFIG_PATH__
EOS
)

escape_for_bash() {
    printf '%q' "$1"
}

escaped_conda_env="$(escape_for_bash "$CONDA_ENV")"
escaped_script_dir="$(escape_for_bash "$SCRIPT_DIR")"
escaped_py_file="$(escape_for_bash "$SCRIPT_DIR/src/test.py")"
escaped_config_path="$(escape_for_bash "$CONFIG_PATH")"

job_cmd=${job_cmd_template//__CONDA_ENV__/$escaped_conda_env}
job_cmd=${job_cmd//__SCRIPT_DIR__/$escaped_script_dir}
job_cmd=${job_cmd//__PY_FILE__/$escaped_py_file}
job_cmd=${job_cmd//__CONFIG_PATH__/$escaped_config_path}

bsub -q "$QUEUE" -J "$JOB_NAME" bash -lc "$job_cmd"
