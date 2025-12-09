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

echo "Running hyperparameter tuning with config: $CONFIG_PATH"
python "$SCRIPT_DIR/src/tune.py" --config "$CONFIG_PATH"
