#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH="$1"

eval "$(python3 - <<'PY'
import os, sys, yaml

config_path = os.path.realpath(os.environ["CONFIG_PATH"])
try:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"echo 'Error parsing YAML: {e}' >&2")
    sys.exit(1)

bsub_cfg = dict(cfg.get("bsub") or {})
tuning_cfg = cfg.get("tuning") or {}
bsub_cfg.update(tuning_cfg.get("bsub") or {})

queue = bsub_cfg.get("queue", "s")
email = bsub_cfg.get("email", "")
log_file = bsub_cfg.get("log_file", "lsflog/tune_result.log")
job_name = bsub_cfg.get("job_name", "e90_tune")
conda_env = bsub_cfg.get("conda_env", "base")

print(f"QUEUE={queue}")
print(f"EMAIL={email}")
print(f"LOG_FILE={log_file}")
print(f"JOB_NAME={job_name}")
print(f"CONDA_ENV={conda_env}")
print(f"ABS_CONFIG_PATH={config_path}")
PY
)"

echo "--------------------------------------------------"
echo "Submitting Tuning Job"
echo "  Job Name  : $JOB_NAME"
echo "  Queue     : $QUEUE"
echo "  Config    : $ABS_CONFIG_PATH"
echo "--------------------------------------------------"

mkdir -p "$(dirname "$LOG_FILE")"

bsub -q "$QUEUE" \
     -u "$EMAIL" \
     -o "$LOG_FILE" \
     -J "$JOB_NAME" \
     -N <<EOF
#!/bin/bash

source ~/.bashrc
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Check your .bashrc."
    exit 1
fi
conda activate $CONDA_ENV

echo "Job started on host: \$(hostname)"
echo "Time: \$(date)"
echo "Config file: $ABS_CONFIG_PATH"

python src/tune.py -c "$ABS_CONFIG_PATH"
EOF
