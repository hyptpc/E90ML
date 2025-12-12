#!/bin/bash
set -e

# --- 1. Argument Validation ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

CONFIG_PATH=$1

# --- 2. Extract Configuration from YAML ---
# Uses Python to parse the 'bsub' section of the YAML file.
# It converts the config path to an absolute path to ensure accessibility on compute nodes.
eval $(python -c "
import sys, yaml, os

try:
    config_path = '$CONFIG_PATH'
    abs_config_path = os.path.abspath(config_path)

    with open(abs_config_path) as f:
        cfg = yaml.safe_load(f)
    
    # 1. Global bsub settings
    bsub_cfg = cfg.get('bsub', {})
    
    # 2. Training specific bsub settings (Override)
    # trainingセクションの中にbsubがあれば、それを優先して上書きマージする
    training_section = cfg.get('training', {})
    if 'bsub' in training_section:
        bsub_cfg.update(training_section['bsub'])
    
    # Extract values
    queue = bsub_cfg.get('queue', 's')
    email = bsub_cfg.get('email', '')
    log_file = bsub_cfg.get('log_file', 'lsflog/train_result.log')
    job_name = bsub_cfg.get('job_name', 'e90_train')
    conda_env = bsub_cfg.get('conda_env', 'base')

    print(f'QUEUE={queue}')
    print(f'EMAIL={email}')
    print(f'LOG_FILE={log_file}')
    print(f'JOB_NAME={job_name}')
    print(f'CONDA_ENV={conda_env}')
    print(f'ABS_CONFIG_PATH={abs_config_path}')

except Exception as e:
    print(f'Error parsing YAML: {e}', file=sys.stderr)
    sys.exit(1)
")

# --- 3. Job Submission (bsub) ---
# Submits the job to the LSF scheduler using a Here Document.
# This avoids issues with shell metacharacters (like #, source) being parsed by bsub wrappers.

echo "--------------------------------------------------"
echo "Submitting Training Job"
echo "  Job Name  : $JOB_NAME"
echo "  Queue     : $QUEUE"
echo "  Config    : $ABS_CONFIG_PATH"
echo "--------------------------------------------------"

# Ensure the log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# The following block is sent directly to bsub as the job script.
bsub -q "$QUEUE" \
     -u "$EMAIL" \
     -o "$LOG_FILE" \
     -J "$JOB_NAME" \
     -N <<EOF
#!/bin/bash
# ----------------------------------------
# Job Execution Script
# ----------------------------------------

# 1. Initialize Environment
source ~/.bashrc

# 2. Setup Conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Check your .bashrc."
    exit 1
fi
conda activate $CONDA_ENV

# 3. Log Job Info
echo "Job started on host: \$(hostname)"
echo "Time: \$(date)"
echo "Config file: $ABS_CONFIG_PATH"

# 4. Run Training Script
# Note: Using absolute path for config to be safe
python src/train.py -c "$ABS_CONFIG_PATH"

EOF