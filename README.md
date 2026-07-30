<div align="center">

# Machine Learning for E90-QF surpression

Machine-learning analysis tools for the J-PARC E90 experiment.

[![arXiv](https://img.shields.io/badge/arXiv-2606.22750-b31b1b.svg)](https://arxiv.org/abs/2606.22750v1)

</div>

---

## Setup

### 1. Create and activate the Conda environment

```bash
# Linux / KEKCC
conda env create -f environment.yml

# macOS (Apple Silicon)
conda env create -f environment-macos.yml

conda activate pyml
```

### 2. Select the data directory

```bash
# KEKCC: point to the shared data root
export E90ML_DATA_DIR=/ghi/fs02/had/sks/Users/YOUR-DIRECTORY
```

When `E90ML_DATA_DIR` is set, the project uses its `input` directory for ROOT
files and its `output` directory for generated files. Both directories are
created automatically. If the variable is unset, the repository's `data`
directory is used.

The expected layout is:

```text
E90ML_DATA_DIR/
├── input/
│   ├── SigmaNCusp.root
│   ├── QFLambda.root
│   ├── QFSigmaZ.root
│   └── test.root
└── output/
```

## How to run

Environment-specific configurations are provided for macOS and KEKCC:

```bash
# macOS: run directly with MPS when available
python src/tune.py -c param/usr/v1_mac.yaml
python src/train.py -c param/usr/v1_mac.yaml
python src/test.py -c param/usr/v1_mac.yaml

# KEKCC: submit jobs to LSF
./tune.sh  param/usr/v1_kekcc.yaml
./train.sh param/usr/v1_kekcc.yaml
./test.sh  param/usr/v1_kekcc.yaml
```

Personal configuration files are ignored by Git. To create one from a shared
template:

```bash
cp param/usr/v1_mac_demo.yaml param/usr/v1_mac.yaml
cp param/usr/v1_kekcc_demo.yaml param/usr/v1_kekcc.yaml
```

Each environment-specific demo is self-contained and documents all available
settings. Personal configuration files can inherit from the corresponding demo
and override only the values that need to change.

- `v1_mac.yaml` uses `num_workers: 0` for stable local execution.
- `v1_kekcc.yaml` uses LSF settings and worker processes for batch jobs.
- `device: auto` selects CUDA, then MPS, then CPU.

For a controlled Dropout ablation, keep the tuned architecture and optimizer
settings fixed and override only the training Dropout rate:

```yaml
extends: v1_mac.yaml

training:
  dropout_rate_override: 0.0
```

Use distinct checkpoint, model, scaler, history, plot, and test output names
for the Dropout ON and OFF runs. `v1_mac_nodropout_demo.yaml` provides a shared
example.

## Branch differences

- **v1 (recommended):** Standard MLP configuration. Input features are `u` and `dE/dx` (12 variables from t0/t1/t2).
  - Use `v1_mac_demo.yaml` or `v1_kekcc_demo.yaml` as the baseline configuration.
- **v2:** GNN model. Accuracy is not good.
- **v3:** Extension of v1 that computes `open_angle` from `u` and uses it as an input feature. Accuracy is comparable to v1.
