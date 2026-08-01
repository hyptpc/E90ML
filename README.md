<div align="center">

# Machine Learning for E90-QF Suppression

Reproducible machine-learning tools for the J-PARC E90 analysis.

[![arXiv](https://img.shields.io/badge/arXiv-2606.22750-b31b1b.svg)](https://arxiv.org/abs/2606.22750v1)

</div>

---

## Overview

The recommended `v1` pipeline trains a multilayer perceptron to separate the
`SigmaNCusp` signal from the `QFLambda` and `QFSigmaZ` backgrounds. Its 12
inputs are the `u_x`, `u_y`, `u_z`, and `dE/dx` variables for tracks `t0`, `t1`,
and `t2`.

The standard workflow consists of three commands:

1. `src/tune.py`: Optuna hyperparameter optimization;
2. `src/train.py`: final training with validation-loss early stopping;
3. `src/test.py`: inference on the independent test sample.

## Environment setup

Create and activate the environment appropriate for the execution platform.

```bash
# KEKCC/Linux: PyTorch 2.5.1 with the official CUDA 12.4 runtime
conda env create -f environment.yml

# macOS/Apple Silicon: PyTorch with MPS support
conda env create -f environment-macos.yml

conda activate pyml
```

The Linux environment targets CUDA 12.4. Before creating it on a different
Linux system, check the NVIDIA driver with `nvidia-smi`. If that driver cannot
support the configured CUDA runtime, select a compatible PyTorch wheel using
the [official PyTorch installer](https://pytorch.org/get-started/locally/).

Verify the backend after installation:

```bash
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("MPS:", torch.backends.mps.is_available())'
```

The code uses PyTorch, NumPy, pandas, scikit-learn, Optuna, uproot, Awkward
Array, PyYAML, and Matplotlib. SHAP is required only for
`experiments/model_explanation/`.

## Data layout

By default, input files are read from `./data/input` and generated data products
are written below `./data/output`:

```text
data/
├── input/
│   ├── SigmaNCusp.root
│   ├── QFLambda.root
│   ├── QFSigmaZ.root
│   └── test.root
└── output/
```

On KEKCC, point the project to the shared data area instead of copying large
ROOT files into the repository:

```bash
export E90ML_DATA_DIR=/ghi/fs02/had/sks/Users/YOUR-DIRECTORY
```

The pipeline then uses `${E90ML_DATA_DIR}/input` and
`${E90ML_DATA_DIR}/output`, creating both directories when necessary.

## Configuration

Self-contained templates are provided for both platforms:

```bash
cp param/usr/v1_mac_demo.yaml param/usr/v1_mac.yaml
cp param/usr/v1_kekcc_demo.yaml param/usr/v1_kekcc.yaml
```

Personal YAML files are ignored by Git. The demo files document the data split,
search space, random seeds, early stopping, output names, and platform-specific
worker settings. `device: auto` selects CUDA first, then MPS, then CPU.

## Running the pipeline

Run directly on macOS:

```bash
python src/tune.py  -c param/usr/v1_mac.yaml
python src/train.py -c param/usr/v1_mac.yaml
python src/test.py  -c param/usr/v1_mac.yaml
```

Submit the same stages to LSF on KEKCC:

```bash
./tune.sh  param/usr/v1_kekcc.yaml
./train.sh param/usr/v1_kekcc.yaml
./test.sh  param/usr/v1_kekcc.yaml
```

The final test output is written as a ROOT `TTree` for compatibility with the
existing ROOT analysis macros.

## Reproducible demo artifacts

The repository retains the compact artifacts needed to inspect or continue the
documented `ptep_demo` run without repeating every expensive stage:

- `param/tune/ptep_demo.db`: Optuna study database;
- `param/tune/ptep_demo.json`: selected hyperparameters;
- `param/pth/ptep_demo.pth`: final weights selected by minimum validation loss;
- `param/pth/ptep_demo.pkl`: fitted `StandardScaler`.

Per-epoch histories, data-count summaries, trial summaries, and the principal
diagnostic plots are also retained as curated results. Raw input ROOT files,
checkpoints, logs, and regenerable test ROOT outputs remain local.

## Supplementary experiments

Reproducible studies that are not part of the standard tune/train/test pipeline
live under `experiments/`:

- `experiments/dropout_ablation/`: controlled Dropout ON/OFF comparison;
- `experiments/pruner_warmup/`: MedianPruner warmup comparison;
- `experiments/model_explanation/`: multi-seed SHAP feature-importance analysis.

Each directory contains its configuration, executable code, reproduction
instructions, and—where appropriate—a technical report. The exact report
databases and numerical summaries are retained, while `*_demo.yaml` files use
non-conflicting output names for a fresh run.

## Branches

- **v1 (recommended):** 12-variable MLP using track directions and `dE/dx`;
- **v2:** graph neural network study;
- **v3:** `v1` extension including the track opening angle.
