**Machine Learning for E90 QF surpression**
=====

Setup
1) Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate pyml
```

2) Link the KEKCC buffer area and place the training data
```bash
# Example: default buffer area
BUFFER_ROOT=/ghi/fs02/had/sks/Users/YOUR-DIRECTORY

ln -s ${BUFFER_ROOT}/input  data/input
ln -s ${BUFFER_ROOT}/output data/output
```

How to Run (tune/train/test)
Run the shell scripts with a config file.
```bash
./tune.sh  param/usr/demo.yaml
./train.sh param/usr/demo.yaml
./test.sh  param/usr/demo.yaml
```
- `tune.sh` / `train.sh` / `test.sh` submit jobs to LSF (`bsub`).
- Set job name, queue, and logs in YAML under `bsub` or `tuning.bsub` / `training.bsub` / `test.bsub`.
- For direct execution, use `python -m src.tune -c <config>`.

Branch Differences (v1 recommended)
- v1: Standard MLP configuration. Input features are `u` and `dE/dx` (12 variables from t0/t1/t2).
  - `param/usr/demo.yaml` is the baseline config. Recommended for stable results.
- v2: GNN model. Accuracy is not good.
- v3: Extension of v1 (compute `open_angle` from `u` and use it as an input feature). Accuracy is comparable to v1.
