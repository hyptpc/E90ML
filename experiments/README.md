# Experiments

This directory contains reproducible supplementary analyses. They deliberately
live outside `src/`, which is reserved for the standard tune/train/test
pipeline.

- `dropout_ablation/`: compare the standard tuned Dropout model with
  `dropout_rate=0` while keeping all other settings fixed;
- `pruner_warmup/`: compare MedianPruner warmup policies before the primary
  hyperparameter study;
- `model_explanation/`: compute multi-seed SHAP feature-importance plots for a
  trained model.

Experiment-specific configurations without `_demo` reproduce the reported
artifact names. The corresponding `*_demo.yaml` files use non-conflicting
names so another user can launch a fresh run without overwriting the retained
report inputs.
