E90ML
=====

Minimal docs to run tuning/training with the explicit (no defaults) config layout.

Setup
- Input ROOTs are resolved relative to the config; in the sample YAML you can list filenames only (looked up under `data/input`).
- Outputs accept filenames only; the code saves them under fixed folders:
  - Tuning params → `param/tune/`
  - Tuning trials CSV → `data/output/`
  - Model/scaler → `param/pth/`
  - Metrics/predictions/plots → `data/output/`
- Set seeds under `tuning.seed` and `training.seed` (or top-level `seed`) for reproducible splits.
- Include label mapping in the config if labels need remapping (see sample YAML).

Commands
- Tuning: `python -m src.tune -c param/usr/demo.yaml`
- Training: `python -m src.train -c param/usr/demo.yaml`
