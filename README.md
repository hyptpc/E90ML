E90ML
=====

Minimal docs to run tuning/training with the simplified config layout.

Setup
- Place ROOT files under `data/input` (default lookup path).
- Set the filenames only in `param/usr/demo.yaml` under `data.files`.
- Labels are fixed in code: SigmaNCusp=1, QFLambda=2, QFSigmaZ=3. SigmaNCusp is treated as signal; the others are background.
- If you pass just a filename (no slashes) in the config, the code drops it into a fixed location:
  - Tuning results (`best_params_path`) → `param/tune/`
  - Tuning trial CSV (`study_summary_path`) → `data/output/`
  - Model weights (`model_output_path`) → `param/pth/`
  - Metrics/predictions/plots (`metrics_output_path`, `predictions_output_path`, `plots_dir`) → `data/output/`

Commands
- Tuning: `python -m src.tune -c param/usr/demo.yaml`
- Training: `python -m src.train -c param/usr/demo.yaml`
