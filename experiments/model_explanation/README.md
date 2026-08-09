# Model explanation

`explain.py` computes SHAP distributions and global feature importance for the tracked `ptep_demo` model. It is intentionally separate from the standard pipeline because SHAP is an optional, comparatively expensive post-training analysis.

The analysis follows the usual supervised-learning SHAP setup: the background (baseline) represents the model's training distribution, while the explained events come from independent test data. It reproduces the group-aware train/validation split used for `ptep_demo` (`validation_fraction=0.2`, `split_seed=90`) and excludes the validation partition from the background pool. The independent `data/input/test.root` sample is not used for tuning, training, or validation.

For each seed in `(7, 21, 42, 90, 2026)`, the script draws 200 background events from the training partition and 500 explained events from the independent test sample, both without replacement within that seed. The publication-oriented beeswarm pools the five equal-size test explanations, while global importance is reported as the mean and standard deviation of `mean(|SHAP|)` across the five sampling seeds. These are SHAP sampling seeds; assessing training-seed dependence would require retraining multiple models.

Required local input data:

- `data/input/SigmaNCusp.root`;
- `data/input/QFLambda.root`;
- `data/input/QFSigmaZ.root`;
- `data/input/test.root`.

Required tracked artifacts:

- `param/tune/ptep_demo.json`;
- `param/pth/ptep_demo.pth`;
- `param/pth/ptep_demo.pkl`.

Run from the repository root:

```bash
python experiments/model_explanation/explain.py
```

The principal figure is written to `plots/explain/ptep_demo_shap_combined.png`. The seed-wise importance table is generated at `data/output/ptep_demo_shap_seed_stability.csv` for local inspection. The CSV is reproducible from the tracked script, model artifacts, and local ROOT inputs, so it is intentionally ignored by Git rather than versioned.

## Result

The left panel shows the pooled SHAP-value distributions, with color indicating the original feature value. The right panel shows the mean and standard deviation of `mean(|SHAP|)` across the five sampling seeds.

![Multi-seed SHAP feature-importance result](../../plots/explain/ptep_demo_shap_combined.png)
