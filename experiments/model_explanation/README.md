# Model explanation

`explain.py` computes SHAP distributions and global feature importance for the tracked `ptep_demo` model. It is intentionally separate from the standard pipeline because SHAP is an optional, comparatively expensive post-training analysis.

The analysis uses the model's training partition as the SHAP background and an independent test sample for the events to be explained. Validation data are excluded from the background, and the test sample is not used for tuning, training, or validation.

Repeated random sampling reduces dependence on a particular subset of events. The beeswarm plot summarizes the SHAP-value distributions, while global importance is based on the mean absolute SHAP value and includes its variation across samples. This variation describes the stability of the explanation, not sensitivity to the model's training seed.

The script requires the local ROOT input data and the tracked model artifacts used by `ptep_demo`.

Run from the repository root:

```bash
python experiments/model_explanation/explain.py
```

The principal figure is written to `plots/explain/ptep_demo_shap_combined.png`. A detailed importance table is also generated under `data/output/` for local inspection and is not tracked by Git.

## Result

The left panel shows the SHAP-value distributions, with color indicating the original feature value. The right panel summarizes global feature importance and its sampling variation.

![SHAP feature-importance result](../../plots/explain/ptep_demo_shap_combined.png)
