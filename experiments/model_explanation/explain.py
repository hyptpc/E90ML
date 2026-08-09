import csv
import gc
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import shap
import matplotlib
# Use 'Agg' backend to avoid errors on servers without GUI (X11)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Keep reusable training code in src/ while this experiment remains self-contained.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import custom modules
from common import (
    load_data,
    create_model_from_params,
    apply_plot_style,
    DEFAULT_LABEL_MAPPING,
    make_event_groups,
    stratified_group_split_indices,
)

# -----------------------------------------------------------------------------
# User-editable parameters (no YAML required)
# -----------------------------------------------------------------------------
# Data used to fit the model. The training partition supplies the SHAP baseline.
TRAIN_DATA_FILES = [
    PROJECT_ROOT / "data" / "input" / "SigmaNCusp.root",
    PROJECT_ROOT / "data" / "input" / "QFLambda.root",
    PROJECT_ROOT / "data" / "input" / "QFSigmaZ.root",
]
# Independent data whose model predictions are explained.
TEST_DATA_FILE = PROJECT_ROOT / "data" / "input" / "test.root"
TREE_NAME = "g4s2s"
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [
    "t0_ux",
    "t0_uy",
    "t0_uz",
    "t0_dedx",
    "t1_ux",
    "t1_uy",
    "t1_uz",
    "t1_dedx",
    "t2_ux",
    "t2_uy",
    "t2_uz",
    "t2_dedx",
]
LABEL_MAPPING = DEFAULT_LABEL_MAPPING  # remap to binary: signal=1, background=0
# These must match the split used to train ptep_demo.pth.
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 90
SEED = 42

# Model artifacts
SCALER_PATH = PROJECT_ROOT / "param" / "pth" / "ptep_demo.pkl"
MODEL_PATH = PROJECT_ROOT / "param" / "pth" / "ptep_demo.pth"
BEST_PARAMS_PATH = PROJECT_ROOT / "param" / "tune" / "ptep_demo.json"

# Output paths
PLOT_COMBINED_PATH = PROJECT_ROOT / "plots" / "explain" / "ptep_demo_shap_combined.png"
SEED_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "output" / "ptep_demo_shap_seed_stability.csv"
)

# Font settings for this script
FONT_FAMILY = "serif"
FONT_SIZE = 16
LABEL_FONT_SIZE = 20

# SHAP sampling sizes
BACKGROUND_SAMPLES = 200
TEST_SAMPLES = 500
SHAP_SEEDS = (7, 21, 42, 90, 2026)


def _force_plot_fonts(fig) -> None:
    # SHAP resets some text properties; enforce chosen fonts after plotting.
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(FONT_SIZE)

    for axis in fig.get_axes():
        axis.tick_params(axis="both", labelsize=FONT_SIZE, length=4, width=0.8)
        for label in [axis.xaxis.label, axis.yaxis.label]:
            if label is not None:
                label.set_fontsize(LABEL_FONT_SIZE)
                label.set_fontfamily(FONT_FAMILY)
        if axis.get_title():
            axis.title.set_fontsize(LABEL_FONT_SIZE)
            axis.title.set_fontfamily(FONT_FAMILY)


def _save_combined_plot(
    explanation: shap.Explanation,
    seed_explanations: list[shap.Explanation],
    feature_names: list[str],
) -> None:
    """Combine SHAP distributions with seed-averaged global importance."""
    seed_importance = np.stack(
        [np.abs(item.values).mean(axis=0) for item in seed_explanations]
    )
    mean_importance_by_name = dict(zip(feature_names, seed_importance.mean(axis=0)))
    std_importance_by_name = dict(
        zip(feature_names, seed_importance.std(axis=0, ddof=1))
    )

    SEED_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_columns = [f"seed_{seed}" for seed in SHAP_SEEDS]
    with SEED_SUMMARY_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["feature", *seed_columns, "mean_abs_shap", "sd_abs_shap"],
            lineterminator="\n",
        )
        writer.writeheader()
        for feature_index, feature_name in enumerate(feature_names):
            values = seed_importance[:, feature_index]
            writer.writerow(
                {
                    "feature": feature_name,
                    **dict(zip(seed_columns, values)),
                    "mean_abs_shap": values.mean(),
                    "sd_abs_shap": values.std(ddof=1),
                }
            )
    print(f"Saved: {SEED_SUMMARY_PATH}")

    fig = plt.figure(figsize=(18.5, 7.5))
    outer_grid = fig.add_gridspec(1, 2, width_ratios=(0.12, 8.8), wspace=0.08)
    colorbar_axis = fig.add_subplot(outer_grid[0, 0])
    content_grid = outer_grid[0, 1].subgridspec(
        1,
        3,
        width_ratios=(3.0, 0.5, 2.0),
        wspace=0.08,
    )
    beeswarm_axis = fig.add_subplot(content_grid[0, 0])
    label_axis = fig.add_subplot(content_grid[0, 1])
    bar_axis = fig.add_subplot(content_grid[0, 2])

    shap.plots.beeswarm(
        explanation,
        max_display=len(feature_names),
        color=shap.plots.colors.red_blue,
        color_bar=False,
        plot_size=None,
        ax=beeswarm_axis,
        show=False,
        s=18,
    )
    beeswarm_axis.set_xlabel("SHAP value")
    beeswarm_axis.set_axisbelow(True)
    beeswarm_axis.grid(
        axis="y",
        color="#a6a6a6",
        linestyle=(0, (1.2, 2.4)),
        linewidth=0.9,
        alpha=0.9,
    )

    row_positions = beeswarm_axis.get_yticks()
    ordered_names = [label.get_text() for label in beeswarm_axis.get_yticklabels()]
    ordered_mean_importance = [
        mean_importance_by_name[name] for name in ordered_names
    ]
    ordered_std_importance = [
        std_importance_by_name[name] for name in ordered_names
    ]
    beeswarm_axis.set_yticklabels([])
    beeswarm_axis.tick_params(axis="y", left=False)

    label_axis.set_xlim(0.0, 1.0)
    label_axis.set_ylim(beeswarm_axis.get_ylim())
    label_axis.axis("off")
    for position, name in zip(row_positions, ordered_names):
        label_axis.text(0.5, position, name, ha="center", va="center")

    bar_axis.set_ylim(beeswarm_axis.get_ylim())
    bar_axis.set_yticks(row_positions)
    bar_axis.set_yticklabels([])
    bar_axis.tick_params(axis="y", which="both", left=False, labelleft=False)
    bar_axis.set_axisbelow(True)
    bar_axis.grid(
        axis="y",
        color="#a6a6a6",
        linestyle=(0, (1.2, 2.4)),
        linewidth=0.9,
        alpha=0.9,
    )
    bar_axis.errorbar(
        ordered_mean_importance,
        row_positions,
        xerr=ordered_std_importance,
        fmt="o",
        color="#111111",
        ecolor="#111111",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.4,
        markersize=9.5,
        elinewidth=1.3,
        capsize=4.5,
        capthick=1.3,
        clip_on=False,
        zorder=3,
    )
    bar_axis.set_xlabel(r"mean($|\mathrm{SHAP}|$)")
    bar_axis.set_xlim(0.0, 0.5)
    bar_axis.set_xticks(np.arange(0.0, 0.51, 0.1))
    bar_axis.spines["left"].set_visible(False)
    bar_axis.spines["top"].set_visible(False)
    bar_axis.spines["right"].set_visible(False)

    color_scale = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
        cmap=shap.plots.colors.red_blue,
    )
    color_scale.set_array([])
    colorbar = fig.colorbar(color_scale, cax=colorbar_axis)
    colorbar.set_ticks((0.0, 1.0), labels=("Low", "High"))
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(labelsize=FONT_SIZE, length=4, width=0.8)
    colorbar.set_label("Feature value", labelpad=20, fontsize=LABEL_FONT_SIZE)
    colorbar.ax.yaxis.label.set_fontfamily(FONT_FAMILY)

    _force_plot_fonts(fig)
    PLOT_COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_COMBINED_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {PLOT_COMBINED_PATH}")


def _normalize_shap_values(shap_values) -> np.ndarray:
    """Normalize DeepExplainer outputs to (samples, features)."""
    if isinstance(shap_values, list):
        print(f"SHAP returned a list of length: {len(shap_values)}")
        values = shap_values[1 if len(shap_values) > 1 else 0]
    else:
        print("SHAP returned a single tensor.")
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3 and values.shape[2] == 1:
        print(f"Squeezing extra dimension: {values.shape} -> ", end="")
        values = np.squeeze(values, axis=2)
        print(values.shape)
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def _sample_training_backgrounds(feature_names):
    """Reproduce the model's split and sample SHAP baselines from train only."""
    print("Loading training data for SHAP backgrounds...")
    train_df, num_classes = load_data(
        files=[str(path) for path in TRAIN_DATA_FILES],
        tree_name=TREE_NAME,
        features=feature_names,
        label_column=LABEL_COLUMN,
        label_mapping=LABEL_MAPPING,
        fraction=1.0,
        random_state=SPLIT_SEED,
        shuffle=False,
    )
    all_features = train_df[feature_names].values.astype(np.float32)
    all_labels = train_df[LABEL_COLUMN].values.astype(np.int64)
    del train_df

    event_groups = make_event_groups(all_features, all_labels)
    train_indices, validation_indices = stratified_group_split_indices(
        all_labels,
        event_groups,
        VALIDATION_FRACTION,
        SPLIT_SEED,
        "SHAP background split",
    )
    if len(train_indices) < BACKGROUND_SAMPLES:
        raise ValueError("Training partition is too small for SHAP background sampling.")

    backgrounds = {}
    for seed in SHAP_SEEDS:
        rng = np.random.default_rng(seed)
        selected = rng.choice(train_indices, BACKGROUND_SAMPLES, replace=False)
        backgrounds[seed] = all_features[selected].copy()

    print(f"  Training rows available    : {len(train_indices)}")
    print(f"  Validation rows excluded   : {len(validation_indices)}")
    del all_features, all_labels, event_groups, train_indices, validation_indices
    gc.collect()
    return backgrounds, num_classes


def _load_test_features(feature_names):
    """Load the independent test population whose predictions are explained."""
    print("Loading independent test data for SHAP explanations...")
    test_df, num_classes = load_data(
        files=[str(TEST_DATA_FILE)],
        tree_name=TREE_NAME,
        features=feature_names,
        label_column=LABEL_COLUMN,
        label_mapping=LABEL_MAPPING,
        fraction=1.0,
        random_state=SEED,
        shuffle=False,
    )
    test_features = test_df[feature_names].values.astype(np.float32)
    del test_df
    if len(test_features) < TEST_SAMPLES:
        raise ValueError("Independent test data is too small for SHAP sampling.")
    print(f"  Independent test rows      : {len(test_features)}")
    return test_features, num_classes


def _compute_explanation(
    model,
    background_scaled,
    test_scaled,
    test_raw,
    device,
    seed,
    feature_names,
):
    """Explain a seed-specific test sample against a training-data baseline."""
    rng = np.random.default_rng(seed)
    background_data = torch.as_tensor(
        background_scaled, dtype=torch.float32, device=device
    )
    test_indices = rng.choice(len(test_scaled), TEST_SAMPLES, replace=False)
    test_data = torch.as_tensor(
        test_scaled[test_indices], dtype=torch.float32, device=device
    )

    print(f"Computing SHAP values with seed={seed}...")
    explainer = shap.DeepExplainer(model, background_data)
    values = _normalize_shap_values(explainer.shap_values(test_data))
    return shap.Explanation(
        values=values,
        data=test_raw[test_indices],
        feature_names=feature_names,
    )


def run_explanation():
    """Runs SHAP analysis to explain the trained model's predictions."""
    np.random.seed(SEED)

    print("--- Configuration ---")
    print(f"Model Path  : {MODEL_PATH}")
    print(f"Scaler Path : {SCALER_PATH}")
    print(f"Params Path : {BEST_PARAMS_PATH}")
    print(f"Output Dir  : {PLOT_COMBINED_PATH.parent}")
    print("---------------------")

    # -------------------------------------------------------------------------
    # 2. Load training backgrounds and independent test data
    # -------------------------------------------------------------------------
    features = FEATURE_COLUMNS
    print(f"Features list ({len(features)}): {features}")
    background_raw_by_seed, train_num_classes = _sample_training_backgrounds(
        features
    )
    test_raw, test_num_classes = _load_test_features(features)
    if train_num_classes != test_num_classes:
        raise ValueError(
            "Training and test data resolve to different numbers of classes: "
            f"{train_num_classes} != {test_num_classes}."
        )
    num_classes = train_num_classes

    # -------------------------------------------------------------------------
    # 3. Load Scaler & Preprocess
    # -------------------------------------------------------------------------
    print("Loading scaler...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    background_scaled_by_seed = {
        seed: scaler.transform(background)
        for seed, background in background_raw_by_seed.items()
    }
    test_scaled = scaler.transform(test_raw)
    del background_raw_by_seed

    # -------------------------------------------------------------------------
    # 4. Load Model
    # -------------------------------------------------------------------------
    print("Loading model parameters...")
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    device = torch.device("cpu")

    model = create_model_from_params(
        best_params,
        input_dim=len(features),
        num_classes=num_classes,
    )

    print("Loading model weights...")
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(state, dict):
        if "best_model_state_dict" in state:
            state = state["best_model_state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 5. Compute SHAP values for several sampling seeds
    # -------------------------------------------------------------------------
    print(f"  Background samples per seed: {BACKGROUND_SAMPLES}")
    print(f"  Test samples per seed      : {TEST_SAMPLES}")
    print(f"  SHAP sampling seeds        : {SHAP_SEEDS}")
    explanations = [
        _compute_explanation(
            model,
            background_scaled_by_seed[seed],
            test_scaled,
            test_raw,
            device,
            seed,
            features,
        )
        for seed in SHAP_SEEDS
    ]

    # Pool equal-size samples from every seed so the publication plots do not
    # inherit the feature ordering of one arbitrarily selected sample.
    explanation = shap.Explanation(
        values=np.concatenate(
            [explanation.values for explanation in explanations], axis=0
        ),
        data=np.concatenate(
            [explanation.data for explanation in explanations], axis=0
        ),
        feature_names=features,
    )
    shap_vals_to_plot = explanation.values
    X_test_original = explanation.data
    # Keep the beeswarm jitter deterministic while pooling all sampling seeds.
    np.random.seed(SEED)

    # -------------------------------------------------------------------------
    # 6. Generate & Save Combined Plot
    # -------------------------------------------------------------------------
    PLOT_COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["font.size"] = FONT_SIZE

    print("Generating combined SHAP plot...")
    _save_combined_plot(explanation, explanations, features)

if __name__ == "__main__":
    run_explanation()
