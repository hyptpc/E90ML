import json
import pickle
from pathlib import Path
import numpy as np
import torch
import shap
import matplotlib
# Use 'Agg' backend to avoid errors on servers without GUI (X11)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
from common import (
    load_data,
    create_model_from_params,
    apply_plot_style,
    DEFAULT_LABEL_MAPPING,
)

# -----------------------------------------------------------------------------
# User-editable parameters (no YAML required)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_FILES = [
    PROJECT_ROOT / "data" / "input" / "SigmaNCusp_mm.root",
    PROJECT_ROOT / "data" / "input" / "QFLambda_mm.root",
    PROJECT_ROOT / "data" / "input" / "QFSigmaZ_mm.root",
]
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
    "mm",
]
LABEL_MAPPING = DEFAULT_LABEL_MAPPING  # remap to binary: signal=1, background=0
SAMPLE_FRACTION = 0.1  # fraction of data for SHAP (keep small; SHAP is expensive)
SEED = 42

# Model artifacts
SCALER_PATH = PROJECT_ROOT / "param" / "pth" / "example.pkl"
MODEL_PATH = PROJECT_ROOT / "param" / "pth" / "example.pth"
BEST_PARAMS_PATH = PROJECT_ROOT / "param" / "tune" / "tuned_params.json"

# Output paths
PLOT_SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "shap_summary.png"
PLOT_BAR_PATH = PROJECT_ROOT / "data" / "output" / "shap_importance_bar.png"

# SHAP sampling sizes
BACKGROUND_SAMPLES = 200
TEST_SAMPLES = 500


def run_explanation():
    """Runs SHAP analysis to explain the trained model's predictions."""
    np.random.seed(SEED)

    print("--- Configuration ---")
    print(f"Model Path  : {MODEL_PATH}")
    print(f"Scaler Path : {SCALER_PATH}")
    print(f"Params Path : {BEST_PARAMS_PATH}")
    print(f"Output Dir  : {PLOT_SUMMARY_PATH.parent}")
    print("---------------------")

    # -------------------------------------------------------------------------
    # 2. Load Data (Sampled)
    # -------------------------------------------------------------------------
    print("Loading data...")
    files = [str(p) for p in DATA_FILES]
    features = FEATURE_COLUMNS
    tree_name = TREE_NAME
    label_column = LABEL_COLUMN
    label_mapping = LABEL_MAPPING

    print(f"Features list ({len(features)}): {features}")

    # Load a fraction of data. SHAP is computationally expensive.
    df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=float(SAMPLE_FRACTION),
        random_state=SEED,
    )
    
    X_raw = df[features].values.astype(np.float32)
    del df # Free memory

    # -------------------------------------------------------------------------
    # 3. Load Scaler & Preprocess
    # -------------------------------------------------------------------------
    print("Loading scaler...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    X_scaled = scaler.transform(X_raw)

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
        num_classes=num_classes
    )
    
    print("Loading model weights...")
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict):
        if "best_model_state_dict" in state:
            state = state["best_model_state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 5. Compute SHAP Values
    # -------------------------------------------------------------------------
    n_background = BACKGROUND_SAMPLES
    n_test = TEST_SAMPLES
    
    if len(X_scaled) < (n_background + n_test):
        raise ValueError("Not enough data loaded for SHAP sampling.")

    indices = np.random.choice(len(X_scaled), n_background + n_test, replace=False)
    idx_bg = indices[:n_background]
    idx_test = indices[n_background:]

    background_data = torch.tensor(X_scaled[idx_bg], device=device)
    test_data = torch.tensor(X_scaled[idx_test], device=device)

    print(f"  Background samples: {n_background}")
    print(f"  Test samples      : {n_test}")
    
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data)

    # Handle SHAP output format and select Class 1 (Signal)
    if isinstance(shap_values, list):
        print(f"SHAP returned a list of length: {len(shap_values)}")
        shap_vals_to_plot = shap_values[1] 
    else:
        print("SHAP returned a single tensor.")
        shap_vals_to_plot = shap_values

    # [Fix] Squeeze the last dimension if it exists: (500, 13, 1) -> (500, 13)
    if len(shap_vals_to_plot.shape) == 3 and shap_vals_to_plot.shape[2] == 1:
        print(f"Squeezing extra dimension: {shap_vals_to_plot.shape} -> ", end="")
        shap_vals_to_plot = np.squeeze(shap_vals_to_plot, axis=2)
        print(f"{shap_vals_to_plot.shape}")

    # -------------------------------------------------------------------------
    # 6. Generate & Save Plots
    # -------------------------------------------------------------------------
    X_test_original = scaler.inverse_transform(test_data.cpu().numpy())
    
    PLOT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    # Plot A: Summary Plot (Beeswarm)
    print("Generating summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_vals_to_plot, 
        X_test_original, 
        feature_names=features,
        show=False,
        plot_size=(10, 8) 
    )
    plt.savefig(PLOT_SUMMARY_PATH, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_SUMMARY_PATH}")

    # Plot B: Bar Plot (Global Importance)
    print("Generating importance bar plot...")
    plt.figure()
    shap.summary_plot(
        shap_vals_to_plot, 
        X_test_original, 
        feature_names=features, 
        plot_type="bar",
        show=False,
        plot_size=(10, 8)
    )
    plt.savefig(PLOT_BAR_PATH, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_BAR_PATH}")

if __name__ == "__main__":
    run_explanation()
