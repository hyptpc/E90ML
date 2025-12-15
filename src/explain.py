import argparse
import json
import pickle
import sys
import numpy as np
import torch
import shap
import matplotlib
# Use 'Agg' backend to avoid errors on servers without GUI (X11)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
from common import (
    load_config,
    resolve_data_files,
    load_data,
    create_model_from_params,
    resolve_dir,
    apply_plot_style,
    DEFAULT_LABEL_MAPPING,
    PTH_DIR,
    TUNE_DIR,
    OUTPUT_DIR,
)

def run_explanation(config_path):
    """
    Runs SHAP analysis to explain the trained model's predictions.
    """
    
    # -------------------------------------------------------------------------
    # 1. Configuration & Path Definitions
    # -------------------------------------------------------------------------
    config, base_dir = load_config(config_path)
    data_cfg = config.get("data", {})
    explain_cfg = config.get("explain", {})
    train_cfg = config.get("training", {})
    tune_cfg = config.get("tuning", {})

    # Determine best_params_file, checking training then tuning config
    best_params_file = train_cfg.get("best_params_file") or tune_cfg.get("best_params_file")
    if not best_params_file:
        raise ValueError("Could not find 'best_params_file' in training or tuning config.")

    # Define all input/output paths at the beginning for clarity
    paths = {
        # Inputs
        "scaler": resolve_dir(
            train_cfg.get("scaler_output_file"), PTH_DIR, base_dir
        ),
        "model": resolve_dir(
            train_cfg.get("model_output_file"), PTH_DIR, base_dir
        ),
        "best_params": resolve_dir(
            best_params_file, 
            TUNE_DIR, 
            base_dir
        ),
        # Outputs
        "plot_summary": resolve_dir("shap_summary.png", OUTPUT_DIR, base_dir),
        "plot_bar": resolve_dir("shap_importance_bar.png", OUTPUT_DIR, base_dir)
    }

    print("--- Configuration ---")
    print(f"Config File : {config_path}")
    print(f"Model Path  : {paths['model']}")
    print(f"Scaler Path : {paths['scaler']}")
    print(f"Output Dir  : {paths['plot_summary'].parent}")
    print("---------------------")

    # -------------------------------------------------------------------------
    # 2. Load Data (Sampled)
    # -------------------------------------------------------------------------
    print("Loading data...")
    files = resolve_data_files(data_cfg, base_dir)
    features = data_cfg.get("feature_columns")
    tree_name = data_cfg.get("tree_name")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING
    if tree_name is None or features is None or label_column is None:
        raise ValueError("Config must define tree_name, feature_columns, and label_column under data.")

    # Determine sampling fraction for SHAP from YAML (explain.fraction or data.fraction)
    fraction = explain_cfg.get("fraction", data_cfg.get("fraction"))
    if fraction is None:
        raise ValueError("Config must define 'explain.fraction' or 'data.fraction' for SHAP sampling.")
    fraction = float(fraction)
    
    # [Debug] Print features to ensure they are loaded correctly
    print(f"Features list ({len(features)}): {features}")

    # Load a fraction of data. SHAP is computationally expensive.
    df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction, 
        random_state=42
    )
    
    X_raw = df[features].values.astype(np.float32)
    del df # Free memory

    # -------------------------------------------------------------------------
    # 3. Load Scaler & Preprocess
    # -------------------------------------------------------------------------
    print("Loading scaler...")
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    
    X_scaled = scaler.transform(X_raw)

    # -------------------------------------------------------------------------
    # 4. Load Model
    # -------------------------------------------------------------------------
    print("Loading model parameters...")
    with open(paths["best_params"], "r") as f:
        best_params = json.load(f)

    device = torch.device("cpu")
    
    model = create_model_from_params(
        best_params, 
        input_dim=len(features), 
        num_classes=num_classes
    )
    
    print("Loading model weights...")
    state = torch.load(paths["model"], map_location=device)
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
    n_background = 200
    n_test = 500
    
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
    
    paths["plot_summary"].parent.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(paths["plot_summary"], bbox_inches='tight')
    plt.close()
    print(f"Saved: {paths['plot_summary']}")

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
    plt.savefig(paths["plot_bar"], bbox_inches='tight')
    plt.close()
    print(f"Saved: {paths['plot_bar']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP explanation for E90ML model.")
    parser.add_argument("config", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    run_explanation(args.config)
