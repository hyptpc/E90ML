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
    apply_plot_style
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
    train_cfg = config.get("training", {})
    tune_cfg = config.get("tuning", {})

    # Define all input/output paths at the beginning for clarity
    paths = {
        # Inputs
        "scaler": resolve_dir(
            train_cfg.get("scaler_output_file"), base_dir / "param/pth", base_dir
        ),
        "model": resolve_dir(
            train_cfg.get("model_output_file"), base_dir / "param/pth", base_dir
        ),
        "best_params": resolve_dir(
            train_cfg.get("best_params_file", "tuned_params.json"), 
            base_dir / "param/tune", 
            base_dir
        ),
        # Outputs
        "plot_summary": base_dir / "data/output/shap_summary.png",
        "plot_bar": base_dir / "data/output/shap_importance_bar.png"
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
    
    # Load a fraction of data. SHAP is computationally expensive,
    # so we don't need the full dataset (10% is usually enough for analysis).
    df, num_classes = load_data(
        files=files,
        tree_name=data_cfg.get("tree_name"),
        features=features,
        label_column=data_cfg.get("label_column"),
        fraction=0.1, 
        random_state=42
    )
    
    # Convert to numpy array (float32)
    X_raw = df[features].values.astype(np.float32)
    del df # Free memory

    # -------------------------------------------------------------------------
    # 3. Load Scaler & Preprocess
    # -------------------------------------------------------------------------
    print("Loading scaler...")
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the data (Model expects scaled input like N(0,1))
    X_scaled = scaler.transform(X_raw)

    # -------------------------------------------------------------------------
    # 4. Load Model
    # -------------------------------------------------------------------------
    print("Loading model parameters...")
    with open(paths["best_params"], "r") as f:
        best_params = json.load(f)

    # Use CPU for SHAP to avoid complex GPU memory management with DeepExplainer
    device = torch.device("cpu")
    
    model = create_model_from_params(
        best_params, 
        input_dim=len(features), 
        num_classes=num_classes
    )
    
    print("Loading model weights...")
    model.load_state_dict(torch.load(paths["model"], map_location=device))
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 5. Compute SHAP Values
    # -------------------------------------------------------------------------
    # SHAP DeepExplainer needs:
    #   1. Background samples: To establish a "baseline" (e.g., 200 samples).
    #   2. Test samples: The samples we want to explain (e.g., 500 samples).
    
    n_background = 200
    n_test = 500
    
    # Randomly sample indices
    if len(X_scaled) < (n_background + n_test):
        raise ValueError("Not enough data loaded for SHAP sampling.")

    indices = np.random.choice(len(X_scaled), n_background + n_test, replace=False)
    idx_bg = indices[:n_background]
    idx_test = indices[n_background:]

    # Create tensors
    background_data = torch.tensor(X_scaled[idx_bg], device=device)
    test_data = torch.tensor(X_scaled[idx_test], device=device)

    print(f"Calculating SHAP values using DeepExplainer...")
    print(f"  Background samples: {n_background}")
    print(f"  Test samples      : {n_test}")
    
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data)

    # Handle SHAP output format
    # DeepExplainer returns a list of arrays if the model has multiple outputs (or classes).
    # For binary classification (if num_classes=2 with CrossEntropy), it returns [shap_class0, shap_class1].
    # We are interested in Class 1 (Signal).
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1] 
    else:
        # If output is single tensor (e.g. BCEWithLogitsLoss outputting 1 value), use as is.
        shap_vals_to_plot = shap_values

    # -------------------------------------------------------------------------
    # 6. Generate & Save Plots
    # -------------------------------------------------------------------------
    # Inverse transform test data to original units (e.g., MeV) for better readability
    X_test_original = scaler.inverse_transform(test_data.cpu().numpy())
    
    # Ensure output directory exists
    paths["plot_summary"].parent.mkdir(parents=True, exist_ok=True)

    # Apply plotting style
    apply_plot_style()

    # Plot A: Summary Plot (Beeswarm)
    # Shows the distribution of impacts for each feature
    print("Generating summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_vals_to_plot, 
        X_test_original, 
        feature_names=features,
        show=False
    )
    plt.savefig(paths["plot_summary"], bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {paths['plot_summary']}")

    # Plot B: Bar Plot (Global Importance)
    # Shows the average absolute impact of each feature
    print("Generating importance bar plot...")
    plt.figure()
    shap.summary_plot(
        shap_vals_to_plot, 
        X_test_original, 
        feature_names=features, 
        plot_type="bar",
        show=False
    )
    plt.savefig(paths["plot_bar"], bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {paths['plot_bar']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP explanation for E90ML model.")
    parser.add_argument("config", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    run_explanation(args.config)