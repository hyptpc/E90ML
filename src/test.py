import argparse
import pickle
import sys
import json
from pathlib import Path

import pandas as pd
import torch
import uproot
from torch.utils.data import DataLoader

from common import (
    E90Dataset,
    DEFAULT_PTH_DIR,
    DEFAULT_TUNE_DIR,
    DEFAULT_LABEL_MAPPING,
    get_config_value,
    _resolve_seed,
    create_model_from_params,
    load_config,
    resolve_device,
    resolve_dir,
    load_data,
)


def _require(cfg: dict, key: str, section: str):
    """Fetch a required config value, raising if missing/empty."""
    value = cfg.get(key)
    if value in (None, ""):
        raise ValueError(f"Config must set '{section}.{key}'.")
    return value


def save_predictions_to_root(input_path: Path, tree_name: str, predictions: list, output_path: Path):
    """
    Reads the input ROOT file, appends the predicted labels as a new branch named 'out',
    and saves the result to a new ROOT file.
    """
    print(f"Opening input ROOT file: {input_path}")
    
    with uproot.open(input_path) as infile:
        if tree_name not in infile:
            raise KeyError(f"Tree '{tree_name}' not found in {input_path}")
        
        tree = infile[tree_name]
        df = tree.arrays(library="pd")

    if len(predictions) != len(df):
        raise ValueError(
            f"Prediction length ({len(predictions)}) does not match entries in {input_path} ({len(df)}). "
            "Ensure 'test.fraction' is set to 1.0."
        )

    df = df.copy()
    df["out"] = pd.Series(predictions, dtype="int32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert DataFrame to a dictionary of numpy arrays to avoid uproot write errors
    branch_dict = {col: df[col].to_numpy() for col in df.columns}
    
    print(f"Writing output to: {output_path}")
    with uproot.recreate(output_path) as outfile:
        outfile[tree_name] = branch_dict


def evaluate(config, base_dir):
    """
    Main evaluation routine.
    """
    # Determine Project Root based on script location (src/test.py -> parent -> E90ML/)
    project_root = Path(__file__).resolve().parent.parent

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    test_cfg = config.get("test", {})
    tuning_cfg = config.get("tuning", {})

    # 1. Setup Seed
    seed = _resolve_seed(test_cfg.get("seed"), config.get("seed"))

    # -------------------------------------------------------------------------
    # 2. Resolve Input Files
    # -------------------------------------------------------------------------
    # Priority 1: Check if 'input_file' is defined in the [TEST] section
    test_input_raw = _require(test_cfg, "input_file", "test")

    default_input_dir = project_root / "data" / "input"
    test_file_path = resolve_dir(test_input_raw, default_input_dir, project_root)
    if not test_file_path.exists():
        test_file_path = resolve_dir(test_input_raw, default_input_dir, base_dir)

    if not test_file_path.exists():
        raise FileNotFoundError(f"Test input file not found: {test_file_path}")

    # Pass as a LIST for load_data
    files = [str(test_file_path)]
    print(f"Using test input file from config: {test_file_path}")

    # Constraint for ROOT output: Must be exactly one file
    if len(files) != 1:
        raise ValueError(
            f"ROOT output mode expects exactly ONE input file, but found {len(files)}. "
            "Please specify 'test.input_file' in your YAML."
        )

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING
    if tree_name is None or features is None or label_column is None:
        raise ValueError("Config must define tree_name, feature_columns, and label_column under data.")

    # 3. Load Data
    fraction = float(_require(test_cfg, "fraction", "test"))
    
    print("Loading data...")
    data_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction,
        random_state=seed,
        shuffle=False, 
    )

    # 4. Load Scaler
    scaler_raw = get_config_value(training_cfg, "scaler_output_file", "scaler_output_path")
    if not scaler_raw:
        raise ValueError("Config must set training.scaler_output_file.")
    
    # Use project_root for scaler path
    scaler_path = resolve_dir(scaler_raw, DEFAULT_PTH_DIR, project_root)
    if not scaler_path.exists():
        scaler_path = resolve_dir(scaler_raw, DEFAULT_PTH_DIR, base_dir)
        
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train the model first.")
    
    print(f"Loading scaler from: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Transform features
    feature_matrix = scaler.transform(data_df[features].values)
    labels = data_df[label_column].values

    # Create Dataset
    dataset = E90Dataset(feature_matrix, labels)

    # 5. Load Model Parameters
    best_params_raw = get_config_value(training_cfg, "best_params_file", "best_params_path") or get_config_value(
        tuning_cfg, "tune_params_file", "best_params_file", "best_params_path"
    )
    if not best_params_raw:
        raise ValueError("Config must set training.best_params_file or tuning.tune_params_file.")
    
    best_params_path = resolve_dir(best_params_raw, DEFAULT_TUNE_DIR, project_root)
    if not best_params_path.exists():
        best_params_path = resolve_dir(best_params_raw, DEFAULT_TUNE_DIR, base_dir)

    if not best_params_path.exists():
        raise FileNotFoundError(f"Best parameter file not found at {best_params_path}.")
    
    with best_params_path.open() as f:
        params = json.load(f)

    model_params = {
        "n_layers": int(params.get("n_layers", 2)),
        "hidden_units": int(params.get("hidden_units", 128)),
        "dropout_rate": float(params.get("dropout_rate", 0.2)),
    }
    
    batch_size = int(_require(test_cfg, "batch_size", "test"))
    num_workers = int(_require(test_cfg, "num_workers", "test"))

    # 6. Load Trained Model Weights
    model_output_raw = get_config_value(training_cfg, "model_output_file", "model_output_path")
    if not model_output_raw:
        raise ValueError("Config must set training.model_output_file.")
    
    model_output_path = resolve_dir(model_output_raw, DEFAULT_PTH_DIR, project_root)
    if not model_output_path.exists():
        model_output_path = resolve_dir(model_output_raw, DEFAULT_PTH_DIR, base_dir)

    if not model_output_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_output_path}.")

    device = resolve_device(test_cfg.get("device") or training_cfg.get("device") or config.get("device"))
    print(f"Using device: {device}")
    
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)

    state_dict = torch.load(model_output_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    model.load_state_dict(state_dict)
    model.eval()

    # 7. Run Inference
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total = 0
    correct = 0
    ordered_preds = [] 
    ordered_labels = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(inputs)
            
            if num_classes == 2:
                logits = outputs.view(-1)
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).long()
                ordered_preds.extend(preds.cpu().numpy().tolist())
            else:
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                ordered_preds.extend(preds.cpu().numpy().tolist())

            total += labels_batch.size(0)
            correct += (preds.to(labels_batch.device) == labels_batch).sum().item()
            ordered_labels.extend(labels_batch.cpu().numpy().tolist())

    accuracy = correct / total if total > 0 else 0.0
    print(f"Inference complete. Accuracy: {accuracy:.4f}")

    # Event-level summary
    positive_label = 1  # signal is mapped to 1 by load_data when label_mapping is provided
    true_signal_count = sum(1 for lbl in ordered_labels if lbl == positive_label)
    predicted_signal_count = sum(1 for pred in ordered_preds if pred == positive_label)
    true_signal_predicted = sum(
        1 for pred, lbl in zip(ordered_preds, ordered_labels) if pred == positive_label and lbl == positive_label
    )

    def _pct(numerator: int, denominator: int) -> float:
        return (numerator / denominator * 100.0) if denominator else 0.0

    print("Event counts:")
    print(f"  Total events: {total}")
    print(f"  SigmaNCusp (true signal) events: {true_signal_count}")
    print(f"  Predicted as signal: {predicted_signal_count}")
    print(
        "  SigmaNCusp predicted as signal: "
        f"{true_signal_predicted} "
        f"({_pct(true_signal_predicted, predicted_signal_count):.2f}% of predicted signals)"
    )

    # 8. Save Output to ROOT
    root_output_raw = get_config_value(test_cfg, "output_file", "root_output_path")
    if not root_output_raw:
        raise ValueError("Config must set 'test.output_file'.")

    # [FIX] Explicitly construct the output path relative to E90ML/data/output
    # This overrides potential misconfiguration in DEFAULT_OUTPUT_DIR
    default_output_dir = project_root / "data" / "output"
    root_output_path = resolve_dir(root_output_raw, default_output_dir, project_root)
        
    # Determine the input ROOT file path
    if isinstance(files, list):
        input_root_file = Path(files[0])
    elif isinstance(files, dict):
        input_root_file = Path(list(files.values())[0])
    else:
        input_root_file = Path(str(files))

    save_predictions_to_root(
        input_path=input_root_file,
        tree_name=tree_name,
        predictions=ordered_preds,
        output_path=root_output_path,
    )
    print(f"Successfully saved ROOT file with predictions to '{root_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on the dataset using a trained model.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    evaluate(config, base_dir)
