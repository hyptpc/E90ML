import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    E90Dataset,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PTH_DIR,
    DEFAULT_TUNE_DIR,
    DEFAULT_LABEL_MAPPING,
    _resolve_seed,
    create_model_from_params,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    load_data,
)


def _get_config_value(cfg: dict, *keys: str) -> Optional[str]:
    for key in keys:
        value = cfg.get(key)
        if value not in (None, ""):
            return value
    return None


def evaluate(config, base_dir):
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    testing_cfg = config.get("testing", {})
    tuning_cfg = config.get("tuning", {})

    seed = _resolve_seed(testing_cfg.get("seed"), config.get("seed"))

    files = resolve_data_files(data_cfg, base_dir)
    if not files:
        raise ValueError("Config must provide data.files with at least one entry.")

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING
    if tree_name is None or features is None or label_column is None:
        raise ValueError("Config must define tree_name, feature_columns, and label_column under data.")

    fraction = float(testing_cfg.get("fraction", 1.0))

    data_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction,
        random_state=seed,
    )

    scaler_raw = _get_config_value(training_cfg, "scaler_output_file", "scaler_output_path")
    if not scaler_raw:
        raise ValueError("Config must set training.scaler_output_file.")
    scaler_path = resolve_dir(scaler_raw, DEFAULT_PTH_DIR, base_dir)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train the model first.")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_matrix = scaler.transform(data_df[features].values)
    labels = data_df[label_column].values

    dataset = E90Dataset(feature_matrix, labels)

    # Model + hyperparameters
    best_params_raw = _get_config_value(training_cfg, "best_params_file", "best_params_path") or _get_config_value(
        tuning_cfg, "tune_params_file", "best_params_file", "best_params_path"
    )
    if not best_params_raw:
        raise ValueError("Config must set training.best_params_file or tuning.tune_params_file.")
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
    batch_size = int(
        testing_cfg.get(
            "batch_size",
            training_cfg.get("batch_size", params.get("batch_size", 128)),
        )
    )
    num_workers = int(testing_cfg.get("num_workers", training_cfg.get("num_workers", 0)))

    model_output_raw = _get_config_value(training_cfg, "model_output_file", "model_output_path")
    if not model_output_raw:
        raise ValueError("Config must set training.model_output_file.")
    model_output_path = resolve_dir(model_output_raw, DEFAULT_PTH_DIR, base_dir)
    if not model_output_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_output_path}. Train the model first.")

    device = resolve_device(testing_cfg.get("device") or training_cfg.get("device") or config.get("device"))
    print(f"Using device: {device}")
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)

    state_dict = torch.load(model_output_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total = 0
    correct = 0
    running_loss = 0.0
    records = []

    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(inputs)
            if num_classes == 2:
                logits = outputs.view(-1)
                loss = criterion(logits, labels_batch.float())
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).long()
                for true_label, pred_label, prob in zip(labels_batch.cpu(), preds, probs):
                    records.append(
                        {
                            "true_label": int(true_label),
                            "pred_label": int(pred_label),
                            "prob_signal": float(prob),
                            "prob_background": float(1 - prob),
                        }
                    )
            else:
                loss = criterion(outputs, labels_batch)
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                for true_label, pred_label, prob_vec in zip(labels_batch.cpu(), preds, probs):
                    row = {
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                    }
                    for cls_idx in range(num_classes):
                        row[f"prob_{cls_idx}"] = float(prob_vec[cls_idx])
                    records.append(row)

            total += labels_batch.size(0)
            correct += (preds.to(labels_batch.device) == labels_batch).sum().item()
            running_loss += loss.item() * labels_batch.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0

    predictions_output_raw = _get_config_value(
        testing_cfg, "predictions_output_file", "predictions_output_path"
    ) or "test_predictions.csv"
    predictions_output_path = resolve_dir(predictions_output_raw, DEFAULT_OUTPUT_DIR, base_dir)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(predictions_output_path, index=False)
    print(f"Saved test predictions to '{predictions_output_path}'.")

    metrics_output_raw = _get_config_value(testing_cfg, "metrics_output_file", "metrics_output_path") or "test_metrics.json"
    metrics_output_path = resolve_dir(metrics_output_raw, DEFAULT_OUTPUT_DIR, base_dir)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "num_samples": total,
        "num_classes": num_classes,
        "features": features,
        "batch_size": batch_size,
        "model_path": str(model_output_path),
        "scaler_path": str(scaler_path),
        "best_params_path": str(best_params_path),
        "fraction": fraction,
        "seed": seed,
    }
    with metrics_output_path.open("w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved test metrics to '{metrics_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on the dataset using a trained model.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    evaluate(config, base_dir)
