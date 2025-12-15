import argparse
from pathlib import Path

import pandas as pd
import torch
import uproot
# PyG Loader
from torch_geometric.loader import DataLoader as PyGDataLoader

from common import (
    E90GraphDataset,
    create_gnn_model_from_params,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PTH_DIR,
    DEFAULT_TUNE_DIR,
    DEFAULT_LABEL_MAPPING,
    get_config_value,
    split_track_features,
    load_best_params,
    resolve_model_output_path,
    _resolve_seed,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    load_data,
    compute_f1,
)


def save_predictions_to_root(input_path: Path, tree_name: str, predictions: list, output_path: Path):
    with uproot.open(input_path) as infile:
        tree = infile[tree_name]
        df = tree.arrays(library="pd")

    if len(predictions) != len(df):
        raise ValueError(
            f"Prediction length ({len(predictions)}) does not match entries in {input_path} ({len(df)})."
        )

    df = df.copy()
    df["out"] = pd.Series(predictions, dtype="int32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(output_path) as outfile:
        outfile[tree_name] = df


def evaluate(config, base_dir):
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    test_cfg = config.get("test", {})
    tuning_cfg = config.get("tuning", {})

    seed = _resolve_seed(test_cfg.get("seed"), config.get("seed"))

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

    feature_cols_dict = split_track_features(features)

    fraction = float(test_cfg.get("fraction", 1.0))

    data_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction,
        random_state=seed,
        shuffle=False,  # Keep ROOT output ordering intact
    )

    # Scaler is unused for the GNN (preprocessing handled in common.py), so no scaler loading here.

    # Dataset
    dataset = E90GraphDataset(data_df, feature_cols_dict, label_column)

    # Hyperparams load
    params, best_params_path = load_best_params(training_cfg, tuning_cfg, base_dir, DEFAULT_TUNE_DIR)

    batch_size = int(test_cfg.get("batch_size", training_cfg.get("batch_size", params.get("batch_size", 128))))
    num_workers = int(test_cfg.get("num_workers", training_cfg.get("num_workers", 0)))

    model_output_path = resolve_model_output_path(training_cfg, base_dir, DEFAULT_PTH_DIR, must_exist=True)

    device = resolve_device(test_cfg.get("device") or training_cfg.get("device") or config.get("device"))
    print(f"Using device: {device}")
    
    # Input Dim = 5
    model = create_gnn_model_from_params(params, input_dim=5, num_classes=num_classes).to(device)

    state_dict = torch.load(model_output_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()
    data_loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total = 0
    correct = 0
    running_loss = 0.0
    records = []
    all_true = []
    all_pred = []
    ordered_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index, batch.batch)
            
            if num_classes == 2:
                logits = outputs.view(-1)
                loss = criterion(logits, batch.y.float())
                probs = torch.sigmoid(logits).cpu()
                preds = (probs > 0.5).long()
                for true_label, pred_label, prob in zip(batch.y.cpu(), preds, probs):
                    records.append(
                        {
                            "true_label": int(true_label),
                            "pred_label": int(pred_label),
                            "prob_signal": float(prob),
                            "prob_background": float(1 - prob),
                        }
                    )
                ordered_preds.extend(preds.cpu().numpy().tolist())
            else:
                loss = criterion(outputs, batch.y)
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                for true_label, pred_label, prob_vec in zip(batch.y.cpu(), preds, probs):
                    row = {
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                    }
                    for cls_idx in range(num_classes):
                        row[f"prob_{cls_idx}"] = float(prob_vec[cls_idx])
                    records.append(row)
                ordered_preds.extend(preds.cpu().numpy().tolist())

            total += batch.num_graphs
            correct += (preds.to(batch.y.device) == batch.y).sum().item()
            running_loss += loss.item() * batch.num_graphs
            all_true.extend(batch.y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0
    f1 = compute_f1(all_true, all_pred, num_classes) if all_true else 0.0
    print(f"Test F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}")

    predictions_output_raw = get_config_value(
        test_cfg, "predictions_output_file", "predictions_output_path"
    ) or "test_predictions.csv"
    predictions_output_path = resolve_dir(predictions_output_raw, DEFAULT_OUTPUT_DIR, base_dir)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(predictions_output_path, index=False)
    print(f"Saved test predictions to '{predictions_output_path}'.")

    # ROOT output logic (same behavior as previous test.py using files[0])
    if len(files) == 1 and fraction == 1.0:
        root_output_raw = get_config_value(test_cfg, "root_output_file", "root_output_path") or Path(files[0]).name
        root_output_path = resolve_dir(root_output_raw, DEFAULT_OUTPUT_DIR, base_dir)
        # save_predictions_to_root remains pandas-based; adapt if a different output format is needed.
        # Kept identical to the previous test.py behavior.
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation (GNN).")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    evaluate(config, base_dir)
