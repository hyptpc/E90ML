import argparse
import json
import pickle
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    E90Dataset,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PTH_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_TUNE_DIR,
    _resolve_path,
    create_model_from_params,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    load_data,
)


def _plot_history(train_values, val_values, ylabel, out_path):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_values) + 1)
    plt.figure()
    plt.plot(epochs, train_values, label="train")
    plt.plot(epochs, val_values, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def train_final(config, base_dir):
    """
    Train the final model with stratified split, leak-safe scaling, class weighting, and early stopping.
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("tuning", {})

    seed = training_cfg.get("seed") or config.get("seed")
    if seed is None:
        raise ValueError("Config must define 'seed' (preferably under training.seed) for reproducible splits.")
    seed = int(seed)

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

    train_fraction = float(training_cfg["fraction"])
    val_split = float(training_cfg["val_split"])
    num_workers = int(training_cfg["num_workers"])
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    batch_size_override = training_cfg.get("batch_size")

    # Load Data (Full)
    full_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=train_fraction,
        random_state=seed,
    )

    feature_matrix = full_df[features].values
    labels = full_df[label_column].values

    # Stratified Split
    train_features, val_features, train_labels, val_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    # Scale (Fit on Train ONLY)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Save Scaler for future inference
    scaler_output_path = resolve_dir(training_cfg["scaler_output_path"], DEFAULT_PTH_DIR, base_dir)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_output_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to '{scaler_output_path}'.")

    # Datasets & Loaders
    train_dataset = E90Dataset(train_features, train_labels)
    val_dataset = E90Dataset(val_features, val_labels)

    # Load Tuned Hyperparameters
    best_params_raw = training_cfg.get("best_params_path") or tuning_cfg.get("best_params_path")
    best_params_path = resolve_dir(best_params_raw, DEFAULT_TUNE_DIR, base_dir)
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"Best parameter file not found at {best_params_path}. Run tuning first or update the config."
        )

    with best_params_path.open() as f:
        params = json.load(f)
    print("Loaded parameters:", params)

    model_params = {
        "n_layers": int(params.get("n_layers", 2)),
        "hidden_units": int(params.get("hidden_units", 128)),
        "dropout_rate": float(params.get("dropout_rate", 0.2)),
    }
    batch_size = batch_size_override or int(params.get("batch_size", 128))
    lr = float(params.get("lr", 1e-3))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(training_cfg.get("device") or config.get("device"))
    print(f"Using device: {device}")
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Weighted Loss
    if num_classes == 2:
        num_pos = (train_labels == 1).sum()
        num_neg = (train_labels == 0).sum()
        pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device) if num_pos > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None:
            print(f"Using Weighted BCE Loss. Pos Weight: {pos_weight.item():.4f}")
        else:
            print("Using BCE Loss without class weighting (no positive labels found).")
    else:
        criterion = nn.CrossEntropyLoss()

    # Training State
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if num_classes == 2:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if num_classes == 2:
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels.float())
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                val_running_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Early Stopping & Best Model Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            # Optional: save checkpoint here
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    # Load best model weights
    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)

    model_output_path = resolve_dir(training_cfg["model_output_path"], DEFAULT_PTH_DIR, base_dir)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Best model saved to '{model_output_path}'.")

    # Plotting
    plots_dir = resolve_dir(training_cfg["plots_dir"], DEFAULT_PLOTS_DIR, base_dir)
    _plot_history(history["train_loss"], history["val_loss"], "Loss", plots_dir / "loss.png")
    _plot_history(history["train_acc"], history["val_acc"], "Accuracy", plots_dir / "accuracy.png")
    print(f"Saved training curves to '{plots_dir}'.")

    # Metrics
    metrics_output_path = resolve_dir(training_cfg["metrics_output_path"], DEFAULT_OUTPUT_DIR, base_dir)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
        "best_val_acc": best_val_acc,
        "epochs_run": len(history["train_loss"]),
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_classes": num_classes,
        "features": features,
    }
    with metrics_output_path.open("w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved metrics to '{metrics_output_path}'.")

    # Predictions (on FULL dataset or Test set)
    # Here we predict on the Validation dataset
    predictions_output_path = resolve_dir(training_cfg["predictions_output_path"], DEFAULT_OUTPUT_DIR, base_dir)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    val_loader_seq = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    records = []
    with torch.no_grad():
        for inputs, labels in val_loader_seq:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if num_classes == 2:
                probs = torch.sigmoid(outputs.view(-1)).cpu()
                preds = (probs > 0.5).long()
                for true_label, pred_label, prob in zip(labels, preds, probs):
                    records.append(
                        {
                            "true_label": int(true_label),
                            "pred_label": int(pred_label),
                            "prob_signal": float(prob),
                            "prob_background": float(1 - prob),
                        }
                    )
            else:
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                for true_label, pred_label, prob_vec in zip(labels, preds, probs):
                    row = {
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                    }
                    for cls_idx in range(num_classes):
                        row[f"prob_{cls_idx}"] = float(prob_vec[cls_idx])
                    records.append(row)

    pd.DataFrame.from_records(records).to_csv(predictions_output_path, index=False)
    print(f"Saved predictions (validation set) to '{predictions_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Final training script using tuned hyperparameters.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    train_final(config, base_dir)
