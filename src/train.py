import argparse
import copy
import json
import pickle
import random
import gc
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    E90Dataset,
    OUTPUT_DIR,
    PTH_DIR,
    LABEL_MAPPING,
    TUNE_DIR,
    get_config_value,
    apply_plot_style,
    create_model_from_params,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
    compute_f1,
)


def _plot_training_curves(history, out_path: Path):
    import matplotlib.pyplot as plt

    apply_plot_style()

    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_f1"], c="blue", label="train", linestyle="-")
    plt.plot(history["val_f1"], c="red", label="val", linestyle="-")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1-score")
    plt.title("Training and validation F1-score")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], c="blue", label="train", linestyle="-")
    plt.plot(history["val_loss"], c="red", label="val", linestyle="-")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def train_final(config, base_dir):
    """
    Train the final model with stratified split, leak-safe scaling, class weighting, and early stopping.
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("tuning", {})

    seed = _resolve_seed(training_cfg.get("seed"), config.get("seed"))
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    files = resolve_data_files(data_cfg, base_dir)
    if not files:
        raise ValueError("Config must provide data.files with at least one entry.")

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
    if tree_name is None or features is None or label_column is None:
        raise ValueError("Config must define tree_name, feature_columns, and label_column under data.")

    train_fraction = float(training_cfg["fraction"])
    val_split = float(training_cfg["val_split"])
    num_workers = int(training_cfg["num_workers"])
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    batch_size_override = training_cfg.get("batch_size")

    # Load Data (Full)
    print("Loading data...")
    full_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=train_fraction,
        random_state=seed,
    )

    # --- Memory Optimization ---
    print("Processing data...")
    feature_matrix = full_df[features].values.astype(np.float32)
    labels = full_df[label_column].values.astype(np.int64)
    del full_df
    gc.collect()

    train_features, val_features, train_labels, val_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    del feature_matrix, labels
    gc.collect()

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    # ---------------------------

    # Save Scaler for future inference
    scaler_output_raw = get_config_value(training_cfg, "scaler_output_file", "scaler_output_path")
    if not scaler_output_raw:
        raise ValueError("Config must set training.scaler_output_file.")
    scaler_output_path = resolve_dir(scaler_output_raw, PTH_DIR, base_dir)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_output_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to '{scaler_output_path}'.")

    # Datasets & Loaders
    train_dataset = E90Dataset(train_features, train_labels)
    val_dataset = E90Dataset(val_features, val_labels)
    train_label_counts = np.bincount(train_dataset.y, minlength=2) if num_classes == 2 else None
    
    del train_features, val_features, train_labels, val_labels
    gc.collect()

    # Load Tuned Hyperparameters
    best_params_raw = get_config_value(training_cfg, "best_params_file", "best_params_path") or get_config_value(
        tuning_cfg, "tune_params_file", "best_params_file", "best_params_path"
    )
    if not best_params_raw:
        raise ValueError("Config must set training.best_params_file or tuning.tune_params_file.")

    best_params_path = resolve_dir(best_params_raw, TUNE_DIR, base_dir)
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"Best parameter file not found at {best_params_path}. Run tuning first or update the config."
        )

    model_output_raw = get_config_value(training_cfg, "model_output_file", "model_output_path")
    if not model_output_raw:
        raise ValueError("Config must set training.model_output_file.")
    model_output_path = resolve_dir(model_output_raw, PTH_DIR, base_dir)

    checkpoint_raw = get_config_value(training_cfg, "checkpoint_file", "checkpoint_path")
    checkpoint_path = resolve_dir(checkpoint_raw or model_output_raw, PTH_DIR, base_dir)

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

    device = resolve_device(config.get("device"))
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Weighted Loss
    if num_classes == 2:
        num_pos = float(train_label_counts[1]) if train_label_counts is not None else 0.0
        num_neg = float(train_label_counts[0]) if train_label_counts is not None else 0.0
        pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device) if num_pos > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None:
            print(f"Using Weighted BCE Loss. Pos Weight: {pos_weight.item():.4f}")
        else:
            print("Using BCE Loss without class weighting (no positive labels found).")
    else:
        criterion = nn.CrossEntropyLoss()

    # Training State
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint.get("model_state_dict", model.state_dict()))
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            ckpt_history = checkpoint.get("history")
            if ckpt_history:
                history["train_loss"] = ckpt_history.get("train_loss", history["train_loss"])
                history["val_loss"] = ckpt_history.get("val_loss", history["val_loss"])
                history["train_f1"] = ckpt_history.get("train_f1", ckpt_history.get("train_acc", []))
                history["val_f1"] = ckpt_history.get("val_f1", ckpt_history.get("val_acc", []))
            best_val_f1 = float(checkpoint.get("best_val_f1", checkpoint.get("best_val_acc", best_val_f1)))
            best_model_wts = checkpoint.get("best_model_state_dict", best_model_wts)
            no_improve_count = int(checkpoint.get("no_improve_count", no_improve_count))
            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"Resuming training from epoch {start_epoch + 1} using checkpoint '{checkpoint_path}'.")
        else:
            model.load_state_dict(checkpoint)
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Loaded weights-only checkpoint from '{checkpoint_path}'. Training will restart from epoch 1.")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_preds = []
        train_targets = []
        for inputs, labels in train_loader:
            # inputs/labels are already Tensor on CPU from E90Dataset
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
            train_total += labels.size(0)
            train_targets.extend(labels.cpu().numpy().tolist())
            train_preds.extend(preds.detach().cpu().numpy().tolist())

        train_loss = running_loss / train_total if train_total > 0 else 0.0
        train_f1 = compute_f1(train_targets, train_preds, num_classes) if train_targets else 0.0

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_targets = []
        val_preds = []
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
                val_targets.extend(labels.cpu().numpy().tolist())
                val_preds.extend(preds.cpu().numpy().tolist())

        val_loss = val_running_loss / val_total if val_total > 0 else 0.0
        val_f1 = compute_f1(val_targets, val_preds, num_classes) if val_targets else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        # Early Stopping & Best Model Saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            # Optional: save checkpoint here
        else:
            no_improve_count += 1

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_model_state_dict": best_model_wts,
            "best_val_f1": best_val_f1,
            "history": history,
            "no_improve_count": no_improve_count,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)

        if no_improve_count >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    # Load best model weights
    print(f"Training finished. Best Val F1: {best_val_f1:.4f}")
    model.load_state_dict(best_model_wts)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Best model saved to '{model_output_path}'.")

    # Plotting
    plot_output_raw = get_config_value(
        training_cfg, "plot_output_file", "plot_output_path", "plots_path", "plots_dir"
    )
    plot_output_path = resolve_dir(plot_output_raw or "training_curves.png", OUTPUT_DIR, base_dir)
    if plot_output_path.suffix == "":
        plot_output_path = plot_output_path / "training_curves.png"

    _plot_training_curves(history, plot_output_path)
    print(f"Saved training curves to '{plot_output_path}'.")

    # Training history (loss/F1 per epoch)
    history_output_raw = get_config_value(
        training_cfg, "history_output_file", "metrics_output_file", "metrics_output_path"
    )
    if not history_output_raw:
        raise ValueError("Config must set training.history_output_file.")
    history_output_path = resolve_dir(history_output_raw, OUTPUT_DIR, base_dir)
    history_output_path.parent.mkdir(parents=True, exist_ok=True)

    with history_output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_f1", "val_f1"])
        for idx in range(len(history["train_loss"])):
            writer.writerow(
                [
                    idx + 1,
                    history["train_loss"][idx],
                    history["val_loss"][idx],
                    history["train_f1"][idx],
                    history["val_f1"][idx],
                ]
            )
    print(f"Saved training history to '{history_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Final training script using tuned hyperparameters.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    train_final(config, base_dir)
