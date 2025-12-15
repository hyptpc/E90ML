import argparse
import copy
import json
import pickle
import random
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
# PyG Loader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split

from common import (
    E90GraphDataset,
    create_gnn_model_from_params,
    OUTPUT_DIR,
    PTH_DIR,
    LABEL_MAPPING,
    TUNE_DIR,
    get_config_value,
    split_track_features,
    load_best_params,
    resolve_model_output_path,
    apply_plot_style,
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
    Train the final GNN model (CPU Optimized for KEKCC).
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("tuning", {})

    seed = _resolve_seed(training_cfg.get("seed"), config.get("seed"))
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    files = resolve_data_files(data_cfg, base_dir)
    if not files:
        raise ValueError("Config must provide data.files with at least one entry.")

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = LABEL_MAPPING

    feature_cols_dict = split_track_features(features)

    train_fraction = float(training_cfg["fraction"])
    val_split = float(training_cfg["val_split"])
    # Default num_workers to 0 for KEKCC (save memory)
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    batch_size_override = training_cfg.get("batch_size")

    # Load Data (Pandas DataFrame)
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

    # Split Data
    train_df, val_df = train_test_split(
        full_df,
        test_size=val_split,
        stratify=full_df[label_column],
        random_state=seed,
    )
    
    # Free memory immediately
    del full_df
    gc.collect()

    # Create Graph Datasets
    print("Creating graph datasets...")
    train_dataset = E90GraphDataset(train_df, feature_cols_dict, label_column)
    val_dataset = E90GraphDataset(val_df, feature_cols_dict, label_column)
    
    # Free DataFrames
    del train_df, val_df
    gc.collect()

    # Class weights calculation (optional, purely purely on labels)
    # If needed, calculate from train_dataset.data_list or keep simple
    # For memory saving, we skip re-iterating unless necessary

    # Load Tuned Hyperparameters
    params, best_params_path = load_best_params(training_cfg, tuning_cfg, base_dir, TUNE_DIR)
    model_output_path = resolve_model_output_path(training_cfg, base_dir, PTH_DIR, must_exist=False)
    checkpoint_raw = get_config_value(training_cfg, "checkpoint_file", "checkpoint_path")
    checkpoint_path = resolve_dir(checkpoint_raw or model_output_path.name, PTH_DIR, base_dir)
    print("Loaded parameters:", params)

    batch_size = batch_size_override or int(params.get("batch_size", 128))
    lr = float(params.get("lr", 1e-3))

    # DataLoader settings tuned for CPU use
    train_loader = PyGDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False  # CPU-only run
    )
    val_loader = PyGDataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )

    # Force CPU usage for KEKCC
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    INPUT_DIM = 5
    model = create_gnn_model_from_params(params, input_dim=INPUT_DIM, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    # Training State
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_f1 = checkpoint.get("best_val_f1", best_val_f1)
        best_model_wts = checkpoint.get("best_model_state_dict", best_model_wts)
        no_improve_count = checkpoint.get("no_improve_count", 0)
        saved_history = checkpoint.get("history")
        if saved_history:
            history = saved_history
        print(f"Resumed from checkpoint: {checkpoint_path} (start at epoch {start_epoch + 1})")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch.x, batch.edge_index, batch.batch)
            
            if num_classes == 2:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, batch.y.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                loss = criterion(outputs, batch.y)
                preds = torch.argmax(outputs, dim=1)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            train_total += batch.num_graphs
            train_targets.extend(batch.y.cpu().numpy().tolist())
            train_preds.extend(preds.detach().cpu().numpy().tolist())

        train_loss = running_loss / train_total if train_total > 0 else 0.0
        train_f1 = compute_f1(train_targets, train_preds, num_classes) if train_targets else 0.0

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                outputs = model(batch.x, batch.edge_index, batch.batch)
                
                if num_classes == 2:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, batch.y.float())
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    loss = criterion(outputs, batch.y)
                    preds = torch.argmax(outputs, dim=1)
                
                val_running_loss += loss.item() * batch.num_graphs
                val_total += batch.num_graphs
                val_targets.extend(batch.y.cpu().numpy().tolist())
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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Collect garbage to keep memory use low
        gc.collect()

        # Save checkpoint for resuming
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_model_state_dict": best_model_wts,
            "history": history,
            "no_improve_count": no_improve_count,
            "params": params,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)

        if no_improve_count >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training finished. Best Val F1: {best_val_f1:.4f}")
    model.load_state_dict(best_model_wts)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Best model saved to '{model_output_path}'.")

    # Plotting
    plots_cfg = training_cfg.get("plots", {})
    plot_output_raw = get_config_value(
        plots_cfg, "training_curve", "training_curves", "plot_file", "plot_path"
    ) or get_config_value(
        training_cfg, "plot_output_file", "plot_output_path", "plots_path", "plots_dir"
    )
    project_root = base_dir.parents[1] if len(base_dir.parents) > 1 else base_dir
    default_plots_dir = resolve_dir(plots_cfg.get("save_dir", "plots"), project_root / "plots", base_dir)
    plot_output_path = resolve_dir(plot_output_raw or "training_curves.png", default_plots_dir, base_dir)
    if plot_output_path.suffix == "":
        plot_output_path = plot_output_path / "training_curves.png"

    _plot_training_curves(history, plot_output_path)
    print(f"Saved training curves to '{plot_output_path}'.")

    # Metrics
    metrics_output_raw = get_config_value(training_cfg, "metrics_output_file", "metrics_output_path")
    if metrics_output_raw:
        metrics_output_path = resolve_dir(metrics_output_raw, OUTPUT_DIR, base_dir)
        metrics_payload = {
            "best_val_f1": best_val_f1,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_classes": num_classes,
        }
        with metrics_output_path.open("w") as f:
            json.dump(metrics_payload, f, indent=4)
        print(f"Saved metrics to '{metrics_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Final training script (GNN CPU).")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    train_final(config, base_dir)
