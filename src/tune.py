import argparse
import json
import gc
from pathlib import Path

import numpy as np
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# PyG Loader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split

# Optuna Visualization (Matplotlib backend for static images)
import optuna.visualization.matplotlib as ovm

from common import (
    E90GraphDataset,
    create_gnn_model_from_params,
    LABEL_MAPPING,
    TUNE_DIR,
    get_config_value,
    split_track_features,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
    compute_f1,
    apply_plot_style,
)


def objective(trial, config, base_dir):
    data_cfg = config.get("data", {})
    tuning_cfg = config.get("tuning", {})
    
    # 1. Hyperparameter sampling for the GNN
    search_space = tuning_cfg.get("search_space", {})
    
    # Parameters explored for the GNN
    params = {
        "n_layers": trial.suggest_int("n_layers", search_space.get("n_layers", {}).get("min", 2), search_space.get("n_layers", {}).get("max", 5)),
        "hidden_units": trial.suggest_int("hidden_units", search_space.get("hidden_units", {}).get("min", 32), search_space.get("hidden_units", {}).get("max", 256)),
        "dropout_rate": trial.suggest_float("dropout", search_space.get("dropout", {}).get("min", 0.0), search_space.get("dropout", {}).get("max", 0.5)),
        "lr": trial.suggest_float("lr", search_space.get("lr", {}).get("min", 1e-4), search_space.get("lr", {}).get("max", 1e-2), log=True),
        "batch_size": trial.suggest_categorical("batch_size", search_space.get("batch_size", [64, 128, 256])),
    }

    # 2. Data Loading (Inside objective for simplicity, or cache externally)
    seed = _resolve_seed(tuning_cfg.get("seed"), config.get("seed"))
    files = resolve_data_files(data_cfg, base_dir)
    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping", LABEL_MAPPING)

    feature_cols_dict = split_track_features(features)

    fraction = float(tuning_cfg.get("fraction", 0.1))

    full_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction,
        random_state=seed,
    )
    
    train_df, val_df = train_test_split(
        full_df,
        test_size=float(tuning_cfg.get("val_split", 0.2)),
        stratify=full_df[label_column],
        random_state=seed,
    )
    del full_df
    gc.collect()

    train_dataset = E90GraphDataset(train_df, feature_cols_dict, label_column)
    val_dataset = E90GraphDataset(val_df, feature_cols_dict, label_column)

    num_workers = int(tuning_cfg.get("num_workers", 0))
    train_loader = PyGDataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = PyGDataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=num_workers)

    # 3. Model Building
    device = resolve_device(config.get("device", "cpu"))
    model = create_gnn_model_from_params(params, input_dim=5, num_classes=num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    # 4. Training Loop
    epochs = int(tuning_cfg.get("epochs", 5))
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.batch)
            if num_classes == 2:
                loss = criterion(outputs.squeeze(1), batch.y.float())
            else:
                loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)
                if num_classes == 2:
                    preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).long()
                else:
                    preds = torch.argmax(outputs, dim=1)
                val_targets.extend(batch.y.cpu().numpy().tolist())
                val_preds.extend(preds.cpu().numpy().tolist())
        
        val_f1 = compute_f1(val_targets, val_preds, num_classes)
        best_val_f1 = max(best_val_f1, val_f1)
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


def save_optuna_plots(study, plots_dir: Path, filenames: dict):
    """Save Optuna analysis plots to PNG files."""
    apply_plot_style()
    plt.switch_backend('Agg')

    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving optimization plots to {plots_dir} ...")

    def _tight_layout_safe(obj):
        try:
            if hasattr(obj, "tight_layout"):
                obj.tight_layout()
            elif hasattr(obj, "figure") and hasattr(obj.figure, "tight_layout"):
                obj.figure.tight_layout()
        except Exception as exc:
            print(f"Skipping tight_layout due to: {exc}")

    def _save_fig(obj, path: Path):
        target = obj
        if hasattr(obj, "savefig"):
            target = obj
        elif hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):
            target = obj.figure
        else:
            print(f"Skipping save for {path} (object has no savefig).")
            return
        target.savefig(path)

    try:
        # 1. Optimization History
        fig = ovm.plot_optimization_history(study)
        _tight_layout_safe(fig)
        _save_fig(fig, plots_dir / filenames.get("optimization_history", "opt_history.png"))
        plt.close() # Free memory

        # 2. Slice plot (relationship between each parameter and objective)
        fig = ovm.plot_slice(study)
        _tight_layout_safe(fig)
        _save_fig(fig, plots_dir / filenames.get("slice", "opt_slice.png"))
        plt.close()

        # 3. Param Importances
        try:
            fig = ovm.plot_param_importances(study)
            _tight_layout_safe(fig)
            _save_fig(fig, plots_dir / filenames.get("param_importances", "opt_importances.png"))
            plt.close()
        except Exception as e:
            print(f"Skipping param_importances plot (needs more than 1 param): {e}")

    except Exception as e:
        print(f"Error during plotting: {e}")


def run_tuning(config, base_dir):
    tuning_cfg = config.get("tuning", {})
    study_name = tuning_cfg.get("study_name", "e90_gnn_study")
    storage_path = resolve_dir(tuning_cfg.get("study_db_file", "tune.db"), TUNE_DIR, base_dir)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"
    direction = tuning_cfg.get("direction", "maximize")
    n_trials = int(tuning_cfg.get("n_trials", 20))

    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config, base_dir), n_trials=n_trials)

    print("Best params:", study.best_params)
    
    # Save best params
    out_file = resolve_dir(tuning_cfg.get("tune_params_file", "best_params.json"), TUNE_DIR, base_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(study.best_params, f, indent=4)

    # [ADDED] Save Plots
    plots_cfg = tuning_cfg.get("plots", {})
    plots_dir_raw = plots_cfg.get("save_dir", "plots")
    project_root = base_dir.parents[1] if len(base_dir.parents) > 1 else base_dir
    default_plots_dir = (project_root / "plots").resolve()
    plots_dir = (default_plots_dir / Path(plots_dir_raw)).resolve()
    plot_filenames = {
        "optimization_history": plots_cfg.get("optimization_history", "opt_history.png"),
        "slice": plots_cfg.get("slice", "opt_slice.png"),
        "param_importances": plots_cfg.get("param_importances", "opt_importances.png"),
    }
    save_optuna_plots(study, plots_dir, plot_filenames)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning (GNN) with Plotting.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_tuning(config, base_dir)
