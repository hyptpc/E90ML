import argparse
import json
import random
import gc
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)

from common import (
    E90Dataset,
    OUTPUT_DIR,
    LABEL_MAPPING,
    TUNE_DIR,
    get_config_value,
    apply_plot_style,
    create_model_from_params,
    get_augmented_feature_columns,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
    compute_f1,
)


def _int_range(cfg, key):
    item = cfg.get(key)
    return int(item["min"]), int(item["max"])


def _float_range(cfg, key):
    item = cfg.get(key)
    return float(item["min"]), float(item["max"])


def objective_factory(config, base_dir):
    """
    Pre-process data once outside the trial loop to save CPU and memory.
    """
    data_cfg = config.get("data", {})
    tuning_cfg = config.get("tuning", {})
    seed = _resolve_seed(tuning_cfg.get("seed"), config.get("seed"))
    
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
    tune_fraction = float(tuning_cfg["fraction"])
    val_split = float(tuning_cfg["val_split"])

    print("Loading data...")
    feature_matrix, labels, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=tune_fraction,
        random_state=seed,
    )

    # Memory: keep only encoded arrays in scope
    features = get_augmented_feature_columns(features)
    gc.collect()

    # Stratified split once up front
    train_features, val_features, train_labels, val_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    
    del feature_matrix, labels
    gc.collect()

    # Fit scaler once
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Create Tensor datasets once
    train_dataset = E90Dataset(train_features, train_labels)
    val_dataset = E90Dataset(val_features, val_labels)

    del train_features, val_features, train_labels, val_labels
    gc.collect()
    print("Data processing complete.")
    # ---------------------------------

    num_workers = int(tuning_cfg["num_workers"])
    epochs = int(tuning_cfg["epochs"])
    search_space_cfg = tuning_cfg["search_space"]
    
    batch_size_candidates = search_space_cfg.get("batch_size")
    if not isinstance(batch_size_candidates, (list, tuple)):
        batch_size_candidates = [batch_size_candidates]
    batch_size_candidates = [int(v) for v in batch_size_candidates]
    
    device = resolve_device(config.get("device"))
    print(f"Tuning using device: {device}")
    
    n_layers_min, n_layers_max = _int_range(search_space_cfg, "n_layers")
    hidden_min, hidden_max = _int_range(search_space_cfg, "hidden_units")
    dropout_min, dropout_max = _float_range(search_space_cfg, "dropout")
    lr_min, lr_max = _float_range(search_space_cfg, "lr")

    pos_weight = None
    if num_classes == 2:
        num_pos = (train_dataset.y == 1).sum()
        num_neg = (train_dataset.y == 0).sum()
        if num_pos > 0:
            weight_val = float(num_neg) / float(num_pos)
            pos_weight = torch.tensor(weight_val, dtype=torch.float32).to(device)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", batch_size_candidates)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )

        params = {
            "n_layers": trial.suggest_int("n_layers", n_layers_min, n_layers_max),
            "hidden_units": trial.suggest_int("hidden_units", hidden_min, hidden_max),
            "dropout_rate": trial.suggest_float("dropout_rate", dropout_min, dropout_max),
        }

        model = create_model_from_params(params, input_dim=len(features), num_classes=num_classes).to(device)

        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for inputs, labels_batch in train_loader:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                
                optimizer.zero_grad()
                if num_classes == 2:
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels_batch.float())
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_true = []
            val_pred = []
            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    inputs = inputs.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    if num_classes == 2:
                        outputs = model(inputs).squeeze(1)
                        predicted = (torch.sigmoid(outputs) > 0.5).long()
                    else:
                        outputs = model(inputs)
                        predicted = torch.argmax(outputs, dim=1)
                    val_true.extend(labels_batch.cpu().numpy().tolist())
                    val_pred.extend(predicted.cpu().numpy().tolist())

            val_f1 = compute_f1(val_true, val_pred, num_classes) if val_true else 0.0
            
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_f1

    return objective


def run_tuning(config, base_dir):
    tuning_cfg = config.get("tuning", {})
    direction = tuning_cfg["direction"]
    target_trials = int(tuning_cfg["n_trials"])

    best_params_raw = get_config_value(tuning_cfg, "tune_params_file", "best_params_file", "best_params_path")
    best_params_path = resolve_dir(best_params_raw, TUNE_DIR, base_dir)

    trials_raw = get_config_value(tuning_cfg, "study_summary_file", "study_summary_path")
    trials_path = resolve_dir(trials_raw, OUTPUT_DIR, base_dir) if trials_raw else None

    project_root = Path(__file__).resolve().parent.parent
    plots_cfg = tuning_cfg.get("plots", {})
    default_plots_dir = project_root / "plots" / "tuning_result"
    plots_dir_raw = plots_cfg.get("base_dir", default_plots_dir)
    plots_dir = resolve_dir(str(plots_dir_raw), default_plots_dir, project_root)
    plot_paths = {
        "optimization_history": resolve_dir(
            plots_cfg.get("optimization_history_file", "optimization_history.png"), plots_dir, project_root
        ),
        "param_importances": resolve_dir(
            plots_cfg.get("param_importances_file", "param_importances.png"), plots_dir, project_root
        ),
        "param_slice": resolve_dir(
            plots_cfg.get("param_slice_file", "param_slice.png"), plots_dir, project_root
        ),
    }
    
    db_file_raw = get_config_value(tuning_cfg, "study_db_file", "db_file") or "e90_optuna.db"
    db_path = resolve_dir(db_file_raw, TUNE_DIR, base_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path}"
    study_name = tuning_cfg.get("study_name", "e90_hyperopt")

    print(f"Optuna database: {storage_url}")
    print(f"Study Name:      {study_name}")
    
    objective = objective_factory(config, base_dir)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction
    )
    
    completed_trials = len(study.trials)
    remaining_trials = target_trials - completed_trials
    
    if remaining_trials > 0:
        print(f"Resuming study. Completed: {completed_trials}, Remaining: {remaining_trials}, Target: {target_trials}")
        try:
            study.optimize(objective, n_trials=remaining_trials)
        except KeyboardInterrupt:
            print("\nTuning interrupted by user. Progress saved to DB.")
            sys.exit(0)
    else:
        print(f"Study already has {completed_trials} trials (Target: {target_trials}). Skipping optimization.")

    print("Best trial value:", study.best_trial.value)
    print("Best params:", study.best_params)

    best_params = dict(study.best_params)

    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with best_params_path.open("w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best parameters to '{best_params_path}'.")

    if trials_path:
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        df = study.trials_dataframe()
        df.to_csv(trials_path, index=False)
        print(f"Saved tuning trials to '{trials_path}'.")
        
    # --- Visualization ---
    def _strip_titles(fig):
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            suptitle.remove()
            fig._suptitle = None
        for ax in fig.axes:
            if ax.get_title():
                ax.set_title("")

    def _finalize_plot(path, *, title=None, rect=(0, 0, 1, 0.98), adjust=None):
        fig = plt.gcf()
        _strip_titles(fig)
        if title:
            fig.suptitle(title, y=0.99)
        try:
            fig.tight_layout(rect=rect)
        except Exception as e:
            print(f"[warn] tight_layout failed: {e}")
        if adjust:
            adjust(fig)
        fig.savefig(path)
        plt.close(fig)

    def _move_colorbar_to_right(fig, pad=0.02, width=0.02):
        axes = fig.axes
        if len(axes) <= 1: return
        cbar_ax = min(axes, key=lambda ax: ax.get_position().width)
        other_axes = [ax for ax in axes if ax is not cbar_ax]
        if not other_axes: return
        right_edge = max(ax.get_position().x1 for ax in other_axes)
        pos = cbar_ax.get_position()
        new_left = min(right_edge + pad, 0.98 - width)
        new_width = min(width, pos.width, 1.0 - new_left)
        if new_width > 0:
            cbar_ax.set_position([new_left, pos.y0, new_width, pos.height])

    print("Generating tuning plots...")
    for path in plot_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply style locally to ensure it works with Agg backend
    apply_plot_style()
    saved_plots = []

    # 1. Optimization History
    plt.figure()
    plot_optimization_history(study)
    _finalize_plot(plot_paths["optimization_history"], title="Optimization History")
    saved_plots.append(plot_paths["optimization_history"])

    # 2. Hyperparameter Importances
    try:
        plt.figure()
        plot_param_importances(study)
        _finalize_plot(plot_paths["param_importances"], title="Param Importances")
        saved_plots.append(plot_paths["param_importances"])
    except ValueError:
        print("Skipping param_importances plot.")

    # 3. Slice Plot
    plt.figure()
    plot_slice(study)
    _finalize_plot(
        plot_paths["param_slice"],
        title="Param Slice",
        rect=(0, 0, 0.9, 0.92),
        adjust=_move_colorbar_to_right,
    )
    saved_plots.append(plot_paths["param_slice"])

    if saved_plots:
        saved_str = ", ".join(str(p) for p in saved_plots)
        print(f"Saved tuning plots to {saved_str}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for E90 ML model.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_tuning(config, base_dir)
