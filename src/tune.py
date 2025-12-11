import argparse
import json
import random
import gc
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
)

from common import (
    E90Dataset,
    OUTPUT_DIR,
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
)


def _int_range(cfg, key):
    item = cfg.get(key)
    return int(item["min"]), int(item["max"])


def _float_range(cfg, key):
    item = cfg.get(key)
    return float(item["min"]), float(item["max"])


def objective_factory(config, base_dir):
    """
    CPU Optimized: Pre-process data ONCE outside the trial loop to save time and memory.
    """
    data_cfg = config.get("data", {})
    tuning_cfg = config.get("tuning", {})
    seed = _resolve_seed(tuning_cfg.get("seed"), config.get("seed"))
    
    # CPU Optimization: Set threads
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Use all available cores for matrix operations
    # torch.set_num_threads(8) # Uncomment and set manually if needed

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
    dataset_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=tune_fraction,
        random_state=seed,
    )

    # --- Memory Optimization Block ---
    print("Processing data (Split & Scale)...")
    # Extract values and immediately cast to float32 to save RAM
    feature_matrix = dataset_df[features].values.astype(np.float32)
    labels = dataset_df[label_column].values.astype(np.int64)
    
    # Delete DataFrame to free memory
    del dataset_df
    gc.collect()

    # 1. Stratified Split (ONCE)
    train_features, val_features, train_labels, val_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    
    # Delete original arrays
    del feature_matrix, labels
    gc.collect()

    # 2. Fit Scaler (ONCE)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # 3. Create Tensor Datasets (ONCE)
    train_dataset = E90Dataset(train_features, train_labels)
    val_dataset = E90Dataset(val_features, val_labels)

    # Free numpy arrays since data is now in Tensors inside Dataset
    del train_features, val_features, train_labels, val_labels
    gc.collect()
    print("Data processing complete. Starting tuning...")
    # ---------------------------------

    # Tuning parameters
    # For CPU with limited memory, consider reducing num_workers in config (e.g., set to 0 or 2)
    num_workers = int(tuning_cfg["num_workers"])
    epochs = int(tuning_cfg["epochs"])
    n_trials = int(tuning_cfg["n_trials"])
    batch_size_options = tuning_cfg["batch_size_options"]
    search_space_cfg = tuning_cfg["search_space"]
    # Force CPU device
    device = torch.device("cpu")

    # Search space
    n_layers_min, n_layers_max = _int_range(search_space_cfg, "n_layers")
    hidden_min, hidden_max = _int_range(search_space_cfg, "hidden_units")
    dropout_min, dropout_max = _float_range(search_space_cfg, "dropout")
    lr_min, lr_max = _float_range(search_space_cfg, "lr")

    # Calculate pos_weight from the Tensor data directly
    pos_weight = None
    if num_classes == 2:
        # Accessing underlying tensor in dataset
        num_pos = (train_dataset.y == 1).sum()
        num_neg = (train_dataset.y == 0).sum()
        if num_pos > 0:
            pos_weight = torch.tensor(float(num_neg)/float(num_pos), dtype=torch.float32)

    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", batch_size_options)
        
        # CPU: pin_memory=False is usually fine.
        # num_workers: If memory is tight, 0 is safest.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
                # Inputs are already on CPU
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
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    if num_classes == 2:
                        outputs = model(inputs).squeeze(1)
                        predicted = (torch.sigmoid(outputs) > 0.5).long()
                    else:
                        outputs = model(inputs)
                        predicted = torch.argmax(outputs, dim=1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

            accuracy = correct / total if total > 0 else 0.0
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    return objective, n_trials


def run_tuning(config, base_dir):
    tuning_cfg = config.get("tuning", {})
    direction = tuning_cfg["direction"]
    best_params_raw = get_config_value(tuning_cfg, "tune_params_file", "best_params_file", "best_params_path")
    if not best_params_raw:
        raise ValueError("Config must set tuning.tune_params_file (or best_params_file/best_params_path).")
    best_params_path = resolve_dir(best_params_raw, TUNE_DIR, base_dir)

    trials_raw = get_config_value(tuning_cfg, "study_summary_file", "study_summary_path")
    trials_path = resolve_dir(trials_raw, OUTPUT_DIR, base_dir) if trials_raw else None
    plots_cfg = tuning_cfg.get("plots", {})
    plots_dir = resolve_dir(plots_cfg.get("dir", "plots"), OUTPUT_DIR, base_dir)
    plot_paths = {
        "optimization_history": resolve_dir(
            plots_cfg.get("optimization_history_file", "optimization_history.png"), plots_dir, base_dir
        ),
        "param_importances": resolve_dir(
            plots_cfg.get("param_importances_file", "param_importances.png"), plots_dir, base_dir
        ),
        "param_slice": resolve_dir(
            plots_cfg.get("param_slice_file", "param_slice.png"), plots_dir, base_dir
        ),
    }

    objective, n_trials = objective_factory(config, base_dir)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

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
    print("Generating tuning plots...")
    for path in plot_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    saved_plots = []

    # 1. Optimization History
    plt.figure()
    plot_optimization_history(study)
    plt.title("Optimization History")
    plt.tight_layout()
    plt.savefig(plot_paths["optimization_history"])
    plt.close()
    saved_plots.append(plot_paths["optimization_history"])

    # 2. Hyperparameter Importances
    try:
        plt.figure()
        plot_param_importances(study)
        plt.title("Hyperparameter Importances")
        plt.tight_layout()
        plt.savefig(plot_paths["param_importances"])
        plt.close()
        saved_plots.append(plot_paths["param_importances"])
    except ValueError:
        print("Skipping param_importances plot (requires more than one parameter).")

    # 3. Slice Plot
    plt.figure()
    plot_slice(study)
    plt.title("Parameter Slices")
    plt.tight_layout()
    plt.savefig(plot_paths["param_slice"])
    plt.close()
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
