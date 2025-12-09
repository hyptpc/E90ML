import argparse
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    E90Dataset,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_TUNE_DIR,
    create_model_from_params,
    load_config,
    resolve_data_files,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
)


def _get_config_value(cfg: dict, *keys: str):
    for key in keys:
        value = cfg.get(key)
        if value not in (None, ""):
            return value
    return None


def _int_range(cfg, key):
    item = cfg.get(key)
    return int(item["min"]), int(item["max"])


def _float_range(cfg, key):
    item = cfg.get(key)
    return float(item["min"]), float(item["max"])


def objective_factory(config, base_dir):
    """Create an Optuna objective that uses stratified splits and leak-safe scaling."""
    data_cfg = config.get("data", {})
    tuning_cfg = config.get("tuning", {})
    seed = _resolve_seed(tuning_cfg.get("seed"), config.get("seed"))

    files = resolve_data_files(data_cfg, base_dir)
    if not files:
        raise ValueError("Config must provide data.files with at least one entry.")

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = DEFAULT_LABEL_MAPPING
    tune_fraction = float(tuning_cfg["fraction"])
    val_split = float(tuning_cfg["val_split"])

    dataset_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=tune_fraction,
        random_state=seed,
    )

    feature_matrix = dataset_df[features].values
    labels = dataset_df[label_column].values

    # Tuning parameters
    num_workers = int(tuning_cfg["num_workers"])
    epochs = int(tuning_cfg["epochs"])
    n_trials = int(tuning_cfg["n_trials"])
    batch_size_options = tuning_cfg["batch_size_options"]
    search_space_cfg = tuning_cfg["search_space"]
    device = resolve_device(tuning_cfg.get("device") or config.get("device"))

    # Search space (read exactly from config)
    n_layers_min, n_layers_max = _int_range(search_space_cfg, "n_layers")
    hidden_min, hidden_max = _int_range(search_space_cfg, "hidden_units")
    dropout_min, dropout_max = _float_range(search_space_cfg, "dropout")
    lr_min, lr_max = _float_range(search_space_cfg, "lr")

    def objective(trial):
        # 1. Stratified Split (prevents leakage and handles imbalance in split)
        train_features, val_features, train_labels, val_labels = train_test_split(
            feature_matrix,
            labels,
            test_size=val_split,
            stratify=labels,
            random_state=seed,
        )

        # 2. Fit Scaler on TRAIN ONLY (prevents leakage)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # 3. Create Datasets
        train_dataset = E90Dataset(train_features, train_labels)
        val_dataset = E90Dataset(val_features, val_labels)

        batch_size = trial.suggest_categorical("batch_size", batch_size_options)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Model params
        params = {
            "n_layers": trial.suggest_int("n_layers", n_layers_min, n_layers_max),
            "hidden_units": trial.suggest_int("hidden_units", hidden_min, hidden_max),
            "dropout_rate": trial.suggest_float("dropout_rate", dropout_min, dropout_max),
        }

        model = create_model_from_params(params, input_dim=len(features), num_classes=num_classes).to(device)

        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Class Imbalance Handling (Weighted Loss)
        if num_classes == 2:
            # Calculate pos_weight based on training data
            num_pos = (train_labels == 1).sum()
            num_neg = (train_labels == 0).sum()
            pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device) if num_pos > 0 else None
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.CrossEntropyLoss()

        # Training Loop
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
            correct = 0
            total = 0
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
    best_params_raw = _get_config_value(tuning_cfg, "tune_params_file", "best_params_file", "best_params_path")
    if not best_params_raw:
        raise ValueError("Config must set tuning.tune_params_file (or best_params_file/best_params_path).")
    best_params_path = resolve_dir(best_params_raw, DEFAULT_TUNE_DIR, base_dir)

    trials_raw = _get_config_value(tuning_cfg, "study_summary_file", "study_summary_path")
    trials_path = resolve_dir(trials_raw, DEFAULT_OUTPUT_DIR, base_dir) if trials_raw else None

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


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for E90 ML model.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_tuning(config, base_dir)
