import argparse
import json

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from common import (
    DEFAULT_FEATURES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TREE_NAME,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TUNE_DIR,
    DEFAULT_TUNED_PARAMS_NAME,
    DEFAULT_TUNING_SUMMARY_NAME,
    E90Dataset,
    create_model_from_params,
    load_config,
    resolve_named_path,
    resolve_data_files,
    resolve_device,
)


def _int_range(cfg, key, default_min, default_max):
    item = cfg.get(key, {})
    return int(item.get("min", default_min)), int(item.get("max", default_max))


def _float_range(cfg, key, default_min, default_max):
    item = cfg.get(key, {})
    return float(item.get("min", default_min)), float(item.get("max", default_max))


def _prepare_data(config, base_dir, fraction, is_train=True):
    data_cfg = config.get("data", {})
    files = resolve_data_files(data_cfg, base_dir)
    if not files:
        raise ValueError("No data files specified in config.data.files")

    tree_name = data_cfg.get("tree_name", DEFAULT_TREE_NAME)
    features = data_cfg.get("feature_columns") or DEFAULT_FEATURES
    label_column = data_cfg.get("label_column", DEFAULT_LABEL_COLUMN)
    label_mapping = data_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    dataset = E90Dataset(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=fraction,
        is_train=is_train,
    )
    num_classes = 2 if label_mapping else int(data_cfg.get("num_classes") or dataset.num_classes)
    return dataset, features, num_classes


def objective_factory(config, base_dir):
    tuning_cfg = config.get("tuning", {})

    tune_fraction = float(tuning_cfg.get("fraction", 0.3))
    val_split = float(tuning_cfg.get("val_split", 0.2))
    num_workers = int(tuning_cfg.get("num_workers", 2))
    epochs = int(tuning_cfg.get("epochs", 10))
    n_trials = int(tuning_cfg.get("n_trials", 20))
    batch_size_options = tuning_cfg.get("batch_size_options", [64, 128, 256])
    search_space_cfg = tuning_cfg.get("search_space", {})
    device = resolve_device(tuning_cfg.get("device") or config.get("device"))

    n_layers_min, n_layers_max = _int_range(search_space_cfg, "n_layers", 2, 4)
    hidden_min, hidden_max = _int_range(search_space_cfg, "hidden_units", 64, 256)
    dropout_min, dropout_max = _float_range(search_space_cfg, "dropout", 0.1, 0.5)
    lr_min, lr_max = _float_range(search_space_cfg, "lr", 1e-4, 1e-2)

    full_dataset, features, num_classes = _prepare_data(config, base_dir, tune_fraction, is_train=True)

    def objective(trial):
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size == 0 or val_size == 0:
            raise ValueError("Not enough data to perform the requested train/val split.")

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        batch_size = trial.suggest_categorical("batch_size", batch_size_options)
        params = {
            "n_layers": trial.suggest_int("n_layers", n_layers_min, n_layers_max),
            "hidden_units": trial.suggest_int("hidden_units", hidden_min, hidden_max),
            "dropout_rate": trial.suggest_float("dropout_rate", dropout_min, dropout_max),
        }

        model = create_model_from_params(params, input_dim=len(features), num_classes=num_classes).to(device)

        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                if num_classes == 2:
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels.float())
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if num_classes == 2:
                        outputs = model(inputs).squeeze(1)
                        predicted = (torch.sigmoid(outputs) > 0.5).long()
                    else:
                        outputs = model(inputs)
                        predicted = torch.argmax(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total if total > 0 else 0.0
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    return objective, n_trials


def run_tuning(config, base_dir):
    tuning_cfg = config.get("tuning", {})
    direction = tuning_cfg.get("direction", "maximize")
    best_params_path = resolve_named_path(
        tuning_cfg.get("best_params_path"),
        default_dir=DEFAULT_TUNE_DIR,
        default_name=DEFAULT_TUNED_PARAMS_NAME,
        base_dir=base_dir,
    )
    trials_path = tuning_cfg.get("study_summary_path")
    trials_path = (
        resolve_named_path(
            trials_path,
            default_dir=DEFAULT_OUTPUT_DIR,
            default_name=DEFAULT_TUNING_SUMMARY_NAME,
            base_dir=base_dir,
        )
        if trials_path is not None
        else None
    )

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
