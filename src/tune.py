import argparse
import json
import random
import gc
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)

from common import (
    E90Dataset,
    LABEL_MAPPING,
    TUNE_DIR,
    get_config_value,
    apply_plot_style,
    create_model_from_params,
    load_config,
    resolve_data_files,
    resolve_data_dirs,
    resolve_device,
    resolve_dir,
    _resolve_seed,
    load_data,
    compute_f1,
    batchnorm_safe_drop_last,
    validate_fraction,
    validate_stratified_split,
    make_event_groups,
    stratified_group_split_indices,
    sample_indices,
    summarize_labels,
    save_data_summary,
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
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    files = resolve_data_files(data_cfg, base_dir)
    _, data_output_dir = resolve_data_dirs(base_dir)
    if not files:
        raise ValueError("Config must provide data.files with at least one entry.")

    tree_name = data_cfg.get("tree_name")
    features = data_cfg.get("feature_columns")
    label_column = data_cfg.get("label_column")
    label_mapping = data_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
    tune_fraction = validate_fraction(tuning_cfg["fraction"], "tuning.fraction")
    split_seed = _resolve_seed(data_cfg.get("split_seed"), seed)
    outer_val_split = data_cfg.get("val_split")
    if outer_val_split is None:
        outer_val_split = tuning_cfg["val_split"]

    print("Loading data...")
    dataset_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=1.0,
        random_state=split_seed,
        shuffle=False,
    )

    # --- Memory Optimization Block ---
    print("Processing data (Split & Scale)...")
    feature_matrix = dataset_df[features].values.astype(np.float32)
    labels = dataset_df[label_column].values.astype(np.int64)
    
    del dataset_df
    gc.collect()

    # 1. Reserve the same outer validation set used by final training.
    event_groups = make_event_groups(feature_matrix, labels)
    outer_train_indices, outer_val_indices = stratified_group_split_indices(
        labels,
        event_groups,
        outer_val_split,
        split_seed,
        "data outer split",
    )
    outer_validation_summary = summarize_labels(labels[outer_val_indices])

    # 2. Draw the tuning subset only from the outer training partition.
    sampled_pool_indices = sample_indices(len(outer_train_indices), tune_fraction, seed)
    tuning_indices = outer_train_indices[sampled_pool_indices]
    tuning_features = feature_matrix[tuning_indices]
    tuning_labels = labels[tuning_indices]
    tuning_groups = event_groups[tuning_indices]

    # 3. Make an inner grouped split for Optuna.
    val_split = validate_stratified_split(tuning_labels, tuning_cfg["val_split"], "tuning")
    train_indices, val_indices = stratified_group_split_indices(
        tuning_labels,
        tuning_groups,
        val_split,
        seed,
        "tuning inner split",
    )
    train_features = tuning_features[train_indices]
    val_features = tuning_features[val_indices]
    train_labels = tuning_labels[train_indices]
    val_labels = tuning_labels[val_indices]
    train_groups = tuning_groups[train_indices]
    val_groups = tuning_groups[val_indices]

    del (
        feature_matrix,
        labels,
        event_groups,
        outer_train_indices,
        outer_val_indices,
        sampled_pool_indices,
        tuning_indices,
        tuning_features,
        tuning_labels,
        tuning_groups,
        train_indices,
        val_indices,
    )
    gc.collect()

    # 4. Fit Scaler using only the inner training partition.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # 5. Create Tensor Datasets (ONCE)
    train_dataset = E90Dataset(train_features, train_labels)
    val_dataset = E90Dataset(val_features, val_labels)
    data_summary = {
        "stage": "tuning",
        "source_files": [str(path) for path in files],
        "fraction": tune_fraction,
        "seed": seed,
        "split_seed": split_seed,
        "outer_val_split": float(outer_val_split),
        "val_split": val_split,
        "outer_validation_reserved": outer_validation_summary,
        "train_unique_groups": int(np.unique(train_groups).size),
        "validation_unique_groups": int(np.unique(val_groups).size),
        "overlapping_groups": 0,
        "train": summarize_labels(train_dataset.y),
        "validation": summarize_labels(val_dataset.y),
    }
    summary_raw = get_config_value(tuning_cfg, "data_summary_file") or "tuning_data_summary.json"
    save_data_summary(data_summary, resolve_dir(summary_raw, data_output_dir, base_dir))

    del train_features, val_features, train_labels, val_labels, train_groups, val_groups
    gc.collect()
    print("Data processing complete.")
    # ---------------------------------

    num_workers = int(tuning_cfg["num_workers"])
    epochs = int(tuning_cfg["epochs"])
    if epochs < 1:
        raise ValueError("tuning.epochs must be at least 1.")
    search_space_cfg = tuning_cfg["search_space"]
    
    batch_size_candidates = search_space_cfg.get("batch_size")
    if not isinstance(batch_size_candidates, (list, tuple)):
        batch_size_candidates = [batch_size_candidates]
    batch_size_candidates = [int(v) for v in batch_size_candidates]
    if not batch_size_candidates or any(value < 2 for value in batch_size_candidates):
        raise ValueError("All tuning batch sizes must be at least 2 for BatchNorm1d.")
    if num_workers < 0:
        raise ValueError("tuning.num_workers must be non-negative.")
    
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
        drop_last = batchnorm_safe_drop_last(len(train_dataset), batch_size)
        loader_generator = None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            loader_generator = torch.Generator()
            loader_generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=drop_last,
            generator=loader_generator,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
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

        def train_step(inputs, labels_batch):
            model.train()
            inputs = inputs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if num_classes == 2:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels_batch.float())
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

        def evaluate_validation():
            model.eval()
            val_loss_sum = 0.0
            val_sample_count = 0
            val_true = []
            val_pred = []
            with torch.inference_mode():
                for inputs, labels_batch in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)
                    
                    if num_classes == 2:
                        outputs = model(inputs).squeeze(1)
                        loss = criterion(outputs, labels_batch.float())
                        predicted = (torch.sigmoid(outputs) > 0.5).long()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch)
                        predicted = torch.argmax(outputs, dim=1)
                    batch_size_actual = labels_batch.size(0)
                    val_loss_sum += loss.item() * batch_size_actual
                    val_sample_count += batch_size_actual
                    val_true.extend(labels_batch.cpu().numpy().tolist())
                    val_pred.extend(predicted.cpu().numpy().tolist())

            if val_sample_count == 0:
                raise RuntimeError("Validation loader produced no samples.")

            val_loss = val_loss_sum / val_sample_count
            val_f1 = compute_f1(val_true, val_pred, num_classes) if val_true else 0.0
            return val_loss, val_f1

        best_val_loss = float("inf")

        for epoch in range(epochs):
            for inputs, labels_batch in train_loader:
                train_step(inputs, labels_batch)

            val_loss, val_f1 = evaluate_validation()
            best_val_loss = min(best_val_loss, val_loss)

            trial.set_user_attr(f"val_f1_epoch_{epoch + 1}", val_f1)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    return objective


def run_tuning(config, base_dir, plots_only=False):
    tuning_cfg = config.get("tuning", {})
    data_cfg = config.get("data", {})
    _, data_output_dir = resolve_data_dirs(base_dir)
    direction = tuning_cfg["direction"]
    target_trials = int(tuning_cfg["n_trials"])

    best_params_raw = get_config_value(tuning_cfg, "tune_params_file", "best_params_file", "best_params_path")
    best_params_path = resolve_dir(best_params_raw, TUNE_DIR, base_dir)

    trials_raw = get_config_value(tuning_cfg, "study_summary_file", "study_summary_path")
    trials_path = resolve_dir(trials_raw, data_output_dir, base_dir) if trials_raw else None

    project_root = Path(__file__).resolve().parent.parent
    plots_cfg = tuning_cfg.get("plots", {})
    default_plots_dir = project_root / "plots" / "tune"
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
    seed = _resolve_seed(tuning_cfg.get("seed"), config.get("seed"))
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_cfg = tuning_cfg.get("pruner")
    pruner = None
    if pruner_cfg is not None:
        supported_pruner_keys = {
            "n_startup_trials",
            "n_warmup_steps",
            "interval_steps",
            "n_min_trials",
        }
        unknown_keys = set(pruner_cfg) - supported_pruner_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unsupported tuning.pruner settings: {unknown}")
        pruner = optuna.pruners.MedianPruner(
            **{key: int(value) for key, value in pruner_cfg.items()}
        )

    print(f"Optuna database: {storage_url}")
    print(f"Study Name:      {study_name}")
    print(f"Sampler:         TPESampler (seed={seed})")
    if pruner is None:
        print("Pruner:          Optuna default")
    else:
        configured = ", ".join(f"{key}={value}" for key, value in pruner_cfg.items())
        print(f"Pruner:          MedianPruner ({configured})")

    study_options = dict(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
    )
    if pruner is not None:
        study_options["pruner"] = pruner
    study = optuna.create_study(**study_options)
    
    if not plots_only:
        attempted_trials = len(study.trials)
        remaining_trials = target_trials - attempted_trials

        if remaining_trials > 0:
            print(
                f"Resuming study. Existing: {attempted_trials}, "
                f"Remaining: {remaining_trials}, Target: {target_trials}"
            )
            objective = objective_factory(config, base_dir)
            try:
                study.optimize(objective, n_trials=remaining_trials)
            except KeyboardInterrupt:
                print("\nTuning interrupted by user. Progress saved to DB.")
                sys.exit(0)
        else:
            print(
                f"Study already has {attempted_trials} trials "
                f"(Target: {target_trials}). Skipping optimization."
            )

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
    else:
        print("Plots-only mode: skipping data loading and optimization.")
        
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

    def _legend_inside_upper_right(fig):
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            handles, labels = ax.get_legend_handles_labels()
            legend.remove()
            ax.legend(handles, labels, loc="upper right")
            # Optuna initially reserves space for its outside legend. Re-run
            # layout after moving it inside to reclaim the unused right margin.
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            break

    def _set_slice_search_ranges(fig):
        search_space = tuning_cfg["search_space"]
        batch_sizes = search_space["batch_size"]
        if not isinstance(batch_sizes, (list, tuple)):
            batch_sizes = [batch_sizes]
        ranges = {
            "batch_size": (min(batch_sizes), max(batch_sizes)),
            "dropout_rate": (
                float(search_space["dropout"]["min"]),
                float(search_space["dropout"]["max"]),
            ),
            "hidden_units": (
                int(search_space["hidden_units"]["min"]),
                int(search_space["hidden_units"]["max"]),
            ),
            "lr": (
                float(search_space["lr"]["min"]),
                float(search_space["lr"]["max"]),
            ),
            "n_layers": (
                int(search_space["n_layers"]["min"]),
                int(search_space["n_layers"]["max"]),
            ),
        }
        for ax in fig.axes:
            parameter = ax.get_xlabel()
            if parameter == "batch_size":
                # Optuna assigns categorical coordinates only to values that
                # occur in completed trials.  If one configured batch size has
                # no completed trial, blindly replacing the tick labels shifts
                # every plotted point onto the wrong category.  Remap the
                # existing categorical coordinates to the full configured
                # search-space order before adding the missing ticks.
                observed_positions = ax.get_xticks()
                observed_labels = [label.get_text() for label in ax.get_xticklabels()]
                target_by_label = {
                    str(value): position for position, value in enumerate(batch_sizes)
                }
                coordinate_map = {
                    float(position): float(target_by_label[label])
                    for position, label in zip(observed_positions, observed_labels)
                    if label in target_by_label
                }
                for collection in ax.collections:
                    offsets = collection.get_offsets()
                    if offsets.size == 0:
                        continue
                    remapped = np.asarray(offsets, dtype=float).copy()
                    original_x = remapped[:, 0].copy()
                    for source, target in coordinate_map.items():
                        remapped[np.isclose(original_x, source), 0] = target
                    collection.set_offsets(remapped)
                positions = np.arange(len(batch_sizes))
                ax.set_xlim(-0.5, len(batch_sizes) - 0.5)
                ax.set_xticks(positions, labels=[str(value) for value in batch_sizes])
                ax.xaxis.set_minor_locator(NullLocator())
            elif parameter in ranges:
                ax.set_xlim(*ranges[parameter])
        _move_colorbar_to_right(fig)

    print("Generating tuning plots...")
    for path in plot_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply style locally to ensure it works with Agg backend
    apply_plot_style()
    saved_plots = []

    # 1. Optimization History
    plt.figure()
    plot_optimization_history(study)
    _finalize_plot(
        plot_paths["optimization_history"],
        title="Optimization History",
        adjust=_legend_inside_upper_right,
    )
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
        adjust=_set_slice_search_ranges,
    )
    saved_plots.append(plot_paths["param_slice"])

    if saved_plots:
        saved_str = ", ".join(str(p) for p in saved_plots)
        print(f"Saved tuning plots to {saved_str}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for E90 ML model.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from the existing Optuna study without loading data or running trials.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_tuning(config, base_dir, plots_only=args.plots_only)
