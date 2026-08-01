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
from sklearn.preprocessing import StandardScaler

from common import (
    E90Dataset,
    PTH_DIR,
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
    make_event_groups,
    stratified_group_split_indices,
    sample_indices,
    summarize_labels,
    save_data_summary,
)


def _epoch_axis_upper(recorded_epochs: int) -> int:
    """Round the displayed epoch range to the next readable ten-epoch bound."""
    return max(10, int(np.ceil(recorded_epochs / 10.0) * 10))


def _plot_training_curves(history, out_path: Path, f1_ylim, loss_ylim):
    import matplotlib.pyplot as plt

    apply_plot_style()

    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    epoch_values = np.arange(1, len(history["train_f1"]) + 1)
    epoch_upper = _epoch_axis_upper(len(epoch_values))
    plt.plot(epoch_values, history["train_f1"], c="blue", label="train", linestyle="-")
    plt.plot(epoch_values, history["val_f1"], c="red", label="val", linestyle="-")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1-score")
    plt.title("Training and validation F1-score")
    plt.xlim(0, epoch_upper)
    plt.ylim(*f1_ylim)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_values, history["train_loss"], c="blue", label="train", linestyle="-")
    plt.plot(epoch_values, history["val_loss"], c="red", label="val", linestyle="-")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.xlim(0, epoch_upper)
    plt.ylim(*loss_ylim)
    plt.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _evaluate_model(model, data_loader, criterion, device, num_classes):
    """Evaluate a complete dataset without updating Dropout or BatchNorm state."""
    model.eval()
    running_loss = 0.0
    total = 0
    targets = []
    predictions = []

    with torch.inference_mode():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            if num_classes == 2:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

    loss = running_loss / total if total else 0.0
    f1 = compute_f1(targets, predictions, num_classes) if targets else 0.0
    return loss, f1


def _f1_from_confusion(confusion: np.ndarray, num_classes: int) -> float:
    """Compute binary or macro F1 from an online confusion matrix."""
    if num_classes == 2:
        tp = float(confusion[1, 1])
        fp = float(confusion[0, 1])
        fn = float(confusion[1, 0])
        denominator = 2.0 * tp + fp + fn
        return 2.0 * tp / denominator if denominator else 0.0

    scores = []
    for label in range(num_classes):
        tp = float(confusion[label, label])
        fp = float(confusion[:, label].sum() - tp)
        fn = float(confusion[label, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        scores.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


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
    _, data_output_dir = resolve_data_dirs(base_dir)
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

    train_fraction = validate_fraction(training_cfg["fraction"], "training.fraction")
    split_seed = _resolve_seed(data_cfg.get("split_seed"), seed)
    outer_val_split = data_cfg.get("val_split")
    if outer_val_split is None:
        outer_val_split = training_cfg["val_split"]
    num_workers = int(training_cfg["num_workers"])
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    f1_ylim = tuple(float(value) for value in training_cfg.get("f1_ylim", [0.7, 0.8]))
    loss_ylim = tuple(float(value) for value in training_cfg.get("loss_ylim", [0.5, 0.6]))
    batch_size_override = training_cfg.get("batch_size")
    dropout_rate_override = training_cfg.get("dropout_rate_override")
    if patience < 1:
        raise ValueError("training.patience must be at least 1.")
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1.")
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative.")
    if len(f1_ylim) != 2 or f1_ylim[0] >= f1_ylim[1]:
        raise ValueError("training.f1_ylim must contain two increasing values.")
    if len(loss_ylim) != 2 or loss_ylim[0] >= loss_ylim[1]:
        raise ValueError("training.loss_ylim must contain two increasing values.")

    # Load Data (Full)
    print("Loading data...")
    full_df, num_classes = load_data(
        files=files,
        tree_name=tree_name,
        features=features,
        label_column=label_column,
        label_mapping=label_mapping,
        fraction=1.0,
        random_state=split_seed,
        shuffle=False,
    )

    # --- Memory Optimization ---
    print("Processing data...")
    feature_matrix = full_df[features].values.astype(np.float32)
    labels = full_df[label_column].values.astype(np.int64)
    del full_df
    gc.collect()

    event_groups = make_event_groups(feature_matrix, labels)
    train_indices, val_indices = stratified_group_split_indices(
        labels,
        event_groups,
        outer_val_split,
        split_seed,
        "data outer split",
    )
    sampled_train = sample_indices(len(train_indices), train_fraction, seed)
    train_indices = train_indices[sampled_train]
    train_features = feature_matrix[train_indices]
    val_features = feature_matrix[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_groups = event_groups[train_indices]
    val_groups = event_groups[val_indices]
    del feature_matrix, labels, event_groups, train_indices, val_indices, sampled_train
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
    data_summary = {
        "stage": "training",
        "source_files": [str(path) for path in files],
        "fraction": train_fraction,
        "seed": seed,
        "split_seed": split_seed,
        "val_split": float(outer_val_split),
        "train_unique_groups": int(np.unique(train_groups).size),
        "validation_unique_groups": int(np.unique(val_groups).size),
        "overlapping_groups": 0,
        "train": summarize_labels(train_dataset.y),
        "validation": summarize_labels(val_dataset.y),
    }
    summary_raw = get_config_value(training_cfg, "data_summary_file") or "training_data_summary.json"
    save_data_summary(data_summary, resolve_dir(summary_raw, data_output_dir, base_dir))
    
    del train_features, val_features, train_labels, val_labels, train_groups, val_groups
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
        "dropout_rate": (
            float(dropout_rate_override)
            if dropout_rate_override not in (None, "")
            else float(params.get("dropout_rate", 0.2))
        ),
    }
    if not 0.0 <= model_params["dropout_rate"] < 1.0:
        raise ValueError("Dropout rate must satisfy 0 <= rate < 1.")
    if dropout_rate_override not in (None, ""):
        print(
            "Overriding tuned dropout_rate for controlled comparison: "
            f"{params.get('dropout_rate')} -> {model_params['dropout_rate']}"
        )
    batch_size = (
        int(batch_size_override)
        if batch_size_override not in (None, "")
        else int(params.get("batch_size", 128))
    )
    lr = float(params.get("lr", 1e-3))
    if batch_size < 2:
        raise ValueError("training.batch_size must be at least 2 for BatchNorm1d.")
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")

    device = resolve_device(config.get("device"))
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)
    pin_memory = device.type == "cuda"
    drop_last = batchnorm_safe_drop_last(len(train_dataset), batch_size)
    if drop_last:
        print("Dropping the final one-sample training batch to keep BatchNorm statistics valid.")
    train_generator = torch.Generator()
    if seed is not None:
        train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=train_generator,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

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
    # Validation loss is the checkpoint/early-stopping criterion. F1 remains a
    # reported classification metric but can be discontinuous at the threshold.
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
            if "best_val_loss" in checkpoint:
                best_val_loss = float(checkpoint["best_val_loss"])
                best_model_wts = checkpoint.get("best_model_state_dict", best_model_wts)
                no_improve_count = int(checkpoint.get("no_improve_count", no_improve_count))
            else:
                print(
                    "Checkpoint predates validation-loss early stopping; "
                    "resetting the best-value and patience state."
                )
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"Resuming training from epoch {start_epoch + 1} using checkpoint '{checkpoint_path}'.")
            if no_improve_count >= patience:
                print(
                    "Checkpoint already satisfies the early-stopping condition; "
                    "skipping additional training."
                )
                start_epoch = epochs
        else:
            model.load_state_dict(checkpoint)
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Loaded weights-only checkpoint from '{checkpoint_path}'. Training will restart from epoch 1.")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        train_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if num_classes == 2:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels.float())
                predictions = (torch.sigmoid(outputs) > 0.5).long()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

            batch_size_actual = labels.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_sample_count += batch_size_actual
            labels_cpu = labels.detach().cpu().numpy()
            predictions_cpu = predictions.detach().cpu().numpy()
            np.add.at(train_confusion, (labels_cpu, predictions_cpu), 1)

            loss.backward()
            optimizer.step()

        # Standard training metrics: aggregate the actual mini-batch losses and
        # predictions used for optimization while Dropout and BatchNorm are in
        # training mode. Validation remains an inference-mode evaluation.
        train_loss = train_loss_sum / train_sample_count if train_sample_count else 0.0
        train_f1 = _f1_from_confusion(train_confusion, num_classes)
        val_loss, val_f1 = _evaluate_model(model, val_loader, criterion, device, num_classes)

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            "best_val_loss": best_val_loss,
            "history": history,
            "no_improve_count": no_improve_count,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_path)

        if no_improve_count >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    # Load best model weights
    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Best model saved to '{model_output_path}'.")

    # Plotting
    project_root = Path(__file__).resolve().parent.parent
    default_plots_dir = project_root / "plots" / "train"
    plot_output_raw = get_config_value(
        training_cfg, "plot_output_file", "plot_output_path", "plots_path", "plots_dir"
    ) or (default_plots_dir / "training_curves.png")
    plot_output_path = resolve_dir(str(plot_output_raw), default_plots_dir, project_root)
    if plot_output_path.suffix == "":
        plot_output_path = plot_output_path / "training_curves.png"
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_training_curves(history, plot_output_path, f1_ylim, loss_ylim)
    print(f"Saved training curves to '{plot_output_path}'.")

    # Training history (loss/F1 per epoch)
    history_output_raw = get_config_value(
        training_cfg, "history_output_file", "metrics_output_file", "metrics_output_path"
    )
    if not history_output_raw:
        raise ValueError("Config must set training.history_output_file.")
    history_output_path = resolve_dir(history_output_raw, data_output_dir, base_dir)
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
