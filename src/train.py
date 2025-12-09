import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from common import (
    DEFAULT_FEATURES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TREE_NAME,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_METRICS_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_PREDICTIONS_NAME,
    DEFAULT_PTH_DIR,
    DEFAULT_TUNED_PARAMS_NAME,
    DEFAULT_TUNE_DIR,
    E90Dataset,
    create_model_from_params,
    load_config,
    resolve_named_dir,
    resolve_named_path,
    resolve_data_files,
    resolve_device,
)


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
    training_cfg = config.get("training", {})
    tuning_cfg = config.get("tuning", {})

    train_fraction = float(training_cfg.get("fraction", 1.0))
    val_split = float(training_cfg.get("val_split", 0.2))
    num_workers = int(training_cfg.get("num_workers", 2))
    epochs = int(training_cfg.get("epochs", 50))
    batch_size_override = training_cfg.get("batch_size")

    dataset, features, num_classes = _prepare_data(config, base_dir, train_fraction, is_train=True)

    best_params_raw = training_cfg.get("best_params_path") or tuning_cfg.get("best_params_path") or DEFAULT_TUNED_PARAMS_NAME
    best_params_path = resolve_named_path(
        best_params_raw,
        default_dir=DEFAULT_TUNE_DIR,
        default_name=DEFAULT_TUNED_PARAMS_NAME,
        base_dir=base_dir,
    )
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

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Not enough data to perform the requested train/val split.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device(training_cfg.get("device") or config.get("device"))
    print(f"Using device: {device}")
    model = create_model_from_params(model_params, input_dim=len(features), num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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

    model_output_path = resolve_named_path(
        training_cfg.get("model_output_path"),
        default_dir=DEFAULT_PTH_DIR,
        default_name=DEFAULT_MODEL_NAME,
        base_dir=base_dir,
    )
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Final model saved to '{model_output_path}'.")

    plots_dir = resolve_named_dir(training_cfg.get("plots_dir"), default_dir=DEFAULT_PLOTS_DIR, base_dir=base_dir)
    _plot_history(history["train_loss"], history["val_loss"], "Loss", plots_dir / "loss.png")
    _plot_history(history["train_acc"], history["val_acc"], "Accuracy", plots_dir / "accuracy.png")
    print(f"Saved training curves to '{plots_dir}'.")

    metrics_output_path = resolve_named_path(
        training_cfg.get("metrics_output_path"),
        default_dir=DEFAULT_OUTPUT_DIR,
        default_name=DEFAULT_METRICS_NAME,
        base_dir=base_dir,
    )
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_classes": num_classes,
        "features": features,
    }
    with metrics_output_path.open("w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved metrics to '{metrics_output_path}'.")

    predictions_output_path = resolve_named_path(
        training_cfg.get("predictions_output_path"),
        default_dir=DEFAULT_OUTPUT_DIR,
        default_name=DEFAULT_PREDICTIONS_NAME,
        base_dir=base_dir,
    )
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    records = []
    with torch.no_grad():
        for inputs, labels in full_loader:
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

    import pandas as pd

    pd.DataFrame.from_records(records).to_csv(predictions_output_path, index=False)
    print(f"Saved predictions to '{predictions_output_path}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Final training script using tuned hyperparameters.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config, base_dir = load_config(args.config)
    train_final(config, base_dir)
