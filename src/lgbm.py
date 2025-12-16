import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from common import (
    OUTPUT_DIR,
    LABEL_MAPPING,
    apply_plot_style,
    load_config,
    load_data,
    resolve_data_files,
    resolve_dir,
    _resolve_seed,
)


def _require(cfg: dict, key: str, section: str):
    value = cfg.get(key)
    if value in (None, ""):
        raise ValueError(f"Config must set '{section}.{key}'.")
    return value


def _build_output_paths(training_cfg: dict, base_dir: Path) -> dict:
    output_dir_raw = training_cfg.get("output_dir", "lgbm")
    output_dir = resolve_dir(output_dir_raw, OUTPUT_DIR, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    default_plots_dir = project_root / "plots" / "lgbm"
    plots_dir_raw = training_cfg.get("plots_dir", default_plots_dir)
    plots_dir = resolve_dir(str(plots_dir_raw), default_plots_dir, project_root)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "dir": output_dir,
        "model": output_dir / training_cfg.get("model_output_file", "lgbm.txt"),
        "importance_plot": plots_dir / training_cfg.get("feature_importance_file", "feature_importance.png"),
        "roc_plot": plots_dir / training_cfg.get("roc_curve_file", "roc_curve.png"),
    }


def _default_params(num_classes: int, seed: int) -> dict:
    return {
        "objective": "binary" if num_classes == 2 else "multiclass",
        "metric": "binary_logloss" if num_classes == 2 else "multi_logloss",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": seed,
        "n_jobs": -1,
        "importance_type": "gain",
    }


def train_lgbm(config, base_dir):
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    seed = _resolve_seed(training_cfg.get("seed", 42), config.get("seed"))

    paths = _build_output_paths(training_cfg, base_dir)

    files = resolve_data_files(data_cfg, base_dir)
    print("Loading data...")
    df, num_classes = load_data(
        files=files,
        tree_name=_require(data_cfg, "tree_name", "data"),
        features=_require(data_cfg, "feature_columns", "data"),
        label_column=_require(data_cfg, "label_column", "data"),
        label_mapping=data_cfg.get("label_mapping", LABEL_MAPPING),
        fraction=float(training_cfg.get("fraction", 1.0)),
        random_state=seed,
    )

    label_column = data_cfg["label_column"]
    features = [c for c in df.columns if c != label_column]
    print(f"Features used ({len(features)}): {features}")

    X = df[features]
    y = df[label_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(_require(training_cfg, "val_split", "training")),
        stratify=y,
        random_state=seed,
    )

    params = _default_params(num_classes, seed)
    params.update(training_cfg.get("params", {}))
    if num_classes > 2:
        params["num_class"] = num_classes

    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=int(training_cfg.get("early_stopping_rounds", 50)), verbose=True),
        lgb.log_evaluation(period=int(training_cfg.get("log_period", 100))),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["validation"],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_val)
    average = "binary" if num_classes == 2 else "macro"
    f1 = f1_score(y_val, y_pred, average=average)
    print(f"\nValidation F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    apply_plot_style()
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, importance_type="gain", max_num_features=20, height=0.5)
    plt.title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(paths["importance_plot"])
    print(f"Saved feature importance to {paths['importance_plot']}")

    if num_classes == 2:
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        fpr, tpr, _ = roc_curve(y_val, y_prob)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(paths["roc_plot"])
        print(f"Saved ROC curve to {paths['roc_plot']}")

    booster = model.booster_
    booster.save_model(str(paths["model"]))
    print(f"Saved model to {paths['model']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM model for E90ML.")
    parser.add_argument("-c", "--config", required=True, help="Path to config file (yaml/json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg, base = load_config(args.config)
    train_lgbm(cfg, base)
