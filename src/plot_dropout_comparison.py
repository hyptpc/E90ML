"""Compare Dropout ON/OFF training histories with fixed publication axes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import apply_plot_style


def load_history(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("train_loss", "val_loss", "train_f1", "val_f1")
    if not rows or any(key not in rows[0] for key in required):
        raise ValueError(f"Invalid or empty training history: {path}")
    return {key: [float(row[key]) for row in rows] for key in required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout", required=True, help="History CSV for tuned Dropout.")
    parser.add_argument("--no-dropout", required=True, help="History CSV for dropout_rate=0.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()

    dropout = load_history(args.dropout)
    no_dropout = load_history(args.no_dropout)
    apply_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    styles = (
        (dropout, "Dropout ON", "-"),
        (no_dropout, "Dropout OFF", "--"),
    )
    for history, condition, linestyle in styles:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(
            epochs,
            history["train_f1"],
            color="tab:blue",
            linestyle=linestyle,
            label=f"train, {condition}",
        )
        axes[0].plot(
            epochs,
            history["val_f1"],
            color="tab:red",
            linestyle=linestyle,
            label=f"validation, {condition}",
        )
        axes[1].plot(
            epochs,
            history["train_loss"],
            color="tab:blue",
            linestyle=linestyle,
            label=f"train, {condition}",
        )
        axes[1].plot(
            epochs,
            history["val_loss"],
            color="tab:red",
            linestyle=linestyle,
            label=f"validation, {condition}",
        )

    axes[0].set(
        title="Dropout comparison: F1-score",
        xlabel="epoch",
        ylabel="F1-score",
        xlim=(0, args.max_epochs),
        ylim=(0, 1),
    )
    axes[1].set(
        title="Dropout comparison: loss",
        xlabel="epoch",
        ylabel="loss",
        xlim=(0, args.max_epochs),
        ylim=(0, 0.8),
    )
    for axis in axes:
        axis.grid()
        axis.legend(fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved Dropout comparison to '{args.output}'.")


if __name__ == "__main__":
    main()
