"""Compare Dropout ON/OFF histories with focused and readable plot ranges."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import apply_plot_style


def load_history(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("train_loss", "val_loss", "train_f1", "val_f1")
    if not rows or any(key not in rows[0] for key in required):
        raise ValueError(f"Invalid or empty training history: {path}")
    return {key: [float(row[key]) for row in rows] for key in required}


def epoch_axis_upper(*histories):
    recorded_epochs = max(len(history["train_loss"]) for history in histories)
    return max(10, int(np.ceil(recorded_epochs / 10.0) * 10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout", required=True, help="History CSV for tuned Dropout.")
    parser.add_argument("--no-dropout", required=True, help="History CSV for dropout_rate=0.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--f1-range", nargs=2, type=float, default=(0.7, 0.8))
    parser.add_argument("--loss-range", nargs=2, type=float, default=(0.5, 0.6))
    args = parser.parse_args()

    dropout = load_history(args.dropout)
    no_dropout = load_history(args.no_dropout)
    epoch_upper = epoch_axis_upper(dropout, no_dropout)
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
        xlim=(0, epoch_upper),
        ylim=tuple(args.f1_range),
    )
    axes[1].set(
        title="Dropout comparison: loss",
        xlabel="epoch",
        ylabel="loss",
        xlim=(0, epoch_upper),
        ylim=tuple(args.loss_range),
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
