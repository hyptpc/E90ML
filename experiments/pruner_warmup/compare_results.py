"""Compare MedianPruner studies with zero and ten warmup epochs."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import apply_plot_style


def load_study(path, name):
    resolved = Path(path).expanduser().resolve()
    return optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{resolved}",
    )


def trial_duration_seconds(trial):
    if trial.datetime_start is None or trial.datetime_complete is None:
        return np.nan
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def prune_epoch(trial):
    if trial.state != TrialState.PRUNED or not trial.intermediate_values:
        return np.nan
    return max(trial.intermediate_values) + 1


def study_rows(study, label):
    return [
        {
            "study": label,
            "trial": trial.number,
            "state": trial.state.name,
            "batch_size": trial.params.get("batch_size"),
            "objective": trial.value,
            "last_epoch": (
                max(trial.intermediate_values) + 1
                if trial.intermediate_values
                else np.nan
            ),
            "prune_epoch": prune_epoch(trial),
            "duration_seconds": trial_duration_seconds(trial),
        }
        for trial in study.trials
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup0-db", required=True)
    parser.add_argument("--warmup0-study", required=True)
    parser.add_argument("--warmup10-db", required=True)
    parser.add_argument("--warmup10-study", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--paired-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    study0 = load_study(args.warmup0_db, args.warmup0_study)
    study10 = load_study(args.warmup10_db, args.warmup10_study)
    rows = [*study_rows(study0, "warmup=0"), *study_rows(study10, "warmup=10")]

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    paired_rows = []
    by_number0 = {trial.number: trial for trial in study0.trials}
    by_number10 = {trial.number: trial for trial in study10.trials}
    for number in sorted(set(by_number0) & set(by_number10))[:10]:
        trial0 = by_number0[number]
        trial10 = by_number10[number]
        paired_rows.append(
            {
                "trial": number,
                "same_parameters": trial0.params == trial10.params,
                "warmup0_state": trial0.state.name,
                "warmup10_state": trial10.state.name,
                "warmup0_objective": trial0.value,
                "warmup10_objective": trial10.value,
                "warmup0_last_epoch": (
                    max(trial0.intermediate_values) + 1
                    if trial0.intermediate_values
                    else np.nan
                ),
                "warmup10_last_epoch": (
                    max(trial10.intermediate_values) + 1
                    if trial10.intermediate_values
                    else np.nan
                ),
            }
        )
    args.paired_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.paired_summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)

    apply_plot_style()
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    labels = ("warmup=0", "warmup=10")
    studies = (study0, study10)
    # The five completed startup trials have identical coordinates in both
    # studies. Preserve their exact trial/objective coordinates and distinguish
    # the overlaid observations using marker shape, fill, size, and draw order.
    for study, label, color, marker, facecolor, size, zorder in zip(
        studies,
        labels,
        ("tab:blue", "tab:orange"),
        ("o", "D"),
        ("none", "tab:orange"),
        (60, 24),
        (3, 2),
    ):
        completed = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE and trial.value is not None
        ]
        axis.scatter(
            [trial.number for trial in completed],
            [trial.value for trial in completed],
            label=label,
            edgecolors=color,
            facecolors=facecolor,
            marker=marker,
            s=size,
            alpha=0.78,
            linewidths=1.3 if label == "warmup=0" else 0.6,
            zorder=zorder,
        )
    axis.set(
        title="Completed-trial objective values",
        xlabel="trial",
        ylabel="best weighted validation BCE",
        xlim=(-2, 102),
    )
    axis.legend(loc="upper right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved comparison summary to '{args.summary}'.")
    print(f"Saved paired summary to '{args.paired_summary}'.")
    print(f"Saved comparison plot to '{args.output}'.")


if __name__ == "__main__":
    main()
