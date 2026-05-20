"""Compare loss and metrics across multiple runs, showing inline plots."""
import argparse
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from visualize import load_log_history, smooth_data


def extract_series(history: list[dict[str, Any]], key: str,
                   max_steps: int | None = None) -> tuple[list[float], list[int], list[float]]:
    """
    Extract (epochs, steps, values) for a given metric key.

    :param history: Trainer log history entries.
    :param key: Metric name to extract (e.g., 'loss', 'eval_loss').
    :param max_steps: Optional maximum step to include in the series.
    :return: Tuple of (epochs, steps, values).
    """
    epochs, steps, vals = [], [], []
    for entry in history:
        if key in entry:
            if max_steps is not None and entry["step"] > max_steps:
                continue

            if "epoch" in entry:
                epochs.append(entry["epoch"])
            steps.append(entry["step"])
            vals.append(entry[key])
    return epochs, steps, vals


def plot_loss_type(runs: list[dict[str, Any]], loss_key: str, x_axis: str = "epoch",
                   smooth: bool = True, smooth_threshold: int = 50, max_steps: int | None = None):
    """
    Plot training or evaluation loss across multiple runs.

    :param runs: List of runs with 'label' and 'history'.
    :param loss_key: Loss key ('loss' or 'eval_loss').
    :param x_axis: X-axis type ('epoch' or 'step').
    :param smooth: Whether to apply smoothing.
    :param smooth_threshold: Minimum points before smoothing is applied.
    :param max_steps: Optional maximum step to include in the plot.
    :raises ValueError: If x_axis is not 'epoch' or 'step'.
    """
    if x_axis not in ("epoch", "step"):
        raise ValueError("x_axis must be 'epoch' or 'step'")

    plt.figure(figsize=(8, 4))
    for run in runs:
        label = run["label"]
        epochs, steps, vals = extract_series(run["history"], loss_key,
                                             max_steps=max_steps)
        if not vals:
            continue  # e.g. no eval_loss in some runs
        x = epochs if x_axis == "epoch" else steps
        y = vals

        if smooth and len(y) > smooth_threshold:
            y = smooth_data(y)
            x = smooth_data(x)

        plt.plot(x, y, label=label, marker=None)

    pretty_key = "Train Loss" if loss_key == "loss" else "Eval Loss"
    plt.title(f"{pretty_key} vs {x_axis.capitalize()}")
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(pretty_key)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(runs: list[dict[str, Any]], metric: str, smooth: bool = True,
                           smooth_threshold: int = 50, max_steps: int | None = None):
    """
    Plot a metric across runs vs both epoch and step.

    :param runs: List of runs with 'label' and 'history'.
    :param metric: Metric name to plot.
    :param smooth: Whether to apply smoothing.
    :param smooth_threshold: Minimum points before smoothing is applied.
    :param max_steps: Optional maximum step to include in the plots.
    """
    # Epoch‐level
    plt.figure(figsize=(8, 4))
    for run in runs:
        label = run["label"]
        epochs, _, vals = extract_series(run["history"], metric,
                                         max_steps=max_steps)
        if not vals:
            continue

        if smooth and len(vals) > smooth_threshold:
            vals = smooth_data(vals)
            epochs = smooth_data(epochs)
        plt.plot(epochs, vals, label=label)

    plt.title(f"{metric} vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step‐level
    plt.figure(figsize=(8, 4))
    for run in runs:
        label = run["label"]
        _, steps, vals = extract_series(run["history"], metric)
        if not vals:
            continue

        if smooth and len(vals) > smooth_threshold:
            vals = smooth_data(vals)
            steps = smooth_data(steps)
        plt.plot(steps, vals, label=label)

    plt.title(f"{metric} vs Step")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Visualize comparison of train/eval loss and extra metrics across multiple runs."""
    parser = argparse.ArgumentParser(
        description="Compare train/eval loss (and extras) across runs, show inline."
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="+",
        required=True,
        help="Checkpoint directories (must contain trainer_state.json)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each checkpoint dir (same order/count)"
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[],
        help="Extra metric keys to compare (e.g. rouge-1 accuracy)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Only plot points with step <= max_steps",
    )
    args = parser.parse_args()

    if len(args.checkpoint_paths) != len(args.labels):
        parser.error("Must supply same count of --checkpoint_dirs and --labels")

    runs = []
    for checkpoint_path, lbl in zip(args.checkpoint_paths, args.labels):
        history = load_log_history(Path(checkpoint_path))
        runs.append({"label": lbl, "history": history})

    plot_loss_type(runs, loss_key="loss",  x_axis="step")
    plot_loss_type(runs, loss_key="eval_loss", x_axis="step")

    for m in args.metrics:
        plot_metric_comparison(runs, m)


if __name__ == "__main__":
    main()
