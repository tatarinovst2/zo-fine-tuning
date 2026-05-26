"""Compare loss and metrics across multiple runs, showing inline plots."""
import argparse
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from visualize import load_log_history, smooth_data


def extract_series(history: list[dict[str, Any]], key: str, step_scale: float = 1.0,
                   max_steps: int | None = None,     # cap raw trainer steps
                   max_x: float | None = None,       # cap scaled x (e.g. forwards)
) -> tuple[list[float], list[float]]:
    """
    Extract (epochs, scaled_steps, values) for a given metric key.

    :param history: Trainer log history entries.
    :param key: Metric name to extract (e.g., 'loss', 'eval_loss').
    :param step_scale: Multiplier applied to entry["step"] (e.g. forwards per step).
    :param max_steps: Optional maximum raw step to include.
    :param max_x: Optional maximum scaled step (x) to include.
    :return: Tuple of (epochs, scaled_steps, values).
    """
    xs: list[float] = []
    vals: list[float] = []

    for entry in history:
        if key not in entry:
            continue
        if "step" not in entry:
            continue

        raw_step = entry["step"]
        if max_steps is not None and raw_step > max_steps:
            continue

        x = raw_step * step_scale
        if max_x is not None and x > max_x:
            continue

        xs.append(x)
        vals.append(entry[key])

    return xs, vals


def plot_loss_type(runs: list[dict[str, Any]], loss_key: str,
                   x_name: str | None = None,  # only used when x_axis == "step"
                   smooth: bool = True, smooth_threshold: int = 50, max_steps: int | None = None,
                   max_x: float | None = None):
    """
    Plot training or evaluation loss across multiple runs.

    :param runs: List of runs with 'label' and 'history'.
    :param loss_key: Loss key ('loss' or 'eval_loss').
    :param x_name: X-axis name to use when x_axis is 'step' (e.g. 'Forwards').
    :param smooth: Whether to apply smoothing.
    :param smooth_threshold: Minimum points before smoothing is applied.
    :param max_steps: Optional maximum step to include in the plot.
    :param max_x: Optional maximum scaled step (x) to include in the plot (e.g. forwards cap).
    """
    plt.figure(figsize=(8, 4))
    for run in runs:
        label = run["label"]
        step_scale = run.get("step_scale", 1.0)

        xs, vals = extract_series(run["history"], loss_key, step_scale=step_scale,
                                  max_steps=max_steps, max_x=max_x)
        if not vals:
            continue  # e.g. no eval_loss in some runs

        x = xs
        y = vals

        if smooth and len(y) > smooth_threshold:
            y = smooth_data(y)
            x = smooth_data(x)

        plt.plot(x, y, label=label, marker=None)

    pretty_key = "Train Loss" if loss_key == "loss" else "Eval Loss"
    plt.title(f"{pretty_key} vs {(x_name or 'Step')}")
    plt.xlabel(x_name or "Step")
    plt.ylabel(pretty_key)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(runs: list[dict[str, Any]], metric: str, x_name: str = "Step",
                           smooth: bool = True, smooth_threshold: int = 50,
                           max_steps: int | None = None, max_x: float | None = None):
    """
    Plot a metric across runs vs both epoch and step.

    :param runs: List of runs with 'label' and 'history'.
    :param metric: Metric name to plot.
    :param x_name: X-axis name to use when plotting against scaled steps (e.g. 'Forward').
    :param smooth: Whether to apply smoothing.
    :param smooth_threshold: Minimum points before smoothing is applied.
    :param max_steps: Optional maximum step to include in the plots.
    :param max_x: Optional maximum scaled step (x) to include in the plots.
    """
    plt.figure(figsize=(8, 4))
    for run in runs:
        label = run["label"]
        step_scale = run.get("step_scale", 1.0)

        xs, vals = extract_series(run["history"], metric, step_scale=step_scale,
                                  max_steps=max_steps, max_x=max_x)
        if not vals:
            continue

        if smooth and len(vals) > smooth_threshold:
            vals = smooth_data(vals)
            xs = smooth_data(xs)

        plt.plot(xs, vals, label=label)

    plt.title(f"{metric} vs {x_name}")
    plt.xlabel(x_name)
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
        "--step_scales",
        nargs="+",
        type=float,
        default=None,
        help="Per-run multiplier applied to logged trainer step (e.g. forwards/step). "
             "Must match number of checkpoints. Default: all 1.0",
    )
    parser.add_argument(
        "--x_name",
        type=str,
        default="Forwards",
        help="X-axis name to use when plotting against scaled steps (e.g. 'Forward').",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Only include points with raw trainer step <= max_steps",
    )
    parser.add_argument(
        "--max_x",
        type=float,
        default=None,
        help="Only include points with scaled x <= max_x (e.g. forwards cap).",
    )
    args = parser.parse_args()

    if len(args.checkpoint_paths) != len(args.labels):
        parser.error("Must supply same count of --checkpoint_paths and --labels")

    if args.step_scales is None:
        step_scales = [1.0] * len(args.checkpoint_paths)
    else:
        if len(args.step_scales) != len(args.checkpoint_paths):
            parser.error("Must supply same count of --step_scales and --checkpoint_paths")
        step_scales = args.step_scales

    runs = []
    for checkpoint_path, label, scale in zip(args.checkpoint_paths, args.labels, step_scales):
        history = load_log_history(Path(checkpoint_path))
        runs.append({"label": label, "history": history, "step_scale": scale})

    plot_loss_type(runs, loss_key="loss", x_name=args.x_name,
                   max_steps=args.max_steps, max_x=args.max_x)
    plot_loss_type(runs, loss_key="eval_loss", x_name=args.x_name,
                   max_steps=args.max_steps, max_x=args.max_x)

    for metric in args.metrics:
        plot_metric_comparison(runs, metric, x_name=args.x_name, max_steps=args.max_steps,
                               max_x=args.max_x)


if __name__ == "__main__":
    main()
