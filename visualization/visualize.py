"""Visualize loss and metrics from a Hugging Face Trainer checkpoint."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


def load_log_history(checkpoint_path: Path) -> list[dict[str, Any]]:
    """
    Load Trainer log history from a checkpoint directory or a .jsonl file.

    :param checkpoint_path: Path to checkpoint containing trainer_state.json or a .jsonl itself.
    :raises FileNotFoundError: If trainer_state.json is missing.
    :raises KeyError: If 'log_history' is missing in trainer_state.json.
    :return: List of log history entries.
    """
    if (checkpoint_path.exists() and not checkpoint_path.is_dir() and
            checkpoint_path.suffix == ".json"):
        state_file_path = checkpoint_path
    else:
        state_file_path = checkpoint_path / "trainer_state.json"

    if not state_file_path.exists():
        raise FileNotFoundError(f"No trainer_state.json in {checkpoint_path}!")

    with open(state_file_path, "r", encoding="utf-8") as jsonl_file:
        state = json.load(jsonl_file)

    if "log_history" not in state:
        raise KeyError(f"'log_history' missing in {state_file_path}")

    return state["log_history"]


def smooth_data(y: list[float], window_size_ratio: float = 0.02) -> list[float]:
    """
    Smooth the data using a relative window length with extending border values.

    :param y: The data to smooth.
    :param window_size_ratio: The ratio of the window size to the data length (e.g., 0.02 for 2%).
    :return: Smoothed data with the same length as the input.
    """
    length = len(y)
    window_size = max(3, int(length * window_size_ratio))

    if window_size >= length:
        return y

    if window_size % 2 == 0:
        pad_left = window_size // 2
        pad_right = (window_size // 2) - 1
    else:
        pad_left = pad_right = window_size // 2

    padded = np.pad(y, (pad_left, pad_right), mode='edge')
    # Use 'valid' mode so that the output length is len(padded) - window_size + 1, which equals N
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    return smoothed.tolist()


def plot_graphs_based_on_log_history(log_history: list[dict], output_dir: str | Path,
                                     metrics: list[str]) -> None:
    """
    Plot the graphs based on the log_history.

    :param log_history: The list of all logs from the Trainer.
    :param output_dir: The directory in which the plots will be created.
    :param metrics: The metrics which to create apart from training and test loss.
    """
    parsed_output_directory = Path(output_dir)

    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-epoch.png",
                                plot_epochs=True)
    plot_training_and_test_loss(log_history, parsed_output_directory / "loss-plot-step.png",
                                plot_epochs=False)

    for metric in metrics:
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-epoch.png", plot_epochs=True)
        plot_metric(metric, log_history,
                    parsed_output_directory / f"{metric}-plot-step.png", plot_epochs=False)


def plot_metric(metric: str, log_history: list[dict], output_path: str | Path,
                plot_epochs: bool = True, window_size_ratio: float = 0.01,
                applying_smoothing_threshold: int = 100) -> None:
    """
    Plot the metric using information from the log history.

    :param metric: The metric to plot (e.g. "rouge-1").
    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    :param window_size_ratio: The ratio of window size relative to data length for smoothing.
    :param applying_smoothing_threshold: The min number of data points required to apply smoothing.
    """
    metric_values = []
    steps = []
    epochs = []

    for entry in log_history:
        if metric.strip() in entry:
            metric_values.append(entry[metric])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])

    if len(metric_values) > applying_smoothing_threshold:
        metric_values = smooth_data(metric_values, window_size_ratio=window_size_ratio)
        if plot_epochs:
            epochs = smooth_data(epochs, window_size_ratio=window_size_ratio)
        else:
            steps = smooth_data(steps, window_size_ratio=window_size_ratio)

    plt.figure(figsize=(8, 4))

    if plot_epochs:
        plt.plot(epochs, metric_values, label=metric, marker='o', linestyle='-', color='0.5')
        plt.title(f"{metric}")
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, metric_values, label=metric, marker='o', linestyle='-', color='0.5')
        plt.title(f"{metric}")
        plt.xlabel('Steps')

    plt.ylabel(metric)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_training_and_test_loss(log_history: list[dict], output_path: str | Path,
                                plot_epochs: bool = True, window_size_ratio: float = 0.01,
                                apply_smoothing_threshold: int = 100) -> None:
    """
    Plot the training and test loss using information from the log history.

    :param log_history: The log history from the trainer.
    :param output_path: The path to save the plot to.
    :param plot_epochs: Whether to plot epochs or steps on the x-axis.
    :param window_size_ratio: The ratio of window size relative to data length for smoothing.
    :param apply_smoothing_threshold: The minimum number of data points required to apply smoothing.
    :raises ValueError: If the train losses and test losses have different lengths.
    """
    train_losses = []
    test_losses = []
    steps = []
    epochs = []

    for entry in log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            steps.append(entry['step'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            test_losses.append(entry['eval_loss'])

    if len(train_losses) != len(test_losses) and test_losses:
        print(f"Train losses: {train_losses}, test losses: {test_losses}, "
              f"steps: {steps}, epochs: {epochs}")
        raise ValueError("Train losses and test losses have different lengths")

    if len(train_losses) > apply_smoothing_threshold:
        train_losses = smooth_data(train_losses, window_size_ratio=window_size_ratio)
        if plot_epochs:
            epochs = smooth_data(epochs, window_size_ratio=window_size_ratio)
        else:
            steps = smooth_data(steps, window_size_ratio=window_size_ratio)

        if test_losses:
            test_losses = smooth_data(test_losses, window_size_ratio=window_size_ratio)

    plt.figure(figsize=(8, 4))

    if plot_epochs:
        plt.plot(epochs, train_losses, label='Train loss',
                 linestyle='-', color='0.4')
        if test_losses:
            plt.plot(epochs, test_losses, label='Val loss',
                     linestyle='-', color='0.8')
        plt.title('Epoch Loss')
        plt.xlabel('Epochs')
    else:
        plt.plot(steps, train_losses, label='Train loss',
                 linestyle='-', color='0.4')
        if test_losses:
            plt.plot(steps, test_losses, label='Val loss',
                     linestyle='-', color='0.8')
        plt.title('Step Loss')
        plt.xlabel('Steps')

    plt.ylabel('Loss')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(output_path)


def main() -> None:
    """Plot the graphs for a given checkpoint."""
    parser = argparse.ArgumentParser(
        description="Plot the graphs for a given checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint to plot."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output plots. Defaults to the checkpoint directory."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='*',
        default=[],
        help="Additional metrics to plot apart from training and test loss (e.g. 'rouge-1')."
    )

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir

    log_history = load_log_history(args.checkpoint_dir)
    plot_graphs_based_on_log_history(log_history, args.output_dir, args.metrics)


if __name__ == "__main__":
    main()
