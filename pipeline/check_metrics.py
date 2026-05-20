"""Compute metrics from a predictions JSONL file."""
import argparse
from pathlib import Path

from metrics import get_accuracy, get_bleu_score, get_boxed_accuracy, get_rouge_score
from utils import read_jsonl


def compute_metrics(labels: list[str], predictions: list[str], task_type: str) -> None:
    """
    Compute and print metrics based on the task type.

    :param labels: List of ground truth labels.
    :param predictions: List of model predictions.
    :param task_type: Type of task ('classification', 'generation', or 'boxed_generation').
    """
    if task_type == "classification":
        accuracy, correct = get_accuracy(predictions, labels)
        print(f"Accuracy: {accuracy*100:.2f}%  ({correct}/{len(labels)})")
    elif task_type in ("generation", "boxed_generation"):
        rouge_l = get_rouge_score(predictions, labels)
        bleu = get_bleu_score(predictions, labels)
        print(f"ROUGE-L: {rouge_l:.4f}")
        print(f"BLEU:    {bleu:.4f}")

        if task_type == "boxed_generation":
            boxed_accuracy, boxed_correct = get_boxed_accuracy(predictions, labels)
            print(f"Boxed Generation Accuracy: {boxed_accuracy*100:.2f}%  "
                  f"({boxed_correct}/{len(labels)})")
    else:
        raise ValueError("task_type must be one of {'classification', 'generation', "
                         "'boxed_generation'}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for metrics computation.

    :return: An argparse.Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute metrics from predictions JSONL.")
    parser.add_argument("--predictions_file", required=True,
                        help="Path to predictions JSONL written by inference.py")
    parser.add_argument("--task_type", required=True,
                        choices=["classification", "generation", "boxed_generation"],
                        help="Task type for metric selection.")
    return parser.parse_args()


def main() -> None:
    """Load predictions and compute task-appropriate metrics."""
    args = parse_args()

    path = Path(args.predictions_file)
    rows = read_jsonl(path)

    labels = [row.get("label", "") for row in rows]
    predictions = [row.get("prediction", "") for row in rows]

    compute_metrics(labels, predictions, args.task_type)


if __name__ == "__main__":
    main()
