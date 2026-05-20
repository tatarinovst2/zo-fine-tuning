"""A script that provides metrics, such as ROUGE or BLEU."""
import re

from evaluate import load
from sympy import simplify
from sympy.parsing.latex import parse_latex


def extract_boxed(text: str) -> str:
    """
    Extract the content inside the last \\boxed{} in the text.

    :param text: The input text containing the solution with \\boxed{}.
    :return: The content inside the last \\boxed{}, or an empty string if not found.
    """
    matches = re.findall(r'\\boxed\{([^}]*)}', text)

    if matches:
        return matches[-1].strip()

    return ""


def math_expressions_equal(pred: str, gold: str) -> bool:
    """
    Compare two math expressions symbolically using SymPy.

    :param pred: Predicted expression.
    :param gold: Ground truth expression.
    :return: True if mathematically equivalent.
    """
    def normalize_math_expression(expr: str) -> str:
        expr = expr.strip()

        replacements = {
            "\\left": "",
            "\\right": "",
            " ": "",
            "\n": "",
            "\t": "",
        }

        for old, new in replacements.items():
            expr = expr.replace(old, new)

        return expr

    pred = normalize_math_expression(pred)
    gold = normalize_math_expression(gold)

    if pred == gold:
        return True

    try:
        pred_expr = parse_latex(pred)
        gold_expr = parse_latex(gold)

        return simplify(pred_expr - gold_expr) == 0

    except Exception:
        return pred.lower() == gold.lower()


def get_boxed_accuracy(predictions: list[str], labels: list[str]) -> tuple[float, int]:
    """
    Compute boxed-answer accuracy for MATH-style datasets.

    Extracts \\boxed{} answers and compares them symbolically.

    :param predictions: Model-generated solutions.
    :param labels: Ground-truth solutions.
    :return: (accuracy, correct_count)
    """
    correct = 0

    for pred_text, gold_text in zip(predictions, labels):
        pred_boxed = extract_boxed(pred_text)
        gold_boxed = extract_boxed(gold_text)

        if not pred_boxed or not gold_boxed:
            continue

        if math_expressions_equal(pred_boxed, gold_boxed):
            correct += 1

    accuracy = correct / len(labels) if labels else 0.0

    return accuracy, correct


def get_accuracy(predictions: list[str], labels: list[str]) -> tuple[float, float]:
    """
    Calculate accuracy given predictions and labels.

    :param predictions: List of predicted labels.
    :param labels: List of true labels.
    :return: Tuple of (accuracy, number of correct predictions).
    """
    correct = sum(
        pred.strip().lower() == label.strip().lower()
        for pred, label in zip(predictions, labels)
    )
    accuracy = (correct / len(labels)) if labels else 0.0
    return accuracy, correct


def get_bleu_score(predictions: list[str], labels: list[str]) -> float:
    """
    Compute the BLEU score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :return: The BLEU score.
    """
    blue_metric = load("sacrebleu")

    blue_results = blue_metric.compute(predictions=predictions, references=labels)
    return blue_results["score"]


def get_rouge_score(predictions: list[str], labels: list[str]) -> float:
    """
    Compute the ROUGE score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :return: The score for ROUGE-L.
    """
    rouge_metric = load("rouge")

    rouge_results = rouge_metric.compute(predictions=predictions, references=labels)

    return rouge_results["rougeL"]


def get_bert_score(predictions: list[str], labels: list[str],
                   device: str = "cpu") -> float:
    """
    Compute the BERT score for a list of predictions and labels.

    :param predictions: The list of predictions.
    :param labels: The list of labels.
    :param device: The PyTorch device to use (e.g. "cuda", "mps" or "cpu")
    :return: The score F1.
    """
    bert_metric = load("bertscore")

    bert_results = bert_metric.compute(predictions=predictions, references=labels,
                                       device=device)

    return sum(bert_results["f1"]) / len(bert_results["f1"])
