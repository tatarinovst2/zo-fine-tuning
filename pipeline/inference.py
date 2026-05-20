"""
Run inference for classification and generation tasks.

Classification mode uses next-token scoring with ``AutoModelForCausalLM``:
    prompt -> next-token logits -> restrict to verbalizer first-token ids -> argmax
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from check_metrics import compute_metrics
from utils import (compute_next_token_option_logits, get_current_torch_device,
                   get_first_token_verbalizer_ids, load_causal_lm_model_for_inference,
                   load_verbalizers_json, read_jsonl, resolve_dtype,
                   select_one_verbalizer_per_label, set_seed)


def _load_tokenizer(args: argparse.Namespace) -> PreTrainedTokenizerBase:
    """Load tokenizer for inference.

    :param args: Parsed CLI arguments.
    :return: Loaded tokenizer.
    """
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def run_classification_inference(  # pylint: disable=too-many-locals
    args: argparse.Namespace,
    data_file: Path,
) -> list[dict[str, Any]]:
    """Run next-token option-LM classification inference.

    The model receives only the prompt. The score for each class is the next-token logit
    corresponding to the first token of that class's selected verbalizer.

    :param args: Parsed CLI arguments.
    :param data_file: Path to a JSONL with entries containing ``prompt`` and optionally ``answer``.
    :raises ValueError: If ``--verbalizers_json`` is missing.
    :return: Prediction entries.
    """
    if not args.verbalizers_json:
        raise ValueError("Classification inference requires --verbalizers_json.")

    dataset_name = (args.dataset_name or data_file.parent.name).lower()

    spec = load_verbalizers_json(Path(args.verbalizers_json), dataset_name)
    label_names = list(spec.label_names)
    option_texts = select_one_verbalizer_per_label(spec)

    tokenizer = _load_tokenizer(args)
    tokenizer.padding_side = "right"

    verbalizer_token_ids = get_first_token_verbalizer_ids(
        tokenizer=tokenizer,
        label_names=label_names,
        option_texts=option_texts,
        verbose=True,
    )

    print(f"Dataset name: {dataset_name}")
    print(f"Labels: {label_names}")
    print(f"Selected verbalizers: {option_texts}")
    print(f"Verbalizer first-token ids: {verbalizer_token_ids}")

    model = load_causal_lm_model_for_inference(args)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    device = get_current_torch_device()
    _prepare_model_for_inference(args, model)

    data = read_jsonl(data_file)
    outputs: list[dict[str, Any]] = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(data), args.batch_size),
            desc="Running next-token classification inference...",
        ):
            batch = data[i:i + args.batch_size]
            prompts = [ex["prompt"] for ex in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            )

            logits = compute_next_token_option_logits(
                model=model,
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                verbalizer_token_ids=verbalizer_token_ids,
            )

            pred_indices = logits.argmax(dim=-1).detach().cpu().tolist()
            logits_cpu = logits.detach().cpu().tolist()

            for pred_idx, score_row, ex in zip(pred_indices, logits_cpu, batch):
                outputs.append(
                    {
                        "prompt": ex["prompt"],
                        "prediction": label_names[pred_idx],
                        "prediction_verbalizer": option_texts[pred_idx],
                        "prediction_token_id": verbalizer_token_ids[pred_idx],
                        "label": ex.get("answer", "")
                    }
                )

    return outputs


def _decode_generations(
    tokenizer: PreTrainedTokenizerBase,
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> list[str]:
    """Decode only generated continuations.

    :param tokenizer: Tokenizer.
    :param generated_ids: Generated ids from ``model.generate``.
    :param input_ids: Original padded prompt input ids.
    :return: Decoded continuations.
    """
    outs: list[str] = []
    prompt_len = input_ids.size(1)

    for i in range(generated_ids.size(0)):
        cont = generated_ids[i, prompt_len:]
        outs.append(tokenizer.decode(cont, skip_special_tokens=True).strip())

    return outs


def run_generation_inference(
    args: argparse.Namespace,
    data_file: Path,
) -> list[dict[str, Any]]:
    """Run generation inference.

    :param args: Parsed CLI arguments.
    :param data_file: Path to a JSONL with entries containing ``prompt`` and optionally ``answer``.
    :return: Prediction entries.
    """
    tokenizer = _load_tokenizer(args)
    tokenizer.padding_side = "left"

    device = get_current_torch_device()
    model = load_causal_lm_model_for_inference(args)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    _prepare_model_for_inference(args, model)

    data = read_jsonl(data_file)
    outputs: list[dict[str, Any]] = []

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        for i in tqdm(range(0, len(data), args.batch_size),
                      desc="Running generation inference..."):
            batch = data[i:i + args.batch_size]
            prompts = [ex["prompt"] for ex in batch]

            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_length)

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       **gen_kwargs)

            conts = _decode_generations(
                tokenizer=tokenizer,
                generated_ids=generated,
                input_ids=input_ids,
            )

            for cont, ex in zip(conts, batch):
                outputs.append(
                    {
                        "prompt": ex["prompt"],
                        "prediction": cont,
                        "label": ex.get("answer", ""),
                    }
                )

    return outputs


def _maybe_compute_metrics(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    """Optionally compute metrics.

    :param args: Parsed CLI arguments.
    :param entries: Prediction entries.
    """
    if not args.compute_metrics:
        return

    labels = [entry.get("label", "") for entry in entries]
    predictions = [entry.get("prediction", "") for entry in entries]

    compute_metrics(labels=labels, predictions=predictions, task_type=args.task_type)


def _prepare_model_for_inference(args: argparse.Namespace,
                                 model: torch.nn.Module) -> torch.nn.Module:
    """
    Check dtype and set it if needed.

    :param args: Parsed CLI arguments.
    :param model: Loaded model.
    :return: Model moved to the correct device and dtype for inference.
    """
    device = get_current_torch_device()
    target_dtype = resolve_dtype(args.dtype)

    cfg_dtype = getattr(model.config, "dtype", None)
    if cfg_dtype is None:
        cfg_dtype = getattr(model.config, "torch_dtype", None)
    print(f"Model original dtype from config: {cfg_dtype}")

    if target_dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=target_dtype)

    model.eval()

    print(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    return model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference for classification or generation tasks."
    )

    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to input .jsonl file with prompt column."
    )
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["classification", "generation", "boxed_generation"],
        help="Task type."
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="HF model name or local checkpoint dir."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default="",
        help="Base model name/path when --model_name_or_path is a PEFT/LoRA adapter.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="",
        help="Tokenizer path/name. Useful when --model_name_or_path is a PEFT adapter directory.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Where to write predictions .jsonl.")

    parser.add_argument(
        "--verbalizers_json",
        default="verbalizers.json",
        help="Path to verbalizers json.")
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Dataset name for verbalizers lookup. If empty, inferred from data_file parent dir.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Dtype for inference (auto, fp16, bf16, fp32)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max prompt/input length."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling for generation."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top_p."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search width."
    )

    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Compute metrics after inference."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    return parser.parse_args()


def main() -> None:
    """Run inference and save predictions."""
    args = parse_args()
    set_seed(args.seed)

    data_file = Path(args.data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"data_file not found: {data_file}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_type == "classification":
        preds = run_classification_inference(args, data_file)
    else:
        preds = run_generation_inference(args, data_file)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for item in preds:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(preds)} predictions to {output_path}")

    _maybe_compute_metrics(args, preds)


if __name__ == "__main__":
    main()
