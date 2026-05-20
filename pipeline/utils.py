"""Utility functions for the pipeline."""
import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase


def get_current_torch_device() -> str:
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_dtype(dtype_str: str) -> torch.dtype | None:
    """
    Map CLI dtype string to torch.dtype. Returns None for 'auto'.

    :param dtype_str: Dtype string from CLI.
    :raises ValueError: If the dtype string is unrecognized.
    :return: Corresponding torch.dtype or None for 'auto'.
    """
    dtype_input = (dtype_str or "auto").lower().strip()
    if dtype_input == "auto":
        return None
    if dtype_input in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_input in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_input in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


class VerbalizerSpec:  # pylint: disable=too-few-public-methods
    """Container for label names and their verbalizers.

    :param label_names: Ordered list of label names.
    :param verbalizers: List of verbalizer lists. ``verbalizers[i]`` belongs to ``label_names[i]``.
    """

    def __init__(self, label_names: list[str], verbalizers: list[list[str]]) -> None:
        self.label_names = label_names
        self.verbalizers = verbalizers


def load_verbalizers_json(path: Path, dataset_name: str) -> VerbalizerSpec:
    """Load verbalizers for a dataset from a multi-dataset JSON file.

    The expected JSON structure is::

        {
          "configs": [
            {
              "datasets": ["boolq"],
              "label_names": ["no", "yes"],
              "verbalizers": [[" no"], [" yes"]]
            }
          ]
        }

    :param path: Path to the verbalizers JSON file.
    :param dataset_name: Dataset name to look up.
    :raises ValueError: If the JSON structure is invalid or the dataset is missing.
    :return: Loaded verbalizer specification.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    if "configs" not in obj or not isinstance(obj["configs"], list):
        raise ValueError(f"Verbalizers JSON must contain a 'configs' list: {path}")

    for cfg in obj["configs"]:
        if dataset_name not in cfg.get("datasets", []):
            continue

        if "label_names" not in cfg or "verbalizers" not in cfg:
            raise ValueError(f"Config for dataset {dataset_name!r} must contain label_names "
                             f"and verbalizers.")

        if len(cfg["label_names"]) != len(cfg["verbalizers"]):

            raise ValueError(f"label_names/verbalizers length mismatch for "
                             f"dataset {dataset_name!r}.")

        return VerbalizerSpec(
            label_names=list(cfg["label_names"]),
            verbalizers=[list(v) for v in cfg["verbalizers"]],
        )

    raise ValueError(f"No verbalizer config found for dataset {dataset_name!r} in {path}")


def build_answer_to_label_map(spec: VerbalizerSpec) -> dict[str, int]:
    """Build a normalized answer/verbalizer string to label-id map.

    This lets gold labels such as ``"Yes"``, ``" yes"``, or ``"yes"`` map to the same class.

    :param spec: Verbalizer specification.
    :return: Mapping from normalized text to label id.
    """
    mapping: dict[str, int] = {}

    for idx, label_name in enumerate(spec.label_names):
        mapping[label_name.strip().lower()] = idx

    for idx, verbalizer_list in enumerate(spec.verbalizers):
        for verbalizer in verbalizer_list:
            mapping[verbalizer.strip().lower()] = idx

    return mapping


def select_one_verbalizer_per_label(spec: VerbalizerSpec, warn: bool = True) -> list[str]:
    """Select the first verbalizer for each label.

    The next-token objective uses the first token of this selected verbalizer.

    :param spec: Verbalizer specification.
    :param warn: Whether to print a warning if a label has multiple verbalizers.
    :raises ValueError: If a label has no verbalizers.
    :return: One selected verbalizer string per label.
    """
    selected: list[str] = []

    for label_name, verbalizer_list in zip(spec.label_names, spec.verbalizers):
        if len(verbalizer_list) == 0:
            raise ValueError(f"Label {label_name!r} has no verbalizers.")

        if warn and len(verbalizer_list) > 1:
            print(
                f"Warning: label {label_name!r} has multiple verbalizers {verbalizer_list}. "
                f"Using only the first one: {verbalizer_list[0]!r}."
            )

        selected.append(verbalizer_list[0])

    return selected


def get_first_token_verbalizer_ids(tokenizer: PreTrainedTokenizerBase, label_names: list[str],
                                   option_texts: list[str], verbose: bool = True) -> list[int]:
    """Map each selected verbalizer to its first tokenizer token id.

    Multi-token verbalizers are allowed, but only the first token is used. A collision check is
    performed because two labels must not map to the same first token.

    :param tokenizer: Tokenizer.
    :param label_names: Ordered label names.
    :param option_texts: One selected verbalizer string per label.
    :param verbose: Whether to print the tokenization mapping.
    :raises ValueError: If a verbalizer is empty after tokenization or if two labels collide.
    :return: List of first-token ids, one per label.
    """
    token_ids: list[int] = []
    seen: dict[int, tuple[str, str]] = {}

    if verbose:
        print("Verbalizer first-token mapping:")

    for label_name, option_text in zip(label_names, option_texts):
        ids = tokenizer.encode(option_text, add_special_tokens=False)

        if len(ids) == 0:
            raise ValueError(f"Verbalizer {option_text!r} for label {label_name!r} "
                             f"tokenized to zero tokens.")

        first_id = int(ids[0])
        first_token = tokenizer.convert_ids_to_tokens([first_id])[0]

        if verbose:
            if len(ids) > 1:
                tokens = tokenizer.convert_ids_to_tokens(ids)
                print(
                    f"  label={label_name!r}, verbalizer={option_text!r} tokenized "
                    f"to {ids} / {tokens}; using first token id={first_id}, token={first_token!r}"
                )
            else:
                print(
                    f"  label={label_name!r}, verbalizer={option_text!r} -> "
                    f"id={first_id}, token={first_token!r}"
                )

        if first_id in seen:
            prev_label, prev_text = seen[first_id]
            raise ValueError(
                "Verbalizer first-token collision detected. "
                f"Label {prev_label!r} with verbalizer {prev_text!r} and label {label_name!r} "
                f"with verbalizer {option_text!r} both map to token "
                f"id {first_id} ({first_token!r}). Choose different verbalizers."
            )

        seen[first_id] = (label_name, option_text)
        token_ids.append(first_id)

    return token_ids


def load_causal_lm_model_for_inference(
        args: argparse.Namespace
) -> PeftModel | PreTrainedModel:
    """Load a causal LM or a PEFT/LoRA adapter for inference.

    If ``args.model_name_or_path`` points to a directory containing ``adapter_config.json``,
    it is treated as a PEFT adapter.

    :param args: Parsed CLI arguments. Must contain ``model_name_or_path`` and optionally
        ``base_model_name_or_path``.
    :raises ImportError: If a PEFT adapter is detected but ``peft`` is not installed.
    :raises ValueError: If the base model for a PEFT adapter cannot be determined.
    :return: Loaded causal LM model.
    """
    model_path = Path(args.model_name_or_path)
    is_peft_adapter = model_path.exists() and (model_path / "adapter_config.json").exists()

    if is_peft_adapter:
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_model_name = args.base_model_name_or_path or peft_config.base_model_name_or_path

        if not base_model_name:
            raise ValueError(
                "Could not determine base model for PEFT adapter. "
                "Please pass --base_model_name_or_path."
            )

        print(f"Loading base model for PEFT adapter: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        print(f"Loading PEFT adapter: {args.model_name_or_path}")
        return PeftModel.from_pretrained(base_model, args.model_name_or_path)

    print(f"Loading causal LM: {args.model_name_or_path}")
    return AutoModelForCausalLM.from_pretrained(args.model_name_or_path)


def get_base_causal_lm(model: PreTrainedModel) -> PreTrainedModel:
    """Return the underlying causal LM.

    For PEFT models, this returns the base model with adapter modules injected.

    :param model: Causal LM or PEFT-wrapped causal LM.
    :return: Underlying causal LM.
    """
    get_base_model = getattr(model, "get_base_model", None)

    if callable(get_base_model):
        try:
            return get_base_model()
        except Exception:
            pass
    return model


def get_lm_backbone(base_model: PreTrainedModel) -> torch.nn.Module:
    """Get the decoder/backbone module before the LM head.

    :param base_model: Underlying causal LM.
    :raises RuntimeError: If no known backbone attribute is found.
    :return: Backbone module.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(base_model, attr):
            return getattr(base_model, attr)

    raise RuntimeError(
        "Could not locate causal LM backbone. Expected one of attributes: "
        "'model', 'transformer', or 'gpt_neox'."
    )


def get_lm_head(base_model: PreTrainedModel) -> torch.nn.Module:
    """Get the model's existing LM head / output embeddings.

    :param base_model: Underlying causal LM.
    :raises RuntimeError: If output embeddings cannot be located.
    :return: LM head module.
    """
    output_embeddings = base_model.get_output_embeddings()

    if output_embeddings is None:
        raise RuntimeError("Could not locate output embeddings / LM head.")

    if not hasattr(output_embeddings, "weight"):
        raise RuntimeError("Output embeddings module has no .weight parameter.")

    return output_embeddings


def compute_next_token_option_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    verbalizer_token_ids: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Compute next-token logits restricted to verbalizer-token ids.

    Runs the transformer backbone, extracts the hidden state at the final non-padding prompt token,
    and applies only selected rows of the existing LM head.

    :param model: Causal LM or PEFT-wrapped causal LM.
    :param input_ids: Input token ids of shape ``[batch_size, seq_len]``.
    :param attention_mask: Attention mask of shape ``[batch_size, seq_len]`` or ``None``.
    :param verbalizer_token_ids: Token ids corresponding to class verbalizers.
    :raises TypeError: If the LM head is not a linear layer.
    :return: Classification logits of shape ``[batch_size, num_labels]``.
    """
    base_model = get_base_causal_lm(model)
    backbone = get_lm_backbone(base_model)
    lm_head = get_lm_head(base_model)

    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"Expected nn.Linear lm_head, got {type(lm_head)!r}")

    backbone_outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    if hasattr(backbone_outputs, "last_hidden_state"):
        hidden_states = backbone_outputs.last_hidden_state
    else:
        hidden_states = backbone_outputs[0]

    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    # Select the final non-padding token position for each sequence.
    if attention_mask is None:
        last_indices = torch.full((batch_size,), seq_len - 1,
                                  dtype=torch.long, device=device)
    else:
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        last_indices = (attention_mask.to(device).long() * positions).max(dim=1).values

    # Gather hidden states at the selected final token positions.
    last_hidden = hidden_states[torch.arange(batch_size, device=device), last_indices]

    # Normalize verbalizer token ids to a tensor on the correct device.
    if not isinstance(verbalizer_token_ids, torch.Tensor):
        option_token_ids = torch.tensor(verbalizer_token_ids, dtype=torch.long, device=device)
    else:
        option_token_ids = verbalizer_token_ids.to(device)

    # Restrict the LM head to the verbalizer-token rows only.
    selected_weight = lm_head.weight.index_select(0, option_token_ids)

    lm_head_bias = getattr(lm_head, "bias", None)
    if lm_head_bias is not None:
        selected_bias = lm_head_bias.index_select(0, option_token_ids)
    else:
        selected_bias = None

    # Compute logits (linear).
    logits = last_hidden @ selected_weight.T

    if selected_bias is not None:
        logits = logits + selected_bias

    return logits


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    :param seed: The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file where each line is a JSON object.

    :param path: Path to the jsonl file.
    :return: A list of dicts with the contents of the jsonl file.
    """
    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def shuffle_in_place(items: list[Any], seed: int) -> None:
    """
    Shuffle a list in place deterministically using the provided seed.

    :param items: The list to shuffle.
    :param seed: The seed used for shuffling.
    """
    rng = random.Random(seed)
    rng.shuffle(items)
