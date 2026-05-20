"""A script to run fine-tuning experiments for classification and generation tasks."""
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Type
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftMixedModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,
                          PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments, BatchEncoding,
                          PretrainedConfig)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm

from metrics import get_bleu_score, get_boxed_accuracy, get_rouge_score
from trainers.fzootrainer import FZOOTrainer, FZOOTrainingArguments
from trainers.hizootrainer import HiZOOTrainer, HiZOOTrainingArguments
from trainers.lozomtrainer import LOZOMTrainer, LOZOMTrainingArguments
from trainers.lozotrainer import LOZOTrainer, LOZOTrainingArguments
from trainers.mezotrainer import MeZOTrainer, MeZOTrainingArguments
from trainers.mezoforeachtrainer import MeZOForEachTrainer
from trainers.zoadamtrainer import ZOAdamTrainer, ZOAdamTrainingArguments
from trainers.zomuontrainer import ZOMuonTrainer, ZOMuonTrainingArguments
from trainers.zosgdmmt import ZOSGDMMTTrainer, ZOSGDMMTTrainingArguments
from utils import (build_answer_to_label_map, compute_next_token_option_logits,
                   get_first_token_verbalizer_ids, load_verbalizers_json, read_jsonl,
                   select_one_verbalizer_per_label, set_seed, shuffle_in_place)


# Argument dataclass

@dataclass
class ExperimentArguments:  # pylint: disable=too-many-instance-attributes
    """
    Experiment arguments (non-HF training knobs).

    :param dataset_dir: Path to directory containing train.jsonl, val.jsonl, and test.jsonl.
    :param task_type: Task type: classification or generation.
    :param model_name_or_path: Base model name or path (e.g., Qwen/Qwen2-1.5B).
    :param trainer_type: Trainer type (e.g., hf; placeholder for future trainers like mezo).
    :param verbalizers_json: Path to verbalizers JSON file (required for classification tasks).
    :param max_length: Maximum input length for classification prompts.
    :param max_input_length: Maximum prompt length for generation tasks.
    :param max_target_length: Maximum target/answer length for generation tasks.
    :param calculate_generation_metrics_per_evals: Compute generation metrics every N eval steps.
    :param use_lora: Whether to enable LoRA adapters.
    :param lora_r: LoRA rank (r).
    :param lora_alpha: LoRA alpha scaling factor.
    :param lora_dropout: Dropout probability applied in LoRA layers.
    :param lora_target_modules: LoRA target modules (e.g., "q_proj,v_proj", auto-detect if empty).
    :param custom_lora_weights_init: Optional LoRA init method (e.g., "pissa", empty by default).
    """
    dataset_dir: str = field(
        metadata={"help": "Path to directory with train.jsonl, val.jsonl, test.jsonl."}
    )
    task_type: str = field(
        metadata={"help": "Task type: 'classification' or 'generation'."}
    )
    model_name_or_path: str = field(
        metadata={"help": "Base model name/path, e.g. Qwen/Qwen2-1.5B."}
    )
    trainer_type: str = field(
        default="hf", metadata={"help": "Trainer type: hf, mezo, lozo, etc."}
    )
    load_fp16: bool = field(
        default=False, metadata={"help": "Whether to load the base model in fp16."}
    )
    load_bf16: bool = field(
        default=False, metadata={"help": "Whether to load the base model in bf16."}
    )

    # Classification-specific
    verbalizers_json: str = field(
        default="", metadata={"help": "Path to verbalizers json."}
    )
    max_length: int = field(
        default=512, metadata={"help": "Max prompt length for next-token classification."}
    )

    # Generation-specific
    max_input_length: int = field(
        default=1024, metadata={"help": "Max prompt length for generation."}
    )
    max_target_length: int = field(
        default=128, metadata={"help": "Max target length for generation."}
    )
    calculate_generation_metrics_per_evals: int = field(
        default=-1,
        metadata={"help": "Calculate generation metrics every N evals."},
    )

    # LoRA
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA adapters."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="",
        metadata={"help": "Comma-separated LoRA target modules. Empty = PEFT auto-detect."},
    )
    custom_lora_weights_init: str = field(
        default="",
        metadata={"help": "Optional custom LoRA init, e.g. 'pissa'. Empty = default."},
    )


# LoRA helpers

def _parse_target_modules(spec: str) -> list[str] | None:
    """
    Parse a comma-separated list of module names; return None if empty.

    :param spec: Comma-separated string (e.g., "q_proj,v_proj").
    :return: List of names or None if empty.
    """
    if not spec:
        return None

    items = [x.strip() for x in spec.split(",")]
    return [x for x in items if x]


def patch_lora(model: Any) -> None:
    """
    Patch a PEFT-wrapped model to ensure inputs require grad when gradient checkpointing is used.

    :param model: The model (possibly PEFT-wrapped).
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(module, input, output):  # pylint: disable=redefined-builtin,unused-argument
        try:
            output.requires_grad_(True)
        except Exception:
            pass

    try:
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    except Exception:
        pass


def apply_lora(model: PreTrainedModel, exp_args: ExperimentArguments,
               task_type: TaskType) -> PeftModel | PeftMixedModel:
    """
    Wrap a model with LoRA adapters according to exp_args.

    :param model: The HF model to wrap.
    :param exp_args: ExperimentArguments with LoRA params.
    :param task_type: The PEFT TaskType (SEQ_CLS or CAUSAL_LM).
    :return: The PEFT-wrapped model.
    """
    target_modules = _parse_target_modules(exp_args.lora_target_modules)

    peft_config = LoraConfig(
        task_type=task_type,
        r=exp_args.lora_r,
        lora_alpha=exp_args.lora_alpha,
        lora_dropout=exp_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=(exp_args.custom_lora_weights_init   # type: ignore[arg-type]
                           if exp_args.custom_lora_weights_init
                           else True),
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    patch_lora(peft_model)

    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model


# Next-token option classification patch

def patch_causal_lm_for_next_token_option_classification(
    model: PreTrainedModel | PeftModel | PeftMixedModel,
    verbalizer_token_ids: list[int],
) -> PreTrainedModel | PeftModel | PeftMixedModel:
    """Patch a causal LM for next-token option classification.

    The patched model receives prompt-only inputs and returns classification logits over the
    selected verbalizer-token ids.

    :param model: Causal LM or PEFT-wrapped causal LM.
    :param verbalizer_token_ids: One first-token verbalizer id per class.
    :raises ValueError: If ``verbalizer_token_ids`` is empty.
    :return: Patched model.
    """
    if hasattr(model, "original_forward_for_next_token_option_classification"):
        return model

    if len(verbalizer_token_ids) == 0:
        raise ValueError("verbalizer_token_ids must be non-empty.")

    # model.original_forward_for_next_token_option_classification = model.forward
    model.verbalizer_token_ids_for_classification = torch.tensor(verbalizer_token_ids,
                                                                 dtype=torch.long)

    def next_token_option_classification_forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """Forward pass for next-token option classification.

        :param input_ids: Prompt input ids of shape ``[batch_size, seq_len]``.
        :param attention_mask: Attention mask of shape ``[batch_size, seq_len]``.
        :param labels: Integer class labels of shape ``[batch_size]``.
        :param return_dict: Whether to return a Hugging Face model output.
        :param kwargs: Additional keyword arguments. ``use_cache`` is ignored.
        :return: ``SequenceClassifierOutput`` or tuple.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        kwargs.pop("use_cache", None)

        class_logits = compute_next_token_option_logits(
            model=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            verbalizer_token_ids=self.verbalizer_token_ids_for_classification,
        )

        loss = None
        if labels is not None:
            labels = labels.to(class_logits.device).long()
            loss = CrossEntropyLoss()(class_logits, labels)

        if return_dict is False:
            if loss is None:
                return (class_logits,)
            return loss, class_logits

        return SequenceClassifierOutput(
            loss=loss,
            logits=class_logits,
        )

    # model.forward = MethodType(next_token_option_classification_forward, model)
    setattr(model, "forward", MethodType(next_token_option_classification_forward, model))
    return model


# Classification dataset/collator

class NextTokenClassificationDataset:
    """
    Prompt-only dataset for next-token verbalizer classification.

    Each example is:
        input_ids = tokenize(prompt)
        labels = label_id

    The model predicts the next token after the final prompt token.
    """

    def __init__(self, examples: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase,
                 ans2label: dict[str, int], max_length: int = 512) -> None:
        self.data: list[dict[str, Any]] = []

        prompts = [ex["prompt"] for ex in examples]

        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_length,
        )

        for ex, input_ids, attention_mask in zip(
                examples, encoded["input_ids"], encoded["attention_mask"]):
            if len(input_ids) == 0:
                raise ValueError(f"Prompt tokenized to zero tokens: {ex['prompt']!r}")

            answer = ex["answer"]
            norm_answer = answer.strip().lower()

            if norm_answer not in ans2label:
                raise ValueError(f"Answer {answer!r} not found in label/verbalizer map.")

            label = ans2label[norm_answer]

            self.data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label,
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class NextTokenClassificationCollator:  # pylint: disable=too-few-public-methods
    """
    Padding collator for next-token classification.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> BatchEncoding:
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [int(feature["labels"]) for feature in features],
            dtype=torch.long,
        )

        return batch


# Generation dataset/collator

class CausalLMDataset:
    """
    A dataset for casual LM fine-tuning with prompt => target supervision.

    For each example:
      - Input is [prompt tokens] + [target tokens]
      - Labels are [-100 for prompt tokens] + [target tokens]
    """
    def __init__(self, examples: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase,
                 max_input_length: int = 1024, max_target_length: int = 128,
                 add_eos_to_target: bool = True) -> None:
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.prompt_lengths = []

        eos_id = tokenizer.eos_token_id

        for ex in examples:
            prompt = ex["prompt"]
            target = ex["answer"] + (
                tokenizer.eos_token if (add_eos_to_target and eos_id is not None) else ""
            )

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_input_length,
            )["input_ids"]

            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max_target_length,
            )["input_ids"]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            attn = [1] * len(input_ids)

            self.input_ids.append(input_ids)
            self.attention_mask.append(attn)
            self.labels.append(labels)
            self.prompt_lengths.append(len(prompt_ids))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "prompt_length": self.prompt_lengths[idx],
        }


class CausalLMDataCollator:  # pylint: disable=too-few-public-methods
    """A padding collator with precomputed labels using -100 on non-loss tokens."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> BatchEncoding:
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].size(1)
        padding_side = self.tokenizer.padding_side
        padded_labels = []

        for feature in features:
            labels = feature["labels"]
            diff = max_len - len(labels)

            if diff > 0:
                if padding_side == "right":
                    labels = labels + [self.label_pad_token_id] * diff
                else:
                    labels = [self.label_pad_token_id] * diff + labels
            else:
                labels = labels[:max_len]

            padded_labels.append(labels)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# Metrics

def classification_compute_metrics_fn() -> Callable[[EvalPrediction], dict[str, float]]:
    """
    Build a compute_metrics function for classification tasks.

    :return: A function suitable for HF Trainer compute_metrics.
    """
    def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
        logits = pred.predictions

        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        labels = pred.label_ids
        acc = (preds == labels).astype(np.float32).mean().item() if len(labels) > 0 else 0.0

        return {"accuracy": float(acc)}

    return compute_metrics


class GenerationEvalCallback(TrainerCallback):
    """Runs full-corpus batched generation on the eval dataset and logs ROUGE-L and BLEU."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Any,
        data_collator: Callable,
        trainer: Trainer,
        batch_size: int = 4,
        max_new_tokens: int = 64,
        skip_special_tokens: bool = True,
        calculate_generation_metrics_per_evals: int = -1,
        boxed_generation: bool = False
    ) -> None:
        """
        Initialize the callback.

        :param tokenizer: The tokenizer for decoding generations.
        :param eval_dataset: The eval dataset (dicts with 'input_ids', 'attention_mask', 'labels').
        :param data_collator: A collator to pad batches from the eval dataset.
        :param trainer: The Trainer instance to log metrics to.
        :param batch_size: Batch size for generation.
        :param max_new_tokens: Max new tokens to generate.
        :param skip_special_tokens: Whether to skip special tokens when decoding.
        :param calculate_generation_metrics_per_evals: Calculate generation metrics every N evals.
        :param boxed_generation: Whether to calculate accuracy based on boxed answer.
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens
        self.trainer = trainer
        self.calculate_generation_metrics_per_evals = calculate_generation_metrics_per_evals
        self.boxed_generation = boxed_generation

    def _run_generation_loop(self, model: PreTrainedModel,
                             dataloader: DataLoader) -> tuple[list[str], list[str]]:
        """
        Run generation on the entire eval dataset and collect predictions and references.

        :param model: The model to use for generation (should be a causal LM).
        :param dataloader: An eval DataLoader, with 'input_ids', 'attention_mask', and 'labels'.
        :raises TypeError: If the model does not support .generate().
        :return: A tuple of (predictions, references), where each is a list of strings.
        """
        device = next(model.parameters()).device

        preds = []
        refs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating..."):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Build prompt-only batch
                prompt_lens = []
                for i in range(batch["labels"].size(0)):
                    idx = (batch["labels"][i] != -100).nonzero(as_tuple=True)[0]
                    prompt_lens.append(idx[0].item() if len(idx) else 0)

                trimmed_input_ids = [input_ids[i, :prompt_lens[i]] for i in
                                     range(input_ids.size(0))]
                trimmed_attention_mask = [attention_mask[i, :prompt_lens[i]] for i in
                                          range(input_ids.size(0))]

                trimmed_batch = self.tokenizer.pad(
                    {"input_ids": trimmed_input_ids, "attention_mask": trimmed_attention_mask},
                    return_tensors="pt",
                )

                # Generate
                generate = getattr(model, "generate", None)
                if not callable(generate):
                    raise TypeError(f"Model does not support generate(): {type(model)!r}")

                gen_out = generate(
                    input_ids=trimmed_batch["input_ids"].to(device),
                    attention_mask=trimmed_batch["attention_mask"].to(device),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )

                # Decode ONLY newly generated tokens
                gen_only = gen_out[:, trimmed_batch["input_ids"].size(1):]
                pred_texts = self.tokenizer.batch_decode(
                    gen_only,
                    skip_special_tokens=self.skip_special_tokens
                )
                preds.extend([pred.strip() for pred in pred_texts])

                # Decode refs from labels (target tokens only)
                label_ids = batch["labels"].clone()

                if self.tokenizer.pad_token_id is None:
                    raise ValueError("tokenizer.pad_token_id must be set to decode refs safely.")

                label_ids[label_ids == -100] = self.tokenizer.pad_token_id
                ref_texts = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                refs.extend([ref.strip() for ref in ref_texts])

        return preds, refs

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    model: PreTrainedModel | None = None, **kwargs) -> None:
        """
        Calculate and report generation metrics at the end of an evaluation phase.

        :param args: TrainingArguments for the current training run.
        :param state: TrainerState with information about the current state (e.g., global_step).
        :param control: TrainerControl to control the training flow (not used here).
        :param model: The model being evaluated. Should be a causal LM that supports .generate().
        :param kwargs: Additional keyword arguments (not used here).
        :raises RuntimeError: If the trainer instance is not set in the callback.
        :return: None
        """
        if not getattr(state, "is_world_process_zero", True):
            return

        if model is None or self.eval_dataset is None:
            return

        if self.calculate_generation_metrics_per_evals == -1 or not args.eval_steps:
            return

        if state.global_step % (args.eval_steps * self.calculate_generation_metrics_per_evals) != 0:
            return

        was_training = model.training

        model.eval()
        self.tokenizer.padding_side = "left"

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=False,
        )

        preds, refs = self._run_generation_loop(model, dataloader)

        rouge_l = float(get_rouge_score(preds, refs))
        bleu = float(get_bleu_score(preds, refs))

        if self.trainer is not None:
            if self.boxed_generation:
                accuracy, _ = get_boxed_accuracy(preds, refs)
                accuracy = float(accuracy)

                log = {
                    "eval_rougeL": rouge_l,
                    "eval_bleu": bleu,
                    "eval_accuracy": accuracy
                }
                self.trainer.log(log)
            else:
                log = {
                    "eval_rougeL": rouge_l,
                    "eval_bleu": bleu,
                }
                self.trainer.log(log)

            print(log)
        else:
            raise RuntimeError("Trainer is not set in callback.")

        print("Sample refs:", refs[:5])
        print("Sample preds:", preds[:5])

        if was_training:
            model.train()
            self.tokenizer.padding_side = "right"


# Trainer factory

TRAINER_REGISTRY: dict[str, type[Trainer]] = {
    "mezo": MeZOTrainer,  # MeZO, ZO-SGD-CON and ZO-SGD-SIGN
    "lozo": LOZOTrainer,
    "lozom": LOZOMTrainer,
    "zosgdmmt": ZOSGDMMTTrainer,
    "zoadam": ZOAdamTrainer,  # ZO-Adam and R-AdaZO
    "fzoo": FZOOTrainer,
    "hizoo": HiZOOTrainer,
    "zomuon": ZOMuonTrainer,
    "mezoforeach": MeZOForEachTrainer
}

HF_ALIASES = {"hf", "adamw", "hf_adamw", "default"}


def get_trainer(trainer_type: str, model: PreTrainedModel |  PeftModel | PeftMixedModel,  # pylint: disable=too-many-arguments
                args: TrainingArguments, train_dataset: Any, eval_dataset: Any,
                tokenizer: PreTrainedTokenizerBase, data_collator: Callable | None = None,
                compute_metrics: Callable | None = None) -> Trainer:
    """
    Construct a trainer instance.

    :param trainer_type: The trainer type string ("hf", "mezo", etc).
    :param model: The model to train.
    :param args: TrainingArguments (or subclass).
    :param train_dataset: The training dataset.
    :param eval_dataset: The evaluation dataset.
    :param tokenizer: The tokenizer.
    :param data_collator: An optional data collator.
    :param compute_metrics: Optional metrics callback.
    :return: A Trainer-like object.
    """
    key = (trainer_type or "hf").lower()

    if key in HF_ALIASES:
        trainer_cls = Trainer
    else:
        trainer_cls = TRAINER_REGISTRY.get(key)

        if trainer_cls is None:
            print(f"Warning: trainer_type={trainer_type!r} is not implemented; "
                  f"falling back to HF Trainer.")
            trainer_cls = Trainer

    return trainer_cls(model=model, args=args, train_dataset=train_dataset,
                       eval_dataset=eval_dataset, data_collator=data_collator,
                       tokenizer=tokenizer, compute_metrics=compute_metrics)


# Main train functions

def train_classification(exp_args: ExperimentArguments, train_args: TrainingArguments,
                         dataset_dir: Path, output_dir: Path) -> None:
    """
    Train a causal LM for next-token verbalizer classification.

    :param exp_args: ExperimentArguments with dataset/model config.
    :param train_args: TrainingArguments to control optimization and logging.
    :param dataset_dir: Directory with train/val/test jsonl files.
    :param output_dir: Output directory to store checkpoints and logs.
    """
    train_args.output_dir = str(output_dir)

    # Important because we patch the model forward and want to control accepted columns.
    train_args.remove_unused_columns = False

    train_data = read_jsonl(dataset_dir / "train.jsonl")
    val_data = read_jsonl(dataset_dir / "val.jsonl")

    shuffle_in_place(train_data, train_args.seed)
    shuffle_in_place(val_data, train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(exp_args.model_name_or_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dataset_name = dataset_dir.name.lower()
    verbal_spec = load_verbalizers_json(Path(exp_args.verbalizers_json), dataset_name)

    label_names = verbal_spec.label_names
    option_texts = select_one_verbalizer_per_label(verbal_spec)
    verbalizer_token_ids = get_first_token_verbalizer_ids(
        tokenizer=tokenizer,
        label_names=label_names,
        option_texts=option_texts,
    )
    ans2label = build_answer_to_label_map(verbal_spec)

    print("Classification labels:", label_names)
    print("Classification selected verbalizer strings:", option_texts)
    print("Classification verbalizer first-token ids:", verbalizer_token_ids)

    config = AutoConfig.from_pretrained(exp_args.model_name_or_path)
    config.pad_token_id = tokenizer.pad_token_id
    config.id2label = dict(enumerate(label_names))
    config.label2id = {name: i for i, name in enumerate(label_names)}
    config.num_labels = len(label_names)

    dtype = torch.float32
    if exp_args.load_bf16:
        dtype = torch.bfloat16
    elif exp_args.load_fp16:
        dtype = torch.float16

    model: PreTrainedModel | PeftModel | PeftMixedModel = AutoModelForCausalLM.from_pretrained(
        exp_args.model_name_or_path, config=config, dtype=dtype
    )
    print(f"Model loaded in dtype {model.dtype}")

    if not isinstance(model.config, PretrainedConfig):
        raise ValueError("Expected model.config to be a PretrainedConfig instance.")

    model.config.pad_token_id = tokenizer.pad_token_id

    # Recommended for training decoder-only models.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if exp_args.use_lora:
        if not isinstance(model, PreTrainedModel):
            raise ValueError("Expected model to be a PreTrainedModel when applying LoRA.")
        model = apply_lora(model, exp_args, task_type=TaskType.CAUSAL_LM)

    model = patch_causal_lm_for_next_token_option_classification(
        model=model,
        verbalizer_token_ids=verbalizer_token_ids,
    )

    train_ds = NextTokenClassificationDataset(train_data, tokenizer=tokenizer, ans2label=ans2label,
                                              max_length=exp_args.max_length)

    val_ds = NextTokenClassificationDataset(val_data, tokenizer=tokenizer, ans2label=ans2label,
                                            max_length=exp_args.max_length)

    collator = NextTokenClassificationCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_args.load_best_model_at_end = True
    train_args.metric_for_best_model = "accuracy"
    train_args.greater_is_better = True

    if getattr(train_args, "optim", None) is None:
        train_args.optim = "adamw_torch"

    trainer = get_trainer(trainer_type=exp_args.trainer_type, model=model, args=train_args,
                          train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
                          data_collator=collator,
                          compute_metrics=classification_compute_metrics_fn())

    trainer.train()
    # trainer.save_model(str(output_dir / "last"))


def train_generation(exp_args: ExperimentArguments, train_args: TrainingArguments,
                     dataset_dir: Path, output_dir: Path) -> None:
    """
    Train a causal LM for free generation tasks (e.g., summarization).

    :param exp_args: ExperimentArguments with dataset/model config.
    :param train_args: TrainingArguments to control optimization and logging.
    :param dataset_dir: Directory with train/val/test jsonl files.
    :param output_dir: Output directory to store checkpoints and logs.
    """
    train_args.output_dir = str(output_dir)

    train_data = read_jsonl(dataset_dir / "train.jsonl")
    val_data = read_jsonl(dataset_dir / "val.jsonl")

    shuffle_in_place(train_data, train_args.seed)
    shuffle_in_place(val_data, train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(exp_args.model_name_or_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dtype = torch.float32
    if exp_args.load_bf16:
        dtype = torch.bfloat16
    elif exp_args.load_fp16:
        dtype = torch.float16

    model: PreTrainedModel | PeftModel | PeftMixedModel = AutoModelForCausalLM.from_pretrained(
        exp_args.model_name_or_path, dtype=dtype
    )
    print(f"Model loaded in dtype {model.dtype}")

    if not isinstance(model.config, PretrainedConfig):
        raise ValueError("Expected model.config to be a PretrainedConfig instance.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_tokens = 256 + 128

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if exp_args.use_lora:
        if not isinstance(model, PreTrainedModel):
            raise ValueError("Expected model to be a PreTrainedModel when applying LoRA.")
        model = apply_lora(model, exp_args, task_type=TaskType.CAUSAL_LM)

    train_ds = CausalLMDataset(train_data, tokenizer=tokenizer,
                               max_input_length=exp_args.max_input_length,
                               max_target_length=exp_args.max_target_length,
                               add_eos_to_target=True)

    val_ds = CausalLMDataset(val_data, tokenizer=tokenizer,
                             max_input_length=exp_args.max_input_length,
                             max_target_length=exp_args.max_target_length,
                             add_eos_to_target=True)

    collator = CausalLMDataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    if getattr(train_args, "optim", None) is None:
        train_args.optim = "adamw_torch"

    assert model.generation_config is not None, "Model must have generation_config for generation."

    model.generation_config.max_new_tokens = 128
    model.generation_config.max_length = 256 + 128

    trainer = get_trainer(trainer_type=exp_args.trainer_type, model=model, args=train_args,
                          train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
                          data_collator=collator, compute_metrics=None)

    gen_cb = GenerationEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=val_ds,
        data_collator=collator,
        trainer=trainer,
        batch_size=train_args.per_device_eval_batch_size,
        max_new_tokens=model.generation_config.max_new_tokens,
        skip_special_tokens=True,
        calculate_generation_metrics_per_evals=exp_args.calculate_generation_metrics_per_evals,
    )

    trainer.add_callback(gen_cb)

    trainer.train()
    # trainer.save_model(str(output_dir / "last"))


# TrainingArguments registry

TRAINING_ARGS_REGISTRY = {
    "mezo": MeZOTrainingArguments,
    "lozo": LOZOTrainingArguments,
    "lozom": LOZOMTrainingArguments,
    "zosgdmmt": ZOSGDMMTTrainingArguments,
    "fastermezo": MeZOTrainingArguments,
    "zoadam": ZOAdamTrainingArguments,
    "fzoo": FZOOTrainingArguments,
    "hizoo": HiZOOTrainingArguments,
    "zomuon": ZOMuonTrainingArguments,
    "mezoforeach": MeZOTrainingArguments
}


def get_training_arguments(experiment_args: ExperimentArguments) -> Type[TrainingArguments]:
    """
    Return a TrainingArguments instance (or subclass) based on the experiment_args.trainer_type.

    :param experiment_args: The ExperimentArguments containing the trainer_type and other config.
    :return: An instance of TrainingArguments or a subclass specific to the trainer type.
    """
    key = (experiment_args.trainer_type or "hf").lower()

    if key in HF_ALIASES:
        return TrainingArguments

    return TRAINING_ARGS_REGISTRY.get(key, TrainingArguments)


def main() -> None:
    """Run the experiment according to the specified configuration."""
    parser = HfArgumentParser(ExperimentArguments)  # type: ignore[arg-type]
    experiment_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args_cls = get_training_arguments(experiment_args)

    parser = HfArgumentParser(training_args_cls)  # type: ignore[arg-type]
    training_args = parser.parse_args_into_dataclasses(remaining)[0]

    set_seed(training_args.seed)

    dataset_dir = Path(experiment_args.dataset_dir)
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_args.task_type not in {"classification", "generation", "boxed_generation"}:
        raise ValueError("task_type must be one of {'classification', 'generation', "
                         "'boxed_generation'}")

    if experiment_args.task_type == "classification":
        if not experiment_args.verbalizers_json:
            raise ValueError("For classification tasks, --verbalizers_json must be provided.")

        train_classification(experiment_args, training_args, dataset_dir, output_dir)
    else:
        train_generation(experiment_args, training_args, dataset_dir, output_dir)


if __name__ == "__main__":
    main()
