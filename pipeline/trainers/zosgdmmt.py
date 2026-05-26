# pylint: disable=duplicate-code
"""Module for ZO-SGD-MMT trainer which combines MeZO gradient estimation with classical momentum."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class ZOSGDMMTTrainingArguments(TrainingArguments):
    """
    TrainingArguments is a class containing the key arguments for Trainer.

    :param trainer_mode: Mode of training (`regular`, `zo`).
    :param zo_eps: MeZO hyperparameter epsilon.
    :param zo_num_directions: Number of directions for MeZO.
    """

    trainer_mode: str = field(
        default="zo", metadata={"help": "Mode of training (`regular`, `zo`)."}
    )
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=1, metadata={"help": "Number of directions for MeZO."})
    zo_subtype: str = field(
        default="mezo", metadata={"help": "Subtype (`mezo`, `zo-sgd-con`, `zo-sgd-sign`"}
    )
    zo_momentum_beta: float = field(
        default=0.9, metadata={"help": "Momentum coefficient for ZO-SGD-MMT."}
    )

    accumulate_in_full_precision: bool = field(
        default=False,
        metadata={"help": "Whether to accumulate gradients in full precision for better stability."}
    )
    use_master_weights: bool = field(
        default=False, metadata={"help": "Whether to keep master FP32 weights."}
    )


class ZOSGDMMTTrainer(BaseZOTrainer):  # pylint: disable=too-many-instance-attributes
    """ZO-SGD-MMT Trainer: Combines MeZO gradient estimation with classical momentum updates."""

    def __init__(self, **kwargs):
        """
        Initialize the ZOSGDMMTTrainer.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            ZO-SGD-MMT-specific configuration is expected through ``ZOSGDMMTTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args
        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)
        self.zo_momentum_beta = getattr(args, "zo_momentum_beta", 0.9)

        self.momentum_step = 0

        self.projected_grad = None
        self.zo_random_seed = None

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
        self.zo_random_seed: int | None = None

        self.trainable_params = [
            (n, p) for n, p in self.model.named_parameters() if p.requires_grad
        ]

        if not self.trainable_params:
            raise ValueError("No trainable parameters found.")

        if self.trainer_mode == "zo":
            for p in self.model.parameters():
                p.requires_grad = False

        self.use_master_weights = getattr(args, "use_master_weights", False)
        self.accumulate_in_full_precision = getattr(args, "accumulate_in_full_precision", False)

        # Choose dtype
        self.model_dtype = next(self.model.parameters()).dtype
        if self.model_dtype in (torch.float32, torch.float64):
            if self.use_master_weights:
                print("Model is in full precision; disabling master weights.")
            if self.accumulate_in_full_precision:
                print("Model is in full precision; disabling full precision accumulation.")

            self.use_master_weights = False
            self.accumulate_in_full_precision = False

        self.accum_dtype = (torch.float32 if self.accumulate_in_full_precision
                                   else self.model_dtype)
        self.master_dtype = torch.float32

        if self.use_master_weights:
            self.master_params = {
                name: param.data.detach().clone().float()
                for name, param in self.trainable_params
            }
        else:
            self.master_params = None

        self.momentum_buffers = {
            name: torch.zeros_like(param.data, dtype=self.accum_dtype)
            for name, param in self.trainable_params
        }

        print(f"ZOSGDMMTTrainer initialized with mode: {self.trainer_mode}, "
              f"zo_eps: {self.zo_eps}, zo_num_directions: {self.zo_num_directions}, "
              f"zo_momentum_beta: {self.zo_momentum_beta}, "
              f"accumulate_in_full_precision: {self.accumulate_in_full_precision}, "
              f"use_master_weights: {self.use_master_weights}\n"
              f"Trainable params amount: {len(self.trainable_params)}")

    ########################
    # MeZO-specific Methods
    ########################

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        """
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.trainable_params:
            z = torch.normal(
                mean=0, std=1, size=param.data.size(),
                device=param.device, dtype=param.dtype
            )
            param.data.add_(z, alpha=scaling_factor * self.zo_eps)

    def zo_step(self, model: PreTrainedModel, inputs: dict[str, Any]) -> torch.Tensor:
        """
        Estimate the gradient by performing multiple directional forward passes.

        :param model: The model.
        :param inputs: Batch of inputs.
        :return: The computed loss.
        """
        directions = []
        if self.batch_zo_seeds is None:
            self.batch_zo_seeds = [
                np.random.randint(1_000_000_000) for _ in range(self.zo_num_directions)
            ]

        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            self.zo_perturb_parameters(+1.0)
            loss_plus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(-2.0)
            loss_minus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(+1.0)

            grad_est = ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()
            directions.append((seed, grad_est))

        self.zo_direction_accumulator.extend(directions)
        self.zo_accumulation_count += 1
        self._maybe_empty_cache()

        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if not self.zo_direction_accumulator:
            return

        one_minus_beta = 1.0 - self.zo_momentum_beta
        norm = one_minus_beta / self.zo_num_directions

        for name in self.momentum_buffers:
            self.momentum_buffers[name] *= self.zo_momentum_beta

        seed_group: dict[int, float] = {}
        for seed, grad_estimate in self.zo_direction_accumulator:
            normalized = grad_estimate / self.args.gradient_accumulation_steps
            if seed in seed_group:
                seed_group[seed] += normalized
            else:
                seed_group[seed] = normalized

        for seed, grad_sum in seed_group.items():
            torch.manual_seed(seed)

            for name, param in self.trainable_params:
                z = torch.normal(
                    mean=0, std=1, size=param.data.size(),
                    device=param.device, dtype=param.dtype
                ).to(self.accum_dtype)

                # g = grad_sum * z

                # if self.args.weight_decay > 0.0 and (
                #        "bias" not in name and "layer_norm" not in name and "layernorm" not in name
                # ):
                #     wd_target = (
                #         self.master_params[name]
                #         if self.use_master_weights
                #         else param.data.to(torch.float32)
                #     )
                #     g += self.args.weight_decay * wd_target

                self.momentum_buffers[name].add_(z, alpha=norm * grad_sum)

        self.momentum_step += 1
        bias_corr = 1.0 - self.zo_momentum_beta ** self.momentum_step

        for name, param in self.trainable_params:
            v = self.momentum_buffers[name] / bias_corr

            if self.use_master_weights:
                self.master_params[name].add_(v, alpha=-learning_rate)
                param.data.copy_(self.master_params[name].to(param.dtype))
            else:
                param.data.add_(v.to(param.dtype), alpha=-learning_rate)

        self.zo_direction_accumulator.clear()
        self.batch_zo_seeds = None
        self.zo_accumulation_count = 0
