# pylint: disable=duplicate-code
"""
Module for a MeZOTrainer implementation.

MeZOTrainer implementation is based on the paper
'Fine-Tuning Language Models with Just Forward Passes'
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class MeZOTrainingArguments(TrainingArguments):
    """
    Class containing the key arguments for MeZOTrainer.

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


class MeZOTrainer(BaseZOTrainer):
    """MeZOTrainer implements the MeZO algorithm for training models without gradients."""

    def __init__(self, **kwargs):
        """
        Initialize the MeZOTrainer.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            MeZO-specific configuration is expected through ``MeZOTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)
        self.zo_subtype = getattr(args, "zo_subtype", "mezo")

        if self.zo_subtype not in ("mezo", "zo-sgd-con", "zo-sgd-sign"):
            raise ValueError(f"Invalid zo_subtype: {self.zo_subtype}. "
                             f"Must be one of 'mezo', 'zo-sgd-con', 'zo-sgd-sign'.")

        self.projected_grad = None
        self.zo_random_seed: int | None = None

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0

        self.batch_zo_seeds = None

        self.trainable_params = [
            (p[0], p[1]) for p in self.model.named_parameters() if p[1].requires_grad
        ]

        if not self.trainable_params:
            raise ValueError("No trainable parameters found.")

        if self.trainer_mode == "zo":
            for param in self.model.parameters():
                param.requires_grad = False  # Ensure no accidental gradient computation

        print(f"MeZOTrainer initialized with mode: {self.trainer_mode}, "
              f"zo_eps: {self.zo_eps}, zo_num_directions: {self.zo_num_directions}, "
              f"zo_subtype: {self.zo_subtype}\n"
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
                np.random.randint(1000000000) for _ in range(self.zo_num_directions)
            ]

        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            self.zo_perturb_parameters(scaling_factor=1.0)
            loss_plus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=-2.0)
            loss_minus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=1.0)

            if self.zo_subtype == "zo-sgd-con":
                loss_zero = self.zo_forward(model, inputs)

                if loss_zero < loss_plus and loss_zero < loss_minus:
                    print(f"{loss_zero} is less than both {loss_plus} and {loss_minus}, "
                          f"skipping update for seed {seed}.")
                    continue

            grad_estimate = ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()

            if self.zo_subtype == "zo-sgd-sign":
                grad_estimate = np.sign(grad_estimate)

            directions.append((seed, grad_estimate))

        self.zo_direction_accumulator.extend(directions)
        self.zo_accumulation_count += 1

        self._maybe_empty_cache()

        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            return

        seed_group: dict[int, float] = {}
        for seed, grad_estimate in self.zo_direction_accumulator:
            if seed in seed_group:
                seed_group[seed] += grad_estimate / (
                        self.args.gradient_accumulation_steps * self.zo_num_directions)
            else:
                seed_group[seed] = grad_estimate / (
                        self.args.gradient_accumulation_steps * self.zo_num_directions)

        for seed, grad_sum in seed_group.items():
            torch.manual_seed(seed)

            for name, param in self.trainable_params:
                z = torch.normal(
                    mean=0, std=1, size=param.data.size(),
                    device=param.device, dtype=param.dtype
                )

                if self.args.weight_decay > 0.0 and (
                        "bias" not in name and "layer_norm" not in name and "layernorm" not in name
                ):
                    param.data.add_(-learning_rate * grad_sum * z -
                                    learning_rate * self.args.weight_decay * param.data)
                else:
                    param.data.add_(z, alpha=-learning_rate * grad_sum)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
