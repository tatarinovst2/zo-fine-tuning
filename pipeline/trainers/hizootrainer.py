# pylint: disable=duplicate-code
"""
Module for a HiZOOTrainer implementation.

HiZOOTrainer implementation is based on the paper
'SECOND-ORDER FINE-TUNING WITHOUT PAIN FOR LLMS: A HESSIAN INFORMED ZEROTH-ORDER OPTIMIZER'
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class HiZOOTrainingArguments(TrainingArguments):
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
    zo_hessian_smooth: float = field(
        default=1e-8, metadata={"help": "Smoothing factor for Hessian estimation."}
    )


class HiZOOTrainer(BaseZOTrainer):
    """HiZOO trainer implementation."""

    def __init__(self, **kwargs):
        """
        Initialize HiZOOTrainer with additional arguments for HiZOO training.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            HiZOO-specific configuration is expected through ``HiZOOTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)

        self.projected_grad = None
        self.zo_random_seed: int | None = None

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0

        self.batch_zo_seeds = None

        self.trainable_params = [
            (p[0], p[1]) for p in self.model.named_parameters() if p[1].requires_grad
        ]

        self.hessian_matrix = {}
        for name, param in self.trainable_params:
            self.hessian_matrix[name] = torch.ones(
                size=param.data.size(), device=param.data.device, dtype=param.data.dtype
            )

        if not self.trainable_params:
            raise ValueError("No trainable parameters found.")

        if self.trainer_mode == "zo":
            for param in self.model.parameters():
                param.requires_grad = False  # Ensure no accidental gradient computation

        self.zo_hessian_smooth = getattr(args, "zo_hessian_smooth", 1e-8)

        print(
            f"HiZOOTrainer initialized with mode: {self.trainer_mode}, zo_eps: {self.zo_eps}, "
            f"zo_num_directions: {self.zo_num_directions}, "
            f"zo_hessian_smooth: {self.zo_hessian_smooth}\n"
            f"Trainable params amount: {len(self.trainable_params)}"
        )

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        """
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.trainable_params:
            z = torch.normal(
                mean=0, std=1, size=param.data.size(), device=param.device, dtype=param.dtype
            )
            denom = torch.sqrt(self.hessian_matrix[name])
            param.data.addcdiv_(z, denom, value=scaling_factor * self.zo_eps)

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

        loss_plus_sum = None

        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            loss_zero = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=1.0)
            loss_plus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=-2.0)
            loss_minus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=1.0)

            grad_est = ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()
            hess_numerator = (torch.abs(
                loss_plus + loss_minus - 2 * loss_zero
            )).item()

            directions.append((seed, grad_est, hess_numerator))

            loss_plus_sum = loss_plus if loss_plus_sum is None else (loss_plus_sum + loss_plus)

        self.zo_direction_accumulator.extend(directions)
        self.zo_accumulation_count += 1

        self._maybe_empty_cache()

        return loss_plus_sum / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            print("No accumulated directions to update.")
            return

        seed_group = {}

        for seed, grad_est, hess_est in self.zo_direction_accumulator:
            scale = 1 / (self.args.gradient_accumulation_steps * self.zo_num_directions)

            if seed not in seed_group:
                seed_group[seed] = {"grad": 0.0, "hess": 0.0}

            seed_group[seed]["grad"] += grad_est * scale
            seed_group[seed]["hess"] += hess_est * scale

        for seed, values in seed_group.items():
            torch.manual_seed(seed)

            grad_sum = values["grad"]
            hess_sum = values["hess"]

            # print(self.hessian_matrix[self.trainable_params[2][0]])
            for name, param in self.trainable_params:
                z = torch.normal(
                    mean=0, std=1, size=param.data.size(), device=param.device, dtype=param.dtype
                )

                hessian_temp = self.hessian_matrix[name] * z  # alloc 1 full tensor
                hessian_temp.mul_(z)  # now hessian_temp == H * z * z

                hessian_estimator = hessian_temp  # alias
                hessian_estimator.mul_(
                    hess_sum * self.zo_hessian_smooth / (2.0 * (self.zo_eps ** 2))
                )

                self.hessian_matrix[name].mul_(1.0 - self.zo_hessian_smooth).add_(hessian_estimator)

                denom = torch.sqrt(self.hessian_matrix[name])  # alloc 1 full tensor
                precond_grad = z  # reuse z buffer (no new tensor)
                precond_grad.mul_(grad_sum).div_(denom)

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    precond_grad.add_(param.data, alpha=float(self.args.weight_decay))
                    param.data.add_(precond_grad, alpha=-learning_rate)
                else:
                    param.data.add_(precond_grad, alpha=-learning_rate)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
