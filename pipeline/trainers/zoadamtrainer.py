# pylint: disable=duplicate-code
"""Module for ZOAdamTrainer which combines MeZO gradient estimation with Adam logic."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class ZOAdamTrainingArguments(TrainingArguments):
    """
    TrainingArguments is a class containing the key arguments for Trainer.

    Args:
        trainer_mode (str): Mode of training (`regular`, `zo`).
        zo_eps (float): MeZO hyperparameter epsilon.
        zo_num_directions (int): Number of directions for MeZO.
    """

    trainer_mode: str = field(
        default="zo", metadata={"help": "Mode of training (`regular`, `zo`)."}
    )
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=1, metadata={"help": "Number of directions for MeZO."})

    zo_adam_beta1: float = field(default=0.9, metadata={"help": "Adam β₁ hyperparameter."})
    zo_adam_beta2: float = field(default=0.999, metadata={"help": "Adam β₂ hyperparameter."})
    zo_adam_eps: float = field(default=1e-4, metadata={"help": "Adam ε hyperparameter."})

    zo_adam_type: str = field(
        default="zoadam", metadata={"help": "Type of ZOAdam variant (`zoadam`, `radazo`)."}
    )

    accumulate_in_full_precision: bool = field(
        default=False,
        metadata={"help": "Whether to accumulate gradients in full precision for better stability."}
    )
    use_master_weights: bool = field(
        default=False, metadata={"help": "Whether to maintain master weights in fp32 for updates."}
    )


class ZOAdamTrainer(BaseZOTrainer):  # pylint: disable=too-many-instance-attributes
    """Class combining MeZO gradient estimation with Adam logic."""

    def __init__(self, **kwargs):
        """
        Initialize ZO-Adam trainer.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            ZOAdam-specific configuration is expected through ``ZOAdamTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)
        args = self.args

        # MeZO
        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)

        # Adam
        self.adam_beta1 = getattr(args, "zo_adam_beta1", 0.9)
        self.adam_beta2 = getattr(args, "zo_adam_beta2", 0.999)
        self.adam_eps = getattr(args, "zo_adam_eps", 1e-4)
        self.adam_step = 0

        self.zoadam_type = getattr(args, "zo_adam_type", "zoadam")  # "zoadam" or "radazo"

        # state for MeZO
        self.zo_direction_accumulator = []
        self.batch_zo_seeds = None
        self.zo_random_seed: int | None = None

        # model params
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
                n: p.data.detach().clone().float()
                for n, p in self.trainable_params
            }
        else:
            self.master_params = None

        # Adam buffers
        self.first_moment_buffers = {
            n: torch.zeros_like(p.data, dtype=self.accum_dtype)
            for n, p in self.trainable_params
        }
        self.second_moment_buffers = {
            n: torch.zeros_like(p.data, dtype=self.accum_dtype)
            for n, p in self.trainable_params
        }

        print(f"ZOAdamTrainer initialized with mode: {self.trainer_mode}, "
              f"zo_eps: {self.zo_eps}, zo_num_directions: {self.zo_num_directions}, "
              f"zo_adam_beta1: {self.adam_beta1}, zo_adam_beta2: {self.adam_beta2}, "
              f"zo_adam_eps: {self.adam_eps}, zo_adam_type: {self.zoadam_type}, "
              f"use_master_weights: {self.use_master_weights}\n"
              f"Trainable params amount: {len(self.trainable_params)}")

    ########################
    # MeZO‐specific
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
        if self.batch_zo_seeds is None:
            self.batch_zo_seeds = [
                np.random.randint(1_000_000_000)
                for _ in range(self.zo_num_directions)
            ]

        loss_plus = None
        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            self.zo_perturb_parameters(+1.0)
            loss_p = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(-2.0)
            loss_m = self.zo_forward(model, inputs)

            # restore
            self.zo_perturb_parameters(+1.0)

            grad_est = ((loss_p - loss_m) / (2 * self.zo_eps)).item()
            self.zo_direction_accumulator.append((seed, grad_est))
            loss_plus = loss_p if loss_plus is None else loss_plus

        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if not self.zo_direction_accumulator:
            return

        # 1) Precompute normalization constants
        one_minus_b1 = 1.0 - self.adam_beta1
        norm1 = one_minus_b1 / self.zo_num_directions

        one_minus_b2 = 1.0 - self.adam_beta2
        norm2 = one_minus_b2 / self.zo_num_directions

        # 2) Decay existing moments
        for name in self.first_moment_buffers:
            self.first_moment_buffers[name] *= self.adam_beta1
        for name in self.second_moment_buffers:
            self.second_moment_buffers[name] *= self.adam_beta2

        # 3) Group grads by seed
        seed_group: dict[int, float] = {}
        for seed, grad_est in self.zo_direction_accumulator:
            seed_group.setdefault(seed, 0.0)
            seed_group[seed] += grad_est / self.args.gradient_accumulation_steps

        # 4) Accumulate into moments
        for seed, grad_sum in seed_group.items():
            torch.manual_seed(seed)
            for name, param in self.trainable_params:
                z = torch.normal(
                    mean=0, std=1, size=param.data.size(),
                    device=param.device, dtype=param.dtype
                ).to(self.accum_dtype)

                grad_sum_sq = grad_sum * grad_sum
                self.first_moment_buffers[name].add_(z, alpha=norm1 * grad_sum)

                if self.zoadam_type == "zoadam":
                    self.second_moment_buffers[name].addcmul_(z, z, value=norm2 * grad_sum_sq)

                elif self.zoadam_type == "radazo":
                    m_buf = self.first_moment_buffers[name]
                    self.second_moment_buffers[name].addcmul_(m_buf, m_buf, value=norm2)

        # 5) Phase 2: bias correction & parameter update
        self.adam_step += 1
        bias_corr1 = 1.0 - self.adam_beta1 ** self.adam_step
        bias_corr2 = 1.0 - self.adam_beta2 ** self.adam_step

        for name, param in self.trainable_params:
            denom = self.second_moment_buffers[name].clone()
            denom.div_(bias_corr2)
            denom.sqrt_()
            denom.add_(self.adam_eps)

            update = self.first_moment_buffers[name].clone()
            update.div_(bias_corr1)
            update.div_(denom)

            if self.use_master_weights:
                # AdamW‐style weight decay
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    self.master_params[name].mul_(1 - learning_rate * self.args.weight_decay)

                self.master_params[name].add_(update, alpha=-learning_rate)
                param.data.copy_(self.master_params[name].to(param.dtype))
            else:
                # AdamW‐style weight decay
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data.mul_(1 - learning_rate * self.args.weight_decay)

                param.data.add_(update.to(param.dtype), alpha=-learning_rate)

        # 6) cleanup
        self.zo_direction_accumulator.clear()
        self.batch_zo_seeds = None
