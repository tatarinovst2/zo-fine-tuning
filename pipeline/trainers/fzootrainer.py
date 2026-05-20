# pylint: disable=duplicate-code
"""
Module for a FZOOTrainer implementation.

FZOOTrainer implementation is based on the paper
'Fast Zeroth-Order Optimizer for Fine‑Tuning Large Language Models towards Adam‑Scale Speed'
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class FZOOTrainingArguments(TrainingArguments):
    """
    Class containing the key arguments for FZOOTrainer.

    :param trainer_mode: Mode of training (`regular`, `zo`).
    :param zo_eps: MeZO hyperparameter epsilon.
    :param zo_num_directions: Number of directions for MeZO.
    :param fzoo_use_rademacher: If True, sample z from {-1,+1}. If False, sample Gaussian noise.
    """

    trainer_mode: str = field(
        default="zo", metadata={"help": "Mode of training (`regular`, `zo`)."}
    )
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=8, metadata={"help": "Number of directions for MeZO."})
    fzoo_use_rademacher: bool = field(
        default=True, metadata={"help": "Use Rademacher +/-1 noise (else Gaussian)."}
    )
    use_master_weights: bool = field(
        default=False, metadata={"help": "Whether to keep master FP32 weights for stability."}
    )


class FZOOTrainer(BaseZOTrainer):
    """FZOOTrainer: One-sided, baseline-subtracted, std-normalized zeroth-order optimizer."""

    def __init__(self, **kwargs):
        """
        Initialize FZOOTrainer with additional arguments for FZOO training.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            FZOO-specific configuration is expected through ``FZOOTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 8)
        self.fzoo_use_rademacher = getattr(args, "fzoo_use_rademacher", True)

        self.use_master_weights = getattr(args, "use_master_weights", False)

        self.trainable_params = [
            (p[0], p[1]) for p in self.model.named_parameters() if p[1].requires_grad
        ]

        if not self.trainable_params:
            raise ValueError("No trainable parameters found.")

        if self.trainer_mode == "zo":
            for param in self.model.parameters():
                param.requires_grad = False  # Ensure no accidental gradient computation

        self.zo_random_seeds = []
        self.projected_grad = None
        self.zo_random_seed: int | None = None

        if self.use_master_weights:
            self.master_params = {
                name: param.data.detach().clone().float()
                for name, param in self.trainable_params
            }
        else:
            self.master_params = None

        # No gradient accumulation
        assert getattr(args, "gradient_accumulation_steps", 1) == 1,\
            ("FZOOTrainer does not support gradient accumulation. "
             "Please set gradient_accumulation_steps=1.")

        assert self.zo_num_directions >= 2, "FZOO needs >=2 directions or std=0."

        print(f"FZOOTrainer initialized with mode: {self.trainer_mode}, zo_eps: {self.zo_eps}, "
              f"zo_num_directions: {self.zo_num_directions}, "
              f"fzoo_use_rademacher: {self.fzoo_use_rademacher}\n"
              f"Trainable params amount: {len(self.trainable_params)}, "
              f"use_master_weights: {self.use_master_weights}")

    def _sample_z(self, like: torch.Tensor) -> torch.Tensor:
        """
        Sample a random noise tensor z of the same shape as `like`.

        :param like: A tensor whose shape and device will be used for sampling z.
        :return: A random noise tensor z of the same shape and device as `like`.
        """
        if self.fzoo_use_rademacher:
            return torch.randint(
                0, 2, size=like.data.size(), device=like.data.device, dtype=like.data.dtype
            ) * 2 - 1  # {-1, +1}

        return torch.normal(
            mean=0.0, std=1.0, size=like.size(), device=like.device, dtype=like.dtype
        )

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        """
        torch.manual_seed(self.zo_random_seed)
        for _, param in self.trainable_params:
            z = self._sample_z(param.data)
            param.data.add_(z, alpha=scaling_factor * self.zo_eps)

    def zo_step(self, model: PreTrainedModel, inputs: dict[str, Any]) -> torch.Tensor:
        """
        Estimate the gradient by performing multiple directional forward passes.

        :param model: The model.
        :param inputs: Batch of inputs.
        :return: The computed loss.
        """
        self.zo_random_seeds = [
            np.random.randint(1_000_000_000) for _ in range(self.zo_num_directions)
        ]
        loss1s = []

        with torch.no_grad():
            for seed in self.zo_random_seeds:
                self.zo_random_seed = seed

                self.zo_perturb_parameters(scaling_factor=1.0)
                loss1 = self.zo_forward(model, inputs).detach()
                loss1s.append(loss1)

                self.zo_perturb_parameters(scaling_factor=-1.0)

            loss1s_t = torch.stack([l.to(dtype=torch.float32) for l in loss1s], dim=0)

            baseline_loss = self.zo_forward(model, inputs).detach()

            std = torch.std(loss1s_t, unbiased=False)
            # std = torch.clamp(std, min=1e-12)

            denom = float(self.zo_num_directions * std.item())

            self.projected_grad = (loss1s_t - baseline_loss.to(loss1s_t.device,
                                                               loss1s_t.dtype)) / denom

            self._maybe_empty_cache()

        return baseline_loss

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if self.projected_grad is None or len(self.zo_random_seeds) == 0:
            print("FZOO: nothing to update (no seeds / projected_grad).")
            return

        with torch.no_grad():
            for idx, seed in enumerate(self.zo_random_seeds):
                torch.manual_seed(int(seed))
                g_i = self.projected_grad[idx].item()

                for name, param in self.trainable_params:
                    z = self._sample_z(param.data)

                    if self.use_master_weights:
                        z = z.to(torch.float32)

                    if self.use_master_weights:
                        self.master_params[name].add_(z, alpha=-learning_rate * g_i)
                    else:
                        param.data.add_(z, alpha=-learning_rate * g_i)

            for name, param in self.trainable_params:
                if getattr(self.args, "weight_decay", 0.0) > 0 and (
                        "bias" not in name and "layer_norm" not in name and "layernorm" not in name
                ):
                    wd_target = (
                        self.master_params[name] if self.use_master_weights else param.data
                    )
                    wd_update = learning_rate * self.args.weight_decay * wd_target
                    if self.use_master_weights:
                        self.master_params[name] -= wd_update
                    else:
                        param.data -= wd_update

                if self.use_master_weights:
                    param.data.copy_(self.master_params[name].to(param.dtype))

        self.zo_random_seeds = []
        self.projected_grad = None
