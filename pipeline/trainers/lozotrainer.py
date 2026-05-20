# pylint: disable=duplicate-code
"""
Module for a LOZOTrainer implementation.

LOZOTrainer implementation is based on the paper
'Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures'
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class LOZOTrainingArguments(TrainingArguments):
    """
    Class containing the key arguments for LOZOTrainer.

    :param trainer_mode: Mode of training (`regular`, `zo`).
    :param zo_eps: MeZO hyperparameter epsilon.
    :param zo_num_directions: Number of directions for MeZO.
    :param zo_rank_r: LOZO hyperparameter: rank r for low-rank perturbations.
    :param zo_step_interval: LOZO hyperparameter: steps between V refresh.
    """

    trainer_mode: str = field(
        default="zo", metadata={"help": "Mode of training (`regular`, `zo`)."}
    )
    zo_eps: float = field(default=1e-3, metadata={"help": "MeZO hyperparameter epsilon."})
    zo_num_directions: int = field(default=1, metadata={"help": "Number of directions for MeZO."})

    zo_rank_r: int = field(
        default=2, metadata={"help": "LOZO hyperparameter: rank r for low-rank perturbations."}
    )
    zo_step_interval: int = field(
        default=50, metadata={"help": "LOZO hyperparameter: steps between V refresh."}
    )


class LOZOTrainer(BaseZOTrainer):
    """Low-rank ZO-SGD algortihm implementation."""

    def __init__(self, **kwargs):
        """
        Initialize LOZOMTrainer with additional arguments for LOZO training.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            LOZO-specific configuration is expected through ``LOZOTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)

        self.zo_rank_r = getattr(args, "zo_rank_r", 2)
        self.zo_step_interval = getattr(args, "zo_step_interval", 50)

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
                param.requires_grad = False

        self.v = {}  # LOZO factor cache: per-parameter right factor V
        self.step = 0  # Step counter used for V refresh cadence

        print(f"LOZOTrainer initialized with mode: {self.trainer_mode}, zo_eps: {self.zo_eps}, "
              f"zo_num_directions: {self.zo_num_directions}, zo_rank_r: {self.zo_rank_r}, "
              f"zo_step_interval: {self.zo_step_interval}\n"
              f"Trainable params amount: {len(self.trainable_params)}")

    ########################
    # LOZO-specific Methods
    ########################

    def random_gaussian_matrix(self, m: int, n: int, device: torch.device, dtype: torch.dtype,
                               random_seed: int | None = None) -> torch.Tensor:
        """
        Generate a random Gaussian matrix of shape (m, n) with the specified device and dtype.

        :param m: Number of rows.
        :param n: Number of columns.
        :param device: Device on which to create the matrix.
        :param dtype: Data type of the matrix.
        :param random_seed: Optional random seed for reproducibility.
        :return: A random Gaussian matrix of shape (m, n).
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        return torch.randn(m, n, device=device, dtype=dtype)

    def _ensure_v_cached(self, refresh: bool) -> None:
        """
        Ensure the correct V for each parameter is available. If `refresh`, resample V everywhere.

        :param refresh: Whether to refresh V for all eligible parameters.
        """
        if refresh:
            self.v = {}

        for name, param in self.trainable_params:
            if param.data.ndim >= 2:
                in_dim = param.data.size(1)
                # Only (re)create if missing or shape mismatch
                need_new = (name not in self.v) or (self.v[name].size(0) != in_dim) or (
                        self.v[name].size(1) != self.zo_rank_r)
                if need_new:
                    # IMPORTANT: We do NOT tie V sampling to the per-direction seed.
                    # V is shared across all directions for this step
                    # unless refresh cadence dictates otherwise.
                    v_mat = torch.randn(in_dim, self.zo_rank_r,
                                        device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v_mat

    def zo_perturb_parameters(self, scaling_factor: float = 1.0):
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        """
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.trainable_params:
            if param.data.ndim >= 2:
                v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=self.zo_rank_r,
                                                device=param.data.device, dtype=param.data.dtype)
                param.data.add_(u @ v.t(), alpha=scaling_factor * self.zo_eps)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                 device=param.data.device, dtype=param.data.dtype)
                param.data.add_(z, alpha=scaling_factor * self.zo_eps)

    def zo_step(self, model: PreTrainedModel, inputs: dict[str, Any]) -> torch.Tensor:
        """
        Estimate the gradient by performing multiple directional forward passes.

        :param model: The model.
        :param inputs: Batch of inputs.
        :return: The computed loss.
        """
        # Determine if we need to refresh V for this step
        # Follow LOZO's convention: refresh every `zo_step_interval` steps
        # when step % zo_step_interval == 0
        refresh_v = self.step % self.zo_step_interval == 0
        self._ensure_v_cached(refresh=refresh_v)

        if self.batch_zo_seeds is None:
            self.batch_zo_seeds = [
                np.random.randint(1000000000) for _ in range(self.zo_num_directions)
            ]

        directions = []

        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            self.zo_perturb_parameters(scaling_factor=1.0)
            loss_plus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=-2.0)
            loss_minus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=1.0)

            grad_estimate = ((loss_plus - loss_minus) / (2 * self.zo_eps)).item()
            directions.append((seed, grad_estimate))

        self.zo_direction_accumulator.extend(directions)
        self.zo_accumulation_count += 1

        self._maybe_empty_cache()

        self.step += 1

        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            print("No accumulated directions to update (LOZO).")
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
                if param.data.ndim >= 2:
                    v = self.v[name]
                    u = self.random_gaussian_matrix(m=param.data.size(0), n=self.zo_rank_r,
                                                    device=param.data.device,
                                                    dtype=param.data.dtype)
                    if self.args.weight_decay > 0.0 and (
                            "bias" not in name and "layer_norm" not in name and
                            "layernorm" not in name
                    ):
                        param.data.add_(-learning_rate * (
                                grad_sum * u @ v.t() + self.args.weight_decay * param.data))
                    else:
                        param.data.add_(u @ v.t(), alpha=-learning_rate * grad_sum)
                else:
                    # 1D params use vector z
                    z = torch.normal(mean=0, std=1, size=param.data.size(),
                                     device=param.data.device, dtype=param.data.dtype)
                    if self.args.weight_decay > 0.0 and (
                            "bias" not in name and "layer_norm" not in name and
                            "layernorm" not in name
                    ):
                        param.data.add_(-learning_rate * (
                                grad_sum * z + self.args.weight_decay * param.data))
                    else:
                        param.data.add_(z, alpha=-learning_rate * grad_sum)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
