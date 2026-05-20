# pylint: disable=duplicate-code
"""
Module for a LOZOMTrainer implementation (with momentum).

LOZOMTrainer implementation is based on the paper
'Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures'
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


@dataclass
class LOZOMTrainingArguments(TrainingArguments):
    """
    Class containing the key arguments for LOZOMTrainer.

    :param trainer_mode: Mode of training (`regular`, `zo`).
    :param zo_eps: MeZO hyperparameter epsilon.
    :param zo_num_directions: Number of directions for MeZO.
    :param zo_rank_r: LOZO hyperparameter: rank r for low-rank perturbations.
    :param zo_step_interval: LOZO hyperparameter: steps between V refresh.
    :param momentum_beta: LOZOM hyperparameter: momentum decay factor.
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

    momentum_beta: float = field(
        default=0.9, metadata={"help": "LOZOM hyperparameter: momentum decay factor."}
    )

    accumulate_in_full_precision: bool = field(
        default=False,
        metadata={"help": "Whether to accumulate gradients in full precision for better stability."}
    )
    use_master_weights: bool = field(
        default=False, metadata={"help": "Whether to maintain master weights in fp32 for updates."}
    )


class LOZOMTrainer(BaseZOTrainer):  # pylint: disable=too-many-instance-attributes
    """Low-rank ZO-SGD algortihm implementation with momentum."""

    def __init__(self, **kwargs):
        """
        Initialize LOZOMTrainer with additional arguments for LOZO training.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            LOZOM-specific configuration is expected through ``LOZOMTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)

        self.zo_rank_r = getattr(args, "zo_rank_r", 2)
        self.zo_step_interval = getattr(args, "zo_step_interval", 50)

        self.momentum_beta = getattr(args, "momentum_beta", 0.9)

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

        self.use_master_weights = getattr(args, "use_master_weights", False)
        self.accumulate_in_full_precision = getattr(args, "accumulate_in_full_precision", False)

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

        # LOZO factor cache
        self.v = {}  # Current right factor V per parameter (shape: in_dim x rank)
        self.v_old = {}  # Previous V per parameter for momentum rotation across refresh
        self.exp_avg_m = {}  # Momentum buffers per parameter (U-like for 2D; vector for 1D params)

        self.step = 0  # Step counter used for V refresh cadence

        print(f"LOZOMTrainer initialized with mode: {self.trainer_mode}, zo_eps: {self.zo_eps}, "
              f"zo_num_directions: {self.zo_num_directions}, zo_rank_r: {self.zo_rank_r}, "
              f"zo_step_interval: {self.zo_step_interval}, momentum_beta: {self.momentum_beta}\n"
              f"accumulate_in_full_precision: {self.accumulate_in_full_precision}, "
              f"use_master_weights: {self.use_master_weights}\n"
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
                need_new = (name not in self.v) or (self.v[name].size(0) != in_dim) or (
                        self.v[name].size(1) != self.zo_rank_r)
                if need_new:
                    # V is shared across all directions for this step
                    # unless refresh cadence dictates otherwise.
                    v_mat = torch.randn(in_dim, self.zo_rank_r,
                                        device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v_mat

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
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
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                param.data.add_(z, alpha=scaling_factor * self.zo_eps)

    def zo_step(self, model: PreTrainedModel, inputs: dict[str, Any]) -> torch.Tensor:
        """
        Estimate the gradient by performing multiple directional forward passes.

        :param model: The model.
        :param inputs: Batch of inputs.
        :return: The computed loss.
        """
        # Determine if we need to refresh V for this step
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

    #############################
    # Momentum-based ZO Updating
    #############################

    WEIGHT_DECAY_EXEMPT_SUBSTRINGS = (
        "bias",
        "layer_norm",
        "layernorm"
    )

    def _is_weight_decay_exempt(self, name: str) -> bool:
        """
        Check if the parameter name indicates exemption from weight decay.

        :param name: Parameter name to check.
        :return: True if the parameter is exempt from weight decay, False otherwise.
        """
        name_l = name.lower()
        return any(substr in name_l for substr in self.WEIGHT_DECAY_EXEMPT_SUBSTRINGS)

    def _apply_update(self, name: str, param: torch.nn.Parameter, grad: torch.Tensor,
                      learning_rate: float, weight_decay: float) -> None:
        """
        Apply weight decay plus the given update.

        :param name: Parameter name (used for checking weight decay exemption).
        :param param: The parameter to update.
        :param grad: The gradient to apply.
        :param learning_rate: The learning rate for the update.
        :param weight_decay: The weight decay factor.
        """
        if self.use_master_weights:
            master = self.master_params[name]
            grad = grad.to(self.master_dtype)

            if weight_decay != 0.0 and not self._is_weight_decay_exempt(name):
                master -= learning_rate * (grad + weight_decay * master)
            else:
                master -= learning_rate * grad

            param.data.copy_(master.to(dtype=param.data.dtype))
        else:
            grad = grad.to(param.data.dtype)

            if weight_decay != 0.0 and not self._is_weight_decay_exempt(name):
                param.data.add_(-learning_rate * (grad + weight_decay * param.data))
            else:
                param.data.add_(grad, alpha=-learning_rate)

    def _zo_update_matrix_param(self, name: str, param: torch.nn.Parameter, step_mod: int,
                                learning_rate: float) -> None:
        """
        Update a matrix parameter using the momentum-based update rule and subspace logic.

        :param name: Parameter name.
        :param param: The parameter to update.
        :param step_mod: Current step modulo the V refresh interval.
        :param learning_rate: The learning rate for the update.
        """
        v = self.v[name].to(self.accum_dtype)  # current right factor (in_dim x rank)
        # U factor sampled with the same seed as used during zo_step
        u = self.random_gaussian_matrix(m=param.data.size(0), n=self.zo_rank_r,
                                        device=param.data.device,
                                        dtype=param.data.dtype).to(self.accum_dtype)

        if name not in self.exp_avg_m:  # Initialize momentum if missing
            self.exp_avg_m[name] = self.projected_grad * u.to(self.accum_dtype)

        # Handle V rotation and momentum update depending on step cadence
        if step_mod == 0:
            if name in self.v_old:  # Rotation if we have a previous V
                v_old = self.v_old[name].to(self.accum_dtype)
                n = v_old.shape[0]  # in_dim
                # Rotate previous momentum from old V to new V basis, then add grad
                old_avg_m = self.momentum_beta * (self.exp_avg_m[name] @ v_old.t() @ v).to(
                    self.accum_dtype) / n
                self.exp_avg_m[name] = (old_avg_m + (1 - self.momentum_beta) *
                                        self.projected_grad * u.to(self.accum_dtype))
            else:
                # No previous V to rotate from; start fresh with current grad
                self.exp_avg_m[name] = self.projected_grad * u.to(self.accum_dtype)
        elif step_mod == self.zo_step_interval - 1:
            # On last step before refresh: save current V and do standard moment update
            self.v_old[name] = v
            self.exp_avg_m[name] = (self.momentum_beta * self.exp_avg_m[name] +
                                    (1 - self.momentum_beta) * self.projected_grad *
                                    u.to(self.accum_dtype))
        else:  # Regular momentum update (no rotation)
            self.exp_avg_m[name] = (self.momentum_beta * self.exp_avg_m[name] +
                                    (1 - self.momentum_beta) * self.projected_grad *
                                    u.to(self.accum_dtype))

        grad = self.exp_avg_m[name].to(self.accum_dtype) @ v.t().to(self.accum_dtype)
        self._apply_update(name, param, grad, learning_rate, self.args.weight_decay)

    def _zo_update_vector_param(self, name: str, param: torch.nn.Parameter,
                                learning_rate: float) -> None:
        """
        Update a vector parameter using the momentum-based update rule.

        :param name: Parameter name.
        :param param: The parameter to update.
        :param learning_rate: The learning rate for the update.
        """
        z = torch.normal(mean=0, std=1, size=param.data.size(),
                         device=param.data.device, dtype=param.data.dtype)

        if name not in self.exp_avg_m:  # Initialize momentum if missing
            self.exp_avg_m[name] = self.projected_grad * z.to(self.accum_dtype)
        else:
            self.exp_avg_m[name] = (self.momentum_beta * self.exp_avg_m[name] +
                                    (1 - self.momentum_beta) * self.projected_grad *
                                    z.to(self.accum_dtype))

        grad = self.exp_avg_m[name]
        self._apply_update(name, param, grad, learning_rate, self.args.weight_decay)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            print("No accumulated directions to update (LOZOM).")
            return

        seed_group: dict[int, float] = {}
        denom = self.args.gradient_accumulation_steps * self.zo_num_directions
        for seed, grad_estimate in self.zo_direction_accumulator:
            if seed in seed_group:
                seed_group[seed] += grad_estimate / denom
            else:
                seed_group[seed] = grad_estimate / denom

        for seed, grad_sum in seed_group.items():
            self.zo_random_seed = seed
            torch.manual_seed(seed)

            # step_mod = self.step % self.zo_step_interval
            step_mod = (self.step - 1) % self.zo_step_interval

            self.projected_grad = grad_sum

            for name, param in self.trainable_params:
                if param.data.ndim >= 2:
                    self._zo_update_matrix_param(name, param, step_mod, learning_rate)
                else:
                    self._zo_update_vector_param(name, param, learning_rate)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
