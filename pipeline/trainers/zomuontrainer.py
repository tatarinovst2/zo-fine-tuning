# pylint: disable=duplicate-code
"""
Module for a ZO-Muon trainer implementation.

ZOMuonTrainer implementation of the algorithm proposed in paper
'Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization'
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, TrainingArguments

from .basezotrainer import BaseZOTrainer


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the matrix using the Newton-Schulz iteration.

    :param grad: The original gradient matrix.
    :param steps: Number of iterations for the Newton-Schulz method.
    :param eps: Small constant to ensure numerical stability.
    :return: The orthogonalized gradient matrix after applying the Newton-Schulz iteration.
    """
    # batched Muon implementation by @scottjmaddox, put into practice by @YouJiacheng
    assert grad.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    x_mat = grad.float()
    if grad.size(-2) > grad.size(-1):
        x_mat = x_mat.mT

    x_mat /= (x_mat.norm(dim=(-2, -1), keepdim=True) + eps)  # Ensure spectral norm is at most 1

    for _ in range(steps):
        a_mat = x_mat @ x_mat.mT
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        b_mat = b * a_mat + c * a_mat @ a_mat
        x_mat = a * x_mat + b_mat @ x_mat

    if grad.size(-2) > grad.size(-1):
        x_mat = x_mat.mT

    return x_mat.to(grad.dtype)


@dataclass
class ZOMuonTrainingArguments(TrainingArguments):
    """
    Class containing the key arguments for ZOMuonTrainer.

    :param trainer_mode: Mode of training (`regular`, `zo`).
    :param zo_eps: Perturbation scale for gradient estimation.
    :param zo_num_directions: Number of random directions to sample.
    :param zo_rank_r: Rank r for low-rank subspace sampling.
    :param zo_step_interval: Steps between refreshing the subspace.
    :param zo_non_muon_learning_rate: Learning rate for 1D, input or output weights.
    """

    trainer_mode: str = field(default="zo")
    zo_eps: float = field(default=1e-3)
    zo_num_directions: int = field(default=4)

    zo_rank_r: int = field(default=64)
    zo_step_interval: int = field(default=100)

    zo_non_muon_learning_rate: float = field(
        default=1e-7,
        metadata={"help": "Learning rate for 1D parameters (biases, LayerNorms)"}
    )


class ZOMuonTrainer(BaseZOTrainer):  # pylint: disable=too-many-instance-attributes
    """
    ZO-Muon implementation.

    Based on 'Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization'
    """

    def __init__(self, **kwargs):
        """
        Initialize ZOMuonTrainer with gradient orthogonalization and low-rank subspace sampling.

        :param kwargs: Keyword arguments forwarded to ``BaseZOTrainer``.
            ZOMuon-specific configuration is expected through ``ZOMuonTrainingArguments``.
        :raises ValueError: If no trainable parameters are found in the model.
        """
        super().__init__(**kwargs)

        args = self.args

        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_rank_r = getattr(args, "zo_rank_r", 64)
        self.zo_step_interval = getattr(args, "zo_step_interval", 100)
        self.zo_num_directions = getattr(args, "zo_num_directions", 4)
        self.zo_non_muon_learning_rate = getattr(args, "zo_non_muon_learning_rate", 1e-7)

        self.trainable_params = [
            (n, p) for n, p in self.model.named_parameters() if p.requires_grad
        ]

        if not self.trainable_params:
            raise ValueError("No trainable params")

        if self.trainer_mode == "zo":
            for p in self.model.parameters():
                p.requires_grad = False

        self.p_matrices: dict[str, torch.Tensor] = {}  # Subspace matrices
        # For perturbations for each parameter and seed, used in the update step
        self.saved_u: dict[str, dict[int, torch.Tensor]] = {}

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds: list[int] | None = None
        self.zo_random_seed = 0

        self.step_count = 0

        print(f"ZOMuonTrainer initialized with mode: {self.trainer_mode}, "
              f"zo_eps: {self.zo_eps}, zo_rank_r: {self.zo_rank_r}, "
              f"zo_step_interval: {self.zo_step_interval}, "
              f"zo_num_directions: {self.zo_num_directions}, "
              f"zo_non_muon_learning_rate: {self.zo_non_muon_learning_rate}\n"
              f"Trainable params amount: {len(self.trainable_params)}")

    def _sample_p_mat(self, m: int, r: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        """
        Sample a random orthogonal matrix P of shape (m, r) using QR decomposition.

        :param m: Number of rows (parameter dimension).
        :param r: Number of columns (rank of the subspace).
        :param device: Device to create the matrix on.
        :param dtype: Data type of the matrix.
        :return: A random orthogonal matrix P of shape (m, r).
        """
        r = min(m, r)
        a_mat = torch.randn(m, r, device=device, dtype=torch.float32)
        q_mat, r_mat = torch.linalg.qr(a_mat, mode="reduced")  # pylint: disable=not-callable

        signs = torch.sign(torch.diagonal(r_mat))
        signs[signs == 0] = 1
        q_mat = q_mat * signs

        return q_mat.to(dtype)

    def _get_p_mat(self, name: str, param: torch.nn.Parameter) -> torch.Tensor:
        """
        Get the subspace matrix P for a given parameter, refreshing it if necessary.

        :param name: Name of the parameter.
        :param param: The parameter tensor.
        :return: The subspace matrix P for the parameter.
        """
        m = param.shape[0]

        if (
            name not in self.p_matrices
            or self.step_count % self.zo_step_interval == 0
            or self.p_matrices[name].shape[0] != m
        ):
            self.p_matrices[name] = self._sample_p_mat(
                m, self.zo_rank_r, param.device, param.dtype
            )

        return self.p_matrices[name]

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        :raises ValueError: If the ZO random seed is not set.
        """
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.trainable_params:
            if name not in self.saved_u:
                self.saved_u[name] = {}

            if param.ndim >= 2:
                m, n = param.data.shape

                p_mat = self._get_p_mat(name, param)
                r = p_mat.shape[1]

                u = torch.randn(r, n, device=param.device, dtype=param.dtype)

                self.saved_u[name][self.zo_random_seed] = u

                perturb = p_mat @ u
                param.data.add_(perturb, alpha=scaling_factor * self.zo_eps)
            else:  # 1D
                z = torch.randn_like(param)
                self.saved_u[name][self.zo_random_seed] = z
                param.data.add(z, alpha=scaling_factor * self.zo_eps)

    def zo_step(self, model: PreTrainedModel, inputs: dict[str, Any]) -> torch.Tensor:
        """
        Estimate the gradient by performing multiple directional forward passes.

        :param model: The model.
        :param inputs: Batch of inputs.
        :return: The computed loss.
        """
        self.step_count += 1

        self.saved_u = {}  # Clear saved perturbations for this step

        if self.batch_zo_seeds is None:
            self.batch_zo_seeds = [
                np.random.randint(1000000000)
                for _ in range(self.zo_num_directions)
            ]

        loss_zero = self.zo_forward(model, inputs)

        for seed in self.batch_zo_seeds:
            self.zo_random_seed = seed

            self.zo_perturb_parameters(scaling_factor=1.0)
            loss_plus = self.zo_forward(model, inputs)

            self.zo_perturb_parameters(scaling_factor=-1.0)

            grad_est = ((loss_plus - loss_zero) / self.zo_eps).item()

            self.zo_direction_accumulator.append((seed, grad_est))

        self.zo_accumulation_count += 1

        return loss_zero / self.args.gradient_accumulation_steps

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            return

        scale = 1.0 / (self.zo_accumulation_count * self.zo_num_directions)

        seed_group = {}
        for seed, grad in self.zo_direction_accumulator:
            if seed not in seed_group:
                seed_group[seed] = 0.0

            seed_group[seed] += grad * scale

        for name, param in self.trainable_params:
            if param.ndim >= 2:
                p_mat = self._get_p_mat(name, param)
                r = p_mat.shape[1]

                lowdim_grad = torch.zeros(
                    r, param.data.shape[1], device=param.device, dtype=param.dtype
                )

                for seed, grad_sum in seed_group.items():
                    u = self.saved_u[name][seed]
                    lowdim_grad.add_(u * grad_sum)

                ortho_lowdim_grad = zeropower_via_newtonschulz5(lowdim_grad)
                final_grad = p_mat @ ortho_lowdim_grad

                if self.args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        final_grad.add_(param.data, alpha=self.args.weight_decay)

                param.data.add_(final_grad, alpha=-learning_rate)
            else:
                grad = torch.zeros_like(param)

                for seed, grad_sum in seed_group.items():
                    z = self.saved_u[name][seed]
                    grad.add_(z * grad_sum)

                if self.args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        grad.add_(param.data, alpha=self.args.weight_decay)

                param.data.add_(grad, alpha=-self.zo_non_muon_learning_rate)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
