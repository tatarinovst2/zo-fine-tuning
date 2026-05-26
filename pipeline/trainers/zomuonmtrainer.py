# pylint: disable=duplicate-code
"""
Module for a ZO-Muon trainer implementation WITH momentum.

ZOMuonMTrainer: ZOMuon + momentum in the low-rank subspace.
Minimal changes:
- add momentum buffers (with bias correction)
- rotate momentum buffers when P refreshes
- add optional accumulate_in_full_precision + use_master_weights logic
"""
from dataclasses import dataclass, field

import torch

from .zomuontrainer import zeropower_via_newtonschulz5, ZOMuonTrainer, ZOMuonTrainingArguments


@dataclass
class ZOMuonMTrainingArguments(ZOMuonTrainingArguments):
    """
    Class containing the key arguments for ZOMuonMTrainer.

    Extends ``ZOMuonTrainingArguments`` with momentum and optional precision controls.

    :param momentum_beta: Momentum decay factor (EMA).
    :param accumulate_in_full_precision: Whether to accumulate momentum buffers in fp32.
    :param use_master_weights: Whether to maintain fp32 master weights for updates.
    """

    momentum_beta: float = field(
        default=0.9,
        metadata={"help": "Momentum decay factor (EMA) for low-dim (and vector) gradients."},
    )

    accumulate_in_full_precision: bool = field(
        default=False,
        metadata={"help": "Whether to accumulate momentum buffers in fp32 for better stability."},
    )
    use_master_weights: bool = field(
        default=False,
        metadata={"help": "Whether to maintain master weights in fp32 for updates."},
    )


class ZOMuonMTrainer(ZOMuonTrainer):  # pylint: disable=too-many-instance-attributes
    """
    ZOMuon with momentum (minimal patch on top of ZOMuonTrainer).

    - Momentum stored in low-dim gradient space for matrix params (shape r x in_dim).
    - Momentum stored as vector for 1D params.
    - Bias correction is applied to the momentum buffer before using it.
    - When P refreshes, rotate low-dim momentum: M <- (P_new^T P_old) M

    Precision options (minimal borrow from LOZOM):
    - accumulate_in_full_precision: store momentum buffers in fp32
    - use_master_weights: keep fp32 master params, update them, then copy back
    """

    WEIGHT_DECAY_EXEMPT_SUBSTRINGS = (
        "bias",
        "layer_norm",
        "layernorm",
    )

    def __init__(self, **kwargs):
        """
        Initialize ZOMuonMTrainer.

        :param kwargs: Keyword arguments forwarded to ``ZOMuonTrainer``.
            ZOMuonM-specific configuration is expected through ``ZOMuonMTrainingArguments``.
        """
        super().__init__(**kwargs)

        args = self.args

        self.momentum_beta = getattr(args, "momentum_beta", 0.9)
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

        self.accum_dtype = torch.float32 if self.accumulate_in_full_precision else self.model_dtype
        self.master_dtype = torch.float32

        if self.use_master_weights:
            self.master_params = {
                name: param.data.detach().clone().to(dtype=self.master_dtype)
                for name, param in self.trainable_params
            }
        else:
            self.master_params = None

        self.exp_avg_m: dict[str, torch.Tensor] = {}
        self.exp_avg_step: dict[str, int] = {}

        print(
            f"ZOMuonMTrainer initialized.\n"
            f"momentum_beta: {self.momentum_beta}\n"
            f"accumulate_in_full_precision: {self.accumulate_in_full_precision}, "
            f"use_master_weights: {self.use_master_weights}"
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

        Uses fp32 master weights if enabled; otherwise updates params in-place.

        :param name: Parameter name (used for checking weight decay exemption).
        :param param: The parameter to update.
        :param grad: The gradient to apply.
        :param learning_rate: The learning rate for the update.
        :param weight_decay: The weight decay factor.
        """
        if self.use_master_weights:
            master = self.master_params[name]
            grad_f = grad.to(dtype=self.master_dtype)

            if weight_decay != 0.0 and not self._is_weight_decay_exempt(name):
                master.add_(-learning_rate * (grad_f + weight_decay * master))
            else:
                master.add_(grad_f, alpha=-learning_rate)

            param.data.copy_(master.to(dtype=param.data.dtype))
        else:
            grad_d = grad.to(dtype=param.data.dtype)

            if weight_decay != 0.0 and not self._is_weight_decay_exempt(name):
                param.data.add_(-learning_rate * (grad_d + weight_decay * param.data))
            else:
                param.data.add_(grad_d, alpha=-learning_rate)

    def _get_p_mat(self, name: str, param: torch.nn.Parameter) -> torch.Tensor:
        """
        Get the subspace matrix P for a given parameter, refreshing it if necessary.

        :param name: Name of the parameter.
        :param param: The parameter tensor.
        :return: The subspace matrix P for the parameter.
        """
        m = param.shape[0]

        need_refresh = (
            name not in self.p_matrices
            or self.step_count % self.zo_step_interval == 0
            or self.p_matrices[name].shape[0] != m
        )

        if need_refresh:
            old_p = self.p_matrices.get(name, None)
            new_p = self._sample_p_mat(m, self.zo_rank_r, param.device, param.dtype)

            if old_p is not None and name in self.exp_avg_m:
                buf = self.exp_avg_m[name]
                if buf.ndim == 2 and old_p.ndim == 2 and new_p.ndim == 2:
                    # rot = new_p.float().mT @ old_p.float()
                    # rotated = rot @ buf.float()  # (r_new, in_dim)

                    self.exp_avg_m[name] = (
                            new_p.mT @ old_p @ buf
                    ).to(dtype=self.accum_dtype, device=buf.device)

            self.p_matrices[name] = new_p

        return self.p_matrices[name]

    def zo_update(self, learning_rate: float) -> None:  # pylint: disable=too-many-locals
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            return

        scale = 1.0 / (self.zo_accumulation_count * self.zo_num_directions)

        seed_group: dict[int, float] = {}
        for seed, grad in self.zo_direction_accumulator:
            if seed not in seed_group:
                seed_group[seed] = 0.0
            seed_group[seed] += grad * scale

        weight_decay = float(getattr(self.args, "weight_decay", 0.0))

        for name, param in self.trainable_params:
            if param.ndim >= 2:
                p_mat = self._get_p_mat(name, param)
                r = p_mat.shape[1]
                in_dim = param.data.shape[1]

                lowdim_grad = torch.zeros(
                    r, in_dim, device=param.device, dtype=self.accum_dtype
                )

                for seed, grad_sum in seed_group.items():
                    u = self.saved_u[name][seed].to(dtype=self.accum_dtype)
                    lowdim_grad.add_(u, alpha=grad_sum)

                if name not in self.exp_avg_m or self.exp_avg_m[name].shape != lowdim_grad.shape:
                    self.exp_avg_m[name] = lowdim_grad.detach().clone().mul_(
                        1.0 - self.momentum_beta
                    )
                    self.exp_avg_step[name] = 1
                else:
                    self.exp_avg_m[name].mul_(self.momentum_beta).add_(
                        lowdim_grad,alpha=1.0 - self.momentum_beta
                    )
                    self.exp_avg_step[name] = self.exp_avg_step.get(name, 0) + 1

                bias_corr = 1.0 - (self.momentum_beta ** self.exp_avg_step[name])
                m_hat = self.exp_avg_m[name] / bias_corr

                ortho_lowdim_grad = zeropower_via_newtonschulz5(m_hat).to(dtype=self.accum_dtype)
                final_grad = p_mat.to(dtype=self.accum_dtype) @ ortho_lowdim_grad

                self._apply_update(name=name, param=param, grad=final_grad,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay)
            else:
                grad_vec = torch.zeros(param.data.shape, device=param.device,
                                       dtype=self.accum_dtype)

                for seed, grad_sum in seed_group.items():
                    z = self.saved_u[name][seed].to(dtype=self.accum_dtype)
                    grad_vec.add_(z, alpha=grad_sum)

                if name not in self.exp_avg_m or self.exp_avg_m[name].shape != grad_vec.shape:
                    self.exp_avg_m[name] = grad_vec.detach().clone().mul_(1.0 - self.momentum_beta)
                    self.exp_avg_step[name] = 1
                else:
                    self.exp_avg_m[name].mul_(self.momentum_beta).add_(
                        grad_vec, alpha=1.0 - self.momentum_beta
                    )
                    self.exp_avg_step[name] = self.exp_avg_step.get(name, 0) + 1

                bias_corr = 1.0 - (self.momentum_beta ** self.exp_avg_step[name])
                vec_update = self.exp_avg_m[name] / bias_corr

                self._apply_update(name=name, param=param, grad=vec_update,
                                   learning_rate=self.zo_non_muon_learning_rate,
                                   weight_decay=weight_decay)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
