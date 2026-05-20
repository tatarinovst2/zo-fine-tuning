# pylint: disable=duplicate-code
"""
Module for a foreach MeZOTrainer implementation.

MeZOTrainer logic is based on the paper
'Fine-Tuning Language Models with Just Forward Passes'
"""
from typing import Any

import numpy as np
import torch
from torch.nn.parameter import Parameter
from transformers import PreTrainedModel

from .basezotrainer import BaseZOTrainer


class MeZOForEachTrainer(BaseZOTrainer):
    """
    MeZOTrainer implements the MeZO algorithm for training models without gradients.

    This version uses torch._foreach APIs for efficient in-place operations on parameter groups,
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = self.args

        self.trainer_mode = getattr(args, "trainer_mode", "zo")
        self.zo_eps = getattr(args, "zo_eps", 1e-3)
        self.zo_num_directions = getattr(args, "zo_num_directions", 1)

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
        self.zo_random_seed = None

        # Collect trainable params and sort by name for deterministic order
        self.trainable_params = sorted(
            [(name, p) for name, p in self.model.named_parameters() if p.requires_grad],
            key=lambda x: x[0]
        )
        print(f"Trainable parameters amount: {len(self.trainable_params)}")
        print(f"Trainable parameters: {[name for name, _ in self.trainable_params]}")

        if not self.trainable_params:
            raise ValueError("No trainable parameters found.")

        if self.trainer_mode == "zo":
            for param in self.model.parameters():
                print(param.data.size())
                param.requires_grad = False

        self._build_param_groups(mem_cap_bytes=None)

        print(f"MeZOForEachTrainer initialized with mode: {self.trainer_mode}, "
              f"zo_eps: {self.zo_eps}, zo_num_directions: {self.zo_num_directions}, "
              f"Trainable params amount: {len(self.trainable_params)}")

    ########################
    # Grouping utilities
    ########################

    def _build_param_groups(self, mem_cap_bytes = None) -> None:
        """
        Build contiguous groups of parameters (single-device assumption) subject to a memory cap.
        Maintains deterministic order by sorted parameter names.
        """
        params_in_order = self.trainable_params  # already sorted by name
        max_param_bytes = max(param.numel() * param.element_size() for _, param in params_in_order)
        cap = mem_cap_bytes or max_param_bytes

        self._groups = []
        current_group: list[tuple[str, Parameter]] = []
        current_group_bytes = 0

        for name, param in params_in_order:
            param_size_bytes = param.numel() * param.element_size()
            if current_group and current_group_bytes + param_size_bytes > cap:
                self._groups.append(self._finalize_group(current_group))
                current_group = []
                current_group_bytes = 0
            current_group.append((name, param))
            current_group_bytes += param_size_bytes

        if current_group:
            self._groups.append(self._finalize_group(current_group))

        for group in self._groups:
            group["decayed_params"] = [
                param
                for (name, param), use_decay in zip(group["items"], group["decay_mask"])
                if use_decay
            ]

        print(f"Built {len(self._groups)} groups with cap {cap/1e6:.2f} MB.")

    def _finalize_group(self, items):
        names = [n for n, _ in items]
        params = [p for _, p in items]
        numels = [p.numel() for p in params]
        shapes = [tuple(p.shape) for p in params]
        decay_mask = [not ("bias" in n or "layer_norm" in n or "layernorm" in n) for n in names]

        return {
            "items": list(zip(names, params)),
            "params": params,
            "names": names,
            "numels": numels,
            "shapes": shapes,
            "decay_mask": decay_mask,
        }

    def _seeded_generator(self, device, seed):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g

    ########################
    # MeZO-specific Methods
    ########################

    def zo_perturb_parameters(self, scaling_factor: float = 1.0) -> None:
        """
        Perturb the model parameters with a random noise z.

        :param scaling_factor: Scaling factor for the perturbation.
        """
        seed = self.zo_random_seed
        if seed is None:
            raise RuntimeError("zo_random_seed must be set before perturbation")

        base_device = self.trainable_params[0][1].device
        g = self._seeded_generator(device=base_device, seed=seed)

        for grp in self._groups:
            dev = grp["params"][0].device
            dt = grp["params"][0].dtype

            total_elems = sum(grp["numels"])
            noise_flat = torch.empty(total_elems, device=dev, dtype=dt)
            noise_flat.normal_(generator=g)

            splits = noise_flat.split(grp["numels"])
            noise_list = [s.view(sh) for s, sh in zip(splits, grp["shapes"])]

            torch._foreach_add_(grp["params"], noise_list, alpha=scaling_factor * self.zo_eps)  # pylint: disable=protected-access

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

        return loss_plus / (self.args.gradient_accumulation_steps * self.zo_num_directions)

    def zo_update(self, learning_rate: float) -> None:
        """
        Update model parameters based on the estimated gradient.

        :param learning_rate: The learning rate used for the update.
        """
        if len(self.zo_direction_accumulator) == 0:
            print("No accumulated directions to update.")
            return

        seed_group: dict[int, float] = {}
        for seed, grad_estimate in self.zo_direction_accumulator:
            seed_group[seed] = seed_group.get(seed, 0.0) + grad_estimate / (
                self.args.gradient_accumulation_steps * self.zo_num_directions
            )

        base_device = self.trainable_params[0][1].device

        for seed, grad_sum in seed_group.items():
            g = self._seeded_generator(device=base_device, seed=seed)

            for grp in self._groups:
                dev = grp["params"][0].device
                dt = grp["params"][0].dtype

                total_elems = sum(grp["numels"])
                noise_flat = torch.empty(total_elems, device=dev, dtype=dt)
                noise_flat.normal_(generator=g)

                splits = noise_flat.split(grp["numels"])
                noise_list = [s.view(sh) for s, sh in zip(splits, grp["shapes"])]

                if self.args.weight_decay:
                    decayed = grp["decayed_params"]
                    if decayed:
                        torch._foreach_mul_(decayed, 1.0 - learning_rate * self.args.weight_decay)  # pylint: disable=protected-access

                torch._foreach_add_(grp["params"], noise_list, alpha=-learning_rate * grad_sum)  # pylint: disable=protected-access

        self.zo_direction_accumulator = []
        self.zo_accumulation_count = 0
        self.batch_zo_seeds = None
