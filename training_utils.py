"""Small reusable pieces for robust detector training."""

import copy
import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ModelEMA:
    """Maintain an exponential moving average of model parameters and floating buffers."""

    def __init__(self, model: nn.Module, decay: float, warmup_tau: float = 0.0) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        if warmup_tau < 0:
            raise ValueError("EMA warm-up tau cannot be negative")
        self.decay = decay
        self.warmup_tau = warmup_tau
        self.updates = 0
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move averaged state toward the latest model state."""
        self.updates += 1
        decay = self.decay
        if self.warmup_tau:
            decay *= 1 - math.exp(-self.updates / self.warmup_tau)
        averaged_state = self.model.state_dict()
        current_state = model.state_dict()
        for name, averaged_value in averaged_state.items():
            current_value = current_state[name].detach()
            if averaged_value.is_floating_point():
                averaged_value.mul_(decay).add_(current_value, alpha=1 - decay)
            else:
                averaged_value.copy_(current_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the averaged model state without a wrapper prefix."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore an averaged model state."""
        self.model.load_state_dict(state_dict)


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Linearly warm up, then decay the learning rate to zero with a cosine."""
    if total_steps < 1:
        raise ValueError("A scheduler requires at least one optimizer step")
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def should_evaluate(epoch: int, epochs: int, interval: int) -> bool:
    """Evaluate periodically, every epoch in the final third, and on the final epoch."""
    completed_epoch = epoch + 1
    final_third = completed_epoch > (2 * epochs) // 3
    return completed_epoch % interval == 0 or final_third or completed_epoch == epochs


def optimizer_steps_per_epoch(batch_count: int, accumulation_steps: int) -> int:
    """Return optimizer updates produced by an epoch of accumulated batches."""
    return math.ceil(batch_count / accumulation_steps)


def seed_worker(worker_id: int) -> None:
    """Seed Python's random module inside a DataLoader worker."""
    del worker_id
    import random

    random.seed(torch.initial_seed() % 2**32)
