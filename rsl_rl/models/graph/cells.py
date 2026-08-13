from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class MLPCell(nn.Module):
    """A rank-2 MLP cell used by :class:`ModelGraph`."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLPCell requires at least one hidden layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2:
            raise ValueError(f"MLPCell expects a rank-2 tensor [B, D], got shape {tuple(input.shape)}.")
        return self.mlp(input)
