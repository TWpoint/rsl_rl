from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TokenProjectionCell(nn.Module):
    """Project every token in a rank-3 sequence to a shared feature dimension."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(
                f"TokenProjectionCell expects [B, T, D], got shape {tuple(input.shape)}."
            )
        return self.projection(input)


class TokenMLPCell(nn.Module):
    """Apply the same MLP independently to every token in a rank-3 tensor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("TokenMLPCell requires at least one hidden layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(f"TokenMLPCell expects [B, N, D], got shape {tuple(input.shape)}.")
        return self.mlp(input)


class TopologyProjectionCell(nn.Module):
    """Project fixed entity topology coordinates into batched token features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        topology: list[list[float]] | tuple[tuple[float, ...], ...],
    ) -> None:
        super().__init__()
        coordinates = torch.as_tensor(topology, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
            raise ValueError("TopologyProjectionCell topology must have shape [N, C].")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("topology", coordinates)
        self.projection = nn.Linear(coordinates.shape[1], output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(
                f"TopologyProjectionCell expects entity tokens [B, N, D], got shape {tuple(input.shape)}."
            )
        if input.shape[1] != self.topology.shape[0]:
            raise ValueError(
                f"TopologyProjectionCell expected {self.topology.shape[0]} entities, got {input.shape[1]}."
            )
        topology_tokens = self.projection(self.topology.to(dtype=input.dtype))
        return topology_tokens.unsqueeze(0).expand(input.shape[0], -1, -1)


class TokenAddCell(nn.Module):
    """Add two token tensors with identical shapes."""

    def __init__(self, input_dim: tuple[int, int] | list[int], output_dim: int) -> None:
        super().__init__()
        if len(input_dim) != 2 or input_dim[0] != input_dim[1]:
            raise ValueError("TokenAddCell requires two inputs with matching feature dimensions.")
        if output_dim != input_dim[0]:
            raise ValueError(f"TokenAddCell output_dim must be {input_dim[0]}, got {output_dim}.")
        self.input_dim = list(input_dim)
        self.output_dim = output_dim

    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        first, second = input
        if first.ndim != 3 or second.ndim != 3:
            raise ValueError("TokenAddCell inputs must both have shape [B, N, D].")
        if first.shape != second.shape:
            raise ValueError(
                f"TokenAddCell input shapes must match, got {tuple(first.shape)} and {tuple(second.shape)}."
            )
        return first + second


class TokenMergeCell(nn.Module):
    """Merge two rank-3 token tensors by interleaving or concatenating them."""

    def __init__(
        self,
        input_dim: tuple[int, int] | list[int],
        output_dim: int,
        mode: str = "interleave",
        dim: int = 1,
    ) -> None:
        super().__init__()
        if len(input_dim) != 2:
            raise ValueError("TokenMergeCell requires exactly two inputs.")
        if mode not in {"interleave", "concatenate"}:
            raise ValueError("TokenMergeCell mode must be 'interleave' or 'concatenate'.")
        expected_output_dim = self.infer_output_dim(input_dim, dim=dim)
        if output_dim != expected_output_dim:
            raise ValueError(
                f"TokenMergeCell output_dim must be {expected_output_dim} when dim={dim}, got {output_dim}."
            )
        self.input_dim = list(input_dim)
        self.output_dim = output_dim
        self.mode = mode
        self.dim = dim % 3

    @staticmethod
    def infer_output_dim(input_dim: tuple[int, int] | list[int], dim: int = 1, **_) -> int:
        """Infer the trailing feature dimension after merging the inputs."""
        normalized_dim = dim % 3
        if normalized_dim == 0:
            raise ValueError("TokenMergeCell cannot merge along the batch dimension.")
        if normalized_dim == 2:
            return sum(input_dim)
        if input_dim[0] != input_dim[1]:
            raise ValueError(
                "TokenMergeCell feature dimensions must match when merging along the token dimension."
            )
        return input_dim[0]

    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        first, second = input
        if first.ndim != 3 or second.ndim != 3:
            raise ValueError(
                "TokenMergeCell inputs must both be rank-3 tensors."
            )
        if any(first.shape[index] != second.shape[index] for index in range(3) if index != self.dim):
            raise ValueError("TokenMergeCell input shapes must match outside the merge dimension.")
        if self.mode == "concatenate":
            return torch.cat((first, second), dim=self.dim)

        if first.shape[self.dim] != second.shape[self.dim] + 1:
            raise ValueError(
                "TokenMergeCell interleave mode requires the first input to have exactly one "
                f"more element along dim={self.dim}; got {first.shape[self.dim]} and "
                f"{second.shape[self.dim]}."
            )
        paired = torch.stack(
            (first.narrow(self.dim, 0, second.shape[self.dim]), second),
            dim=self.dim + 1,
        ).flatten(self.dim, self.dim + 1)
        tail = first.narrow(self.dim, first.shape[self.dim] - 1, 1)
        return torch.cat((paired, tail), dim=self.dim)


class TemporalAttentionCell(nn.Module):
    """Encode a temporal observation with a trailing learnable readout token.

    Rotary position embeddings are applied to the queries and keys of every
    token, including the learnable token appended after the observation
    sequence. The cell returns only that token after attention and, when
    enabled, an FFN residual block.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        position_embedding: str = "rope",
        normalization: str = "rms_norm",
        normalization_eps: float = 1.0e-6,
        attention_residual: bool = True,
        ffn_residual: bool = True,
        use_ffn: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("TemporalAttentionCell output_dim must be divisible by num_heads.")
        self.head_dim = output_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("TemporalAttentionCell head dimension must be even for RoPE.")
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported TemporalAttentionCell activation '{activation}'.")
        if position_embedding not in {"rope", "none"}:
            raise ValueError(
                "TemporalAttentionCell position_embedding must be either 'rope' or 'none'."
            )
        if normalization not in {"rms_norm", "layer_norm"}:
            raise ValueError(
                "TemporalAttentionCell normalization must be either 'rms_norm' or 'layer_norm'."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.position_embedding = position_embedding
        self.attention_residual = attention_residual
        self.ffn_residual = ffn_residual
        self.use_ffn = use_ffn
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.readout_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        norm_class = nn.RMSNorm if normalization == "rms_norm" else nn.LayerNorm
        self.attention_norm = norm_class(output_dim, eps=normalization_eps)
        # Only the trailing readout token is consumed by this single-block
        # encoder. It therefore needs one query, while every token still
        # contributes a key and value.
        self.query_projection = nn.Linear(output_dim, output_dim)
        self.key_value_projection = nn.Linear(output_dim, 2 * output_dim)
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.attention_dropout = dropout
        if use_ffn:
            self.ffn_norm = norm_class(output_dim, eps=normalization_eps)
            hidden_dim = ffn_dim or 4 * output_dim
            activation_layer = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                activation_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            )
        nn.init.normal_(self.readout_token, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError(
                f"TemporalAttentionCell expects [B, T, D], got shape {tuple(input.shape)}."
            )
        batch_size = input.shape[0]
        tokens = self.input_projection(input)
        tokens = torch.cat((tokens, self.readout_token.expand(batch_size, -1, -1)), dim=1)

        normalized = self.attention_norm(tokens)
        query = self.query_projection(normalized[:, -1:]).reshape(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value_projection(normalized).reshape(
            batch_size, tokens.shape[1], 2, self.num_heads, self.head_dim
        )
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        if self.position_embedding == "rope":
            query = self._apply_rope(query, position_offset=tokens.shape[1] - 1)
            key = self._apply_rope(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, 1, self.output_dim)
        attended = self.output_projection(attended)
        readout = tokens[:, -1:] + attended if self.attention_residual else attended
        if self.use_ffn:
            ffn_output = self.ffn(self.ffn_norm(readout))
            readout = readout + ffn_output if self.ffn_residual else ffn_output
        return readout[:, 0]

    def _apply_rope(self, tensor: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        sequence_length = tensor.shape[-2]
        frequency = 1.0 / (
            self.rope_base
            ** (torch.arange(0, self.head_dim, 2, device=tensor.device, dtype=torch.float32) / self.head_dim)
        )
        position = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=tensor.device,
            dtype=torch.float32,
        )
        angles = torch.outer(position, frequency).to(dtype=tensor.dtype)
        cosine = angles.cos()[None, None]
        sine = angles.sin()[None, None]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)


class CommandSelfAttentionCell(nn.Module):
    """Update command tokens by attending over an unchanged readout and the commands.

    The readout participates as context but is not part of the output. Only the
    command positions are queried and updated, so the original readout can be
    passed unchanged to a following cross-attention block.
    """

    def __init__(
        self,
        input_dim: tuple[int, int] | list[int],
        output_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        normalization: str = "rms_norm",
        normalization_eps: float = 1.0e-6,
        attention_residual: bool = True,
        ffn_residual: bool = True,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if len(input_dim) != 2:
            raise ValueError("CommandSelfAttentionCell requires readout and command inputs.")
        if output_dim % num_heads != 0:
            raise ValueError("CommandSelfAttentionCell output_dim must be divisible by num_heads.")
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported CommandSelfAttentionCell activation '{activation}'.")
        if normalization not in {"rms_norm", "layer_norm"}:
            raise ValueError("CommandSelfAttentionCell normalization must be 'rms_norm' or 'layer_norm'.")

        self.input_dim = list(input_dim)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.attention_residual = attention_residual
        self.ffn_residual = ffn_residual
        self.use_ffn = use_ffn
        self.readout_projection = nn.Linear(input_dim[0], output_dim)
        self.command_projection = nn.Linear(input_dim[1], output_dim)
        norm_class = nn.RMSNorm if normalization == "rms_norm" else nn.LayerNorm
        self.attention_norm = norm_class(output_dim, eps=normalization_eps)
        self.query_projection = nn.Linear(output_dim, output_dim)
        self.key_value_projection = nn.Linear(output_dim, 2 * output_dim)
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.attention_dropout = dropout
        if use_ffn:
            self.ffn_norm = norm_class(output_dim, eps=normalization_eps)
            hidden_dim = ffn_dim or 4 * output_dim
            activation_layer = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                activation_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            )

    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        readout_input, command_input = input
        if readout_input.ndim != 2:
            raise ValueError(
                f"CommandSelfAttentionCell readout expects [B, D], got {tuple(readout_input.shape)}."
            )
        if command_input.ndim != 3:
            raise ValueError(
                f"CommandSelfAttentionCell commands expect [B, N, D], got {tuple(command_input.shape)}."
            )
        if readout_input.shape[0] != command_input.shape[0]:
            raise ValueError("CommandSelfAttentionCell input batch sizes must match.")

        batch_size, num_commands = command_input.shape[:2]
        readout = self.readout_projection(readout_input).unsqueeze(1)
        commands = self.command_projection(command_input)
        context = torch.cat((readout, commands), dim=1)
        query = self.query_projection(self.attention_norm(commands)).reshape(
            batch_size, num_commands, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value_projection(self.attention_norm(context)).reshape(
            batch_size, num_commands + 1, 2, self.num_heads, self.head_dim
        )
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, num_commands, self.output_dim)
        attended = self.output_projection(attended)
        encoded = commands + attended if self.attention_residual else attended
        if self.use_ffn:
            ffn_output = self.ffn(self.ffn_norm(encoded))
            encoded = encoded + ffn_output if self.ffn_residual else ffn_output
        return encoded


class CrossAttentionCell(nn.Module):
    """Attend from one query feature to a sequence of context tokens."""

    def __init__(
        self,
        input_dim: tuple[int, int] | list[int],
        output_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        normalization: str = "rms_norm",
        normalization_eps: float = 1.0e-6,
        attention_residual: bool = True,
        ffn_residual: bool = True,
    ) -> None:
        super().__init__()
        if len(input_dim) != 2:
            raise ValueError("CrossAttentionCell requires query and key/value inputs.")
        if output_dim % num_heads != 0:
            raise ValueError("CrossAttentionCell output_dim must be divisible by num_heads.")
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported CrossAttentionCell activation '{activation}'.")
        if normalization not in {"rms_norm", "layer_norm"}:
            raise ValueError("CrossAttentionCell normalization must be 'rms_norm' or 'layer_norm'.")

        self.input_dim = list(input_dim)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.attention_residual = attention_residual
        self.ffn_residual = ffn_residual
        self.query_input_projection = nn.Linear(input_dim[0], output_dim)
        self.context_input_projection = nn.Linear(input_dim[1], output_dim)
        norm_class = nn.RMSNorm if normalization == "rms_norm" else nn.LayerNorm
        self.query_norm = norm_class(output_dim, eps=normalization_eps)
        self.context_norm = norm_class(output_dim, eps=normalization_eps)
        self.query_projection = nn.Linear(output_dim, output_dim)
        self.key_value_projection = nn.Linear(output_dim, 2 * output_dim)
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.attention_dropout = dropout
        self.ffn_norm = norm_class(output_dim, eps=normalization_eps)
        hidden_dim = ffn_dim or 4 * output_dim
        activation_layer = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        query_input, context_input = input
        if query_input.ndim != 2:
            raise ValueError(
                f"CrossAttentionCell query expects [B, D], got shape {tuple(query_input.shape)}."
            )
        if context_input.ndim != 3:
            raise ValueError(
                f"CrossAttentionCell context expects [B, N, D], got shape {tuple(context_input.shape)}."
            )
        if query_input.shape[0] != context_input.shape[0]:
            raise ValueError("CrossAttentionCell query and context batch sizes must match.")

        batch_size, num_tokens = context_input.shape[:2]
        query_token = self.query_input_projection(query_input).unsqueeze(1)
        context_tokens = self.context_input_projection(context_input)
        query = self.query_projection(self.query_norm(query_token)).reshape(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value_projection(self.context_norm(context_tokens)).reshape(
            batch_size, num_tokens, 2, self.num_heads, self.head_dim
        )
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, 1, self.output_dim)
        attended = self.output_projection(attended)
        encoded = query_token + attended if self.attention_residual else attended
        ffn_output = self.ffn(self.ffn_norm(encoded))
        encoded = encoded + ffn_output if self.ffn_residual else ffn_output
        return encoded[:, 0]


class InterleavedCausalAttentionCell(nn.Module):
    """Causally encode ``[o0, a0, ..., o4, e]`` and return the readout token ``e``."""

    def __init__(
        self,
        input_dim: tuple[int, int] | list[int],
        output_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        position_embedding: str = "rope",
        normalization: str = "rms_norm",
        normalization_eps: float = 1.0e-6,
        attention_residual: bool = True,
        ffn_residual: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if len(input_dim) != 2:
            raise ValueError("InterleavedCausalAttentionCell requires observation and action inputs.")
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads.")
        head_dim = output_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE.")
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported activation '{activation}'.")
        if position_embedding not in {"rope", "none"}:
            raise ValueError("position_embedding must be 'rope' or 'none'.")
        if normalization not in {"rms_norm", "layer_norm"}:
            raise ValueError("normalization must be 'rms_norm' or 'layer_norm'.")

        self.input_dim = list(input_dim)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.position_embedding = position_embedding
        self.rope_base = rope_base
        self.attention_residual = attention_residual
        self.ffn_residual = ffn_residual
        self.observation_projection = nn.Linear(input_dim[0], output_dim)
        self.action_projection = nn.Linear(input_dim[1], output_dim)
        self.readout_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        norm_class = nn.RMSNorm if normalization == "rms_norm" else nn.LayerNorm
        self.attention_norm = norm_class(output_dim, eps=normalization_eps)
        self.qkv_projection = nn.Linear(output_dim, 3 * output_dim)
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.attention_dropout = dropout
        self.ffn_norm = norm_class(output_dim, eps=normalization_eps)
        hidden_dim = ffn_dim or 4 * output_dim
        activation_layer = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.readout_token, std=0.02)

    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        observations, actions = input
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("Interleaved inputs must have shapes [B, 5, Do] and [B, 4, Da].")
        if observations.shape[1] != 5 or actions.shape[1] != 4:
            raise ValueError(
                f"Expected 5 observations and 4 actions, got {observations.shape[1]} and {actions.shape[1]}."
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("Observation and action batch sizes must match.")

        observation_tokens = self.observation_projection(observations)
        action_tokens = self.action_projection(actions)
        history_tokens = torch.stack(
            tuple(token for index in range(4) for token in (observation_tokens[:, index], action_tokens[:, index])),
            dim=1,
        )
        tokens = torch.cat(
            (history_tokens, observation_tokens[:, -1:], self.readout_token.expand(observations.shape[0], -1, -1)),
            dim=1,
        )

        normalized = self.attention_norm(tokens)
        query, key, value = self.qkv_projection(normalized).reshape(
            observations.shape[0], tokens.shape[1], 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if self.position_embedding == "rope":
            query = self._apply_rope(query)
            key = self._apply_rope(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(observations.shape[0], tokens.shape[1], self.output_dim)
        attended = self.output_projection(attended)
        encoded = tokens + attended if self.attention_residual else attended
        ffn_output = self.ffn(self.ffn_norm(encoded))
        encoded = encoded + ffn_output if self.ffn_residual else ffn_output
        return encoded[:, -1]

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        frequency = 1.0 / (
            self.rope_base
            ** (torch.arange(0, self.head_dim, 2, device=tensor.device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(tensor.shape[-2], device=tensor.device, dtype=torch.float32)
        angles = torch.outer(positions, frequency).to(dtype=tensor.dtype)
        cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)
