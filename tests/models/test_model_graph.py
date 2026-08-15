# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import ModelGraph
from rsl_rl.models.graph.cells import TrackingAttentionBlock, TokenMergeCell
from rsl_rl.models.mlp_model import MLPModel


def _make_graph(obs, nodes=None, routes=None, output="nodes.decoder.output"):
    return ModelGraph(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        nodes=nodes or {"decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16, 8], "activation": "elu"}}},
        routes=routes or [{"source": "inputs.policy", "target": "nodes.decoder.input"}],
        output=output,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )


def test_single_mlp_node_model_protocol():
    obs = TensorDict({"policy": torch.randn(7, 11)}, batch_size=[7])
    graph = _make_graph(obs)

    deterministic = graph(obs)
    stochastic = graph(obs, stochastic_output=True)
    loss = -graph.get_output_log_prob(stochastic).mean()
    loss.backward()

    assert deterministic.shape == (7, 3)
    assert stochastic.shape == (7, 3)
    assert graph.output_mean.shape == (7, 3)
    assert graph.output_entropy.shape == (7,)
    assert any(parameter.grad is not None for parameter in graph.parameters())


def test_linear_output_cell():
    obs = TensorDict({"policy": torch.randn(7, 11)}, batch_size=[7])
    graph = _make_graph(
        obs,
        nodes={"decoder": {"cell": {"class_name": "LinearCell"}}},
    )

    output = graph(obs)
    output.square().mean().backward()

    assert output.shape == (7, 3)
    assert graph.nodes["decoder"].linear.weight.shape == (3, 11)
    assert graph.nodes["decoder"].linear.weight.grad is not None


def test_single_mlp_node_matches_mlp_model():
    obs = TensorDict({"policy": torch.randn(7, 11)}, batch_size=[7])
    graph = _make_graph(obs)
    mlp = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        hidden_dims=[16, 8],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    graph.nodes["decoder"].mlp.load_state_dict(mlp.mlp.state_dict())

    torch.testing.assert_close(graph(obs), mlp(obs))


def test_multiple_routes_are_concatenated_in_declaration_order():
    obs = TensorDict({"left": torch.randn(4, 2), "right": torch.randn(4, 5)}, batch_size=[4])
    graph = ModelGraph(
        obs,
        {"actor": ["left", "right"]},
        "actor",
        3,
        nodes={"decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [8]}}},
        routes=[
            {"source": "inputs.left", "target": "nodes.decoder.input"},
            {"source": "inputs.right", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )
    assert graph(obs).shape == (4, 3)
    assert graph.nodes["decoder"].input_dim == 7


def test_graph_rejects_cycles():
    obs = TensorDict({"policy": torch.randn(2, 4)}, batch_size=[2])
    with pytest.raises(ValueError, match="cycle"):
        _make_graph(
            obs,
            nodes={
                "a": {"cell": {"class_name": "MLPCell", "hidden_dims": [4], "output_dim": 4}},
                "b": {"cell": {"class_name": "MLPCell", "hidden_dims": [4]}},
            },
            routes=[
                {"source": "nodes.b.output", "target": "nodes.a.input"},
                {"source": "nodes.a.output", "target": "nodes.b.input"},
            ],
            output="nodes.b.output",
        )


def test_temporal_attention_and_command_branches():
    obs = TensorDict({"policy": torch.randn(5, 10, 12), "command": torch.randn(5, 7)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["policy", "command"]},
        "actor",
        3,
        nodes={
            "temporal_encoder": {
                "cell": {
                    "class_name": "TemporalAttentionCell",
                    "output_dim": 32,
                    "num_heads": 4,
                    "ffn_dim": 64,
                    "position_embedding": "rope",
                    "normalization": "rms_norm",
                    "normalization_eps": 1.0e-6,
                    "attention_residual": True,
                    "ffn_residual": True,
                }
            },
            "command_encoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16], "output_dim": 16}},
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [32, 16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.temporal_encoder.input"},
            {"source": "inputs.command", "target": "nodes.command_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.decoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    output = graph(obs)
    output.square().mean().backward()
    assert output.shape == (5, 3)
    assert graph.nodes["decoder"].input_dim == 48
    assert graph.nodes["temporal_encoder"].readout_token.grad is not None


def test_temporal_attention_without_ffn():
    obs = TensorDict({"policy": torch.randn(5, 10, 12)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        nodes={
            "temporal_encoder": {
                "cell": {
                    "class_name": "TemporalAttentionCell",
                    "output_dim": 32,
                    "num_heads": 4,
                    "use_ffn": False,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.temporal_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    temporal_encoder = graph.nodes["temporal_encoder"]
    assert not hasattr(temporal_encoder, "ffn")
    output = graph(obs)
    assert output.shape == (5, 3)
    output.square().mean().backward()


def test_token_projection_before_temporal_attention():
    obs = TensorDict({"policy": torch.randn(5, 5, 12)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        nodes={
            "policy_projection": {"cell": {"class_name": "TokenProjectionCell", "output_dim": 128}},
            "temporal_encoder": {
                "cell": {
                    "class_name": "TemporalAttentionCell",
                    "output_dim": 32,
                    "num_heads": 4,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.policy_projection.input"},
            {"source": "nodes.policy_projection.output", "target": "nodes.temporal_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    output = graph(obs)
    output.square().mean().backward()

    assert output.shape == (5, 3)
    assert graph.nodes["policy_projection"].projection.weight.shape == (128, 12)
    assert graph.nodes["temporal_encoder"].input_dim == 128


def test_cross_attention_from_temporal_query_to_command_tokens():
    obs = TensorDict({"policy": torch.randn(5, 10, 12), "command": torch.randn(5, 14, 81)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["policy", "command"]},
        "actor",
        3,
        nodes={
            "temporal_encoder": {"cell": {"class_name": "TemporalAttentionCell", "output_dim": 32, "num_heads": 4}},
            "command_projection": {
                "cell": {
                    "class_name": "TokenMLPCell",
                    "hidden_dims": [64],
                    "output_dim": 32,
                }
            },
            "topology_projection": {
                "cell": {
                    "class_name": "TopologyProjectionCell",
                    "output_dim": 32,
                    "topology": [[0, 0, 0]] * 14,
                }
            },
            "command_token_adder": {"cell": {"class_name": "TokenAddCell", "input_mode": "list", "output_dim": 32}},
            "command_encoder": {
                "cell": {
                    "class_name": "CrossAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                    "ffn_dim": 64,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.temporal_encoder.input"},
            {"source": "inputs.command", "target": "nodes.command_projection.input"},
            {"source": "inputs.command", "target": "nodes.topology_projection.input"},
            {"source": "nodes.command_projection.output", "target": "nodes.command_token_adder.input"},
            {"source": "nodes.topology_projection.output", "target": "nodes.command_token_adder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.command_encoder.input"},
            {"source": "nodes.command_token_adder.output", "target": "nodes.command_encoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    output = graph(obs)
    output.square().mean().backward()

    assert output.shape == (5, 3)
    assert graph.nodes["command_projection"].input_dim == 81
    assert graph.nodes["command_encoder"].input_dim == [32, 32]
    assert graph.nodes["command_encoder"].query_projection.weight.grad is not None
    assert graph.nodes["topology_projection"].projection.weight.grad is not None
    assert graph.nodes["temporal_encoder"].input_projection.weight.grad is not None


def test_command_self_attention_updates_only_commands_before_cross_attention():
    obs = TensorDict({"readout": torch.randn(5, 32), "command": torch.randn(5, 14, 32)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["readout", "command"]},
        "actor",
        3,
        nodes={
            "command_self_attention": {
                "cell": {
                    "class_name": "CommandSelfAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                    "ffn_dim": 64,
                }
            },
            "command_encoder": {
                "cell": {
                    "class_name": "CrossAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.readout", "target": "nodes.command_self_attention.input"},
            {"source": "inputs.command", "target": "nodes.command_self_attention.input"},
            {"source": "inputs.readout", "target": "nodes.command_encoder.input"},
            {"source": "nodes.command_self_attention.output", "target": "nodes.command_encoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    original_readout = obs["readout"].clone()
    updated_commands = graph.nodes["command_self_attention"]([obs["readout"], obs["command"]])
    assert updated_commands.shape == (5, 14, 32)
    torch.testing.assert_close(obs["readout"], original_readout)
    output = graph(obs)
    output.square().mean().backward()
    assert output.shape == (5, 3)
    assert graph.nodes["command_self_attention"].key_value_projection.weight.grad is not None


def test_command_self_attention_without_ffn():
    obs = TensorDict({"readout": torch.randn(5, 32), "command": torch.randn(5, 14, 32)}, batch_size=[5])
    graph = ModelGraph(
        obs,
        {"actor": ["readout", "command"]},
        "actor",
        3,
        nodes={
            "command_self_attention": {
                "cell": {
                    "class_name": "CommandSelfAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                    "use_ffn": False,
                }
            },
            "command_encoder": {
                "cell": {
                    "class_name": "CrossAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.readout", "target": "nodes.command_self_attention.input"},
            {"source": "inputs.command", "target": "nodes.command_self_attention.input"},
            {"source": "inputs.readout", "target": "nodes.command_encoder.input"},
            {
                "source": "nodes.command_self_attention.output",
                "target": "nodes.command_encoder.input",
            },
            {"source": "nodes.command_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    command_self_attention = graph.nodes["command_self_attention"]
    assert not hasattr(command_self_attention, "ffn")
    output = graph(obs)
    assert output.shape == (5, 3)
    output.square().mean().backward()
    assert command_self_attention.key_value_projection.weight.grad is not None


def test_tracking_attention_blocks_preserve_tokens_and_stack():
    batch_size = 5
    feature_dim = 32
    temporal_tokens = torch.randn(batch_size, 9, feature_dim)
    raw_command_tokens = torch.randn(batch_size, 14, feature_dim)
    original_raw_command_tokens = raw_command_tokens.clone()
    block = TrackingAttentionBlock(feature_dim=feature_dim, num_heads=4, ffn_dim=64)

    block_output = block(temporal_tokens, raw_command_tokens)
    readout_output = block.forward_readout(temporal_tokens, raw_command_tokens)

    assert block_output.shape == (batch_size, 9, feature_dim)
    torch.testing.assert_close(readout_output, block_output[:, -1])
    torch.testing.assert_close(raw_command_tokens, original_raw_command_tokens)
    assert not hasattr(block.temporal_self_attention, "input_projection")
    assert isinstance(block.command_self_attention.readout_projection, torch.nn.Identity)
    assert isinstance(block.command_self_attention.command_projection, torch.nn.Identity)
    assert not hasattr(block.cross_attention, "query_input_projection")
    assert not hasattr(block.cross_attention, "context_input_projection")

    obs = TensorDict(
        {"temporal": temporal_tokens, "command": raw_command_tokens},
        batch_size=[batch_size],
    )
    graph = ModelGraph(
        obs,
        {"actor": ["temporal", "command"]},
        "actor",
        3,
        nodes={
            "attention_blocks": {
                "cell": {
                    "class_name": "StackedTrackingAttentionCell",
                    "input_mode": "list",
                    "output_dim": feature_dim,
                    "num_blocks": 3,
                    "num_heads": 4,
                    "ffn_dim": 64,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.temporal", "target": "nodes.attention_blocks.input"},
            {"source": "inputs.command", "target": "nodes.attention_blocks.input"},
            {"source": "nodes.attention_blocks.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    output = graph(obs)
    output.square().mean().backward()

    attention_stack = graph.nodes["attention_blocks"]
    stacked_tokens = torch.cat((temporal_tokens, attention_stack.readout_token.expand(batch_size, -1, -1)), dim=1)
    for attention_block in attention_stack.blocks:
        stacked_tokens = attention_block(stacked_tokens, raw_command_tokens)
    expected_readout = attention_stack.final_norm(stacked_tokens)[:, -1]
    optimized_readout = attention_stack([temporal_tokens, raw_command_tokens])

    torch.testing.assert_close(optimized_readout, expected_readout)
    assert output.shape == (batch_size, 3)
    assert len(attention_stack.blocks) == 3
    assert attention_stack.residual_scale == pytest.approx(1.0 / (6.0**0.5))
    assert attention_stack.readout_token.grad is not None
    assert attention_stack.final_norm.weight.grad is not None
    assert all(block.cross_attention.ffn.output.weight.grad is not None for block in attention_stack.blocks)


def test_projected_observation_and_action_tokens_are_interleaved():
    obs = TensorDict(
        {
            "policy": torch.randn(5, 5, 12),
            "action": torch.randn(5, 4, 6),
        },
        batch_size=[5],
    )
    graph = ModelGraph(
        obs,
        {"actor": ["policy", "action"]},
        "actor",
        3,
        nodes={
            "policy_projection": {"cell": {"class_name": "TokenProjectionCell", "output_dim": 128}},
            "action_projection": {"cell": {"class_name": "TokenProjectionCell", "output_dim": 128}},
            "token_interleaver": {
                "cell": {
                    "class_name": "TokenMergeCell",
                    "input_mode": "list",
                    "mode": "interleave",
                    "dim": 1,
                }
            },
            "temporal_encoder": {
                "cell": {
                    "class_name": "TemporalAttentionCell",
                    "output_dim": 32,
                    "num_heads": 4,
                }
            },
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.policy_projection.input"},
            {"source": "inputs.action", "target": "nodes.action_projection.input"},
            {
                "source": "nodes.policy_projection.output",
                "target": "nodes.token_interleaver.input",
            },
            {
                "source": "nodes.action_projection.output",
                "target": "nodes.token_interleaver.input",
            },
            {
                "source": "nodes.token_interleaver.output",
                "target": "nodes.temporal_encoder.input",
            },
            {"source": "nodes.temporal_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    policy_tokens = graph.nodes["policy_projection"](obs["policy"])
    action_tokens = graph.nodes["action_projection"](obs["action"])
    interleaved = graph.nodes["token_interleaver"]([policy_tokens, action_tokens])
    torch.testing.assert_close(interleaved[:, 0::2], policy_tokens)
    torch.testing.assert_close(interleaved[:, 1::2], action_tokens)

    output = graph(obs)
    output.square().mean().backward()

    assert output.shape == (5, 3)
    assert interleaved.shape == (5, 9, 128)
    assert graph.nodes["policy_projection"].projection.weight.grad is not None
    assert graph.nodes["action_projection"].projection.weight.grad is not None


def test_token_merge_mode_and_dimension_are_configurable():
    first = torch.randn(5, 4, 12)
    second = torch.randn(5, 4, 6)
    merge = TokenMergeCell(
        input_dim=[12, 6],
        output_dim=18,
        mode="concatenate",
        dim=-1,
    )

    torch.testing.assert_close(merge([first, second]), torch.cat((first, second), dim=-1))


def test_interleaved_causal_attention_with_non_concatenated_action_group():
    obs = TensorDict(
        {
            "policy": torch.randn(5, 5, 12),
            "last_action": TensorDict({"actions": torch.randn(5, 4, 6)}, batch_size=[5]),
            "command": torch.randn(5, 7),
        },
        batch_size=[5],
    )
    graph = ModelGraph(
        obs,
        {"actor": ["policy", "last_action", "command"]},
        "actor",
        3,
        nodes={
            "temporal_encoder": {
                "cell": {
                    "class_name": "InterleavedCausalAttentionCell",
                    "input_mode": "list",
                    "output_dim": 32,
                    "num_heads": 4,
                    "ffn_dim": 64,
                }
            },
            "command_encoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16], "output_dim": 16}},
            "decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [32, 16]}},
        },
        routes=[
            {"source": "inputs.policy", "target": "nodes.temporal_encoder.input"},
            {"source": "inputs.last_action", "target": "nodes.temporal_encoder.input"},
            {"source": "inputs.command", "target": "nodes.command_encoder.input"},
            {"source": "nodes.temporal_encoder.output", "target": "nodes.decoder.input"},
            {"source": "nodes.command_encoder.output", "target": "nodes.decoder.input"},
        ],
        output="nodes.decoder.output",
    )

    output = graph(obs)
    output.square().mean().backward()

    assert output.shape == (5, 3)
    assert graph.nodes["decoder"].input_dim == 48
    assert graph.nodes["temporal_encoder"].input_dim == [12, 6]
    assert graph.nodes["temporal_encoder"].readout_token.grad is not None
