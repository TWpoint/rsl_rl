import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import ModelGraph
from rsl_rl.models.mlp_model import MLPModel


def _make_graph(obs, nodes=None, routes=None, output="nodes.decoder.output"):
    return ModelGraph(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        nodes=nodes
        or {"decoder": {"cell": {"class_name": "MLPCell", "hidden_dims": [16, 8], "activation": "elu"}}},
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
