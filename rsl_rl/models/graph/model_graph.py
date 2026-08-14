from __future__ import annotations

import copy
from collections import defaultdict, deque
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.graph.cells import (
    CommandSelfAttentionCell,
    CrossAttentionCell,
    InterleavedCausalAttentionCell,
    MLPCell,
    TemporalAttentionCell,
    TokenAddCell,
    TokenMergeCell,
    TokenMLPCell,
    TokenProjectionCell,
    TopologyProjectionCell,
)
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


@dataclass(frozen=True)
class _Route:
    source: str
    target: str


class ModelGraph(nn.Module):
    """A restricted acyclic Node/Cell/Route model.

    Version one supports named graph inputs, single-output nodes, identity routes,
    and concatenating multiple routes targeting the same node input.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        nodes: dict[str, dict],
        routes: list[dict],
        output: str,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        self.output_endpoint = output
        self._input_dims = self._validate_inputs(obs)
        self._routes = [_Route(**route) for route in routes]
        self._node_cfgs = copy.deepcopy(nodes)
        self._incoming, self._execution_order = self._compile_graph()

        self.obs_normalization = obs_normalization
        self.input_normalizers = nn.ModuleDict(
            {
                name: EmpiricalNormalization(dim) if obs_normalization else nn.Identity()
                for name, dim in self._input_dims.items()
            }
        )

        if distribution_cfg is not None:
            dist_cfg = distribution_cfg.copy()
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            graph_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            graph_output_dim = output_dim
        if not isinstance(graph_output_dim, int):
            raise ValueError("ModelGraph currently supports distributions with a scalar input dimension only.")

        self.nodes = nn.ModuleDict()
        self._node_input_modes: dict[str, str] = {}
        endpoint_dims = {f"inputs.{name}": dim for name, dim in self._input_dims.items()}
        for node_name in self._execution_order:
            sources = self._incoming[node_name]
            input_dim = sum(endpoint_dims[source] for source in sources)
            node_cfg = self._node_cfgs[node_name]
            cell_cfg = node_cfg.get("cell", node_cfg).copy()
            cell_class_name = cell_cfg.pop("class_name")
            input_mode = cell_cfg.pop("input_mode", "concat")
            if input_mode not in {"concat", "list"}:
                raise ValueError(f"Unsupported input_mode '{input_mode}' for node '{node_name}'.")
            self._node_input_modes[node_name] = input_mode
            cell_class = self._resolve_cell(cell_class_name)
            cell_input_dim = [endpoint_dims[source] for source in sources] if input_mode == "list" else input_dim
            node_output_dim = cell_cfg.pop("output_dim", None)
            endpoint = f"nodes.{node_name}.output"
            if node_output_dim is None:
                if endpoint == self.output_endpoint:
                    node_output_dim = graph_output_dim
                elif hasattr(cell_class, "infer_output_dim"):
                    node_output_dim = cell_class.infer_output_dim(cell_input_dim, **cell_cfg)
                else:
                    raise ValueError(f"Node '{node_name}' must define cell.output_dim because it is not graph output.")
            self.nodes[node_name] = cell_class(input_dim=cell_input_dim, output_dim=node_output_dim, **cell_cfg)
            endpoint_dims[endpoint] = node_output_dim

        if self.output_endpoint not in endpoint_dims:
            raise ValueError(f"Graph output endpoint '{self.output_endpoint}' does not exist.")
        if endpoint_dims[self.output_endpoint] != graph_output_dim:
            raise ValueError(
                f"Graph output dimension {endpoint_dims[self.output_endpoint]} does not match required dimension"
                f" {graph_output_dim}."
            )
        self.output_dim = output_dim
        if self.distribution is not None:
            output_cell = self.nodes[self.output_endpoint.split(".")[1]]
            self.distribution.init_mlp_weights(getattr(output_cell, "mlp", output_cell))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        tensors = {
            f"inputs.{name}": self.input_normalizers[name](self._input_tensor(obs[name], name))
            for name in self._input_dims
        }
        for node_name in self._execution_order:
            parts = [tensors[source] for source in self._incoming[node_name]]
            if self._node_input_modes[node_name] == "list":
                node_input = parts
            else:
                node_input = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            tensors[f"nodes.{node_name}.output"] = self.nodes[node_name](node_input)
        graph_output = tensors[self.output_endpoint]
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(graph_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(graph_output)
        return graph_output

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            for name in self._input_dims:
                self.input_normalizers[name].update(self._input_tensor(obs[name], name))  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore

    def _validate_inputs(self, obs: TensorDict) -> dict[str, int]:
        dims = {}
        for name in self.obs_groups:
            if name not in obs:
                raise ValueError(f"Graph input observation group '{name}' is missing.")
            input_tensor = self._input_tensor(obs[name], name)
            if input_tensor.ndim not in (2, 3):
                raise ValueError(
                    f"Graph input '{name}' must be rank 2 or 3, got shape {tuple(input_tensor.shape)}."
                )
            dims[name] = input_tensor.shape[-1]
        return dims

    @staticmethod
    def _input_tensor(value, name: str) -> torch.Tensor:
        """Unwrap a non-concatenated observation group containing one term."""
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "keys"):
            keys = list(value.keys())
            if len(keys) == 1:
                tensor = value[keys[0]]
                if isinstance(tensor, torch.Tensor):
                    return tensor
        raise ValueError(
            f"Graph input observation group '{name}' must be a tensor or contain exactly one tensor term."
        )

    def _compile_graph(self) -> tuple[dict[str, list[str]], list[str]]:
        incoming: dict[str, list[str]] = defaultdict(list)
        dependencies: dict[str, set[str]] = {name: set() for name in self._node_cfgs}
        consumers: dict[str, set[str]] = defaultdict(set)
        for route in self._routes:
            target_parts = route.target.split(".")
            if len(target_parts) != 3 or target_parts[0] != "nodes" or target_parts[2] != "input":
                raise ValueError(f"Invalid route target '{route.target}'; expected nodes.<name>.input.")
            target_node = target_parts[1]
            if target_node not in self._node_cfgs:
                raise ValueError(f"Route target node '{target_node}' does not exist.")
            source_parts = route.source.split(".")
            if len(source_parts) == 2 and source_parts[0] == "inputs":
                if source_parts[1] not in self._input_dims:
                    raise ValueError(f"Route source input '{source_parts[1]}' is not declared for this model.")
            elif len(source_parts) == 3 and source_parts[0] == "nodes" and source_parts[2] == "output":
                source_node = source_parts[1]
                if source_node not in self._node_cfgs:
                    raise ValueError(f"Route source node '{source_node}' does not exist.")
                dependencies[target_node].add(source_node)
                consumers[source_node].add(target_node)
            else:
                raise ValueError(f"Invalid route source '{route.source}'.")
            incoming[target_node].append(route.source)
        missing = [name for name in self._node_cfgs if not incoming[name]]
        if missing:
            raise ValueError(f"Graph nodes without an input route: {missing}.")
        ready = deque(name for name, deps in dependencies.items() if not deps)
        order = []
        while ready:
            name = ready.popleft()
            order.append(name)
            for consumer in consumers[name]:
                dependencies[consumer].remove(name)
                if not dependencies[consumer]:
                    ready.append(consumer)
        if len(order) != len(self._node_cfgs):
            raise ValueError("Model graph contains a cycle.")
        return dict(incoming), order

    @staticmethod
    def _resolve_cell(class_name: str):
        if class_name == "MLPCell":
            return MLPCell
        if class_name == "TokenProjectionCell":
            return TokenProjectionCell
        if class_name == "TokenMLPCell":
            return TokenMLPCell
        if class_name == "TopologyProjectionCell":
            return TopologyProjectionCell
        if class_name == "TokenAddCell":
            return TokenAddCell
        if class_name == "TokenMergeCell":
            return TokenMergeCell
        if class_name == "TemporalAttentionCell":
            return TemporalAttentionCell
        if class_name == "CommandSelfAttentionCell":
            return CommandSelfAttentionCell
        if class_name == "InterleavedCausalAttentionCell":
            return InterleavedCausalAttentionCell
        if class_name == "CrossAttentionCell":
            return CrossAttentionCell
        return resolve_callable(class_name)
