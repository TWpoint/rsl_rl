# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPO algorithm."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _build_ppo(**overrides: object) -> tuple[PPO, TensorDict]:
    """Build a PPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    critic = _make_critic(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=0.01,
    )
    defaults.update(overrides)
    ppo = PPO(actor, critic, storage, **defaults)
    return ppo, obs


class TestGAEComputation:
    """Tests for generalized advantage estimation in ``compute_returns``."""

    def test_gae_returns_hand_computed(self) -> None:
        """Verify GAE returns match a hand-computed example with known rewards, values, and dones."""
        num_envs, num_steps = 1, 3
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [0.0, 0.0, 0.0]

        for i in range(num_steps):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), values[i])
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 2.0)
        # Manually compute GAE (backward pass)
        # Step 2: delta = r2 + gamma * V_last - V2 = 3.0 + 0.99*2.0 - 1.5 = 3.48
        #          adv2 = 3.48
        # Step 1: delta = r1 + gamma * V2 - V1 = 2.0 + 0.99*1.5 - 1.0 = 2.485
        #          adv1 = 2.485 + gamma*lam*adv2 = 2.485 + 0.99*0.95*3.48 = 2.485 + 3.27294 = 5.75794
        # Step 0: delta = r0 + gamma * V1 - V0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        #          adv0 = 1.49 + gamma*lam*adv1 = 1.49 + 0.99*0.95*5.75794 = 1.49 + 5.41484... = 6.90484...
        expected_adv = [
            1.49 + 0.99 * 0.95 * (2.485 + 0.99 * 0.95 * 3.48),
            2.485 + 0.99 * 0.95 * 3.48,
            3.48,
        ]
        expected_returns = [expected_adv[i] + values[i] for i in range(3)]

        # Use the actual critic to produce last_values override
        with torch.no_grad():
            storage.values[0] = torch.full((num_envs, 1), values[0])
            storage.values[1] = torch.full((num_envs, 1), values[1])
            storage.values[2] = torch.full((num_envs, 1), values[2])

        # Call compute_returns with a custom last_values by monkeypatching critic
        original_critic_call = ppo.critic.forward
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)
        ppo.critic.forward = original_critic_call

        for i in range(num_steps):
            assert torch.allclose(
                storage.returns[i, 0, 0],
                torch.tensor(expected_returns[i]),
                atol=1e-4,
            ), f"Return mismatch at step {i}: got {storage.returns[i, 0, 0].item()}, expected {expected_returns[i]}"

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """When a done flag is set, the advantage should not bootstrap from the next value."""
        num_envs, num_steps = 1, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        # Step 0: done=True, so step 1 is a fresh episode
        for i, (r, v, d) in enumerate([(1.0, 0.5, 1.0), (2.0, 1.0, 0.0)]):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), v)
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), r)
            t.dones = torch.full((num_envs,), d)
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 3.0)
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)

        # Step 0: done=True, so next_is_not_terminal = 0
        # delta0 = r0 - V0 = 1.0 - 0.5 = 0.5 (no bootstrap because done)
        # Step 1: delta1 = r1 + gamma * V_last - V1 = 2.0 + 0.99*3.0 - 1.0 = 3.97
        # adv1 = 3.97
        # adv0 = 0.5 (no bootstrap because done at step 0)
        expected_return_0 = 0.5 + 0.5  # adv0 + V0
        expected_return_1 = 3.97 + 1.0  # adv1 + V1

        assert torch.allclose(storage.returns[0, 0, 0], torch.tensor(expected_return_0), atol=1e-4)
        assert torch.allclose(storage.returns[1, 0, 0], torch.tensor(expected_return_1), atol=1e-4)

    def test_advantage_normalization_global(self) -> None:
        """With normalize_advantage_per_mini_batch=False, advantages should have mean~0, std~1."""
        ppo, obs = _build_ppo(normalize_advantage_per_mini_batch=False)

        for _ in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = ppo.actor(obs, stochastic_output=True).detach()
            t.values = ppo.critic(obs).detach()
            t.actions_log_prob = ppo.actor.get_output_log_prob(t.actions).detach()
            t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            ppo.storage.add_transition(t)

        ppo.compute_returns(obs)

        adv = ppo.storage.advantages.flatten()
        assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"
        assert abs(adv.std().item() - 1.0) < 0.1, "Advantages should be unit-std"

    def test_advantage_normalization_uses_moments_from_all_ranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distributed normalization should use the combined advantage distribution."""
        ppo, _ = _build_ppo(
            normalize_advantage_per_mini_batch=False,
            multi_gpu_cfg={"global_rank": 0, "world_size": 2},
        )
        local_advantages = torch.tensor([1.0, 3.0])
        remote_advantages = torch.tensor([5.0, 7.0])

        remote_moments = torch.tensor(
            [
                remote_advantages.double().sum(),
                remote_advantages.double().square().sum(),
                remote_advantages.numel(),
            ],
            dtype=torch.float64,
        )

        def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
            assert op == torch.distributed.ReduceOp.SUM
            tensor.add_(remote_moments)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        normalized = ppo._normalize_advantages(local_advantages)
        global_advantages = torch.cat((local_advantages, remote_advantages))
        expected = (local_advantages - global_advantages.mean()) / (global_advantages.std() + 1e-8)

        assert torch.allclose(normalized, expected)


class TestTimeoutBootstrapping:
    """Tests for timeout bootstrapping in ``process_env_step``."""

    def test_timeout_adds_bootstrap_to_reward(self) -> None:
        """When time_outs is set, stored reward should include gamma * value * timeout."""
        ppo, obs = _build_ppo()

        # Manually act to populate transition.values
        ppo.act(obs)
        stored_values = ppo.transition.values.clone()

        raw_reward = torch.ones(NUM_ENVS)
        dones = torch.ones(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0  # Only env 0 times out

        ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        # The stored reward for env 0 should be: 1.0 + gamma * value[0]
        stored_reward_env0 = ppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + ppo.gamma * stored_values[0, 0].item()
        assert abs(stored_reward_env0 - expected) < 1e-5

        # Env 1 should have raw reward only
        stored_reward_env1 = ppo.storage.rewards[0, 1, 0].item()
        assert abs(stored_reward_env1 - 1.0) < 1e-5


class TestPPOLosses:
    """Tests for PPO loss computation correctness."""

    def test_surrogate_loss_clipping(self) -> None:
        """When ratio deviates beyond clip_param, the clipped branch should dominate."""
        clip_param = 0.2

        # Construct a scenario: positive advantages, ratio > 1 + clip
        advantages = torch.tensor([1.0, 1.0, 1.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        # New log probs that give ratio = exp(0.5) ≈ 1.65, which is > 1 + 0.2
        new_log_probs = torch.tensor([0.5, 0.5, 0.5])

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        loss = torch.max(surrogate, surrogate_clipped).mean()

        # The clipped branch should be -advantages * (1 + clip_param) = -1.2
        # The unclipped branch should be -advantages * 1.65 ≈ -1.65
        # max(-1.65, -1.2) = -1.2, so clipped branch dominates
        expected_clipped = (-advantages * (1.0 + clip_param)).mean()
        assert torch.allclose(loss, expected_clipped, atol=1e-5)

    def test_elementwise_value_loss_uses_worse_clipped_or_unclipped_error(self) -> None:
        """Clipped value loss should retain the worse candidate for every element."""
        ppo, _ = _build_ppo(clip_param=0.2, use_clipped_value_loss=True)
        old_values = torch.tensor([[1.0], [1.0]])
        new_values = torch.tensor([[2.0], [1.3]])
        returns = torch.tensor([[1.5], [2.0]])

        effective_loss = ppo._compute_elementwise_value_loss(new_values, old_values, returns)

        # Env 0: new=2.0, old=1.0, clipped_new=1.2
        #   unclipped: (2.0 - 1.5)^2 = 0.25
        #   clipped: (1.2 - 1.5)^2 = 0.09
        #   max = 0.25
        # Env 1: new=1.3, old=1.0, clipped_new=1.2
        #   unclipped: (1.3 - 2.0)^2 = 0.49
        #   clipped: (1.2 - 2.0)^2 = 0.64
        #   max = 0.64
        expected = torch.tensor([[0.25], [0.64]])
        assert torch.allclose(effective_loss, expected)
        assert effective_loss.mean().item() == pytest.approx((0.25 + 0.64) / 2)

    def test_elementwise_unclipped_value_loss_is_squared_error(self) -> None:
        """Disabling clipping should return ordinary squared errors without reduction."""
        ppo, _ = _build_ppo(use_clipped_value_loss=False)
        old_values = torch.tensor([[100.0, -100.0], [4.0, 5.0]])
        new_values = torch.tensor([[2.0, -1.0], [0.5, 4.0]])
        returns = torch.tensor([[1.5, 1.0], [2.5, 3.0]])

        effective_loss = ppo._compute_elementwise_value_loss(new_values, old_values, returns)

        assert torch.equal(effective_loss, (new_values - returns).pow(2))
        assert effective_loss.shape == new_values.shape

    def test_default_value_loss_recording_hook_is_noop(self) -> None:
        """Base PPO should not add logging state when no subclass opts into the hook."""
        ppo, _ = _build_ppo()
        effective_loss = torch.tensor([[0.25], [0.64]])

        result = ppo._record_value_loss(effective_loss)

        assert result is None

    def test_update_mean_reduction_and_detached_recording_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Base update should average all value elements and pass a detached tensor to the hook."""
        ppo, obs = _build_ppo(num_learning_epochs=1, num_mini_batches=1)
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        ppo.compute_returns(obs)

        recorded_losses: list[torch.Tensor] = []

        def fake_elementwise_loss(
            values: torch.Tensor,
            old_values: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            assert values.shape == old_values.shape == returns.shape
            return values * 0.0 + 3.25

        def record_value_loss(effective_value_loss: torch.Tensor) -> None:
            recorded_losses.append(effective_value_loss)

        monkeypatch.setattr(ppo, "_compute_elementwise_value_loss", fake_elementwise_loss)
        monkeypatch.setattr(ppo, "_record_value_loss", record_value_loss)

        losses = ppo.update()

        assert losses["value"] == pytest.approx(3.25)
        assert len(recorded_losses) == 1
        assert torch.equal(recorded_losses[0], torch.full_like(recorded_losses[0], 3.25))
        assert not recorded_losses[0].requires_grad
        assert recorded_losses[0].grad_fn is None


class TestGradientClipping:
    """Tests for joint actor and critic gradient clipping."""

    def test_actor_and_critic_use_one_joint_global_norm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All PPO parameters should be passed to one clip_grad_norm_ call."""
        ppo, _ = _build_ppo(max_grad_norm=0.1)
        calls: list[tuple[list[torch.nn.Parameter], float]] = []

        def fake_clip_grad_norm(parameters: object, max_norm: float) -> torch.Tensor:
            calls.append((list(parameters), max_norm))  # type: ignore[arg-type]
            return torch.tensor(0.2)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm)

        total_norm = ppo._clip_gradients()

        expected_parameters = list(ppo.actor.parameters()) + list(ppo.critic.parameters())
        assert len(calls) == 1
        assert [id(parameter) for parameter in calls[0][0]] == [id(parameter) for parameter in expected_parameters]
        assert calls[0][1] == 0.1
        assert total_norm.item() == pytest.approx(0.2)


class TestAdaptiveLearningRate:
    """Tests for adaptive KL-based learning rate scheduling."""

    def test_default_lr_bounds_match_legacy_values(self) -> None:
        """Default bounds should preserve the previous hard-coded scheduler range."""
        ppo, _obs = _build_ppo()

        assert ppo.learning_rate_min == 1e-5
        assert ppo.learning_rate_max == 1e-2

    def test_lr_decreases_when_kl_too_high(self) -> None:
        """LR should decrease when KL > 2 * desired_kl."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        ppo._adapt_learning_rate(torch.tensor(0.03))  # > 2 * 0.01

        assert ppo.learning_rate < initial_lr
        assert ppo.learning_rate == max(1e-5, initial_lr / 1.5)
        assert ppo.optimizer.param_groups[0]["lr"] == ppo.learning_rate

    def test_lr_increases_when_kl_too_low(self) -> None:
        """LR should increase when 0 < KL < desired_kl / 2."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        ppo._adapt_learning_rate(torch.tensor(0.002))  # < 0.01 / 2 = 0.005

        assert ppo.learning_rate > initial_lr
        assert ppo.learning_rate == min(1e-2, initial_lr * 1.5)
        assert ppo.optimizer.param_groups[0]["lr"] == ppo.learning_rate

    def test_lr_unchanged_in_stable_range(self) -> None:
        """LR should remain unchanged when KL is in [desired_kl/2, 2*desired_kl]."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        ppo._adapt_learning_rate(torch.tensor(0.01))  # Exactly desired_kl — in stable range

        assert ppo.learning_rate == initial_lr

    @pytest.mark.parametrize(
        ("learning_rate", "kl_mean", "expected_learning_rate"),
        [
            (2.5e-4, 0.03, 2e-4),
            (7e-4, 0.002, 8e-4),
        ],
    )
    def test_custom_lr_bounds_are_enforced(
        self, learning_rate: float, kl_mean: float, expected_learning_rate: float
    ) -> None:
        """Adaptive scheduling should clamp the actor rate to configured bounds."""
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            learning_rate=learning_rate,
            learning_rate_min=2e-4,
            learning_rate_max=8e-4,
        )

        ppo._adapt_learning_rate(torch.tensor(kl_mean))

        assert ppo.learning_rate == expected_learning_rate
        assert ppo.actor_learning_rate == expected_learning_rate
        assert ppo.critic_learning_rate == expected_learning_rate
        assert ppo.optimizer.param_groups[0]["lr"] == expected_learning_rate

    @pytest.mark.parametrize(
        ("learning_rate_min", "learning_rate_max", "match"),
        [
            (0.0, 1e-2, "learning_rate_min must be positive"),
            (-1e-5, 1e-2, "learning_rate_min must be positive"),
            (1e-5, 0.0, "learning_rate_max must be positive"),
            (1e-5, -1e-2, "learning_rate_max must be positive"),
            (1e-2, 1e-5, "learning_rate_min must be less than or equal"),
        ],
    )
    def test_invalid_lr_bounds_raise_value_error(
        self, learning_rate_min: float, learning_rate_max: float, match: str
    ) -> None:
        """Adaptive learning-rate bounds must be positive and ordered."""
        with pytest.raises(ValueError, match=match):
            _build_ppo(learning_rate_min=learning_rate_min, learning_rate_max=learning_rate_max)


class TestSeparateLearningRates:
    """Tests for optional actor/critic optimizer parameter groups."""

    def test_legacy_learning_rate_keeps_single_parameter_group(self) -> None:
        ppo, _obs = _build_ppo(learning_rate=3e-4)

        assert len(ppo.optimizer.param_groups) == 1
        assert ppo.actor_learning_rate == 3e-4
        assert ppo.critic_learning_rate == 3e-4
        assert ppo.optimizer.param_groups[0]["lr"] == 3e-4

    def test_separate_learning_rates_create_actor_and_critic_groups(self) -> None:
        ppo, _obs = _build_ppo(actor_learning_rate=5e-5, critic_learning_rate=5e-4)

        assert [group["name"] for group in ppo.optimizer.param_groups] == ["actor", "critic"]
        assert [group["lr"] for group in ppo.optimizer.param_groups] == [5e-5, 5e-4]
        actor_parameter_ids = {id(parameter) for parameter in ppo.actor.parameters()}
        critic_parameter_ids = {id(parameter) for parameter in ppo.critic.parameters()}
        assert {id(parameter) for parameter in ppo.optimizer.param_groups[0]["params"]} == actor_parameter_ids
        assert {id(parameter) for parameter in ppo.optimizer.param_groups[1]["params"]} == critic_parameter_ids

    def test_adaptive_schedule_changes_only_split_actor_rate(self) -> None:
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            actor_learning_rate=5e-5,
            critic_learning_rate=5e-4,
        )

        ppo._adapt_learning_rate(torch.tensor(0.03))

        assert ppo.actor_learning_rate == 5e-5 / 1.5
        assert ppo.critic_learning_rate == 5e-4
        assert ppo.learning_rate == ppo.actor_learning_rate
        assert ppo.optimizer.param_groups[0]["lr"] == ppo.actor_learning_rate
        assert ppo.optimizer.param_groups[1]["lr"] == 5e-4

    def test_shared_adaptation_applies_common_rate_to_all_groups(self) -> None:
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            actor_learning_rate=2e-5,
            critic_learning_rate=1e-3,
            shared_kl_adaptation=True,
        )

        ppo._adapt_learning_rate(torch.tensor(0.01))

        assert ppo.learning_rate == 2e-5
        assert ppo.actor_learning_rate == 2e-5
        assert ppo.critic_learning_rate == 2e-5
        assert [group["lr"] for group in ppo.optimizer.param_groups] == [2e-5, 2e-5]

    def test_shared_kl_matches_stabilized_reference_expression(self) -> None:
        ppo, _obs = _build_ppo(shared_kl_adaptation=True)
        old_mean = torch.tensor([[0.2, -0.1]])
        old_std = torch.tensor([[0.05, 0.1]])
        new_mean = torch.tensor([[0.3, -0.2]])
        new_std = torch.tensor([[0.08, 0.12]])

        actual = ppo._compute_adaptive_kl((old_mean, old_std), (new_mean, new_std))
        expected = torch.sum(
            torch.log(new_std / old_std + 1.0e-5)
            + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
            - 0.5,
            dim=-1,
        )

        assert torch.equal(actual, expected)

    def test_unspecified_side_falls_back_to_legacy_rate(self) -> None:
        ppo, _obs = _build_ppo(learning_rate=1e-3, actor_learning_rate=5e-5)

        assert ppo.actor_learning_rate == 5e-5
        assert ppo.critic_learning_rate == 1e-3


class TestFusedOptimizer:
    """Tests for the optional CUDA fused optimizer path."""

    def test_fused_optimizer_requires_cuda(self) -> None:
        with pytest.raises(ValueError, match="requires a CUDA device"):
            _build_ppo(optimizer_fused=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused Adam")
    def test_fused_optimizer_supports_separate_learning_rates(self) -> None:
        ppo, _obs = _build_ppo(
            device="cuda:0",
            optimizer_fused=True,
            actor_learning_rate=5e-5,
            critic_learning_rate=5e-4,
        )

        assert ppo.optimizer.defaults["fused"] is True
        assert [group["lr"] for group in ppo.optimizer.param_groups] == [5e-5, 5e-4]
