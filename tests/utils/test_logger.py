# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from rsl_rl.utils.logger import Logger


def _make_logger(rnd: bool = False) -> Logger:
    logger = Logger(
        log_dir=None,
        cfg={"algorithm": {"rnd_cfg": {"weight": 1.0} if rnd else None}},
        env_cfg={},
        num_envs=3,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    # process_env_step intentionally skips all bookkeeping when logging is
    # disabled. A sentinel is sufficient for testing the buffering path.
    logger.writer = object()  # type: ignore
    return logger


def test_episode_statistics_are_buffered_in_step_order():
    logger = _make_logger()

    logger.process_env_step(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        dones=torch.tensor([False, True, False]),
        extras={},
    )
    logger.process_env_step(
        rewards=torch.tensor([4.0, 5.0, 6.0]),
        dones=torch.tensor([True, False, True]),
        extras={},
    )

    assert list(logger.rewbuffer) == []
    logger._flush_episode_buffers()

    assert list(logger.rewbuffer) == [2.0, 5.0, 9.0]
    assert list(logger.lenbuffer) == [1.0, 2.0, 2.0]
    torch.testing.assert_close(logger.cur_reward_sum, torch.tensor([0.0, 5.0, 0.0]))
    torch.testing.assert_close(logger.cur_episode_length, torch.tensor([0.0, 1.0, 0.0]))
    assert logger._episode_done_masks == []


def test_intrinsic_and_extrinsic_rewards_are_flushed_together():
    logger = _make_logger(rnd=True)

    logger.process_env_step(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        dones=torch.tensor([False, True, False]),
        extras={},
        intrinsic_rewards=torch.tensor([0.5, 1.5, 2.5]),
    )
    logger._flush_episode_buffers()

    assert list(logger.rewbuffer) == [3.5]
    assert list(logger.erewbuffer) == [2.0]
    assert list(logger.irewbuffer) == [1.5]


def test_non_writer_rank_does_not_accumulate_rollout_lists():
    logger = _make_logger()
    logger.writer = None

    for _ in range(100):
        logger.process_env_step(
            rewards=torch.ones(3),
            dones=torch.tensor([False, True, False]),
            extras={},
        )

    assert logger._episode_done_masks == []
    assert logger._episode_rewards == []
    assert logger._episode_lengths == []
