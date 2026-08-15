# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.utils import check_nan


def test_check_nan_accepts_finite_environment_outputs():
    obs = TensorDict({"policy": torch.ones(4, 3), "critic": torch.zeros(4, 2)}, batch_size=[4])
    check_nan(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("observation", "observation group 'policy'"),
        ("rewards", "rewards returned"),
        ("dones", "dones returned"),
    ],
)
def test_check_nan_reports_the_first_bad_environment_output(target: str, message: str):
    obs = TensorDict({"policy": torch.ones(4, 3)}, batch_size=[4])
    rewards = torch.ones(4)
    dones = torch.zeros(4)
    if target == "observation":
        obs["policy"][1, 2] = torch.nan
    elif target == "rewards":
        rewards[2] = torch.nan
    else:
        dones[3] = torch.nan

    with pytest.raises(ValueError, match=message):
        check_nan(obs, rewards, dones)
