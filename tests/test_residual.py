import os

import pytest

if not os.environ.get("HITSZ_ROOT"):
    pytest.skip("Set HITSZ_ROOT to run the native residual PPO tests", allow_module_level=True)

import torch
from tensordict import TensorDict

from ruma_ppo.training.hitsz_joint import GateActorCritic
from ruma_ppo.training.residual import ResidualGateActorCritic, apply_vertical_delta


def test_residual_starts_at_base_command_and_value():
    groups = {"policy": ["depth", "state"], "critic": ["depth", "state"]}
    base = GateActorCritic(None, groups, 3).requires_grad_(False)
    model = ResidualGateActorCritic(base, bound_mps=0.8, init_noise_std=0.1)
    model.train()
    assert not base.training
    assert all(not p.requires_grad for p in base.parameters())

    obs = TensorDict({
        "gate_future": torch.randn(4, 3), "base_action": torch.randn(4, 3),
        "base_value": torch.randn(4, 1), "state": torch.randn(4, 14), "gate": torch.randn(4, 6),
    }, batch_size=[4])
    delta = model.act_inference(obs)
    assert delta.shape == (4, 1)
    assert torch.equal(apply_vertical_delta(obs["base_action"], delta), obs["base_action"])
    assert torch.equal(model.evaluate(obs), obs["base_value"])

    with torch.no_grad():
        model.actor[-1].bias.fill_(100.0)
    delta = model.act_inference(obs)
    torch.testing.assert_close(delta, torch.full_like(delta, 0.8))
    command = apply_vertical_delta(obs["base_action"], delta)
    assert torch.equal(command[:, :2], obs["base_action"][:, :2])
    torch.testing.assert_close(command[:, 2:3], obs["base_action"][:, 2:3] + delta)
