import os

import pytest

if not os.environ.get("HITSZ_ROOT"):
    pytest.skip("Set HITSZ_ROOT to run the native PPO tests", allow_module_level=True)

import torch
from tensordict import TensorDict

from ruma_ppo.training.hitsz_joint import (
    GateActorCritic,
    GateFutureConfig,
    OccupancyConfig,
    gate_future_features,
)


@pytest.mark.parametrize("batched_aperture", [False, True])
def test_gate_future_features(batched_aperture):
    position = torch.tensor([
        [-5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.15],
        [-5.0, 0.0, 9.0],
        [-5.0, 0.0, 0.0],
        [-5.0, 0.0, 2.4],
        [-5.0, 0.0, 1.0],
    ])
    velocity = torch.tensor([[10.0, 0.0, 0.0]] * 6)
    velocity[3, 0] = 0.0
    velocity[4, 2] = 2.0
    velocity[5, 0] = 0.5
    center = torch.zeros_like(position)
    basis = torch.eye(3).expand(6, -1, -1)
    active = torch.tensor([True, True, False, True, True, True])
    half_aperture = torch.tensor([3.0, 1.5])
    if batched_aperture:
        half_aperture = half_aperture.repeat(6, 1)
        half_aperture[5, 1] = 2.0
    features = gate_future_features(position, velocity, center, basis, half_aperture, active)
    low_speed_height = 0.5 if batched_aperture else 2 / 3
    expected = torch.tensor([
        [0.25, 0.0, 1.0],
        [0.25, 0.1, 0.9],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.25, 1.5, -1.0],
        [1.0, low_speed_height, 1 - low_speed_height],
    ])
    torch.testing.assert_close(features, expected)


@torch.inference_mode()
def test_gate_future_base_equivalence():
    groups = {"policy": ["depth", "state"], "critic": ["depth", "state"]}
    occupancy = OccupancyConfig((0.5, 1.0), (8, 8, 2), ((0, 16), (-8, 8), (-4, 4)), 0.1, 8.0)
    base = GateActorCritic(None, groups, 3, occupancy=occupancy, recurrent_type="sru").eval()
    model = GateActorCritic(
        None, groups, 3, occupancy=occupancy, recurrent_type="sru",
        gate_future=GateFutureConfig(enabled=True),
    ).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"actor_future.weight", "critic_future.weight"}
    assert not unexpected
    assert torch.count_nonzero(model.actor_future.weight) == 0
    assert torch.count_nonzero(model.critic_future.weight) == 0

    generator = torch.Generator().manual_seed(17)
    obs = TensorDict({
        "depth": torch.rand(4, 1600, generator=generator),
        "state": torch.rand(4, 14, generator=generator),
        "gate": torch.rand(4, 6, generator=generator),
        "gate_future": torch.rand(4, 3, generator=generator),
    }, batch_size=[4])
    for _ in range(2):
        assert (model.act_inference(obs) - base.act_inference(obs)).abs().max() < 1e-6
        assert (model.evaluate(obs) - base.evaluate(obs)).abs().max() < 1e-6
