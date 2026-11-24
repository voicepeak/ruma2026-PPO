"""Warm-start the vertical residual with rollouts from the geometric teacher."""

import torch
from torch.nn import functional as F

from ruma_ppo.hitsz.legacy_policy import atomic_json, save
from ruma_ppo.training.residual import geometric_height_delta


BC_EPOCHS = 12
BC_BATCH = 4096


@torch.no_grad()
def collect_teacher(runner, steps):
    env, policy = runner.env, runner.alg.policy
    features, targets = [], []
    for _ in range(steps):
        obs = env.get_observations()
        center, basis, active = env.gates.current()
        delta = geometric_height_delta(env.env.p, env.env.v, center, basis, active)
        features.append(policy.get_actor_obs(obs))
        targets.append(delta)
        env.step_with_delta(delta)
    features, targets = torch.stack(features), torch.stack(targets)
    held_out = torch.arange(env.num_envs, device=env.device) % 8 == 0
    return (
        features[:, ~held_out].flatten(0, 1), targets[:, ~held_out].flatten(0, 1),
        features[:, held_out].flatten(0, 1), targets[:, held_out].flatten(0, 1),
    )


@torch.no_grad()
def clone_statistics(policy, features, target):
    predicted = policy.actor(features).tanh() * policy.bound_mps
    active = (target.abs() > 1e-5).squeeze(-1)
    error = (predicted - target).abs()
    return {
        "examples": len(target),
        "teacher_active_examples": int(active.sum()),
        "mae_mps": error.mean().item(),
        "active_mae_mps": error[active].mean().item() if active.any() else None,
        "inactive_mae_mps": error[~active].mean().item() if (~active).any() else None,
        "teacher_active_mean_abs_mps": target[active].abs().mean().item() if active.any() else None,
    }


def behavior_clone(runner, steps):
    policy = runner.alg.policy
    features, targets, val_features, val_targets = collect_teacher(runner, steps)
    active = targets[:, 0].abs() > 1e-5
    corrections = active.nonzero().flatten()
    zeros = (~active).nonzero().flatten()
    samples_per_class = min(len(corrections), len(zeros))
    if not samples_per_class:
        raise RuntimeError("Geometric teacher rollout needs both zero and nonzero corrections")

    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=0.001)
    before = clone_statistics(policy, val_features, val_targets)
    for _ in range(BC_EPOCHS):
        batch = torch.cat([
            indices[torch.randperm(len(indices), device=indices.device)[:samples_per_class]]
            for indices in (corrections, zeros)
        ])
        batch = batch[torch.randperm(len(batch), device=batch.device)]
        for indices in batch.split(BC_BATCH):
            delta = policy.actor(features[indices]).tanh() * policy.bound_mps
            loss = F.mse_loss(delta, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

    result = {
        "teacher_environment_steps": steps * runner.env.num_envs,
        "train_active_examples": len(corrections),
        "train_inactive_examples": len(zeros),
        "epochs": BC_EPOCHS,
        "held_out_before": before,
        "held_out_after": clone_statistics(policy, val_features, val_targets),
    }
    atomic_json(runner.out / "bc.json", result)
    save(runner.out / "bc-eval.pt", {
        "recipe": runner.recipe, "update": 0, "model_state_dict": policy.state_dict(),
    })
    print(f"Geometric teacher BC: {result}", flush=True)
