"""One-axis residual PPO over a frozen native joint policy."""

import torch
from tensordict import TensorDict
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from ruma_ppo.training.hitsz_joint import GateActorCritic, GateEnv, HitszRunner, OccupancyConfig


RESIDUAL_GROUPS = {
    "policy": ["gate_future", "base_action"],
    "critic": ["gate_future", "base_action", "state", "gate"],
}
RESIDUAL_POLICY = {"class_name": "ResidualGateActorCritic", "bound_mps": 0.8, "init_noise_std": 0.1}
SATURATION_FRACTION = 0.95


def make_frozen_base(recipe):
    if recipe["variant"] != "joint" or recipe["additions"].get("gate_future", {}).get("enabled"):
        raise ValueError("A1 requires an original joint base checkpoint")
    config = recipe["train"]["policy"].copy()
    config.pop("class_name")
    config["base_model_path"] = None
    occupancy = OccupancyConfig(**recipe["additions"]["occupancy"])
    model = GateActorCritic(None, recipe["train"]["obs_groups"], 3, occupancy=occupancy, **config)
    return model.requires_grad_(False).eval()


def apply_vertical_delta(base_action, delta):
    command = base_action.clone()
    command[..., 2:3] += delta
    return command


def geometric_height_delta(position, velocity, center, basis, active):
    relative = ((position - center)[:, None] @ basis).squeeze(1)
    gate_velocity = (velocity[:, None] @ basis).squeeze(1)
    ahead = -relative[:, 0]
    closing = gate_velocity[:, 0]
    seconds = (ahead / closing.clamp_min(1.0)).clamp(0, 1)
    predicted_down = relative[:, 2] + seconds * gate_velocity[:, 2]
    upward = (2 * (predicted_down - 1.35)).clamp(0, 0.8)
    proximity = (1 - ahead / 10).clamp(0, 1)
    return (-upward * proximity * (active & (ahead > 0) & (closing > 1.0)))[:, None]


class ResidualGateActorCritic(ActorCritic):
    def __init__(self, base_policy, *, bound_mps, init_noise_std):
        if not 0 < bound_mps <= 0.8:
            raise ValueError("A1 residual bound must be in (0, 0.8] m/s")
        observations = TensorDict({
            "state": torch.zeros(1, 14),
            "gate": torch.zeros(1, 6),
            "gate_future": torch.zeros(1, 3),
            "base_action": torch.zeros(1, 3),
        }, batch_size=[1])
        super().__init__(
            observations, RESIDUAL_GROUPS, 1, actor_hidden_dims=(64, 64),
            critic_hidden_dims=(64, 64), activation="tanh", noise_std_type="log",
            init_noise_std=init_noise_std,
        )
        for head in (self.actor[-1], self.critic[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.base_policy = base_policy
        self.bound_mps = bound_mps

    def train(self, mode=True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def act_inference(self, obs):
        return super().act_inference(obs).tanh() * self.bound_mps

    def evaluate(self, obs, **kwargs):
        return obs["base_value"] + super().evaluate(obs, **kwargs)


class ResidualGateEnv(GateEnv):
    def __init__(self, cfg, env_cuda, gate, future, base_policy, bound_mps):
        super().__init__(cfg, env_cuda, gate, future=future)
        self.base_policy = base_policy
        self.bound_mps = bound_mps
        self.num_actions = 1
        self._residual_observations = None

    @torch.no_grad()
    def get_observations(self):
        if self._residual_observations is None:
            obs = super().get_observations()
            residual = obs.select("state", "gate", "gate_future")
            residual["base_action"] = self.base_policy.act_inference(obs)
            residual["base_value"] = self.base_policy.evaluate(obs)
            self._residual_observations = residual
        return self._residual_observations

    def step(self, actions):
        # PPO stores the Gaussian sample. The same fixed tanh transform is
        # applied on every rollout, so its Jacobian cancels in the PPO ratio.
        return self.step_with_delta(actions.tanh() * self.bound_mps)

    def step_with_delta(self, delta):
        command = apply_vertical_delta(self.get_observations()["base_action"], delta)
        self._residual_observations = None
        obs, reward, done, extras = super().step(command)
        extras["log"].update({
            "residual/mean_delta_z_mps": delta.mean(),
            "residual/mean_abs_delta_z_mps": delta.abs().mean(),
            "residual/saturation_fraction": (delta.abs() >= SATURATION_FRACTION * self.bound_mps).float().mean(),
            "residual/vertical_pi_clip_fraction": (self._current_a_pred_bl[:, 2].abs() > 9.81).float().mean(),
        })
        return obs, reward, done, extras

    def _reset_envs(self, rows):
        super()._reset_envs(rows)
        if rows.numel() == self.num_envs:
            self.base_policy.reset()
        else:
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            done[rows] = True
            self.base_policy.reset(done)
        self._residual_observations = None


class ResidualPPO(PPO):
    def update(self):
        actor = (*self.policy.actor.parameters(), self.policy.log_std)
        critic = tuple(self.policy.critic.parameters())
        squared = torch.zeros(2, device=self.device)
        totals = torch.zeros_like(squared)
        max_norm = torch.zeros_like(squared)
        clipped = torch.zeros((), device=self.device)
        steps = 0

        def measure(index):
            def record(grad):
                squared[index].add_(grad.detach().square().sum())
            return record

        def on_step(*_):
            nonlocal steps
            norm = squared.sqrt()
            totals.add_(norm)
            max_norm.copy_(torch.maximum(max_norm, norm))
            clipped.add_(squared.sum() > self.max_grad_norm ** 2)
            squared.zero_()
            steps += 1

        hooks = [parameter.register_hook(measure(index))
                 for index, group in enumerate((actor, critic)) for parameter in group]
        hooks.append(self.optimizer.register_step_pre_hook(on_step))
        try:
            metrics = super().update()
        finally:
            for hook in hooks:
                hook.remove()
        means = (totals / steps).tolist()
        max_actor, max_critic = max_norm.tolist()
        metrics.update({
            "actor_grad_norm": means[0], "critic_grad_norm": means[1],
            "max_actor_grad_norm": max_actor, "max_critic_grad_norm": max_critic,
            "grad_clip_fraction": (clipped / steps).item(),
        })
        return metrics


class ResidualRunner(HitszRunner):
    def _construct_algorithm(self, obs):
        policy_config = {key: value for key, value in self.policy_cfg.items() if key != "class_name"}
        policy = ResidualGateActorCritic(self.env.base_policy, **policy_config)
        self.alg_cfg.pop("class_name")
        self.alg_cfg.setdefault("rnd_cfg", None)
        self.alg_cfg.setdefault("symmetry_cfg", None)
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device,
        )
        return ResidualPPO(policy, storage, device=self.device, **self.alg_cfg)
