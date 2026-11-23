"""Gate-conditioned HITSZ PPO and optional supervised vehicle forecasting."""

from copy import deepcopy
from dataclasses import dataclass
from math import prod
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F

from evaluation.rollout import gate_crossing
from hitsz_ppo import atomic_json, save
from RL.rl_env_wrapper import RLQuadEnv
from rsl_rl.algorithms import PPO
from rsl_rl.modules import CustomVisionActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


CURRICULUM_STATE = (
    "curriculum_phase", "speed_level", "obstacle_level", "good_success_epochs",
    "curriculum_epoch_steps", "epoch_window_successes", "epoch_window_failures",
    "curriculum_successes", "curriculum_failures", "curriculum_success_rate_ema",
    "last_epoch_success_rate",
)


@dataclass(frozen=True)
class GateConfig:
    aperture_m: tuple
    pass_reward: float
    three_gate_reward: float


@dataclass(frozen=True)
class OccupancyConfig:
    horizons_s: tuple
    grid_shape: tuple
    grid_bounds_m: tuple
    loss_weight: float
    positive_weight: float

    @property
    def output_size(self):
        return len(self.horizons_s) * prod(self.grid_shape)


def gate_features(position, yaw, center, basis, active):
    relative = ((center - position)[:, None] @ yaw).squeeze(1)
    normal = (basis[:, :, :1].transpose(1, 2) @ yaw).squeeze(1)
    scale = relative.new_tensor([24, 6, 4])
    features = torch.cat(((relative / scale).clamp(-2, 2), normal), dim=-1)
    return torch.where(active[:, None], features, 0.0)


class OrderedGates:
    def __init__(self, env, aperture_m):
        self.env = env
        # Upstream stores six boxes per gate. The mean of the two pillars
        # gives the center, and their shared rotation is the gate basis.
        boxes = env.base_voxels_gate[:, :, :3].reshape(env.num_trajectories, -1, 6, 3)
        self.centers = boxes[:, :, :2].mean(dim=2)
        self.bases = env.base_voxels_gate_R[:, ::6]
        ids = torch.arange(boxes.size(1), device=env.device)
        self.stations = (ids // 2) * 21.5 + (ids % 2) * 2.6
        self.counts = (boxes[:, :, 0, 0] != 1e6).sum(dim=-1)
        self.half_aperture = env.p.new_tensor(aperture_m) / 2
        self.next = torch.zeros(env.batch_size, device=env.device, dtype=torch.long)
        self.streak = torch.zeros_like(self.next)
        self.reset(torch.arange(env.batch_size, device=env.device))

    def reset(self, rows):
        e = self.env
        routes = e.env_traj_ids[rows]
        positions = e.p[rows]
        nodes = e.trajectories[routes]
        start = nodes[:, :-1]
        edges = nodes[:, 1:] - start
        delta = positions[:, None] - start
        length_sq = edges.square().sum(dim=-1)
        fraction = ((delta * edges).sum(dim=-1) / length_sq.clamp_min(1e-8)).clamp(0, 1)
        distance_sq = (delta - edges * fraction[..., None]).square().sum(dim=-1)
        closest = distance_sq.masked_fill(length_sq == 0, torch.inf).argmin(dim=-1, keepdim=True)
        s = (e.traj_s[routes, :-1] + fraction * length_sq.sqrt()).gather(1, closest).squeeze(-1)
        self.next[rows] = torch.searchsorted(self.stations, s.contiguous(), right=True)
        self.streak[rows] = 0

    def current(self):
        routes = self.env.env_traj_ids
        index = self.next.clamp_max(self.centers.size(1) - 1)
        return self.centers[routes, index], self.bases[routes, index], self.next < self.counts[routes]

    def features(self, yaw):
        center, basis, active = self.current()
        return gate_features(self.env.p, yaw, center, basis, active)

    def step(self, before, fatal):
        center, basis, active = self.current()
        cross, inside, _ = gate_crossing(before, self.env.p, center, basis, self.half_aperture)
        visited = cross & active & ~fatal.bool()
        passed = visited & inside
        self.next += visited.long()
        self.streak = torch.where(visited & ~inside, 0, self.streak + passed.long())
        completed = self.streak == 3
        self.streak.masked_fill_(completed, 0)
        return passed, completed, visited & ~inside


class VehicleOccupancy:
    """Rasterize future vehicle boxes in the current yaw-aligned frame."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.horizons = env.p.new_tensor(config.horizons_s)
        self.bounds = env.p.new_tensor(config.grid_bounds_m)
        shape = env.p.new_tensor(config.grid_shape)
        self.half_cell = self.bounds.diff(dim=-1).squeeze(-1) / (2 * shape)
        centers = [low + (2 * torch.arange(n, device=env.device) + 1) * half
                   for (low, _), n, half in zip(self.bounds, config.grid_shape, self.half_cell)]
        self.cells = torch.cartesian_prod(*centers)

    def targets(self, yaw):
        e = self.env
        n = e.active_num_cars
        out = torch.zeros((e.batch_size, len(self.horizons), len(self.cells)), dtype=torch.uint8, device=e.device)
        routes = e.env_traj_ids
        future_s = (e.car_s[:, :n, None] - e.car_v[:, :n, None] * self.horizons
                    ) % e.traj_lengths[routes, None, None]
        for route in range(e.num_trajectories):
            rows = (routes == route).nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            s = future_s[rows].contiguous()
            station = e.traj_s[route]
            j = torch.searchsorted(station, s).clamp_min(1).minimum(e.traj_num_points[route] - 1)
            fraction = (s - station[j - 1]) / (station[j] - station[j - 1]).clamp_min(1e-6)
            points = e.trajectories[route]
            basis = e.traj_R[route, j - 1]
            position = torch.lerp(points[j - 1], points[j], fraction[..., None])
            position += (basis[..., 1] * e.car_lat_offset[rows, :n, None]
                         + basis[..., 2] * e.car_vert_offset[rows, :n, None])
            yaw_rows = yaw[rows, None, None]
            center = ((position - e.p[rows, None, None])[..., None, :] @ yaw_rows).squeeze(-2)
            # Enclosing AABB in the current frame provides a conservative coarse label.
            half = (yaw_rows.transpose(-1, -2) @ basis).abs() @ e.voxel_size
            near = ((center + half >= self.bounds[:, 0]) & (center - half <= self.bounds[:, 1])).all(-1)
            row, car, horizon = near.nonzero(as_tuple=True)
            hit = ((self.cells - center[row, car, horizon, None]).abs()
                   <= half[row, car, horizon, None] + self.half_cell).all(-1)
            hit_row, cell = hit.nonzero(as_tuple=True)
            out[rows[row[hit_row]], horizon[hit_row], cell] = 1
        return out.flatten(1)


class GateEnv(RLQuadEnv):
    def __init__(self, cfg, env_cuda, gate, occupancy=None):
        super().__init__(cfg, env_cuda)
        self.gate_cfg = gate
        self.gates = OrderedGates(self.env, gate.aperture_m)
        self.occupancy = VehicleOccupancy(self.env, occupancy) if occupancy else None

    def get_observations(self):
        obs = super().get_observations()
        obs["gate"] = self.gates.features(self._current_R_yaw)
        if self.occupancy is not None:
            obs["vehicle_target"] = self.occupancy.targets(self._current_R_yaw)
        return obs

    def step(self, actions):
        self._before = self.env.p.clone()
        return super().step(actions)

    def _compute_rewards(self):
        rewards, fatal, info = super()._compute_rewards()
        passed, completed, missed = self.gates.step(self._before, fatal)
        bonus = passed * self.gate_cfg.pass_reward + completed * self.gate_cfg.three_gate_reward
        info.update({
            "gate/pass_per_step": passed.float().mean(),
            "gate/three_consecutive_per_step": completed.float().mean(),
            "gate/miss_per_step": missed.float().mean(),
            "reward/gate_bonus": bonus.mean(),
        })
        return rewards + bonus, fatal, info

    def _reset_envs(self, rows):
        super()._reset_envs(rows)
        self.gates.reset(rows)


def zero_projection(input_size, output_size):
    layer = nn.Linear(input_size, output_size, bias=False)
    nn.init.zeros_(layer.weight)
    return layer


class GateActorCritic(CustomVisionActorCritic):
    def __init__(self, *args, occupancy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_gate = zero_projection(6, self.actor_encoder.output_dim)
        self.critic_gate = zero_projection(6, self.critic_encoder.output_dim)
        self.occupancy = occupancy
        if occupancy is not None:
            hidden = self.memory_a.hidden_dim
            self.forecast = nn.Sequential(nn.Linear(hidden, 128), nn.LeakyReLU(0.05),
                                          nn.Linear(128, occupancy.output_size))
            self.forecast_velocity = zero_projection(occupancy.output_size, self.actor_head.out_features)

    def encode(self, obs, encoder, gate):
        depth = obs["depth"].reshape(-1, 1, encoder.img_height, encoder.img_width)
        visual = encoder.stem(depth).reshape(*obs.batch_size, encoder.output_dim)
        return encoder.act(visual + encoder.v_proj(obs["state"]) + gate(obs["gate"]))

    def actor_outputs(self, obs, masks=None, hidden_state=None):
        features = self.encode(obs, self.actor_encoder, self.actor_gate)
        hidden = self.memory_a(features, masks, hidden_state).squeeze(0)
        actions = self.actor_head(self.act_fn(hidden))
        if self.occupancy is None:
            return actions, None
        forecast = self.forecast(hidden)
        actions = actions + self.forecast_velocity(forecast.detach().sigmoid())
        return actions, forecast

    def _actor_mean(self, obs, masks=None, hidden_state=None):
        return self.actor_outputs(obs, masks, hidden_state)[0]

    def evaluate(self, obs, masks=None, hidden_state=None):
        features = self.encode(obs, self.critic_encoder, self.critic_gate)
        hidden = self.memory_c(features, masks, hidden_state).squeeze(0)
        return self.critic_head(self.act_fn(hidden))


class OccupancyPPO(PPO):
    """Native recurrent PPO update with a supervised term in the same backward pass."""

    def update(self):
        policy = self.policy
        cfg = policy.occupancy
        totals = torch.zeros(6, device=self.device)
        batch_iter = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs, actions, values, advantages, returns, old_log, old_mean, old_std, hidden, masks in batch_iter:
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            mean, forecast = policy.actor_outputs(obs, masks, hidden[0])
            policy._update_distribution(mean)
            value = policy.evaluate(obs, masks, hidden[1])
            std = policy.action_std
            entropy = policy.entropy.mean()
            with torch.no_grad():
                kl = (torch.log(std / old_std + 1e-5) + (old_std.square() + (old_mean - mean).square())
                      / (2 * std.square()) - 0.5).sum(-1).mean()
                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl > self.desired_kl * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = (policy.get_actions_log_prob(actions) - old_log.squeeze(-1)).exp()
            adv = advantages.squeeze(-1)
            policy_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - self.clip_param,
                                                                         1 + self.clip_param)).mean()
            value_error = (value - returns).square()
            if self.use_clipped_value_loss:
                clipped = values + (value - values).clamp(-self.clip_param, self.clip_param)
                value_error = torch.maximum(value_error, (clipped - returns).square())
            value_loss = value_error.mean()
            target = unpad_trajectories(obs["vehicle_target"], masks).to(forecast.dtype)
            prediction_loss = F.binary_cross_entropy_with_logits(
                forecast, target, pos_weight=forecast.new_tensor(cfg.positive_weight)
            )
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss = loss + cfg.loss_weight * prediction_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            totals += torch.stack((value_loss, policy_loss, entropy, kl, prediction_loss, target.mean())).detach()

        self.storage.clear()
        names = ("value", "surrogate", "entropy", "kl", "occupancy_bce", "occupancy_target_fraction")
        return dict(zip(names, (totals / (self.num_mini_batches * self.num_learning_epochs)).tolist()))


class HitszRunner(OnPolicyRunner):
    def __init__(self, env, cfg, out, device, recipe, occupancy):
        self.out = out
        self.recipe = recipe
        self.occupancy = occupancy
        self.restored_update = 0
        self.restored_environment_steps = 0
        super().__init__(env, cfg, str(out), device)

    def _construct_algorithm(self, obs):
        if self.recipe["variant"] == "hitsz":
            return super()._construct_algorithm(obs)

        self.policy_cfg.pop("class_name")
        self.alg_cfg.pop("class_name")
        self.alg_cfg.setdefault("rnd_cfg", None)
        self.alg_cfg.setdefault("symmetry_cfg", None)
        policy = GateActorCritic(
            obs, self.cfg["obs_groups"], self.env.num_actions, occupancy=self.occupancy, **self.policy_cfg
        )
        storage = RolloutStorage("rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs,
                                 [self.env.num_actions], self.device)
        algorithm = OccupancyPPO if self.occupancy is not None else PPO
        return algorithm(policy, storage, device=self.device, **self.alg_cfg)

    def environment_steps(self, completed):
        steps_per_update = self.env.num_envs * self.cfg["num_steps_per_env"]
        return self.restored_environment_steps + (completed - self.restored_update) * steps_per_update

    def save(self, path, infos=None):
        completed = self.current_learning_iteration + 1
        environment_steps = self.environment_steps(completed)
        checkpoint = {
            "recipe": self.recipe,
            "update": completed,
            "environment_steps": environment_steps,
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "learning_rate": self.alg.learning_rate,
            "curriculum": {key: getattr(self.env, key) for key in CURRICULUM_STATE},
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device),
            },
        }
        path = Path(path)
        save(path, checkpoint)
        latest = self.out / "latest.pt"
        pending = latest.with_suffix(".pt.tmp")
        pending.unlink(missing_ok=True)
        pending.hardlink_to(path)
        pending.replace(latest)
        atomic_json(self.out / "status.json", {
            "update": completed,
            "variant": self.recipe["variant"],
            "environment_steps": environment_steps,
            "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1),
            "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved(self.device) / 2**20, 1),
        })

    def restore(self, path, *, allow_batch_change=False):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        saved_recipe = ckpt["recipe"]
        comparison_recipe = deepcopy(saved_recipe)
        if allow_batch_change:
            comparison_recipe["environment"]["env"]["batch_size"] = self.env.num_envs
        if comparison_recipe != self.recipe:
            raise ValueError("Resume settings differ from the checkpoint")
        self.alg.policy.load_state_dict(ckpt["model_state_dict"])
        self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.alg.learning_rate = ckpt["learning_rate"]
        self.current_learning_iteration = ckpt["update"]
        self.restored_update = ckpt["update"]
        original_steps_per_update = (saved_recipe["environment"]["env"]["batch_size"]
                                     * saved_recipe["train"]["num_steps_per_env"])
        self.restored_environment_steps = ckpt.get(
            "environment_steps", self.restored_update * original_steps_per_update
        )
        for key, value in ckpt["curriculum"].items():
            setattr(self.env, key, value)
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"], self.device)
        self.env._apply_obstacle_level()
        self.env._reset_envs(torch.arange(self.env.num_envs, device=self.device))
