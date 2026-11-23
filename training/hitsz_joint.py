"""Gate-conditioned HITSZ PPO."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from evaluation.rollout import gate_crossing
from hitsz_ppo import atomic_json, save
from RL.rl_env_wrapper import RLQuadEnv
from rsl_rl.algorithms import PPO
from rsl_rl.modules import CustomVisionActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


@dataclass(frozen=True)
class GateConfig:
    aperture_m: tuple
    pass_reward: float
    three_gate_reward: float


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


class GateEnv(RLQuadEnv):
    def __init__(self, cfg, env_cuda, gate):
        super().__init__(cfg, env_cuda)
        self.gate_cfg = gate
        self.gates = OrderedGates(self.env, gate.aperture_m)

    def get_observations(self):
        obs = super().get_observations()
        obs["gate"] = self.gates.features(self._current_R_yaw)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_gate = zero_projection(6, self.actor_encoder.output_dim)
        self.critic_gate = zero_projection(6, self.critic_encoder.output_dim)

    def encode(self, obs, encoder, gate):
        depth = obs["depth"].reshape(-1, 1, encoder.img_height, encoder.img_width)
        visual = encoder.stem(depth).reshape(*obs.batch_size, encoder.output_dim)
        return encoder.act(visual + encoder.v_proj(obs["state"]) + gate(obs["gate"]))

    def _actor_mean(self, obs, masks=None, hidden_state=None):
        features = self.encode(obs, self.actor_encoder, self.actor_gate)
        hidden = self.memory_a(features, masks, hidden_state).squeeze(0)
        return self.actor_head(self.act_fn(hidden))

    def evaluate(self, obs, masks=None, hidden_state=None):
        features = self.encode(obs, self.critic_encoder, self.critic_gate)
        hidden = self.memory_c(features, masks, hidden_state).squeeze(0)
        return self.critic_head(self.act_fn(hidden))


class HitszRunner(OnPolicyRunner):
    def __init__(self, env, cfg, out, device, recipe):
        self.out = out
        self.recipe = recipe
        super().__init__(env, cfg, str(out), device)

    def _construct_algorithm(self, obs):
        if self.recipe["variant"] == "hitsz":
            return super()._construct_algorithm(obs)

        self.policy_cfg.pop("class_name")
        self.alg_cfg.pop("class_name")
        self.alg_cfg.setdefault("rnd_cfg", None)
        self.alg_cfg.setdefault("symmetry_cfg", None)
        policy = GateActorCritic(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        )
        storage = RolloutStorage("rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs,
                                 [self.env.num_actions], self.device)
        return PPO(policy, storage, device=self.device, **self.alg_cfg)

    def save(self, path, infos=None):
        completed = self.current_learning_iteration + 1
        environment_steps = completed * self.env.num_envs * self.cfg["num_steps_per_env"]
        checkpoint = {
            "recipe": self.recipe,
            "update": completed,
            "environment_steps": environment_steps,
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
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
