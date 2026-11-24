"""10 Hz learning adapter over the existing LiDAR planner and 100 Hz PWM control."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ruma_ppo.env.control.lidar_bypass import candidate_scores, right_of
from ruma_ppo.evaluation.lidar_bypass import LidarBypassFollower
from ruma_ppo.evaluation.rollout import BOUNDS, COLLISION, MISSED_GATE, SUCCESS, TIMEOUT, Progress
from ruma_ppo.hitsz.branches import station
from ruma_ppo.env.sim.pwm_env import DT


PATH_ACTIONS = 5
WAIT = PATH_ACTIONS
HISTORY_FRAMES = 3
OBS_SIZE = HISTORY_FRAMES * PATH_ACTIONS + 16
WAIT_LOCK_MPS = 0.3
RULE_PRIOR_SCALE = 1.5
WAIT_INITIAL_LOGIT = -3.0


@dataclass(frozen=True)
class Rewards:
    progress_per_m: float = 0.08
    gate: float = 1.0
    success: float = 6.0
    time_per_s: float = 0.10
    near_miss: float = 0.04
    risk_radius_m: float = 0.30
    collision: float = -8.0
    bounds: float = -8.0
    missed_gate: float = -5.0
    timeout: float = -4.0


def decision_reward(config, progress, gates, seconds, clearance_m, outcome):
    risk = ((config.risk_radius_m - clearance_m) / config.risk_radius_m).clamp(0, 1)
    reward = config.progress_per_m * progress + config.gate * gates - config.time_per_s * seconds
    reward = reward - config.near_miss * torch.where(clearance_m >= 0, risk, 0.0)
    for state, value in ((SUCCESS, config.success), (COLLISION, config.collision), (BOUNDS, config.bounds),
                         (MISSED_GATE, config.missed_gate), (TIMEOUT, config.timeout)):
        reward = reward + (outcome == state) * value
    return reward


class BypassActorCritic(nn.Module):
    def __init__(self, allow_wait):
        super().__init__()
        actions = PATH_ACTIONS + int(allow_wait)
        self.trunk = nn.Sequential(nn.Linear(OBS_SIZE, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, actions)
        self.critic = nn.Linear(64, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, features, prior):
        hidden = self.trunk(features)
        return self.actor(hidden) + prior, self.critic(hidden).squeeze(-1)


class BypassLearningEnv:
    def __init__(self, course, graph, *, allow_wait, rewards=Rewards()):
        self.course, self.graph = course, graph
        self.follower = LidarBypassFollower(course.e)
        self.allow_wait, self.rewards = allow_wait, rewards
        self.last_station = graph.gates[-1].station_m

    def reset(self, level, seed):
        fingerprint = self.course.start(level, seed, False)
        e = self.course.e
        self.follower.reset(seed)
        self.progress = Progress(e, self.graph)
        self.furthest = station(e).clamp_max(self.last_station)
        self.history = None
        self.preview = None
        self.last_action = torch.zeros_like(self.progress.gates)
        self.waiting_s = e.p.new_zeros(e.batch_size)
        self.hold = e.p.clone()
        self.hold_lateral = e.p.new_zeros(e.batch_size)
        self.hold_locked = torch.zeros_like(self.progress.done)
        return fingerprint

    def observe(self):
        if self.preview is not None:
            raise RuntimeError("Consume each LiDAR observation with one decision")
        p = self.follower
        e = self.course.e
        s, lateral, tangent = p.route_state()
        distances = p.scan_paths(s, lateral)
        self.preview = s, lateral, distances
        scan = (distances.clamp_max(p.brake.lookahead_m) / p.brake.lookahead_m)[:, None, :]
        if self.history is None:
            self.history = scan.expand(-1, HISTORY_FRAMES, -1)
        else:
            self.history = torch.cat((self.history[:, 1:], scan), dim=1)

        right = right_of(tangent)
        velocity = torch.stack(((e.v * tangent).sum(-1), (e.v * right).sum(-1), e.v[:, 2]), dim=-1)
        progress = self.progress
        goal = progress.gates.clamp_max(len(progress.center) - 1)
        to_gate = ((progress.center[goal] - e.p)[:, None, :] @ progress.basis[goal]).squeeze(1)
        gate_scale = e.p.new_tensor([24.0, 6.0, 4.0])
        time_left = 1.0 - progress.flight_s.to(e.p.dtype) / progress.limit_s
        action = F.one_hot(self.last_action, PATH_ACTIONS + 1).to(e.p.dtype)
        features = torch.cat((self.history.flatten(1), (velocity / 10).clamp(-2, 2),
                              lateral[:, None] / 3, (to_gate / gate_scale).clamp(-2, 2),
                              progress.gates[:, None] / len(progress.center), time_left[:, None],
                              action, (self.waiting_s[:, None] / 2).clamp_max(1)), dim=-1)
        scores = candidate_scores(distances, p.lateral_choices, p.choice, p.bypass)
        prior = RULE_PRIOR_SCALE * (scores - scores.amax(dim=-1, keepdim=True))
        if self.allow_wait:
            prior = torch.cat((prior, prior.new_full((len(prior), 1), WAIT_INITIAL_LOGIT)), dim=-1)
        return features, prior

    def step(self, action):
        if self.preview is None:
            raise RuntimeError("Observe before choosing a path")
        p = self.follower
        e = self.course.e
        progress = self.progress
        active = ~progress.done
        waiting = action == WAIT
        begin_wait = waiting & (self.last_action != WAIT)
        self.hold = torch.where(begin_wait[:, None], e.p, self.hold)
        self.hold_locked &= waiting & ~begin_wait
        s, lateral_at_decision, distances = self.preview
        self.hold_lateral = torch.where(begin_wait, lateral_at_decision, self.hold_lateral)
        p.set_choice(s, distances, torch.where(waiting, p.choice, action))
        self.preview = None
        self.last_action = action

        before_gates = progress.gates.clone()
        before_s = progress.flight_s.clone()
        minimum = e.p.new_full((e.batch_size,), torch.inf)
        for _ in range(p.sensor.config.sample_every_ticks):
            current_s, lateral, tangent = p.route_state()
            if self.allow_wait:
                lock = waiting & ~self.hold_locked & (e.v.norm(dim=-1) < WAIT_LOCK_MPS)
                self.hold = torch.where(lock[:, None], e.p, self.hold)
                self.hold_locked |= lock
                center, _, _ = e.get_state_by_s(current_s, e.env_traj_ids)
                braking_pose = center + self.hold_lateral[:, None] * right_of(tangent)
                # Keep the route grade while braking instead of holding the entry altitude.
                along = (e.v[:, :2] * tangent[:, :2]).sum(-1) / tangent[:, :2].square().sum(-1).clamp_min(1e-6)
                down_velocity = along * tangent[:, 2]
                braking_velocity = torch.cat((torch.zeros_like(e.v[:, :2]), down_velocity[:, None]), dim=-1)
                target = torch.where(self.hold_locked[:, None], self.hold, braking_pose)
                target_velocity = torch.where(self.hold_locked[:, None], 0.0, braking_velocity)
                hold = (waiting, target, target_velocity)
            else:
                hold = None
            before = e.p.clone()
            live = ~progress.done
            pwm = p.track_path(current_s, lateral, hold=hold)
            progress.advance(before, p.step(pwm))
            minimum = torch.where(live, torch.minimum(minimum, progress.step_clearance_m), minimum)
            if progress.done.all():
                break

        elapsed = (progress.flight_s - before_s).to(e.p.dtype)
        self.waiting_s = torch.where(waiting, self.waiting_s + elapsed, 0.0)
        new_furthest = torch.maximum(self.furthest, station(e, progress.position).clamp_max(self.last_station))
        gained = new_furthest - self.furthest
        self.furthest = new_furthest
        reward = decision_reward(self.rewards, gained, progress.gates - before_gates, elapsed, minimum, progress.outcome)
        return torch.where(active, reward, 0.0)
