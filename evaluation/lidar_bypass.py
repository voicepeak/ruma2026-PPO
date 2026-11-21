"""Follow a LiDAR-scored lateral path through the existing gate apertures."""

import torch
import torch.nn.functional as F

from control.lidar_brake import first_path_return, lidar_confirms_clear, stopping_speed
from control.lidar_bypass import BypassConfig, choose_candidate, lateral_profile, right_of
from evaluation.rollout import LidarBrakeFollower, REFERENCE_LOOKAHEAD_M
from hitsz_branches import station


class LidarBypassFollower(LidarBrakeFollower):
    def __init__(self, env, config=BypassConfig()):
        self.bypass = config
        super().__init__(env)
        self.lateral_choices = env.p.new_tensor(config.lateral_offsets_m)

    def reset(self, seed):
        super().reset(seed)
        e = self.sim.env
        count = len(self.bypass.lateral_offsets_m)
        self.choice = torch.zeros(e.batch_size, device=e.device, dtype=torch.long)
        self.detouring = torch.zeros_like(self.choice, dtype=torch.bool)
        self.blocking_points = e.p.new_zeros((e.batch_size, count, 3))
        self.blocking_visible = torch.zeros((e.batch_size, count), device=e.device, dtype=torch.bool)
        self.clear_scans = torch.zeros_like(self.blocking_visible, dtype=torch.long)

    def scan_paths(self, s, lateral):
        e = self.sim.env
        cfg, brake = self.bypass, self.brake
        points, valid = self.sensor.capture_lidar()
        relative_points = points @ e.R.transpose(-1, -2)
        memory = self.blocking_points - e.p[:, None]
        local_memory = memory @ e.R
        scan = self.sensor.config
        for index in range(len(self.lateral_choices)):
            clear = lidar_confirms_clear(points, valid, local_memory[:, index], scan.lidar_hw, scan.range_m)
            self.clear_scans[:, index] = torch.where(clear, self.clear_scans[:, index] + 1, 0)
        self.blocking_visible &= self.clear_scans < brake.release_samples
        observed = torch.cat((relative_points, memory), dim=1)
        observed_valid = torch.cat((valid, self.blocking_visible), dim=1)

        ahead, tangent, _ = e.get_state_by_s(s[:, None] + self.offsets, e.env_traj_ids)
        center = ahead - e.p[:, None]
        right = right_of(tangent)
        distances, surfaces = [], []
        for index, goal in enumerate(self.lateral_choices):
            offset = lateral_profile(self.offsets, lateral[:, None], goal, cfg.merge_distance_m)
            if index == 0:
                offset = torch.where(self.detouring[:, None], offset, 0)
            path = center + offset[..., None] * right
            path[:, 0] = 0.0
            distance, surface = first_path_return(observed, observed_valid, path, self.offsets, brake.corridor_radius_m)
            distances.append(distance)
            surfaces.append(surface)
        distances = torch.stack(distances, dim=1)
        surfaces = torch.stack(surfaces, dim=1)
        has_return = torch.isfinite(distances)
        # Preserve an unconfirmed close return even if a new candidate currently avoids it.
        update = has_return & (~self.blocking_visible | (surfaces.norm(dim=-1) <= memory.norm(dim=-1)))
        self.blocking_points = torch.where(update[..., None], e.p[:, None] + surfaces, self.blocking_points)
        self.blocking_visible |= update
        self.clear_scans = torch.where(update, 0, self.clear_scans)

        return distances

    def set_choice(self, s, distances, choice):
        self.choice = choice
        self.obstacle_station = s + distances.gather(1, choice[:, None]).squeeze(1) - self.brake.stand_off_m

    def route_state(self):
        e = self.sim.env
        s = station(e)
        center, tangent, _ = e.get_state_by_s(s, e.env_traj_ids)
        lateral = ((e.p - center) * right_of(tangent)).sum(-1)
        return s, lateral, tangent

    def command(self):
        s, lateral, _ = self.route_state()
        if self.tick % self.sensor.config.sample_every_ticks == 0:
            distances = self.scan_paths(s, lateral)
            choice = choose_candidate(distances, self.lateral_choices, self.choice, self.bypass)
            self.set_choice(s, distances, choice)
        return self.track_path(s, lateral)

    def track_path(self, s, lateral, *, hold=None):
        e = self.sim.env
        self.detouring = (self.choice != 0) | (self.detouring & (lateral.abs() > 0.16))

        brake = self.brake
        limit = stopping_speed(self.obstacle_station - s, brake.deceleration_mps2, brake.reaction_s)
        self.speed_fraction = (limit / self.nominal_mps).clamp(0, 1)
        lookahead = REFERENCE_LOOKAHEAD_M * self.speed_fraction
        reference, tangent, _ = e.get_state_by_s(s + lookahead, e.env_traj_ids)
        right = right_of(tangent)
        goal = self.lateral_choices[self.choice]
        lateral_goal = lateral_profile(lookahead, lateral, goal, self.bypass.merge_distance_m)
        slope = (goal - lateral) / self.bypass.merge_distance_m
        reference = reference + right * torch.where(self.detouring, lateral_goal, 0)[:, None]
        direction = F.normalize(tangent + slope[:, None] * right, dim=-1)
        direction = torch.where(self.detouring[:, None], direction, tangent)
        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        velocity = self.speed_mps * self.speed_fraction[:, None] * direction
        if hold is not None:
            waiting, hold_position, hold_velocity = hold
            reference = torch.where(waiting[:, None], hold_position, reference)
            velocity = torch.where(waiting[:, None], hold_velocity, velocity)
        return self.controller.command(e.p, e.v, e.R, e.w, reference, velocity, yaw)
