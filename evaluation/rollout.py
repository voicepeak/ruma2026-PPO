"""Shared P0 gate, clearance and timeout scoring for PWM and legacy controllers."""

import torch

from control.geometric import GeometricController
from control.lidar_brake import BrakeConfig, first_path_return, lidar_confirms_clear, stopping_speed
from hitsz_branches import station
from sim.pwm_env import DT, PWMEnv, clearance
from sim.render import Renderer, SensorConfig


RUNNING, COLLISION, BOUNDS, MISSED_GATE, SUCCESS, TIMEOUT = range(6)
OUTCOMES = ("running", "collision", "bounds", "missed_gate", "success", "timeout")
REFERENCE_LOOKAHEAD_M = 1.3


class Progress:
    def __init__(self, env, graph, limit_s=10.0):
        if not graph.gates:
            raise ValueError("Select at least one gate for scoring")
        self.env = env
        self.center = env.p.new_tensor([gate.center_ned for gate in graph.gates])
        self.basis = env.p.new_tensor([gate.basis_ned for gate in graph.gates])
        self.half_aperture = env.p.new_tensor([gate.aperture_m for gate in graph.gates]) / 2
        self.gates = torch.zeros(env.batch_size, dtype=torch.long, device=env.device)
        self.minimum_m = clearance(env, env.p, samples=1)
        self.step_clearance_m = self.minimum_m
        self.outcome = torch.where(self.minimum_m < 0, COLLISION, RUNNING)
        self.position = env.p.clone()
        self.flight_s = torch.zeros(env.batch_size, dtype=torch.float64, device=env.device)
        self.time_s, self.limit_s = 0.0, limit_s

    @property
    def done(self):
        return self.outcome != RUNNING

    def advance(self, before, dt):
        e = self.env
        active = ~self.done
        elapsed = min(dt, self.limit_s - self.time_s)
        self.time_s += elapsed
        end = torch.lerp(before, e.p, elapsed / dt)
        self.flight_s += active.to(torch.float64) * elapsed
        self.position = torch.where(active[:, None], end, self.position)
        near = clearance(e, before, end=end)
        self.step_clearance_m = near
        self.minimum_m = torch.where(active, torch.minimum(self.minimum_m, near), self.minimum_m)
        delta, _ = e.get_nearest_traj_state(end)
        bounds = (delta[:, :2].norm(dim=-1) > 3.0) | (delta[:, 2].abs() > 2.5)

        index = self.gates.clamp_max(len(self.center) - 1)
        center, basis = self.center[index], self.basis[index]
        a = ((before - center)[:, None] @ basis).squeeze(1)
        b = ((end - center)[:, None] @ basis).squeeze(1)
        cross = (a[:, 0] < 0) & (b[:, 0] >= 0)
        fraction = (-a[:, 0] / (b[:, 0] - a[:, 0]).clamp_min(1e-8)).clamp(0, 1)
        hit = torch.lerp(a[:, 1:], b[:, 1:], fraction[:, None])
        inside = (hit.abs() <= self.half_aperture[index]).all(dim=-1)
        contact = near < 0
        valid_cross = active & cross & ~contact & ~bounds
        self.gates += (valid_cross & inside).long()
        success = valid_cross & inside & (self.gates == len(self.center))
        self.flight_s -= torch.where(success, (1 - fraction.to(torch.float64)) * elapsed, 0.0)

        for mask, code in ((contact, COLLISION), (bounds, BOUNDS), (valid_cross & ~inside, MISSED_GATE),
                           (success, SUCCESS)):
            self.outcome = torch.where(~self.done & mask, code, self.outcome)
        if self.time_s >= self.limit_s - 1e-9:
            self.outcome = torch.where(self.done, self.outcome, TIMEOUT)

    def rows(self):
        for gates, code, seconds, minimum, position in zip(self.gates.tolist(), self.outcome.tolist(),
                self.flight_s.tolist(), self.minimum_m.tolist(), self.position.tolist()):
            yield {"gates": gates, "outcome": OUTCOMES[code], "flight_s": round(seconds, 6),
                   "min_clearance_m": round(minimum, 6), "near_miss": 0 <= minimum < 0.3,
                   "failure_ned": None if code == SUCCESS else [round(value, 6) for value in position]}


class PWMRouteFollower:
    """State-feedback reference controller for P0 smoke tests and diagnostics."""

    def __init__(self, env, speed_mps=4.0, *, wind=None):
        self.sim = PWMEnv(env, wind=wind)
        self.controller = GeometricController(env)
        self.speed_mps = speed_mps

    def command(self):
        e = self.sim.env
        return self.command_at(station(e) + REFERENCE_LOOKAHEAD_M, self.speed_mps)

    def command_at(self, reference_station, speed_mps):
        e = self.sim.env
        reference, tangent, _ = e.get_state_by_s(reference_station, e.env_traj_ids)
        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        return self.controller.command(e.p, e.v, e.R, e.w, reference, speed_mps * tangent, yaw)

    def step(self, pwm):
        self.sim.step(pwm)
        return DT


class LidarBrakeFollower(PWMRouteFollower):
    """Use LiDAR returns to constrain speed while following the same route."""

    def __init__(self, env, config=BrakeConfig()):
        super().__init__(env)
        self.brake = config
        self.sensor = Renderer(env, SensorConfig())
        self.offsets = torch.linspace(0, config.lookahead_m, config.path_segments + 1, device=env.device)
        gains_ratio = (self.controller.position_gain[0] / self.controller.velocity_gain[0]).item()
        self.nominal_mps = self.speed_mps + REFERENCE_LOOKAHEAD_M * gains_ratio
        self.reset(0)

    def reset(self, seed):
        e = self.sim.env
        self.sensor.reset(seed)
        self.obstacle_station = e.p.new_full((e.batch_size,), torch.inf)
        self.obstacle_point_ned = torch.zeros_like(e.p)
        self.farther_samples = torch.zeros_like(self.obstacle_station, dtype=torch.long)
        self.tick = 0

    def command(self):
        e = self.sim.env
        cfg = self.brake
        s = station(e)
        if self.tick % self.sensor.config.sample_every_ticks == 0:
            points, valid = self.sensor.capture_lidar()
            # The default scanner mount is at the body origin with no yaw offset.
            relative_points = points @ e.R.transpose(-1, -2)
            ahead, _, _ = e.get_state_by_s(s[:, None] + self.offsets, e.env_traj_ids)
            path = ahead - e.p[:, None, :]
            path[:, 0] = 0
            obstacle, surface = first_path_return(relative_points, valid, path, self.offsets, cfg.corridor_radius_m)
            new_station = s + obstacle - cfg.stand_off_m
            old_surface_sensor = ((self.obstacle_point_ned - e.p)[:, None, :] @ e.R).squeeze(1)
            scan = self.sensor.config
            clear = lidar_confirms_clear(points, valid, old_surface_sensor, scan.lidar_hw, scan.range_m)
            farther = (new_station > self.obstacle_station + 0.10) & clear
            self.farther_samples = torch.where(farther, self.farther_samples + 1, 0)
            released = self.farther_samples >= cfg.release_samples
            replace = released | (new_station <= self.obstacle_station)
            self.obstacle_station = torch.where(replace, new_station, self.obstacle_station)
            self.obstacle_point_ned = torch.where((replace & torch.isfinite(new_station))[:, None],
                                                  e.p + surface, self.obstacle_point_ned)
            self.farther_samples = torch.where(released, 0, self.farther_samples)

        limit = stopping_speed(self.obstacle_station - s, cfg.deceleration_mps2, cfg.reaction_s)
        self.speed_fraction = (limit / self.nominal_mps).clamp(0, 1)
        return self.command_at(s + REFERENCE_LOOKAHEAD_M * self.speed_fraction,
                               self.speed_mps * self.speed_fraction[:, None])

    def step(self, pwm):
        dt = super().step(pwm)
        self.tick += 1
        return dt
