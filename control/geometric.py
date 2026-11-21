"""Position/velocity and SO(3) attitude feedback for the HITSZ PWM simulator."""

import math

import torch
import torch.nn.functional as F

from sim.pwm_env import rotate


class GeometricController:
    def __init__(self, env, *, max_tilt_deg=32.0):
        if not 0 < max_tilt_deg < 90:
            raise ValueError("Maximum tilt must lie between 0 and 90 degrees")
        self.mass, self.gravity, self.max_motor_force = env.mass, env.g, env.Fmax
        self.inertia = env.J
        self.inverse_allocator = torch.linalg.inv(env.alloc_mat)
        self.position_gain = env.p.new_tensor([2.4, 2.4, 4.0])
        self.velocity_gain = env.p.new_tensor([3.2, 3.2, 3.8])
        self.attitude_gain = env.p.new_tensor([0.30, 0.44, 0.30])
        self.rate_gain = env.p.new_tensor([0.085, 0.11, 0.11])
        self.tilt_limit = math.tan(math.radians(max_tilt_deg))

    def command(self, p, v, R, w, target_p, target_v, target_yaw):
        """Map a NED reference and FRD body state to four normalized motor commands."""
        acceleration = self.position_gain * (target_p - p) + self.velocity_gain * (target_v - v)
        vertical = (self.gravity - acceleration[:, 2]).clamp(0.35 * self.gravity, 1.7 * self.gravity)
        horizontal = -acceleration[:, :2]
        max_horizontal = vertical[:, None] * self.tilt_limit
        scale = (max_horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        down_force = torch.cat((horizontal * scale, vertical[:, None]), dim=-1)
        down = F.normalize(down_force, dim=-1)
        heading = torch.stack((target_yaw.cos(), target_yaw.sin(), torch.zeros_like(target_yaw)), dim=-1)
        right = F.normalize(torch.cross(down, heading, dim=-1), dim=-1)
        forward = torch.cross(right, down, dim=-1)
        desired = torch.stack((forward, right, down), dim=-1)

        relative = desired.transpose(-1, -2) @ R
        skew = (relative - relative.transpose(-1, -2)) / 2
        error = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=-1)
        torque = -self.attitude_gain * error - self.rate_gain * w
        torque = torque + torch.cross(w, self.inertia * w, dim=-1)
        thrust = self.mass * (down_force * R[..., 2]).sum(-1, keepdim=True)
        thrust = thrust.clamp(0, 4 * self.max_motor_force)

        # Reduce differential thrust uniformly when the four motors run out of headroom.
        wrench = torch.cat((torque, -thrust), dim=-1)
        normalized = rotate(self.inverse_allocator, wrench) / self.max_motor_force
        collective = thrust / (4 * self.max_motor_force)
        differential = normalized - collective
        positive = (1 - collective) / differential.amax(-1, keepdim=True).clamp_min(1e-8)
        negative = collective / (-differential.amin(-1, keepdim=True)).clamp_min(1e-8)
        return (collective + differential * torch.minimum(positive, negative).clamp_max(1)).clamp(0, 1)
