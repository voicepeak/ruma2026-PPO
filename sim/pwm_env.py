"""Direct motor input for the pinned HITSZ full dynamics, in NED/FRD frames."""

import torch
import torch.nn.functional as F


CONTROL_HZ = 100
DT = 1 / CONTROL_HZ


def rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


class PWMEnv:
    """Advance an existing HITSZ Env by one 10 ms motor-command interval."""

    def __init__(self, env, *, wind=None):
        if env.dynamics != "full" or env.keep_grad:
            raise ValueError("PWM requires HITSZ full dynamics with keep_grad=false")
        self.env = env
        self.wind = wind
        self.gravity = env.p.new_tensor([0.0, 0.0, env.g])
        self.identity = torch.eye(3, device=env.device, dtype=env.p.dtype)

    @property
    def hover_pwm(self):
        return self.env.mass * self.env.g / (4 * self.env.Fmax)

    @torch.no_grad()
    def step(self, pwm):
        """Accept (batch, 4) normalized PWM in HITSZ alloc_mat column order."""
        e = self.env
        if pwm.shape != e.f.shape or pwm.device != e.f.device or pwm.dtype != e.f.dtype:
            raise ValueError("PWM must match the shape, device and dtype of the motor state")
        if not torch.isfinite(pwm).all() or ((pwm < 0) | (pwm > 1)).any():
            raise ValueError("PWM values must be finite and in [0, 1]")

        # Match the existing HITSZ motor lag and 100 Hz rigid-body integration.
        e._advance_cars(DT)
        e.f = torch.lerp(e.f, pwm * e.Fmax, DT / e.motor_tau)
        wrench = e.f @ e.alloc_mat.T
        angular_accel = (wrench[:, :3] - torch.cross(e.w, e.J * e.w, dim=-1)) / e.J
        e.w = e.w + angular_accel * DT

        wx, wy, wz = (e.w * DT).unbind(-1)
        zero = torch.zeros_like(wx)
        cross = torch.stack((zero, -wz, wy, wz, zero, -wx, -wy, wx, zero), dim=-1)
        cross = cross.reshape_as(e.R)
        angle = (e.w * DT).norm(dim=-1)[:, None, None]
        delta = self.identity + torch.sinc(angle / torch.pi) * cross
        delta = delta + 0.5 * torch.sinc(angle / (2 * torch.pi)).square() * (cross @ cross)
        rotated = e.R @ delta
        x = F.normalize(rotated[..., 0], dim=-1)
        y = F.normalize(rotated[..., 1] - (rotated[..., 1] * x).sum(-1, keepdim=True) * x, dim=-1)
        e.R = torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)

        airspeed = e.v if self.wind is None else e.v - self.wind.step(DT)
        drag = e.lin_drag_coef * airspeed * airspeed.abs()
        e.a = e.R[..., 2] * (wrench[:, 3:4] / e.mass) + self.gravity - drag
        e.p_old = e.p
        e.p = e.p + e.v * DT + 0.5 * e.a * DT**2
        e.v = e.v + e.a * DT
        e._check_finite({"position": e.p, "velocity": e.v, "rotation": e.R})


def clearance(env, before, samples=3, *, end=None):
    """Minimum signed box clearance along the last segment, using current voxel poses."""
    if end is None:
        end = env.p
    alpha = torch.linspace(0, 1, samples, device=env.device, dtype=env.p.dtype)
    points = torch.lerp(before[:, None, :], end[:, None, :], alpha[None, :, None])
    boxes = env.voxels[:, None, :, :]
    local = (points[:, :, None, :] - boxes[..., :3]).unsqueeze(-2)
    local = (local @ env.voxels_R[:, None, :, :, :]).squeeze(-2)
    excess = local.abs() - boxes[..., 3:]
    distance = excess.clamp_min(0).norm(dim=-1) + excess.amax(dim=-1).clamp_max(0)
    return distance.amin(dim=(1, 2)) - env.margin
