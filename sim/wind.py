"""Seeded NED air velocity for the HITSZ PWM simulation."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class WindSpec:
    steady_ned_mps: tuple = (0.0, 0.0, 0.0)
    gust_ned_mps: tuple = (0.0, 0.0, 0.0)
    gust_start_s: float = 0.0
    gust_duration_s: float = 1.0
    turbulence_std_mps: float = 0.0
    turbulence_timescale_s: float = 0.5


class WindField:
    """A steady wind, smooth finite gust and per-aircraft correlated turbulence."""

    def __init__(self, velocity_template, spec=WindSpec(), seed=0):
        if velocity_template.ndim != 2 or velocity_template.shape[-1] != 3 or not velocity_template.is_floating_point():
            raise ValueError("Expected floating-point aircraft velocities with shape (batch, 3)")
        for vector in (spec.steady_ned_mps, spec.gust_ned_mps):
            if len(vector) != 3 or not all(map(math.isfinite, vector)):
                raise ValueError("Wind vectors must have three finite NED components")
        times = (spec.gust_start_s, spec.gust_duration_s, spec.turbulence_std_mps, spec.turbulence_timescale_s)
        if not all(map(math.isfinite, times)) or min(times) < 0 or min(spec.gust_duration_s, spec.turbulence_timescale_s) <= 0:
            raise ValueError("Wind durations and turbulence parameters are out of range")
        self.spec = spec
        self.steady = velocity_template.new_tensor(spec.steady_ned_mps)
        self.gust = velocity_template.new_tensor(spec.gust_ned_mps)
        self.turbulence = torch.zeros_like(velocity_template)
        self.rng = torch.Generator(device=velocity_template.device)
        self.reset(seed)

    def reset(self, seed):
        self.time_s = 0.0
        self.turbulence.zero_()
        self.rng.manual_seed(seed)

    def step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Wind updates require a positive time step")
        self.time_s += dt
        cfg = self.spec
        phase = (self.time_s - cfg.gust_start_s) / cfg.gust_duration_s
        gust = math.sin(math.pi * phase)**2 if 0 < phase < 1 else 0.0
        if cfg.turbulence_std_mps:
            retain = math.exp(-dt / cfg.turbulence_timescale_s)
            scale = cfg.turbulence_std_mps * math.sqrt(-math.expm1(-2 * dt / cfg.turbulence_timescale_s))
            noise = torch.randn(self.turbulence.shape, device=self.turbulence.device,
                                dtype=self.turbulence.dtype, generator=self.rng)
            self.turbulence = retain * self.turbulence + scale * noise
        return self.steady + gust * self.gust + self.turbulence
