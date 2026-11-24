"""Synchronized synthetic RGB-D cameras and LiDAR from native HITSZ voxels."""

from collections import deque
from dataclasses import dataclass
import math

import torch

from ruma_ppo.env.sim.pwm_env import CONTROL_HZ, rotate


@dataclass(frozen=True)
class Mount:
    yaw_deg: float = 0.0
    offset_m: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SensorConfig:
    cameras: tuple = (Mount(), Mount(90), Mount(180), Mount(-90))
    lidar: Mount = Mount()
    image_hw: tuple = (48, 80)
    lidar_hw: tuple = (20, 1000)
    camera_hfov_deg: float = 94.0
    range_m: float = 50.0
    sample_every_ticks: int = 10
    latency_ticks: int = 0
    range_noise_m: float = 0.0
    dropout: float = 0.0


@dataclass(frozen=True)
class SensorFrame:
    capture_tick: int
    available_tick: int
    position_ned: torch.Tensor
    body_to_ned: torch.Tensor
    camera_depth_m: torch.Tensor
    camera_rgb: torch.Tensor
    lidar_xyz_m: torch.Tensor
    lidar_valid: torch.Tensor

    @property
    def timestamp_s(self):
        return self.capture_tick / CONTROL_HZ


class Renderer:
    def __init__(self, env, config=SensorConfig(), seed=0):
        if not 1 <= len(config.cameras) <= 4 or min(*config.image_hw, *config.lidar_hw) < 1:
            raise ValueError("Expected one to four cameras and positive raster dimensions")
        if math.prod(config.lidar_hw) > 20_000 or not 0 < config.camera_hfov_deg < 180:
            raise ValueError("Invalid LiDAR point budget or camera field of view")
        if config.sample_every_ticks < 1 or config.latency_ticks < 0 or not 0 <= config.dropout <= 1:
            raise ValueError("Invalid sensor cadence, latency, or dropout")
        if not math.isfinite(config.range_m) or config.range_m <= 0 or not math.isfinite(
            config.range_noise_m
        ) or config.range_noise_m < 0:
            raise ValueError("Invalid sensor range or range noise")
        import quadsim_cuda

        self.engine = quadsim_cuda
        self.env, self.config = env, config
        self.mounts = [self._mount(mount) for mount in (*config.cameras, config.lidar)]
        self.fov = math.tan(math.radians(config.camera_hfov_deg) / 2)
        h, w = config.lidar_hw
        elevation = (40 - 59 * (torch.arange(h, device=env.device) + 0.5) / h).deg2rad()
        azimuth = torch.pi * (2 * (torch.arange(w, device=env.device) + 0.5) / w - 1)
        el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
        self.lidar_dirs = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), -el.sin()), dim=-1)
        self.rng = torch.Generator(device=env.device)
        self.reset(seed)

    def _mount(self, mount):
        if len(mount.offset_m) != 3 or not all(map(math.isfinite, (*mount.offset_m, mount.yaw_deg))):
            raise ValueError("A mount needs a finite yaw and three body-frame coordinates")
        yaw = math.radians(mount.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        return self.env.p.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]), self.env.p.new_tensor(mount.offset_m)

    def reset(self, seed):
        self.rng.manual_seed(seed)
        self.pending = deque()
        self.last_tick = -1

    def _range(self, raw):
        cfg = self.config
        valid = torch.isfinite(raw) & (raw > 0) & (raw < cfg.range_m)
        depth = raw.clamp(0, cfg.range_m)
        if cfg.range_noise_m:
            noise = torch.randn(depth.shape, device=depth.device, generator=self.rng)
            depth = (depth + cfg.range_noise_m * noise).clamp(0, cfg.range_m)
            valid &= (depth > 0) & (depth < cfg.range_m)
        if cfg.dropout:
            valid &= torch.rand(depth.shape, device=depth.device, generator=self.rng) >= cfg.dropout
        return torch.where(valid, depth, cfg.range_m), valid

    @torch.no_grad()
    def capture_lidar(self):
        """Render one immediate scan in the LiDAR mount frame, without the cameras."""
        e, cfg = self.env, self.config
        orientation, offset = self.mounts[-1]
        raw = e.p.new_empty((e.batch_size, *cfg.lidar_hw))
        self.engine.render_panorama_cuda(raw, e.empty_cylinders, e.empty_cylinders_R, e.voxels,
                                         e.voxels_R, e.R @ orientation, e.p + rotate(e.R, offset))
        distance, valid = self._range(raw)
        xyz = self.lidar_dirs * torch.where(valid, distance, 0)[..., None]
        return xyz.flatten(1, 2), valid.flatten(1, 2)

    @torch.no_grad()
    def _capture(self, tick):
        e, cfg = self.env, self.config
        depth_frames, rgb_frames = [], []
        gate_start = e.num_voxels + e.num_bg_voxels + e.num_bottom_voxels
        gates = slice(gate_start, gate_start + 6 * e.num_gates)
        for orientation, offset in self.mounts[:-1]:
            rotation, origin = e.R @ orientation, e.p + rotate(e.R, offset)
            raw = e.p.new_empty((e.batch_size, *cfg.image_hw))
            self.engine.render_cuda(raw, e.empty_cylinders, e.empty_cylinders_R,
                                    e.voxels, e.voxels_R, rotation, origin, self.fov)
            gate_depth = torch.empty_like(raw)
            self.engine.render_cuda(gate_depth, e.empty_cylinders, e.empty_cylinders_R,
                                    e.voxels[:, gates], e.voxels_R[:, gates], rotation, origin, self.fov)
            depth, hit = self._range(raw)
            gate_hit = (raw - gate_depth).abs() <= 1e-3
            obstacle_rgb = raw.new_tensor([155, 171, 181])
            gate_rgb = raw.new_tensor([246, 169, 55])
            surface = torch.where(gate_hit[..., None], gate_rgb, obstacle_rgb)
            light = (0.35 + 0.65 * (1 - depth / cfg.range_m))[..., None]
            background = raw.new_tensor([24, 29, 37])
            rgb_frames.append(torch.where(hit[..., None], surface * light, background).to(torch.uint8))
            depth_frames.append(depth)

        xyz, valid = self.capture_lidar()
        return SensorFrame(tick, tick + cfg.latency_ticks, e.p.clone(), e.R.clone(),
                           torch.stack(depth_frames, dim=1), torch.stack(rgb_frames, dim=1), xyz, valid)

    def update(self, tick):
        """Capture and release whole synchronized frames. LiDAR XYZ uses the LiDAR mount frame."""
        if not isinstance(tick, int) or tick <= self.last_tick:
            raise ValueError("Sensor ticks must increase within each episode")
        self.last_tick = tick
        if tick % self.config.sample_every_ticks == 0:
            self.pending.append(self._capture(tick))
        ready = []
        while self.pending and self.pending[0].available_tick <= tick:
            ready.append(self.pending.popleft())
        return ready
