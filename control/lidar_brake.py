"""Point-cloud corridor detection and a deterministic stopping-distance rule."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BrakeConfig:
    lookahead_m: float = 16.0
    path_segments: int = 16
    corridor_radius_m: float = 0.72
    stand_off_m: float = 0.85
    deceleration_mps2: float = 2.5
    reaction_s: float = 0.25
    release_samples: int = 2


def first_path_return(points, valid, path, stations, radius_m):
    """Return the nearest occupied station and its observed surface point.

    Points and path vertices share a local metric frame. Stations are distances
    along the reference route from the current progress estimate. A clear path
    has infinite distance; its returned point is unused.
    """
    start = path[:, :-1]
    segment = path[:, 1:] - start
    length_sq = segment.square().sum(-1)[:, None, :]
    projected = points @ segment.transpose(-1, -2) - (start * segment).sum(-1)[:, None, :]
    fraction = (projected / length_sq.clamp_min(1e-8)).clamp(0, 1)
    start_dist_sq = (points.square().sum(-1, keepdim=True) + start.square().sum(-1)[:, None, :]
                     - 2 * (points @ start.transpose(-1, -2)))
    dist_sq = start_dist_sq - 2 * fraction * projected + fraction.square() * length_sq
    distance = stations[:-1] + fraction * stations.diff()
    inside = valid[:, :, None] & (dist_sq <= radius_m**2)
    nearest, index = torch.where(inside, distance, torch.inf).flatten(1).min(dim=1)
    row = torch.arange(len(points), device=points.device)
    return nearest, points[row, index // segment.shape[1]]


def lidar_confirms_clear(points, valid, target, scan_hw, range_m, *, tolerance_m=0.20):
    """Require a clear ray neighborhood through a previously observed surface point.

    The HITSZ LiDAR has full azimuth and an elevation span from -19° to +40°.
    A point outside the current field of view is unobserved, including after pitch changes.
    """
    height, width = scan_hw
    distance = target.norm(dim=-1)
    horizontal = target[:, :2].norm(dim=-1)
    elevation = torch.atan2(-target[:, 2], horizontal).rad2deg()
    azimuth = torch.atan2(target[:, 1], target[:, 0])
    y = ((40 - elevation) * height / 59).floor().long()
    x = ((azimuth / torch.pi + 1) * width / 2).floor().long()
    offsets = torch.arange(-1, 2, device=points.device)
    rows = (y[:, None] + offsets).clamp(0, height - 1)
    cols = (x[:, None] + offsets) % width
    index = (rows[:, :, None] * width + cols[:, None, :]).flatten(1)
    batch = torch.arange(len(points), device=points.device)[:, None]
    ranges = torch.where(valid[batch, index], points[batch, index].norm(dim=-1), range_m)
    in_view = (y >= 1) & (y < height - 1) & (distance > 0) & (distance + tolerance_m < range_m)
    return in_view & (ranges > distance[:, None] + tolerance_m).all(dim=-1)


def stopping_speed(distance_m, deceleration_mps2, reaction_s):
    """Solve speed * reaction + speed² / (2 * deceleration) <= distance."""
    reaction_speed = deceleration_mps2 * reaction_s
    return (reaction_speed**2 + 2 * deceleration_mps2 * distance_m.clamp_min(0)).sqrt() - reaction_speed
