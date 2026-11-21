"""Lateral route candidates and deterministic selection from LiDAR clearance."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BypassConfig:
    lateral_offsets_m: tuple = (0.0, -1.1, 1.1, -2.05, 2.05)
    merge_distance_m: float = 3.1
    scoring_horizon_m: float = 9.0
    offset_cost: float = 0.45
    switch_cost: float = 0.40


def right_of(tangent):
    """Horizontal right unit vector in the HITSZ NED frame."""
    return F.normalize(torch.stack((-tangent[..., 1], tangent[..., 0], torch.zeros_like(tangent[..., 0])), dim=-1), dim=-1)


def lateral_profile(distance_m, current, goal, merge_distance_m):
    """Reach the chosen offset over a bounded forward distance."""
    return torch.lerp(current, goal, (distance_m / merge_distance_m).clamp(0, 1))


def candidate_scores(visible_m, offsets, previous, config):
    """Score visible progress, small offsets and continuity of the previous path."""
    current = offsets[previous, None]
    reward = visible_m.clamp_max(config.scoring_horizon_m)
    cost = config.offset_cost * offsets.abs() + config.switch_cost * (offsets - current).abs()
    return reward - cost


def choose_candidate(visible_m, offsets, previous, config):
    return candidate_scores(visible_m, offsets, previous, config).argmax(dim=-1)
