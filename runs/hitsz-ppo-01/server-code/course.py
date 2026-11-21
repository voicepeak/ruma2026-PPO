"""Planar, constant-altitude course proxy. Units: metres and seconds."""
import numpy as np
import torch

DT = 0.05
OFFSETS = (-3.6, -2.3, 0., 2.3, 3.6)
SPEED_FACTORS = (0.72, 1.)
CANDIDATES = [(offset, speed) for speed in SPEED_FACTORS for offset in OFFSETS]
OBS_NAMES = ["obstacle_x", "obstacle_y", "obstacle_vy", "radius", "start_speed",
             "turn_sign", "turn_size", "second_gate_x", "candidate_y", "candidate_speed"]


def generate(seed, families, device):
    """Four matched scenes per family: mirror the obstacle, then reverse the turn."""
    rng = np.random.default_rng(seed)
    def param(low, high):
        values = np.repeat(rng.uniform(low, high, families), 4)
        return torch.as_tensor(values, dtype=torch.float32, device=device)[:, None]

    mirror = torch.tensor([1., 1., -1., -1.], device=device).repeat(families)[:, None]
    turn = torch.tensor([1., -1., -1., 1.], device=device).repeat(families)[:, None]
    ox = param(9., 11.5)
    oy = mirror * param(-1.45, 1.45)
    ov = mirror * param(-0.65, 0.65)
    radius = param(0.9, 1.55)
    speed = param(5.5, 8.5)
    amplitude = param(2.9, 4.9)
    gx = param(19., 22.)
    candidates = torch.tensor(CANDIDATES, device=device)
    offset, factor = candidates.T
    values = torch.broadcast_tensors(ox, oy, ov, radius, speed, turn, amplitude, gx, offset, factor)
    features = torch.stack(values, dim=-1)
    # A: geometric rule, estimated clearance at closest approach under constant obstacle velocity.
    eta = ox / (speed * factor)
    heuristic = (offset - (oy + ov * eta)).abs() - radius - 0.32
    rule = heuristic.clamp(max=0.65) - 0.07 * offset.abs() + 0.02 * factor

    x, y = torch.zeros_like(rule), torch.zeros_like(rule)
    vx, vy = speed.expand_as(x).clone(), torch.zeros_like(y)
    gates = torch.zeros_like(x, dtype=torch.long)
    alive = torch.ones_like(x, dtype=torch.bool)
    passed_short = torch.zeros_like(alive)
    completed_time = torch.full_like(x, 8.)
    closest = torch.full_like(x, 100.)
    boundary = 5.8
    for step in range(160):
        first_leg = x < 14.
        # Same follower after the branch for all candidates, using the designated bend.
        blend = ((x + 3. - 14.) / (gx - 14.)).clamp(0., 1.)
        target_y = torch.where(first_leg, offset, turn * amplitude * blend)
        target_vx = torch.where(first_leg, speed * factor, speed)
        ax = 2.0 * (target_vx - vx)
        ay = 3.0 * (target_y - y) - 2.4 * vy
        limit = (8. / (ax.square() + ay.square()).sqrt().clamp_min(1e-6)).clamp(max=1.)
        vx_new, vy_new = vx + ax * limit * DT, vy + ay * limit * DT
        xn, yn = x + (vx + vx_new) * (0.5 * DT), y + (vy + vy_new) * (0.5 * DT)
        # Swept centre distance for a moving disc over a substep.
        rx, ry = x - ox, y - (oy + ov * (step * DT))
        dx, dy = xn - x, yn - y - ov * DT
        frac = (-(rx * dx + ry * dy) / (dx.square() + dy.square()).clamp_min(1e-9)).clamp(0., 1.)
        clearance = ((rx + frac * dx).square() + (ry + frac * dy).square()).sqrt() - radius - 0.32
        closest = torch.where(alive & first_leg, torch.minimum(closest, clearance), closest)
        alive = alive & (clearance > 0.) & (yn.abs() < boundary)
        passed_short |= alive & (x < 14.) & (xn >= 14.)
        # Detection gates are synthetic intervals at three ordered planes.
        for j, (plane, centre, halfwidth) in enumerate(((15., 0., 4.4), (gx, turn * amplitude, 1.5),
                                                         (28., turn * amplitude, 1.3))):
            crosses = alive & (x < plane) & (xn >= plane)
            alpha = ((plane - x) / (xn - x).clamp_min(1e-6)).clamp(0., 1.)
            valid = (y + alpha * (yn - y) - centre).abs() < halfwidth
            scored = crosses & valid & (gates == j)
            gates += scored.long()
            if j == 2:
                completed_time = torch.where(scored, (step + alpha) * DT, completed_time)
            alive &= ~(crosses & ~valid)
        x, y, vx, vy = xn, yn, vx_new, vy_new
    return dict(features=features, rule=rule, short=passed_short, gates=gates, time=completed_time,
                clearance=closest)

