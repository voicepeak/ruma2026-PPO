"""Paired R1 gate tracking in calm air, steady crosswind and a crosswind gust."""

import argparse
import csv
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.rollout import Progress, PWMRouteFollower
from hitsz_ppo import Course, atomic_json
from scripts.train_bypass_ppo import gate_graph
from ruma_ppo.env.sim.pwm_env import CONTROL_HZ, PWMEnv
from ruma_ppo.env.sim.wind import WindField, WindSpec


BATCH = 24
GROUPS = ((2, 85001, 1), (3, 85002, -1))


def wind_spec(mode, right, sign):
    if mode == "calm":
        return WindSpec()
    direction = np.asarray(right) * sign
    steady = tuple(direction * 6)
    if mode == "steady":
        return WindSpec(steady_ned_mps=steady)
    return WindSpec(steady_ned_mps=steady, gust_ned_mps=tuple(direction * 5), gust_start_s=1.0,
                    gust_duration_s=2.0, turbulence_std_mps=0.6, turbulence_timescale_s=0.5)


def check_physics_and_seed():
    course = Course(1)
    e = course.e

    def hover_step(wind):
        course.start(1, 85000, False)
        e.v.zero_()
        e.R.copy_(torch.eye(3, device=e.device)[None])
        e.f.fill_(e.mass * e.g / 4)
        field = WindField(e.v, wind) if wind is not None else None
        sim = PWMEnv(e, wind=field)
        sim.step(torch.full_like(e.f, sim.hover_pwm))
        return e.a.clone()

    calm = hover_step(None)
    still = hover_step(WindSpec())
    torch.testing.assert_close(calm, still, rtol=0, atol=0)
    windy = hover_step(WindSpec(steady_ned_mps=(0.0, 6.0, 0.0)))
    expected = calm.clone()
    expected[:, 1] += e.lin_drag_coef[:, 1] * 36.0
    torch.testing.assert_close(windy, expected, rtol=1e-6, atol=1e-6)

    def trace(seed):
        field = WindField(e.v, WindSpec(turbulence_std_mps=0.6), seed=seed)
        return torch.stack([field.step(1 / CONTROL_HZ) for _ in range(CONTROL_HZ)])

    sample = trace(85000)
    torch.testing.assert_close(trace(85000), sample, rtol=0, atol=0)
    assert not torch.equal(sample, trace(85001))
    return {"calm_equals_zero_wind": True, "crosswind_accel_check_mps2": float(windy[0, 1]),
            "wind_trace_reproducible": True}


def fly(course, graph, level, seed, mode, sign):
    fingerprint = course.start(level, seed, False)
    e = course.e
    gate_offset = e.num_voxels + e.num_bg_voxels + e.num_bottom_voxels
    gate_frames = e.static_voxels[:, gate_offset:].clone()
    e.static_voxels[:, :gate_offset, :3] = 1e6
    e.static_voxels[:, :gate_offset, 3:] = 0
    e.active_num_cars = 0
    e._update_cars_idx()

    spec = wind_spec(mode, [row[1] for row in graph.gates[0].basis_ned], sign)
    field = None if mode == "calm" else WindField(e.v, spec, seed=seed + 1000)
    pilot = PWMRouteFollower(e, wind=field)
    progress = Progress(e, graph)
    peak_horizontal = e.p.new_zeros(e.batch_size)
    peak_vertical = e.p.new_zeros(e.batch_size)
    while not progress.done.all():
        active = ~progress.done
        before = e.p.clone()
        progress.advance(before, pilot.step(pilot.command()))
        delta, _ = e.get_nearest_traj_state(progress.position)
        peak_horizontal = torch.maximum(peak_horizontal, torch.where(active, delta[:, :2].norm(dim=-1), 0.0))
        peak_vertical = torch.maximum(peak_vertical, torch.where(active, delta[:, 2].abs(), 0.0))
    assert torch.equal(e.voxels[:, gate_offset:gate_offset + gate_frames.size(1)], gate_frames)
    assert not (e.car_voxels[..., 3:] != 0).any()
    rows = []
    for row, (score, horizontal, vertical) in enumerate(zip(progress.rows(), peak_horizontal.tolist(), peak_vertical.tolist())):
        rows.append({"level": level, "seed": seed, "row": row, "mode": mode, "initial_sha256": fingerprint,
                     **score, "peak_horizontal_m": horizontal, "peak_vertical_m": vertical})
    return fingerprint, rows, spec


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = check_physics_and_seed()
    graph, course = gate_graph(), Course(BATCH)
    rows, settings = [], []
    modes = ("calm", "steady", "gust_and_turbulence")
    for level, seed, sign in GROUPS:
        expected = None
        for mode in modes:
            fingerprint, scores, spec = fly(course, graph, level, seed, mode, sign)
            expected = fingerprint if expected is None else expected
            assert fingerprint == expected
            rows.extend(scores)
            settings.append({"level": level, "seed": seed, "mode": mode, **asdict(spec)})
            print("WIND_GROUP", level, seed, mode, dict(Counter(r["outcome"] for r in scores)), flush=True)
    level, seed, sign = GROUPS[0]
    replay_hash, replay, _ = fly(course, graph, level, seed, modes[-1], sign)
    first = [r for r in rows if r["mode"] == modes[-1] and r["seed"] == seed]
    assert replay == first and replay_hash == first[0]["initial_sha256"]
    checks["gust_rollout_reproducible"] = True
    with (args.out / "paired.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        times = [row["flight_s"] for row in selected if row["outcome"] == "success"]
        metrics[mode] = {"outcomes": dict(Counter(r["outcome"] for r in selected)),
                         "mean_success_s": float(np.mean(times)) if times else None,
                         "peak_horizontal_m_p50_p95_max": np.percentile([r["peak_horizontal_m"] for r in selected],
                                                                      [50, 95, 100]).tolist(),
                         "min_gateframe_clearance_m": min(r["min_clearance_m"] for r in selected)}
    report = {"paired_scenes": BATCH * len(GROUPS), "gate_graph_sha256": graph.sha256,
              "geometry": "original gate frames with randomized obstacles removed",
              "checks": checks, "profiles": settings, "metrics": metrics}
    atomic_json(args.out / "result.json", report)
    print("WIND_DONE", json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
