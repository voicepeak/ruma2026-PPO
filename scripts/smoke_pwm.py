"""Verify native PWM dynamics, synchronized sensors, and a replayable first-gate flight."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ruma_ppo.hitsz.branches import NEXT_GATE, ROUTE, UP
from ruma_ppo.hitsz.legacy_policy import Course
from ruma_ppo.evaluation.rollout import Progress, PWMRouteFollower
from ruma_ppo.env.maps.gate_graph import GateGraph
from ruma_ppo.env.sim.pwm_env import CONTROL_HZ, PWMEnv
from ruma_ppo.env.sim.render import Renderer, SensorConfig


SEED, LEVEL = 50731, 1


def reset_hover(course):
    fingerprint = course.start(LEVEL, SEED, False)
    e = course.e
    e.v.zero_()
    yaw = torch.atan2(e.R[:, 1, 0], e.R[:, 0, 0])
    c, s = yaw.cos(), yaw.sin()
    z, one = torch.zeros_like(c), torch.ones_like(c)
    e.R.copy_(torch.stack((c, -s, z, s, c, z, z, z, one), dim=-1).reshape_as(e.R))
    e.f.fill_(e.mass * e.g / 4)
    return fingerprint, yaw


def verify_motor_response(course, follower):
    e, sim = course.e, follower.sim
    reset_hover(course)
    origin = e.p.clone()
    hover = torch.full_like(e.f, sim.hover_pwm)
    for _ in range(CONTROL_HZ):
        sim.step(hover)
    hover_drift = (e.p - origin).norm().item()
    if hover_drift > 1e-3 or e.v.norm().item() > 1e-3:
        raise AssertionError(f"Hover drift {hover_drift:.6f} m")

    first_force = e.f.clone()
    sim.step(hover + 0.04)
    expected = first_force + 0.04 * e.Fmax * (1 / CONTROL_HZ) / e.motor_tau
    torch.testing.assert_close(e.f, expected, rtol=1e-6, atol=1e-6)
    if e.v[0, 2].item() >= 0:
        raise AssertionError("Increasing collective thrust must accelerate up in NED")
    reset_hover(course)
    delta = hover.new_tensor([[0.025, 0.025, -0.025, -0.025]])
    sim.step(hover + delta)
    if not (e.w[0, 2] > 0 and e.w[0, :2].abs().max() < 1e-6):
        raise AssertionError("HITSZ motor channel pairs must produce positive yaw with no roll/pitch")

    _, heading = reset_hover(course)
    altitude = e.p[:, 2:3].clone()
    e.v.copy_(e.R[..., 0] * 3)
    for _ in range(CONTROL_HZ):
        target = torch.cat((e.p[:, :2], altitude), dim=-1)
        pwm = follower.controller.command(e.p, e.v, e.R, e.w, target, torch.zeros_like(e.v), heading)
        sim.step(pwm)
    brake_speed = e.v.norm().item()
    if brake_speed >= 1.0:
        raise AssertionError(f"Brake ended at {brake_speed:.3f} m/s from 3 m/s")
    return {"hover_drift_m": hover_drift, "step_force_per_motor_n": expected[0, 0].item(),
            "brake_speed_mps_at_1s": brake_speed}


def fly(course, follower, graph, *, commands=None, sensors=None):
    fingerprint, _ = reset_hover(course)
    e = course.e
    progress = Progress(e, graph)
    if sensors:
        sensors.reset(SEED)
        frames = sensors.update(0)
    else:
        frames = []
    position, rotation, actions = [e.p.clone()], [e.R.clone()], []
    t0 = time.perf_counter()
    while not progress.done.all():
        tick = len(actions)
        if commands is not None and tick >= len(commands):
            raise RuntimeError("Recorded PWM ended before the episode finished")
        pwm = follower.command() if commands is None else commands[tick]
        actions.append(pwm.clone())
        dt = follower.step(pwm)
        progress.advance(position[-1], dt)
        position.append(e.p.clone())
        rotation.append(e.R.clone())
        if sensors:
            frames.extend(sensors.update(tick + 1))
    if not actions or (commands is not None and len(actions) != len(commands)):
        raise RuntimeError(f"Unexpected episode length: {list(progress.rows())}")
    torch.cuda.synchronize()
    rate = len(actions) / (time.perf_counter() - t0)
    trace = {"position_ned": torch.stack(position), "body_to_ned": torch.stack(rotation),
             "pwm": torch.stack(actions)}
    return fingerprint, list(progress.rows())[0], trace, frames, rate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    graph = GateGraph.from_hitsz(UP / ROUTE).select([NEXT_GATE])
    course = Course(1)
    e = course.e
    gate = graph.gates[0]
    part = gate.id * 6
    torch.testing.assert_close(e.p.new_tensor(gate.center_ned),
                               e.base_voxels_gate[0, part:part + 2, :3].mean(0), rtol=0, atol=2e-4)
    torch.testing.assert_close(e.R.new_tensor(gate.basis_ned), e.base_voxels_gate_R[0, part], rtol=0, atol=2e-5)
    follower = PWMRouteFollower(e)
    motors = verify_motor_response(course, follower)
    renderer = Renderer(e, SensorConfig(latency_ticks=2, range_noise_m=0.015, dropout=0.02))
    fingerprint, result, trace, frames, rate = fly(course, follower, graph, sensors=renderer)
    if result["outcome"] != "success" or not frames:
        raise AssertionError(f"Smoke course failed: {result}")
    first = frames[0]
    assert first.capture_tick == 0 and first.available_tick == 2
    assert first.camera_rgb.shape == (1, 4, 48, 80, 3) and first.camera_depth_m.shape == (1, 4, 48, 80)
    assert first.lidar_xyz_m.shape == (1, 20_000, 3) and first.lidar_valid.any()
    assert torch.isfinite(first.lidar_xyz_m).all() and torch.isfinite(first.camera_depth_m).all()
    assert all(frame.available_tick - frame.capture_tick == 2 for frame in frames)

    traj = {key: value.cpu().numpy() for key, value in trace.items()}
    np.savez_compressed(args.out / "flight.npz", **traj)
    with np.load(args.out / "flight.npz") as data:
        commands = torch.from_numpy(data["pwm"]).to(e.device)
    replay_hash, replay_score, replay, _, replay_rate = fly(course, follower, graph, commands=commands)
    assert (replay_hash, replay_score) == (fingerprint, result)
    for key in trace:
        torch.testing.assert_close(trace[key], replay[key], rtol=0, atol=0)

    reset_hover(course)
    renderer.reset(SEED)
    assert not renderer.update(0) and not renderer.update(1)
    repeat_frame, = renderer.update(2)
    for key in ("camera_rgb", "camera_depth_m", "lidar_xyz_m", "lidar_valid"):
        torch.testing.assert_close(getattr(first, key), getattr(repeat_frame, key), rtol=0, atol=0)
    np.savez_compressed(args.out / "first_frame.npz", camera_rgb=first.camera_rgb.cpu().numpy(),
                        camera_depth_m=first.camera_depth_m.cpu().numpy(), lidar_xyz_m=first.lidar_xyz_m.cpu().numpy(),
                        lidar_valid=first.lidar_valid.cpu().numpy(), capture_tick=first.capture_tick,
                        available_tick=first.available_tick, position_ned=first.position_ned.cpu().numpy(),
                        body_to_ned=first.body_to_ned.cpu().numpy())
    report = {"seed": SEED, "obstacle_level": LEVEL, "route": ROUTE, "gate": gate.id,
              "graph_sha256": graph.sha256, "scene_sha256": fingerprint, "control_hz": CONTROL_HZ, **motors,
              "flight_with_sensors_ticks_per_wall_second": round(rate, 2),
              "replay_ticks_per_wall_second": round(replay_rate, 2), "replay_identical": True,
              "sensor_frames_released": len(frames), "first_lidar_returns": first.lidar_valid.sum().item(),
              **result}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
