"""Replay frozen native PPO validation and audit the first gate miss or contact."""

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from ruma_ppo.env.maps.gate_graph import GateGraph
from ruma_ppo.evaluation.harness import BATCH, SUITES, load_native_ppo
from ruma_ppo.evaluation.rollout import COLLISION, MISSED_GATE, Progress, gate_crossing
from ruma_ppo.hitsz.branches import NEXT_GATE, ROUTE, UP
from ruma_ppo.hitsz.legacy_policy import Course


def signed_distance(offset, half_size):
    excess = offset.abs() - half_size
    return excess.clamp_min(0).norm(dim=-1) + excess.amax(dim=-1).clamp_max(0)


def identify_box(env, index):
    for name, count in (
        ("fixed_obstacle", env.num_voxels),
        ("background", env.num_bg_voxels),
        ("ground", env.num_bottom_voxels),
        ("gate_frame", env.num_gates * 6),
        ("moving_vehicle", env.num_cars),
    ):
        if index < count:
            return name, index
        index -= count
    raise ValueError("Collision object outside voxel ranges")


def contact_details(progress, start, end, affected, initial, history):
    env = progress.env
    if not len(affected):
        return {}
    alpha = env.p.new_tensor([0.0] if initial else [0.0, 0.5, 1.0])
    points = torch.lerp(start[affected, None], end[affected, None], alpha[None, :, None])
    boxes, rotations = env.voxels[affected], env.voxels_R[affected]
    distances = signed_distance(
        ((points[:, :, None, :] - boxes[:, None, :, :3]).unsqueeze(-2)
         @ rotations[:, None]).squeeze(-2),
        boxes[:, None, :, 3:],
    ) - env.margin[affected, None, None]
    minimum, winner = distances.flatten(1).min(dim=1)
    torch.testing.assert_close(minimum, progress.step_clearance_m[affected], rtol=0, atol=2e-6)
    sample, collider = winner // boxes.shape[1], winner % boxes.shape[1]
    indices = torch.arange(len(affected), device=env.device)
    center, half_size = boxes[indices, collider, :3], boxes[indices, collider, 3:]
    basis = rotations[indices, collider]
    impact = points[indices, sample]
    locations = env.p.clone()
    locations[affected] = impact
    route_delta, _ = env.get_nearest_traj_state(locations)
    centerline_local = ((impact + route_delta[affected] - center)[:, None] @ basis).squeeze(1)
    centerline_clearance = signed_distance(centerline_local, half_size) - env.margin[affected]

    results = {}
    for i, row in enumerate(affected.tolist()):
        index = int(collider[i])
        kind, local = identify_box(env, index)
        elapsed = float(progress.flight_s[row])
        cutoff = max(0.0, elapsed - 0.5)
        previous_time, positions, velocities, car_boxes, car_rotations = min(
            history, key=lambda frame: abs(frame[0] - cutoff)
        )
        if kind == "moving_vehicle":
            earlier_center = car_boxes[row, local, :3]
            earlier_basis = car_rotations[row, local]
        else:
            earlier_center = center[i]
            earlier_basis = basis[i]
        earlier_clearance = signed_distance(
            (positions[row] - earlier_center) @ earlier_basis, half_size[i]
        ) - env.margin[row]
        results[row] = {
            "collision_object": kind,
            "voxel_index": index,
            "box_index_within_type": local,
            "collision_gate_frame_id": local // 6 if kind == "gate_frame" else None,
            "initial_contact": initial,
            "contact_clearance_m": float(minimum[i]),
            "route_centerline_clearance_m": float(centerline_clearance[i]),
            "route_vertical_offset_at_contact_m": float(-route_delta[row, 2]),
            "impact_speed_mps": float(env.v[row].norm()),
            "prior_sample_s": previous_time,
            "prior_same_object_clearance_m": float(earlier_clearance),
            "prior_speed_mps": float(velocities[row].norm()),
        }
    return results


def miss_details(progress, start, end, next_gate, missed, graph):
    if not len(missed):
        return {}
    index = next_gate[missed]
    center = progress.center[index]
    basis = progress.basis[index]
    half = progress.half_aperture[index]
    cross, inside, fraction = gate_crossing(start[missed], end[missed], center, basis, half)
    assert cross.all() and not inside.any()
    positions = torch.lerp(start[missed], end[missed], fraction[:, None])
    offset = ((positions - center)[:, None] @ basis).squeeze(1)[:, 1:]
    result = {}
    for i, row in enumerate(missed.tolist()):
        lateral, down = offset[i].tolist()
        width, height = half[i].tolist()
        lateral_miss = abs(lateral) > width
        vertical_miss = abs(down) > height
        result[row] = {
            "missed_gate_id": graph.gates[int(index[i])].id,
            "gate_lateral_m": lateral,
            "gate_down_m": down,
            "miss_axes": "both" if lateral_miss and vertical_miss else "vertical" if vertical_miss else "lateral",
            "miss_side": ("below" if down > 0 else "above") if vertical_miss else ("right" if lateral > 0 else "left"),
        }
    return result


def table(path, rows):
    fields = dict.fromkeys(k for row in rows for k in row)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(archived, row):
    assert row["batch_sha256"] == archived["batch_sha256"]
    assert row["outcome"] == archived["outcome"] and row["gates"] == int(archived["gates"])
    for key in ("flight_s", "min_clearance_m"):
        assert abs(row[key] - float(archived[key])) <= 2e-5, (row, archived)
    if row["failure_ned"] is not None:
        np.testing.assert_allclose(row["failure_ned"], json.loads(archived["failure_ned"]), rtol=0, atol=1e-5)


def snapshot(course, time_s):
    env = course.e
    return time_s, env.p.clone(), env.v.clone(), env.car_voxels.clone(), env.car_voxels_R.clone()


def gate_centering_probe(course, predictive):
    from ruma_ppo.training.residual import apply_vertical_delta, geometric_height_delta

    original_step = course.env.step
    progress = None

    def set_progress(value):
        nonlocal progress
        progress = value

    def step(actions):
        index = progress.gates.clamp_max(len(progress.center) - 1)
        center, basis = progress.center[index], progress.basis[index]
        if predictive:
            delta = geometric_height_delta(course.e.p, course.e.v, center, basis, ~progress.done)
            return original_step(apply_vertical_delta(actions, delta))
        relative = ((course.e.p - center)[:, None] @ basis).squeeze(1)
        ahead = -relative[:, 0]
        proximity = (1 - ahead / 10.0).clamp(0, 1)
        upward = (relative[:, 2] - 0.75).clamp(0, 0.8)
        active = (ahead > 0) & ~progress.done
        adjusted = actions.clone()
        adjusted[:, 2] -= upward * proximity * active
        return original_step(adjusted)

    course.env.step = step
    return set_progress


def capture_vehicle_forecast(policy, course):
    from ruma_ppo.training.hitsz_joint import VehicleOccupancy

    raster = VehicleOccupancy(course.e, policy.policy.occupancy)
    history = deque(maxlen=18)
    progress = None

    def set_progress(value):
        nonlocal progress
        progress = value
        history.clear()

    def inference(obs):
        actions, logits = policy.policy.actor_outputs(obs)
        probabilities = logits.sigmoid()
        forecast_residual = policy.policy.forecast_velocity(probabilities)
        history.append((progress.time_s, course.e.p.clone(), course.env._current_R_yaw.clone(),
                        course.e.v.clone(), actions.clone(), probabilities, forecast_residual))
        return actions

    def impact_predictions(progress, contacts):
        result = {}
        env = course.e
        for row, contact in contacts.items():
            if not history:
                continue
            elapsed = float(progress.flight_s[row])
            result[row] = {
                "last_forward_command_mps": float(history[-1][4][row, 0]),
                "last_forecast_forward_adjustment_mps": float(history[-1][6][row, 0]),
            }
            if contact["collision_object"] != "moving_vehicle":
                continue
            car = contact["box_index_within_type"]
            box, rotation = env.car_voxels[row, car], env.car_voxels_R[row, car]
            for horizon_index, seconds in enumerate(raster.horizons.tolist()):
                if elapsed < seconds:
                    continue
                t, position, yaw, velocity, action, prediction, residual = min(
                    history, key=lambda frame: abs(frame[0] - (elapsed - seconds))
                )
                center = (box[:3] - position[row]) @ yaw[row]
                half = (yaw[row].T @ rotation).abs() @ box[3:]
                hits = ((raster.cells - center).abs() <= (half + raster.half_cell)).all(dim=-1)
                label = f"horizon_{seconds:g}s"
                result[row].update({
                    f"{label}_timing_error_s": t + seconds - elapsed,
                    f"{label}_forward_command_mps": float(action[row, 0]),
                    f"{label}_speed_mps": float(velocity[row].norm()),
                    f"{label}_forecast_forward_adjustment_mps": float(residual[row, 0]),
                    f"{label}_target_cells": int(hits.sum()),
                })
                if hits.any():
                    probs = prediction[row].reshape(len(raster.horizons), -1)[horizon_index, hits]
                    result[row].update({
                        f"{label}_peak_probability": float(probs.max()),
                        f"{label}_mean_probability": float(probs.mean()),
                    })
        return result

    policy.policy.act_inference = inference
    return set_progress, impact_predictions


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("checkpoint", type=Path)
    args.add_argument("scene_reference", type=Path, nargs="?")
    args.add_argument("out", type=Path)
    args.add_argument("--expected-model")
    probes = args.add_mutually_exclusive_group()
    probes.add_argument("--probe", action="store_true")
    probes.add_argument("--predictive-probe", action="store_true")
    args.add_argument("--stress", action="store_true")
    args.add_argument("--fresh-seeds", type=int, nargs="+")
    args.add_argument("--levels", type=int, nargs="+")
    options = args.parse_args()
    if (options.scene_reference is None) == (options.fresh_seeds is None):
        args.error("Provide either an archived scenario CSV or --fresh-seeds")
    expected = None
    if options.scene_reference:
        with options.scene_reference.open() as stream:
            expected = {(int(r["level"]), int(r["seed"]), int(r["row"])): r for r in csv.DictReader(stream)
                        if r["model"] == (options.expected_model or "native_joint_2501")}
        assert len(expected) == 192

    graph = GateGraph.from_hitsz(UP / ROUTE).select(range(NEXT_GATE, NEXT_GATE + 3))
    course = Course(BATCH)
    policy, info = load_native_ppo(options.checkpoint, course)
    if options.expected_model:
        assert options.expected_model == info["name"]
    set_probe_progress = gate_centering_probe(course, options.predictive_probe) if options.probe or options.predictive_probe else None
    stress = capture_vehicle_forecast(policy, course) if options.stress else None
    options.out.mkdir()
    rows = []
    default_levels, default_seeds = SUITES["val192"]
    levels, seeds = options.levels or default_levels, options.fresh_seeds or default_seeds
    for level in levels:
        for seed in seeds:
            fingerprint = course.start(level, seed, False)
            env = course.e
            progress = Progress(env, graph)
            if set_probe_progress is not None:
                set_probe_progress(progress)
            if stress:
                stress[0](progress)
            collision_predictions = {}
            history = deque([snapshot(course, 0.0)], maxlen=18)
            initial_clearance = progress.minimum_m.tolist()
            last_crossing = {}
            initial = (progress.outcome == COLLISION).nonzero().flatten()
            contacts = contact_details(progress, env.p, env.p, initial, True, history)
            missed = {}
            tick = 0
            while not progress.done.all():
                start = env.p.clone()
                next_gate = progress.gates.clone()
                alive = ~progress.done
                before_time = progress.time_s
                dt = policy.step(tick, progress)
                end = torch.lerp(start, env.p, min(dt, progress.limit_s - before_time) / dt)
                progress.advance(start, dt)
                crossed = ((progress.gates > next_gate) | (alive & (progress.outcome == MISSED_GATE))).nonzero().flatten()
                index = next_gate[crossed]
                center, basis = progress.center[index], progress.basis[index]
                _, _, fraction = gate_crossing(start[crossed], end[crossed], center, basis, progress.half_aperture[index])
                hit = torch.lerp(start[crossed], end[crossed], fraction[:, None])
                offset = ((hit - center)[:, None] @ basis).squeeze(1)[:, 1:].tolist()
                for row, gate_index, (lateral, down) in zip(crossed.tolist(), index.tolist(), offset, strict=True):
                    last_crossing[row] = {
                        "final_crossing_gate_id": graph.gates[gate_index].id,
                        "final_crossing_lateral_m": lateral,
                        "final_crossing_down_m": down,
                        "final_crossing_lower_margin_m": graph.gates[gate_index].aperture_m[1] / 2 - down,
                    }
                history.append(snapshot(course, progress.time_s))
                new_contacts = contact_details(
                    progress, start, end, (alive & (progress.outcome == COLLISION)).nonzero().flatten(),
                    False, history,
                )
                contacts.update(new_contacts)
                if stress and new_contacts:
                    collision_predictions.update(stress[1](progress, new_contacts))
                missed.update(miss_details(
                    progress, start, end, next_gate,
                    (alive & (progress.outcome == MISSED_GATE)).nonzero().flatten(), graph,
                ))
                tick += 1
            for number, score in enumerate(progress.rows()):
                scene_key = [graph.route, level, seed, number, graph.sha256, fingerprint]
                scene_hash = hashlib.sha256(json.dumps(scene_key, separators=(",", ":")).encode()).hexdigest()
                row = {
                    "route": graph.route, "level": level, "seed": seed, "row": number,
                    "scene_sha256": scene_hash, "batch_sha256": fingerprint,
                    "initial_clearance_m": initial_clearance[number], **score,
                    **contacts.get(number, {}), **missed.get(number, {}), **last_crossing.get(number, {}),
                    **collision_predictions.get(number, {}),
                }
                if expected:
                    reference = expected[level, seed, number]
                    assert fingerprint == reference["batch_sha256"] and scene_hash == reference["scene_sha256"]
                    if options.expected_model:
                        compare(reference, row)
                rows.append(row)
            print(info["name"], level, seed, len(rows), flush=True)

    table(options.out / "scenes.csv", rows)
    misses = [r for r in rows if r["outcome"] == "missed_gate"]
    initial = [r for r in rows if r.get("initial_contact")]
    airborne = [r for r in rows if r.get("initial_contact") is False]
    assert len(initial) + len(airborne) == sum(r["outcome"] == "collision" for r in rows)
    summary = {
        "model": info, "cases": len(rows),
        "gate_centering_probe": "predictive" if options.predictive_probe else options.probe,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "miss_axes": dict(Counter(r["miss_axes"] for r in misses)),
        "miss_sides": dict(Counter(r["miss_side"] for r in misses)),
        "miss_gate_ids": dict(Counter(r["missed_gate_id"] for r in misses)),
        "vertical_miss_median_outside_m": median([abs(r["gate_down_m"]) - 1.5 for r in misses if r["miss_axes"] != "lateral"] or [0.0]),
        "initial_contacts": dict(Counter(r["collision_object"] for r in initial)),
        "airborne_collisions": dict(Counter(r["collision_object"] for r in airborne)),
        "airborne_collisions_by_1_second": sum(r["flight_s"] <= 1.0 for r in airborne),
        "airborne_collision_box_also_blocks_route_centerline": sum(r["route_centerline_clearance_m"] < 0 for r in airborne),
    }
    if options.stress:
        clean = [r for r in rows if r["initial_clearance_m"] >= 0.3]
        impacts = [r for r in clean if r["outcome"] == "collision"]
        clean_start = {
            "cases": len(clean), "outcomes": dict(Counter(r["outcome"] for r in clean)),
            "collider_counts": dict(Counter(r["collision_object"] for r in impacts)),
            "collisions_by_1_second": dict(Counter(r["collision_object"] for r in impacts if r["flight_s"] <= 1.0)),
        }
        summary["start_clearance_at_least_0.3m"] = clean_start
    (options.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
