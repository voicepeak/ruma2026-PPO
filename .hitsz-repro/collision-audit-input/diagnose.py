"""Replay the archived val192 PWM runs and identify the OBB at first contact."""

import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

from evaluation.harness import BATCH, SUITES
from evaluation.rollout import COLLISION, PWMRouteFollower, Progress
from hitsz_branches import NEXT_GATE, ROUTE, UP
from hitsz_ppo import Course
from maps.gate_graph import GateGraph
from sim.pwm_env import clearance


def signed_distance(delta, radius):
    excess = delta.abs() - radius
    return excess.clamp_min(0).norm(dim=-1) + excess.amax(dim=-1).clamp_max(0)


def label(e, index):
    border = 0
    for kind, size in (("route_obstacle", e.num_voxels), ("background", e.num_bg_voxels),
                       ("ground", e.num_bottom_voxels), ("gate_frame", 6 * e.num_gates)):
        if index < border + size:
            return kind, (index - border) // 6 if kind == "gate_frame" else None
        border += size
    return "moving_vehicle", None


def record_contacts(e, survivors, progress, before, previous_centers, tick, rows):
    impacted = (survivors & (progress.outcome == COLLISION)).nonzero().flatten()
    if not len(impacted):
        return
    alpha = e.p.new_tensor([0.0, 0.5, 1.0] if tick else [1.0])
    points = torch.lerp(before[impacted, None], e.p[impacted, None], alpha[None, :, None])
    boxes, rotations = e.voxels[impacted], e.voxels_R[impacted]
    offsets = points[:, :, None, :] - boxes[:, None, :, :3]
    local = (offsets.unsqueeze(-2) @ rotations[:, None]).squeeze(-2)
    dist = signed_distance(local, boxes[:, None, :, 3:]) - e.margin[impacted, None, None]
    idx = torch.arange(len(impacted), device=e.device)
    winner = dist.flatten(1).argmin(1)
    sample, collider = winner // boxes.size(1), winner % boxes.size(1)
    impact = points[idx, sample]
    center, half = boxes[idx, collider, :3], boxes[idx, collider, 3:]
    basis = rotations[idx, collider]
    # Calculate with the original environment indexing, as its route projection is batched.
    full_impact = e.p.clone()
    full_impact[impacted] = impact
    route_delta, _ = e.get_nearest_traj_state(full_impact)
    projection = impact + route_delta[impacted]
    selected_dist = dist[idx, sample, collider]
    torch.testing.assert_close(selected_dist, progress.minimum_m[impacted], rtol=0, atol=2e-6)
    projected_clear = signed_distance(((projection - center)[:, None] @ basis).squeeze(1), half)
    end_clear = signed_distance(((e.p[impacted] - center)[:, None] @ basis).squeeze(1), half)
    dense_alpha = torch.linspace(0, 1, 21, device=e.device)
    own_path = torch.lerp(before[impacted, None], e.p[impacted, None], dense_alpha[None, :, None])
    box_path = torch.lerp(previous_centers[impacted, collider, None], center[:, None], dense_alpha[None, :, None])
    aligned_min = signed_distance((own_path - box_path) @ basis, half[:, None]).amin(dim=1)
    margin = e.margin[impacted]
    for j, row in enumerate(impacted.tolist()):
        index = int(collider[j])
        kind, gate = label(e, index)
        rows[row] = {
            'collision_tick': tick, 'voxel_index': index, 'kind': kind, 'gate_frame_id': gate,
            'gates_before_collision': int(progress.gates[row]), 'speed_mps': float(e.v[row].norm()),
            'down_speed_mps': float(e.v[row, 2]), 'margin_m': float(margin[j]),
            'route_offset_m': float(route_delta[row].norm()), 'centerline_clearance_m': float(projected_clear[j] - margin[j]),
            'hit_clearance_m': float(selected_dist[j]), 'endpoint_clearance_m': float(end_clear[j] - margin[j]),
            'time_aligned_clearance_m': float(aligned_min[j] - margin[j]), 'impact_ned': impact[j].tolist(),
            'center_ned': center[j].tolist(), 'box_half_size_m': half[j].tolist(), 'box_basis_ned': basis[j].tolist(),
        }


def compare_row(actual, baseline):
    assert actual['outcome'] == baseline['outcome'] and actual['gates'] == int(baseline['gates'])
    for key in ('flight_s', 'min_clearance_m'):
        assert abs(actual[key] - float(baseline[key])) < 2e-5, (actual, baseline)
    if actual['failure_ned'] is not None:
        np.testing.assert_allclose(actual['failure_ned'], json.loads(baseline['failure_ned']), rtol=0, atol=1e-5)


@torch.inference_mode()
def main():
    source, out = map(Path, sys.argv[1:])
    with source.open() as stream:
        baseline = {(int(x['level']), int(x['seed']), int(x['row'])): x for x in csv.DictReader(stream)
                    if x['model'] == 'pwm_geometric'}
    graph = GateGraph.from_hitsz(UP / ROUTE).select(range(NEXT_GATE, NEXT_GATE + 3))
    course = Course(BATCH)
    e = course.e
    policy = PWMRouteFollower(e)
    out.mkdir(parents=True, exist_ok=False)
    examples, collisions, scenes = set(), [], []
    for level in SUITES['val192'][0]:
        for seed in SUITES['val192'][1]:
            fingerprint = course.start(level, seed, False)
            initial_speed = e.v.norm(dim=1).tolist()
            initial_force = e.f.sum(dim=1).tolist()
            progress = Progress(e, graph)
            initial_clear = progress.minimum_m.tolist()
            detail = {}
            record_contacts(e, torch.ones_like(progress.done), progress, e.p, e.voxels[..., :3], 0, detail)
            path, velocity, control = [e.p.clone()], [e.v.clone()], []
            while not progress.done.all():
                alive = ~progress.done
                before, centers = e.p.clone(), e.voxels[..., :3].clone()
                pwm = policy.command()
                dt = policy.step(pwm)
                control.append(pwm)
                path.append(e.p.clone())
                velocity.append(e.v.clone())
                progress.advance(before, dt)
                record_contacts(e, alive, progress, before, centers, len(control), detail)
            path, velocity, control = (torch.stack(track).cpu().numpy() for track in (path, velocity, control))
            for row, actual in enumerate(progress.rows()):
                entry = baseline[level, seed, row]
                assert fingerprint == entry['batch_sha256']
                compare_row(actual, entry)
                scene = {'level': level, 'seed': seed, 'row': row, 'scene_sha256': entry['scene_sha256'],
                         'initial_speed_mps': initial_speed[row], 'initial_motor_force_n': initial_force[row],
                         'initial_clearance_m': initial_clear[row], **actual}
                scenes.append(scene)
                if row not in detail:
                    continue
                item = {**scene, **detail[row]}
                collisions.append(item)
                phase = 'initial' if not item['collision_tick'] else f"airborne-after-{item['gates']}"
                example = phase, item['kind']
                if example not in examples:
                    examples.add(example)
                    tick = item['collision_tick']
                    name = f"{phase}-{item['kind']}-l{level}-s{seed}-r{row}.npz"
                    np.savez_compressed(out / name, p_ned=path[:tick + 1, row], v_ned=velocity[:tick + 1, row],
                                        pwm=control[:tick, row], time_s=np.arange(tick + 1) / 100,
                                        collision=json.dumps(item))
            print(f"checked level={level} seed={seed}, collisions={len(detail)}", flush=True)

    assert len(scenes) == len(baseline) == 192 and len(collisions) == 126
    for name, records in (('scenes.csv', scenes), ('collisions.csv', collisions)):
        with (out / name).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    flying = [x for x in collisions if x['collision_tick']]
    report = {'suite': 'val192', 'reproduced_scenes': len(scenes), 'collisions': len(collisions),
              'initial': dict(Counter(x['kind'] for x in collisions if not x['collision_tick'])),
              'airborne': dict(Counter(x['kind'] for x in flying)),
              'airborne_by_gate_count_and_type': {f"{gates}/{kind}": count for (gates, kind), count in
                  sorted(Counter((x['gates'], x['kind']) for x in flying).items())},
              'airborne_endpoint_contacts': sum(x['endpoint_clearance_m'] < 0 for x in flying),
              'airborne_time_aligned_contacts': sum(x['time_aligned_clearance_m'] < 0 for x in flying),
              'airborne_same_box_intersects_centerline_margin': sum(x['centerline_clearance_m'] < 0 for x in flying),
              'airborne_centerline_contacts_by_type': dict(Counter(x['kind'] for x in flying
                                                                 if x['centerline_clearance_m'] < 0)),
              'median_airborne_impact_speed_mps': float(np.median([x['speed_mps'] for x in flying])),
              'median_airborne_offset_m': float(np.median([x['route_offset_m'] for x in flying])),
              'representative_trajectories': sorted(path.name for path in out.glob('*.npz'))}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
