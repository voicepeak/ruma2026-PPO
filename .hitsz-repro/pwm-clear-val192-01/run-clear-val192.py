import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from evaluation.harness import BATCH, SUITES
from evaluation.rollout import Progress, PWMRouteFollower
from hitsz_branches import NEXT_GATE, ROUTE, UP
from hitsz_ppo import Course
from maps.gate_graph import GateGraph

@torch.inference_mode()
def main():
    root = Path('.hitsz-repro')
    out = root / 'pwm-clear-val192-01'
    out.mkdir()
    with (root / 'pwm-val192-01/paired.csv').open() as stream:
        old = {(int(r['level']), int(r['seed']), int(r['row'])): r for r in csv.DictReader(stream)
               if r['model'] == 'pwm_geometric'}
    graph = GateGraph.from_hitsz(UP / ROUTE).select(range(NEXT_GATE, NEXT_GATE + 3))
    course = Course(BATCH)
    e = course.e
    pwm = PWMRouteFollower(e)
    rows = []
    gate_offset = e.num_voxels + e.num_bg_voxels + e.num_bottom_voxels
    for level in SUITES['val192'][0]:
        for seed in SUITES['val192'][1]:
            fingerprint = course.start(level, seed, False)
            assert all(fingerprint == old[level, seed, row]['batch_sha256'] for row in range(BATCH))
            gate_snapshot = e.static_voxels[:, gate_offset:].clone()
            e.static_voxels[:, :gate_offset, :3] = 1e6
            e.static_voxels[:, :gate_offset, 3:] = 0.0
            e.active_num_cars = 0
            e._update_cars_idx()
            progress = Progress(e, graph)
            peak_horizontal = e.p.new_zeros(BATCH)
            peak_vertical = e.p.new_zeros(BATCH)
            while not progress.done.all():
                active = ~progress.done
                before = e.p.clone()
                step = pwm.step(pwm.command())
                progress.advance(before, step)
                rel, _ = e.get_nearest_traj_state(progress.position)
                peak_horizontal = torch.maximum(peak_horizontal, torch.where(active, rel[:, :2].norm(dim=-1), 0))
                peak_vertical = torch.maximum(peak_vertical, torch.where(active, rel[:, 2].abs(), 0))
            assert torch.equal(e.voxels[:, gate_offset:gate_offset + gate_snapshot.shape[1]], gate_snapshot)
            assert not (e.car_voxels[..., 3:] != 0).any()
            for row, (score, horizontal, vertical) in enumerate(zip(progress.rows(), peak_horizontal.tolist(), peak_vertical.tolist())):
                origin = old[level, seed, row]
                rows.append({'level': level, 'seed': seed, 'row': row, 'original_scene_sha256': origin['scene_sha256'],
                             'original_outcome': origin['outcome'], 'original_flight_s': float(origin['flight_s']),
                             **score, 'peak_horizontal_m': horizontal, 'peak_vertical_m': vertical})
            print('CLEAR_BATCH', level, seed, dict(Counter(r['outcome'] for r in rows[-BATCH:])), flush=True)
    assert len(rows) == len(old) == 192
    with (out / 'cases.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    success = [r['flight_s'] for r in rows if r['outcome'] == 'success']
    by_outcome = Counter(r['outcome'] for r in rows)
    report = {'cases': len(rows), 'outcomes': dict(by_outcome),
              'original_outcomes': dict(Counter(r['original_outcome'] for r in rows)),
              'formerly_start_contact_now_success': sum(r['original_outcome'] == 'collision' and r['original_flight_s'] == 0 and r['outcome'] == 'success' for r in rows),
              'formerly_airborne_collision_now_success': sum(r['original_outcome'] == 'collision' and r['original_flight_s'] > 0 and r['outcome'] == 'success' for r in rows),
              'mean_success_s': float(np.mean(success)) if success else None,
              'success_time_range_s': [min(success), max(success)] if success else None,
              'peak_horizontal_m_p50_p95_max': np.percentile([r['peak_horizontal_m'] for r in rows], [50, 95, 100]).tolist(),
              'peak_vertical_m_max': max(r['peak_vertical_m'] for r in rows),
              'min_gateframe_clearance_m': min(r['min_clearance_m'] for r in rows),
              'settings': {'original_initial_states_verified': True, 'retained_geometry': 'HITSZ gate frames',
                           'controller': 'pwm_geometric', 'reference_speed_mps': pwm.speed_mps, 'limit_s': 10}}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

main()
