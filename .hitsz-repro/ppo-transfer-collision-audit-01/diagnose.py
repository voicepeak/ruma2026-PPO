"""Audit first collisions in the frozen transfer replay, with no policy changes."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from evaluation.bypass_learning import BypassActorCritic, BypassLearningEnv
from evaluation.rollout import COLLISION
from hitsz_branches import UP
from hitsz_ppo import Course, DEVICE
from maps.gate_graph import GateGraph
from scripts.eval_bypass_transfer import rule_policy, sha


def collision_recorder():
    path = Path(os.environ['RMUA_RUNTIME']) / 'diagnose-pwm-collisions.py'
    spec = importlib.util.spec_from_file_location('collision_audit_p0', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def run_group(env, policy, level, case_seed, reference, common, recorder):
    fingerprint = env.reset(level, case_seed)
    assert all(row['batch_sha256'] == fingerprint for row in reference.values())
    e, follower, progress = env.course.e, env.follower, env.progress
    contacts, trace, timing = {}, [], {}
    step_original = follower.step
    advance_original = progress.advance
    preview_station = None
    previous_centers = None

    def track_step(pwm):
        nonlocal previous_centers
        previous_centers = e.voxels[..., :3].clone()
        if follower.tick % follower.sensor.config.sample_every_ticks == 0:
            trace.append(torch.stack((e.v.norm(dim=-1), follower.speed_fraction,
                                      follower.obstacle_station - preview_station,
                                      follower.choice.float(), follower.detouring.float()), dim=-1))
        return step_original(pwm)

    def track_advance(before, dt):
        active = ~progress.done
        advance_original(before, dt)
        hits = (active & (progress.outcome == COLLISION)).nonzero().flatten()
        if len(hits):
            recorder.record_contacts(e, active, progress, before, previous_centers, follower.tick, contacts)
            for row in hits.tolist():
                collider = contacts[row]['voxel_index']
                v = ((e.voxels[row, collider, :3] - previous_centers[row, collider]) / dt).tolist()
                timing[row] = {'collider_speed_mps': float(np.linalg.norm(v)), 'collider_velocity_ned': v}

    follower.step = track_step
    progress.advance = track_advance
    try:
        while not progress.done.all():
            features, prior = env.observe()
            preview_station = env.preview[0]
            action = policy(features, prior)[0].argmax(-1)
            env.step(action)
    finally:
        follower.step = step_original

    history = torch.stack(trace).cpu().numpy()
    records = []
    for row, score in enumerate(progress.rows()):
        old = reference[row]
        recorder.compare_row(score, old)
        if row not in contacts:
            continue
        detail = contacts[row]
        until = (detail['collision_tick'] + 9) // 10
        values = history[:until, row]
        last = values[-1]
        records.append({'level':level, 'scene_seed':case_seed, 'row':row, 'scene_sha256':old['scene_sha256'],
                        'common_collision':old['scene_sha256'] in common, **score, **detail, **timing[row],
                        'last_scan_speed_mps':float(last[0]), 'last_scan_speed_fraction':float(last[1]),
                        'last_scan_stop_margin_m':float(last[2]), 'last_choice':int(last[3]),
                        'last_detouring':bool(last[4]),
                        'brake_last_05s':bool((values[-5:, 1] < 1).any()),
                        'brake_last_1s':bool((values[-10:, 1] < 1).any()),
                        'ever_braked':bool((values[:, 1] < 1).any())})
    return records


def main():
    transfer, out, segment_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    protocol = json.loads((transfer/'protocol.json').read_text())
    segment, = (s for s in protocol['segments'] if s['id']==segment_id)
    with (transfer/f'{segment_id}.csv').open(newline='') as stream:
        archive = list(csv.DictReader(stream))
    by_scene = {}
    for row in archive:
        by_scene.setdefault(row['scene_sha256'], []).append(row)
    common = {key for key, scores in by_scene.items() if len(scores)==4 and all(
        r['outcome']=='collision' and r['initial_contact']=='False' for r in scores)}

    graph=GateGraph.from_hitsz(UP/segment['route']).select(segment['gate_ids'])
    course=Course(48, route=segment['route'], start=segment['start_m'], first_gate=segment['gate_ids'][0])
    checkpoint, = [c for c in protocol['weights'] if c['seed']==20261122]
    path=Path(checkpoint['path'])
    assert sha(path)==checkpoint['sha256']
    net=BypassActorCritic(True).to(DEVICE).eval()
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)['policy'])
    recorder=collision_recorder()
    results=[]
    for model,fn in [('rule',rule_policy),('ppo-20261122',net)]:
        env=BypassLearningEnv(course,graph,allow_wait=model!='rule')
        for group in segment['groups']:
            reference={int(r['row']):r for r in archive if r['model']==model and int(r['scene_seed'])==group['seed']}
            assert len(reference)==48
            hits=run_group(env,fn,group['level'],group['seed'],reference,common,recorder)
            results.extend({'segment':segment_id,'model':model,**r} for r in hits)
            print('AUDIT_GROUP',segment_id,group['level'],model,len(hits),flush=True)
    with (out/f'{segment_id}.csv').open('x', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__=='__main__':
    main()
