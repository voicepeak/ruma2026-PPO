import csv,json
from collections import Counter
from pathlib import Path
from statistics import median

root=Path('.hitsz-repro/ppo-transfer-collision-audit-01')
all_rows=[]
for segment in ('r1-early','r1-late','r2-climb','r3-descent'):
    with (root/f'{segment}.csv').open(newline='') as stream: all_rows.extend(csv.DictReader(stream))

def by_kind(rows):
    table={}
    for kind in sorted({r['kind'] for r in rows}):
        subset=[r for r in rows if r['kind']==kind]
        table[kind]={
            'cases':len(subset),
            'median_ego_mps':median(float(r['speed_mps']) for r in subset),
            'median_obstacle_mps':median(float(r['collider_speed_mps']) for r in subset),
            'stopped_ego':sum(float(r['speed_mps'])<.3 for r in subset),
            'ego_below_1_mps':sum(float(r['speed_mps'])<1 for r in subset),
            'median_hit_time_s':median(float(r['flight_s']) for r in subset),
            'no_speed_limit_last_half_second':sum(r['brake_last_05s']=='False' for r in subset),
            'no_speed_limit_entire_episode':sum(r['ever_braked']=='False' for r in subset),
            'centerline_already_blocked':sum(float(r['centerline_clearance_m'])<0 for r in subset),
            'endpoint_contact':sum(float(r['endpoint_clearance_m'])<0 for r in subset)}
    return table

rule=[r for r in all_rows if r['model']=='rule']
ppo=[r for r in all_rows if r['model']=='ppo-20261122']
shared_rule={r['scene_sha256']:r for r in rule if r['common_collision']=='True'}
shared_ppo={r['scene_sha256']:r for r in ppo if r['common_collision']=='True'}
assert len(rule)==116 and len(ppo)==113 and shared_rule.keys()==shared_ppo.keys() and len(shared_rule)==109
paired={'same_type':sum(shared_rule[k]['kind']==r['kind'] for k,r in shared_ppo.items()),
        'same_voxel':sum(shared_rule[k]['voxel_index']==r['voxel_index'] for k,r in shared_ppo.items()),
        'same_tick':sum(shared_rule[k]['collision_tick']==r['collision_tick'] for k,r in shared_ppo.items()),
        'same_voxel_and_tick':sum(shared_rule[k]['voxel_index']==r['voxel_index'] and
                                 shared_rule[k]['collision_tick']==r['collision_tick'] for k,r in shared_ppo.items())}
summary={'rule_airborne':by_kind(rule),'ppo_seed22_airborne':by_kind(ppo),
         'four_policy_common_airborne':by_kind(list(shared_rule.values())),'paired_rule_vs_ppo22':paired,
         'common_segment_type':{f'{segment}/{kind}':n for (segment,kind),n in
             sorted(Counter((r['segment'],r['kind']) for r in shared_rule.values()).items())}}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
