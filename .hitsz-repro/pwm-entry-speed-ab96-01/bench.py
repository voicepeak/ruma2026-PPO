import csv,json,sys,os,importlib.util
from collections import Counter
from pathlib import Path
from statistics import median
import torch
from evaluation.lidar_bypass import LidarBypassFollower
from evaluation.rollout import COLLISION, Progress
from hitsz_branches import UP
from hitsz_ppo import Course
from maps.gate_graph import GateGraph

OUT=Path('.hitsz-repro/pwm-entry-speed-ab96-01')
SOURCES=Path('.hitsz-repro/ppo-transfer384-01/protocol.json')
GROUPS={'r1-late':[(2,84001),(3,84002)],'r2-climb':[(2,84003),(3,84004)]}
CAP=4.0
BATCH=24

@torch.inference_mode()
def run(segment):
    spec=importlib.util.spec_from_file_location('audit', Path(os.environ['RMUA_RUNTIME'])/'diagnose-pwm-collisions.py')
    audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    t, = (s for s in json.loads(SOURCES.read_text())['segments'] if s['id']==segment)
    graph=GateGraph.from_hitsz(UP/t['route']).select(t['gate_ids'])
    assert graph.sha256==t['gate_graph_sha256']
    course=Course(BATCH,route=t['route'],start=t['start_m'],first_gate=t['gate_ids'][0])
    e=course.e; policy=LidarBypassFollower(e)
    rows=[]
    for level,seed in GROUPS[segment]:
        expected=None
        for mode in ('original','entry-4mps'):
            fingerprint=course.start(level,seed,False)
            if expected is None: expected=fingerprint
            assert expected==fingerprint
            policy.reset(seed)
            natural=e.v.norm(dim=-1)
            if mode=='entry-4mps':
                e.v.mul_((CAP/natural.clamp_min(1e-8)).clamp_max(1)[:,None])
            actual=e.v.norm(dim=-1)
            progress=Progress(e,graph)
            init=(progress.outcome==COLLISION).tolist()
            detail={}
            n=0
            while not progress.done.all():
                live=~progress.done
                before=e.p.clone()
                centers=e.voxels[...,:3].clone()
                dt=policy.step(policy.command())
                progress.advance(before,dt); n+=1
                audit.record_contacts(e,live,progress,before,centers,n,detail)
            for row,(score,origin,speed,contact) in enumerate(zip(progress.rows(),natural.tolist(),actual.tolist(),init)):
                rows.append({'segment':segment,'level':level,'seed':seed,'row':row,'mode':mode,
                             'pre_cap_sha256':fingerprint,'original_entry_mps':origin,'entry_mps':speed,
                             'initial_contact':contact,'impact_kind':detail.get(row,{}).get('kind',''),**score})
            print(segment,level,mode,dict(Counter(r['outcome'] for r in rows[-BATCH:] if not r['initial_contact'])),flush=True)
    with (OUT/f'{segment}.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def summarize():
    rows=[]
    for segment in GROUPS:
        with (OUT/f'{segment}.csv').open(newline='') as f: rows.extend(csv.DictReader(f))
    pair={}
    for r in rows:
        pair.setdefault((r['segment'],r['level'],r['seed'],r['row']),{})[r['mode']]=r
    assert len(pair)==96 and all(len(x)==2 for x in pair.values())
    assert all(a['original']['pre_cap_sha256']==a['entry-4mps']['pre_cap_sha256'] and
               a['original']['initial_contact']==a['entry-4mps']['initial_contact'] for a in pair.values())
    initial=sum(a['original']['initial_contact']=='True' for a in pair.values())
    variants={}
    for mode in ('original','entry-4mps'):
        reachable=[r for r in rows if r['mode']==mode and r['initial_contact']=='False']
        crashed=[r for r in reachable if r['outcome']=='collision']
        variants[mode]={'outcomes':dict(Counter(r['outcome'] for r in reachable)),
            'first_second_crashes':sum(float(r['flight_s'])<=1.0 for r in crashed),
            'impact_kinds':dict(Counter(r['impact_kind'] for r in crashed)),
            'median_entry_mps':median(float(r['entry_mps']) for r in reachable)}
    pairs=Counter((x['original']['outcome'],x['entry-4mps']['outcome']) for x in pair.values()
                  if x['original']['initial_contact']=='False')
    report={'scenes':len(pair),'initial_contacts':initial,
            'groups':GROUPS,'entry_cap_mps':CAP,'variants':variants,
            'paired_transitions':{f'{a}->{b}':n for (a,b),n in sorted(pairs.items())}}
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ENTRY_AB_DONE',report,flush=True)

if __name__=='__main__':
    summarize() if sys.argv[1]=='summary' else run(sys.argv[1])
