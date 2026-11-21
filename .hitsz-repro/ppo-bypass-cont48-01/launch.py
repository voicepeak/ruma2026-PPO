import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('.hitsz-repro/ppo-bypass-cont48-01')
ORIGINAL=Path('.hitsz-repro/ppo-bypass')
SEEDS=(20261121,20261122,20261123)
MODULE=[sys.executable,'-u','-m','scripts.train_bypass_ppo']
BASELINE='.hitsz-repro/pwm-lidar-bypass-val192-01/paired.csv'

def read(path):
    return json.loads((path/'status.json').read_text())

def clone(seed, target):
    original=ORIGINAL/f'wait-{seed}'
    assert read(original)['update']==24
    target.mkdir()
    for file in ('latest.pt','config.json','status.json','curves.jsonl'):
        shutil.copy2(original/file,target/file)

def train(seed, path, updates, name):
    with (path/f'{name}.log').open('x') as log:
        subprocess.run([*MODULE,'train','--resume','--wait','--seed',str(seed),'--batch','32',
                        '--updates',str(updates),'--out',str(path)],
                       check=True,stdout=log,stderr=subprocess.STDOUT)

def probe(jobs, phase):
    initial=[read(path)['decisions'] for _,path in jobs]
    t=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(train,seed,path,27,'throughput_probe') for seed,path in jobs]
        errors=[]
        for future in futures:
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                errors.append(str(exc))
    elapsed=time.monotonic()-t
    decisions=sum(read(path)['decisions']-before for (_,path),before in zip(jobs,initial))
    result={'phase':phase,'jobs':len(jobs),'seconds':elapsed,'decisions':decisions,
            'decisions_per_second':decisions/elapsed,'errors':errors}
    print(json.dumps(result),flush=True)
    return result

def finish(seed,path):
    print(f'CONT48_TRAIN {seed}',flush=True)
    train(seed,path,48,'continued_train')
    print(f'CONT48_EVAL {seed}',flush=True)
    with (path/'validation.log').open('x') as log:
        subprocess.run([*MODULE,'eval','--checkpoint',str(path/'latest.pt'),
                        '--baseline',BASELINE,'--out',str(path/'val192')],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    print(f'CONT48_DONE {seed}',flush=True)

def main():
    solo=ROOT/'single-job-probe'
    clone(SEEDS[0],solo)
    jobs=[(seed,ROOT/f'wait-{seed}') for seed in SEEDS]
    for seed,path in jobs:
        clone(seed,path)

    single=probe([(SEEDS[0],solo)],'single')
    if single['errors']:
        raise RuntimeError('Single-worker benchmark failed')
    dual=probe(jobs[:2],'dual')
    gain=dual['decisions_per_second']/single['decisions_per_second']
    slots=2 if not dual['errors'] and gain >= 1.10 else 1
    plan={'single':single,'dual':dual,'aggregate_gain':gain,'selected_slots':slots,
          'target_update':48,'seeds':SEEDS}
    (ROOT/'throughput.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('CONT48_PARALLELISM',slots,flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        futures=[pool.submit(finish,seed,path) for seed,path in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print('CONT48_QUEUE_DONE',flush=True)

if __name__=='__main__':
    main()
