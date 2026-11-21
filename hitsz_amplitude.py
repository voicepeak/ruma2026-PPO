"""Prospective single-factor comparison of lateral amplitudes on new HITSZ scenes."""
import argparse
import csv
from pathlib import Path
import json

import numpy as np
import torch
import yaml

from hitsz_branches import (BranchEnv, CANDIDATES, NEXT_GATE, ROUTE, START, UP,
                            make_env, reset, rollout, seed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", required=True,type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((UP/"RL/configs/config_env.yaml").read_text())
    tc=yaml.safe_load((UP/"RL/configs/config_rl.yaml").read_text())
    batch=48
    cfg["env"]["batch_size"]=batch
    cfg["trajectory"]["paths"]=[str(UP/ROUTE)]
    tc["env"]=cfg["env"]
    assert cfg["env"]["dynamics"]=="full"
    seed(1043)
    env=BranchEnv(tc,make_env(cfg,torch.device("cuda")))
    small=[(x*0.5,v) for x,v in CANDIDATES]
    branches=list(dict.fromkeys([*CANDIDATES,*small]))
    runs=[(2,80819),(3,91513)]
    records=[]
    tables=[]
    hashes={}
    for level, value in runs:
        expected=None
        paths={}
        gates=[]
        for i,branch in enumerate(branches):
            obs,h=reset(env,value,level)
            if expected is None:
                expected=h
                e=env.env
                clear=(e.find_vec_to_nearest_pt()[0].norm(dim=-1)-e.margin).cpu().numpy()
                ids=torch.arange(NEXT_GATE,NEXT_GATE+3,device=e.device)*6
                centres=(e.base_voxels_gate[0,ids,:3]+e.base_voxels_gate[0,ids+1,:3])*0.5
                np.savez_compressed(args.out/f"scene-L{level}.npz",depth=obs["depth"].reshape(batch,20,80).cpu().numpy(),
                    initial_clearance=clear, gate_centres=centres.cpu().numpy(),gate_basis=e.base_voxels_gate_R[0,ids].cpu().numpy(),
                    waypoints=e.trajectories[0].cpu().numpy())
            elif h!=expected:
                raise RuntimeError("Mismatched initial scenes")
            g,r,n,tr=rollout(env,obs,branch)
            gates.append(g)
            paths[f"c{i}"]=tr
            for j in range(batch):
                records.append([level,value,j,i,*branch,int(g[j]),int(r[j]),int(n[j])])
        hashes[str(level)]=expected
        np.savez_compressed(args.out/f"tracks-L{level}.npz",**paths)
        g=np.stack(gates)
        one=g[[branches.index(x) for x in CANDIDATES]].max(0)==3
        two=g[[branches.index(x) for x in small]].max(0)==3
        center=g[branches.index((0.,1.))]==3
        tables.append(dict(level=level,n=batch,original=int(one.sum()),half=int(two.sum()),
            both=int((one&two).sum()),original_only=int((one&~two).sum()),
            half_only=int((~one&two).sum()),neither=int((~one&~two).sum()),centre=int(center.sum()),
            initial_contact=int((clear<0).sum()),initial_clear_total=int((clear>=0).sum()),
            original_if_clear=int((one&(clear>=0)).sum()),half_if_clear=int((two&(clear>=0)).sum())))
        print(tables[-1],flush=True)
    with (args.out/"branches.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["level","seed","case","candidate","offset_m","factor","gates","reason","steps"]);w.writerows(records)
    report=dict(route=ROUTE,start=START,seeds=runs,candidates=branches,levels=tables,
                initial_hash=hashes,outcome_codes=["timeout","collision","bounds_including_possible_contact","missed_gate","three_gates"])
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    with torch.inference_mode():
        main()

