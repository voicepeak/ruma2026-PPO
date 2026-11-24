"""Paired native-simulator comparison, HITSZ released v12 vs frozen PPO checkpoints."""
import argparse
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import onnxruntime as ort
import torch
from hitsz_ppo import ActorCritic, Course, DEV, DEVICE, MAX_DECISIONS


class V12Course(Course):
    def __init__(self, batch, path):
        super().__init__(batch)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path),opts,providers=["CPUExecutionProvider"])
        self.batch_fixed = self.session.get_inputs()[0].shape[0] == 1

    def start(self,*args):
        self.hx = np.zeros((self.batch,192),np.float32)
        return super().start(*args)

    def velocity(self, unused):
        depth = self.obs["depth"].reshape(self.batch,1,20,80).cpu().numpy()
        state = self.obs["state"].cpu().numpy()
        if self.batch_fixed:
            results = [self.session.run(["act","next_hx"],dict(depth=depth[i:i+1],state=state[i:i+1],hx=self.hx[i:i+1])) for i in range(self.batch)]
            act,self.hx = [np.concatenate(col) for col in zip(*results)]
        else:
            act,self.hx = self.session.run(["act","next_hx"],dict(depth=depth,state=state,hx=self.hx))
        if not np.isfinite(act).all() or not np.isfinite(self.hx).all():
            raise RuntimeError("Nonfinite v12 output")
        return torch.from_numpy(act).to(DEVICE)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("v12",type=Path)
    p.add_argument("models",type=Path,nargs="+")
    args=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    ours, theirs = Course(48), V12Course(48,args.v12)
    nets={}
    meta={"v12":{"sha256":sha(args.v12)}}
    for path in args.models:
        ck=torch.load(path,map_location=DEVICE,weights_only=True)
        net=ActorCritic().to(DEVICE).eval()
        net.load_state_dict(ck["policy"])
        tag="ppo_"+str(ck["update"])
        nets[tag]=net
        meta[tag]={"sha256":sha(path),"update":ck["update"]}
    protocol={"dev":DEV,"fresh":[(2,350791),(3,730919)]}
    rows=[]
    comparisons={}
    for split,suite in protocol.items():
        for level,seed in suite:
            expected=None
            batch={}
            for tag,net in [("v12",None),*nets.items()]:
                course=theirs if tag=="v12" else ours
                fingerprint=course.start(level,seed,False)
                if expected and expected!=fingerprint: raise RuntimeError("Unequal scenes")
                expected=fingerprint
                for t in range(MAX_DECISIONS):
                    action=torch.zeros_like(course.gates) if net is None else net(*course.observe())[0].argmax(-1)
                    course.advance(action,t==MAX_DECISIONS-1)
                    if course.done.all():break
                g=course.gates.cpu().numpy()
                batch[tag]=g==3
                for i,(ng,cause,ticks) in enumerate(zip(g,course.reason.tolist(),course.flight_ticks.tolist())):
                    rows.append([split,level,seed,i,tag,int(ng),cause,ticks,fingerprint])
            comparisons[f"{split}_{level}"]={tag:dict(
                hitsz=int(batch["v12"].sum()),ours=int(batch[tag].sum()),
                hitsz_only=int((batch["v12"] & ~batch[tag]).sum()),
                ours_only=int((~batch["v12"] & batch[tag]).sum()),
                both=int((batch["v12"] & batch[tag]).sum()),
                neither=int((~batch["v12"] & ~batch[tag]).sum())) for tag in nets}
    with open("paired.csv","w",newline="") as f:
        w=csv.writer(f);w.writerow(["split","level","seed","case","model","gates","reason","ticks","batch_hash"]);w.writerows(rows)
    Path("result.json").write_text(json.dumps(dict(models=meta,protocol=protocol,results=comparisons),indent=2))
    print(comparisons)
if __name__=="__main__":main()
