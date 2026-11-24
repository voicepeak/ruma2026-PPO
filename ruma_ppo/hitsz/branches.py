"""Paired candidate rollouts in the pinned HITSZ CUDA simulator; research gate score."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

from ruma_ppo.hitsz.course import CANDIDATES

UP = Path(os.environ["HITSZ_ROOT"]).resolve()
sys.path.insert(0, str(UP))
from Env.utils import make_env
from RL.rl_env_wrapper import RLQuadEnv
import quadsim_cuda

ROUTE = "path_point/1-3_2_training.yaml"
START = 400.5
NEXT_GATE = 38  # 19 * 21.5 = 408.5 m on the upstream route
STEPS = 100


class BranchEnv(RLQuadEnv):
    def _reset_envs(self, ids):
        # Finished rows cannot affect other rows; keep advancing without random respawns.
        pass

    def _compute_rewards(self):
        reward, fatal, info = super()._compute_rewards()
        delta = self._current_delta_p_world
        self.bounds = (delta[:, :2].norm(dim=1) > 3.) | (delta[:, 2].abs() > 2.5)
        self.fatal = fatal.bool()
        return reward, fatal, info


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    random.seed(value)


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def reset(env, value, level, *, start=START):
    seed(value)
    env.set_curriculum_levels(2, level, True)
    e = env.env
    station = torch.full((env.num_envs,), float(start), device=e.device)
    pos, direction, segment = e.get_state_by_s(station, e.env_traj_ids)
    e.p.copy_(pos)
    e.p_old.copy_(pos)
    e.R.copy_(e.traj_R[e.env_traj_ids, segment-1])
    e.v.copy_(direction * env.target_speed)
    env._reset_speed_ema(torch.arange(env.num_envs, device=e.device))
    obs = env.get_observations()
    state_hash = digest([e.p, e.v, e.R, e.voxels, e.voxels_R, e.car_s, e.car_v, e.margin,
                         e.lin_drag_coef, e.tilt_time_constant, e.yaw_time_constant,
                         env.target_speed, obs["state"], obs["depth"]])
    return obs, state_hash


def station(e, position=None):
    p = e.p if position is None else position
    nodes = e.trajectories[0]
    edge = nodes[1:] - nodes[:-1]
    t = ((p[:,None]-nodes[:-1])*edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-8)
    t = t.clamp(0, 1)
    k = (nodes[:-1]+t[...,None]*edge-p[:,None]).square().sum(-1).argmin(1)
    row = torch.arange(len(k),device=e.device)
    return e.traj_s[0,k]+t[row,k]*edge[k].norm(dim=-1)


def controller(env, s, offset, factor):
    e = env.env
    ref, tangent, idx = e.get_state_by_s(s+3., e.env_traj_ids)
    lateral = e.traj_R[e.env_traj_ids,idx-1,:,1]
    branch = s < START + 14.
    shift = branch.float() * offset
    v = tangent * env.target_speed * torch.where(branch, factor, 1.)[:,None]
    # Track the offset in the plane normal to the road; use the same follower after 14 m.
    error = ref + shift[:,None]*lateral - e.p
    cross_track = error - (error*tangent).sum(1,keepdim=True)*tangent
    v = v + 1.35 * cross_track
    return torch.bmm(v[:,None], env._current_R_yaw).squeeze(1)


def gate_event(e, before, count):
    part = (NEXT_GATE + count.clamp_max(2)) * 6
    centre = (e.base_voxels_gate[0,part,:3]+e.base_voxels_gate[0,part+1,:3])/2
    basis = e.base_voxels_gate_R[0,part]
    a = torch.bmm((before-centre)[:,None],basis).squeeze(1)
    b = torch.bmm((e.p-centre)[:,None],basis).squeeze(1)
    crossed = (a[:,0]<0) & (b[:,0]>=0)
    fraction = (-a[:,0]/(b[:,0]-a[:,0]).clamp_min(1e-8)).clamp(0,1)
    hit = a + fraction[:,None]*(b-a)
    inside = (hit[:,1].abs()<=3.) & (hit[:,2].abs()<=1.5)
    return crossed, inside


def rollout(env, initial, candidate):
    e = env.env
    n = env.num_envs
    count = torch.zeros(n, device=e.device, dtype=torch.long)
    reason = torch.zeros_like(count)
    length = torch.zeros_like(count)
    trace = [e.p.clone()]
    for step in range(1,STEPS+1):
        before = e.p.clone()
        active = reason == 0
        action = controller(env,station(e),*candidate)
        obs, _, _, _ = env.step(action)
        if not torch.isfinite(e.p).all() or not obs["depth"].isfinite().all():
            raise ValueError("Nonfinite native rollout")
        crossed, inside = gate_event(e, before, count)
        count += (active & ~env.fatal & crossed & inside).long()
        for mask,code in ((env.fatal & env.bounds,2),(env.fatal,1),(crossed & ~inside,3),(count==3,4)):
            reason = torch.where((reason==0) & mask,code,reason)
        length += active.long()
        trace.append(torch.where(active[:,None], e.p, trace[-1]))
        if (reason!=0).all(): break
    return tuple(t.cpu().numpy() for t in (count, reason, length, torch.stack(trace)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch",type=int,default=48)
    args=p.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((UP/"RL/configs/config_env.yaml").read_text())
    train=yaml.safe_load((UP/"RL/configs/config_rl.yaml").read_text())
    cfg["trajectory"]["paths"]=[str(UP/ROUTE)]
    cfg["env"]["batch_size"]=args.batch
    train["env"]=cfg["env"]
    assert cfg["env"]["dynamics"]=="full"
    seed(1043)
    env=BranchEnv(train,make_env(cfg,torch.device("cuda")))
    records=[]
    hashes={}
    outcomes=["timeout","collision","bounds_including_possible_contact","missed_gate","three_gates"]
    with torch.inference_mode():
        for level,value in [(2,30817),(3,69073)]:
            expected=None
            counts=[]
            for c in [*range(len(CANDIDATES)),7]:
                obs,fingerprint=reset(env,value,level)
                if expected is None:
                    expected=fingerprint
                    depth=obs["depth"].reshape(args.batch,20,80)
                    assert depth.std()>0 and depth.isfinite().all()
                    np.savez_compressed(out/f"observations-L{level}.npz",
                        depth=depth.cpu().numpy(),state=obs["state"].cpu().numpy())
                elif expected!=fingerprint:
                    raise ValueError("Paired initial state mismatch")
                gates,why,length,trace=rollout(env,obs,CANDIDATES[c])
                if len(counts)==len(CANDIDATES):
                    for a,b in zip((gates,why,length,trace),repeat_reference):
                        np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
                    break
                if c==7: repeat_reference=(gates,why,length,trace)
                counts.append(gates)
                # Every path from case 0, plus all case outcomes for further audits.
                np.save(out/f"trace-L{level}-c{c}.npy",trace[:,0])
                for i in range(args.batch):
                    records.append([level,i,c,*CANDIDATES[c],int(gates[i]),outcomes[why[i]],int(length[i])])
                print(level,c,int((gates==3).sum()),flush=True)
            hashes[str(level)]=expected
    with (out/"branches.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["density","case","candidate","offset_m","speed_factor","gates","outcome","steps"])
        w.writerows(records)
    counts=np.array([r[5] for r in records]).reshape(2,10,args.batch)
    ok=counts==3
    by_case=ok.sum(1)
    summary={"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"route":ROUTE,
             "start_s_m":START,"gate_ids_zero_based":[38,39,40],"native_extension_sha256":
             hashlib.sha256(Path(quadsim_cuda.__file__).read_bytes()).hexdigest(),
             "upstream_commit":"9ac9657334817ad0bac6db3520bca59e5f864bf3",
             "candidates":CANDIDATES,"levels":[2,3],"seeds":[30817,69073],"batch":args.batch,
             "same_start_hash":hashes,"center_replay_passed":True,
             "success_by_level_candidate":ok.sum(2).tolist(),
             "any_success":(by_case>0).sum(1).tolist(),"all_success":(by_case==10).sum(1).tolist(),
             "none_success":(by_case==0).sum(1).tolist(),
             "center_full_rescued":((by_case>0)&~ok[:,7]).sum(1).tolist(),
             "success_histogram":np.bincount(by_case.ravel(),minlength=11).tolist(),
             "outcomes":{name:sum(r[6]==name for r in records) for name in outcomes}}
    (out/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()

