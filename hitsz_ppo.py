"""On-policy PPO for repeated candidate selection in the native HITSZ 3D environment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import yaml

from hitsz_branches import BranchEnv, UP, ROUTE, START, NEXT_GATE, make_env, reset, seed, station

DEVICE = torch.device("cuda")
OFFSETS = [-1.8, -1.15, 0., 1.15, 1.8]
ACTIONS = [(x, v) for v in (0.72, 1.) for x in OFFSETS]
HOLD_TICKS = 4
MAX_DECISIONS = 25
DEV = [(2, 50731), (3, 50753)]


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(1, 12, 3, 2, 1), nn.SiLU(),
                                    nn.Conv2d(12, 20, 3, 2, 1), nn.SiLU(),
                                    nn.AdaptiveAvgPool2d((3, 10)), nn.Flatten())
        self.trunk = nn.Sequential(nn.Linear(600+21, 128), nn.Tanh(), nn.Linear(128, 96), nn.Tanh())
        self.actor = nn.Linear(96, len(ACTIONS))
        self.critic = nn.Linear(96, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, image, state):
        hidden = self.trunk(torch.cat([self.vision(image), state], dim=-1))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class Course:
    def __init__(self, batch):
        cfg = yaml.safe_load((UP/"RL/configs/config_env.yaml").read_text())
        train = yaml.safe_load((UP/"RL/configs/config_rl.yaml").read_text())
        cfg["trajectory"]["paths"] = [str(UP / ROUTE)]
        cfg["env"]["batch_size"] = batch
        train["env"] = cfg["env"]
        assert cfg["env"]["dynamics"] == "full"
        self.env = BranchEnv(train, make_env(cfg, DEVICE))
        self.e = self.env.env
        self.batch = batch
        self.table = torch.tensor(ACTIONS, device=DEVICE)

    def start(self, level, scene_seed, training):
        self.obs, fingerprint = reset(self.env, scene_seed, level)
        self.first = torch.full((self.batch,), NEXT_GATE, device=DEVICE, dtype=torch.long)
        if training:
            # Half the rows practise the studied bend. The rest start at random gate groups.
            rows = torch.arange(self.batch//2, self.batch, device=DEVICE)
            last_group = int((self.e.traj_lengths[0].item()-38.)//21.5)
            groups = torch.randint(1, last_group+1, (len(rows),), device=DEVICE)
            self.first[rows] = 2*groups
            pos, direction, idx = self.e.get_state_by_s(groups*21.5-8., self.e.env_traj_ids[rows])
            self.e.p[rows] = pos
            self.e.p_old[rows] = pos
            self.e.R[rows] = self.e.traj_R[0,idx-1]
            self.e.v[rows] = direction*self.env.target_speed[rows]
            self.env._reset_speed_ema(rows)
            self.obs = self.env.get_observations()
        self.gates = torch.zeros_like(self.first)
        self.done = torch.zeros(self.batch, dtype=torch.bool, device=DEVICE)
        self.previous = torch.full_like(self.first, 7)
        self.reason = torch.zeros_like(self.first)
        self.flight_ticks = torch.zeros_like(self.first)
        self.speed_sum = torch.zeros_like(self.first,dtype=torch.float32)
        self.xy_error_sum = torch.zeros_like(self.speed_sum)
        self.s = station(self.e)
        return fingerprint

    def geometry(self):
        part = 6*(self.first+self.gates.clamp_max(2))
        centers = (self.e.base_voxels_gate[0,part,:3]+self.e.base_voxels_gate[0,part+1,:3])*0.5
        return centers, self.e.base_voxels_gate_R[0,part]

    def observe(self):
        target, _ = self.geometry()
        delta = torch.bmm((target-self.e.p)[:,None], self.env._current_R_yaw).squeeze(1)
        prev = self.table[self.previous]
        extras = torch.cat([delta.clamp(-40,40)/20, self.gates[:,None]/3,
                            prev[:, :1]/2, prev[:, 1:], self.env.target_speed/10],1)
        state = self.obs["state"].clone()
        state[:,:6] /= 10.
        return self.obs["depth"].reshape(-1,1,20,80), torch.cat([state, extras],1)

    def velocity(self, action):
        offset, factor = self.table[action].T
        ref,tangent,idx = self.e.get_state_by_s(self.s+3, self.e.env_traj_ids)
        right = self.e.traj_R[0,idx-1,:,1]
        error = ref+offset[:,None]*right-self.e.p
        normal = error-(error*tangent).sum(-1,keepdim=True)*tangent
        world = tangent*self.env.target_speed*factor[:,None] + 1.35*normal
        return torch.bmm(world[:,None],self.env._current_R_yaw).squeeze(1)

    def advance(self, action, final):
        self.previous = action
        total = torch.zeros(self.batch,device=DEVICE)
        for _ in range(HOLD_TICKS):
            active = ~self.done
            before = self.e.p.clone()
            cmd = self.velocity(action)
            center,basis = self.geometry()
            self.obs,_,_,_ = self.env.step(cmd)
            a = torch.bmm((before-center)[:,None],basis).squeeze(1)
            b = torch.bmm((self.e.p-center)[:,None],basis).squeeze(1)
            cross = (a[:,0]<0) & (b[:,0]>=0)
            frac = (-a[:,0]/(b[:,0]-a[:,0]).clamp_min(1e-7)).clamp(0,1)
            hit = a+frac[:,None]*(b-a)
            inside = (hit[:,1].abs()<=3) & (hit[:,2].abs()<=1.5)
            fail = self.env.fatal | (cross & ~inside)
            scored = active & ~fail & cross & inside
            self.gates += scored.long()
            success = scored & (self.gates==3)
            new_s = station(self.e)
            reward = 0.06*(new_s-self.s).clamp(-2,2)-0.015 + 2.*scored + 5.*success - 5.*fail
            total += torch.where(active,reward,0.)
            # Aggregate valid-flight ticks only. The vector simulator keeps inactive slots alive.
            self.flight_ticks += active.long()
            self.speed_sum += torch.where(active,self.e.v.norm(dim=1),0.)
            xy_err = self.env._current_delta_p_world[:,:2].norm(dim=1)
            self.xy_error_sum += torch.where(active,xy_err,0.)
            for condition,code in ((self.env.fatal & self.env.bounds,2),
                                   (self.env.fatal,1),(cross & ~inside,3),(success,4)):
                self.reason = torch.where(active & (self.reason==0) & condition,code,self.reason)
            self.s = new_s
            self.done |= fail | success
        if final:
            total -= 2.*(~self.done)
            self.done.fill_(True)
        return total


@torch.no_grad()
def collect(course, net, level, scene_seed, mode):
    fingerprint = course.start(level,scene_seed,mode=="sample")
    data=[]
    for t in range(MAX_DECISIONS):
        valid = ~course.done
        image,state=course.observe()
        logits,value=net(image,state)
        if not logits.isfinite().all(): raise RuntimeError("Nonfinite policy output")
        dist=Categorical(logits=logits)
        if mode=="sample": action=dist.sample()
        elif mode=="greedy": action=logits.argmax(-1)
        else: action=torch.full_like(course.gates,7)
        reward=course.advance(action,t+1==MAX_DECISIONS)
        if mode=="sample":
            data.append((image,state,action,dist.log_prob(action),value,reward,
                         (~course.done).float(),valid))
        if course.done.all(): break
    gates=course.gates.cpu().tolist()
    return data, gates, fingerprint


def update(net,opt,rows):
    # Every row is a complete episode; terminated slots never enter the loss.
    img,state,act,oldlog,val,rew,cont,valid=[torch.stack(col) for col in zip(*rows)]
    advantage=torch.zeros_like(val)
    carry=torch.zeros_like(val[0])
    next_v=torch.zeros_like(carry)
    for t in reversed(range(len(rows))):
        delta=rew[t]+0.99*cont[t]*next_v-val[t]
        carry=delta+0.99*0.95*cont[t]*carry
        advantage[t]=carry
        next_v=val[t]
    ret=(advantage+val)[valid]
    img,state,act,oldlog,adv=(x[valid] for x in (img,state,act,oldlog,advantage))
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
    sums=torch.zeros(5,device=DEVICE)
    samples=0
    for _ in range(4):
        for i in torch.randperm(len(act),device=DEVICE).split(512):
            logits,v=net(img[i],state[i])
            d=Categorical(logits=logits)
            ratio=(d.log_prob(act[i])-oldlog[i]).exp()
            pg=-torch.minimum(adv[i]*ratio,adv[i]*ratio.clamp(0.8,1.2)).mean()
            value_loss=(v-ret[i]).square().mean()
            entropy=d.entropy().mean()
            loss=pg+0.5*value_loss-0.012*entropy
            with torch.no_grad():
                log_ratio=d.log_prob(act[i])-oldlog[i]
                approx_kl=(ratio-1-log_ratio).mean()
                clip_fraction=((ratio-1).abs()>0.2).float().mean()
                sums+=len(i)*torch.stack((pg,value_loss,entropy,approx_kl,clip_fraction))
                samples+=len(i)
            if not loss.isfinite(): raise RuntimeError("Nonfinite PPO loss")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),0.7,error_if_nonfinite=True)
            opt.step()
    names=("PPO/policy_loss","PPO/value_loss","PPO/entropy","PPO/approx_KL","PPO/clip_fraction")
    stats=dict(zip(names,(sums/samples).tolist()))
    stats["Train/episode_reward"]=float(rew.sum(0).mean())
    return len(act),stats


def atomic_json(path, obj):
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def save(path,ck):
    tmp=path.with_suffix(".pt.tmp")
    torch.save(ck,tmp)
    tmp.replace(path)


def evaluate(course,net,mode):
    result={}
    for level,seed_value in DEV:
        _,gates,h=collect(course,net,level,seed_value,mode)
        result[str(level)]={"gates":gates,"hash":h,"completed":gates.count(3)}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--eval-every",type=int,default=25)
    p.add_argument("--updates",type=int,default=0,help="zero runs until STOP or SIGTERM")
    args=p.parse_args()
    if args.batch<4 or args.eval_every<1 or args.updates<0: p.error("invalid run settings")
    folder=args.out
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"trainer.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    seed(20261121)
    tr,dev=Course(args.batch),Course(48)
    net=ActorCritic().to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    step=decisions=0
    best=(-1,-1)
    last=folder/"latest.pt"
    if last.exists():
        ck=torch.load(last,map_location=DEVICE,weights_only=True)
        net.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["optimizer"])
        step,decisions,best=ck["update"],ck["decisions"],tuple(ck["best"])
    tb=SummaryWriter(str(folder/"tensorboard"),purge_step=step+1 if last.exists() else None,flush_secs=10)
    baseline_path=folder/"baseline.json"
    def checkpoint():
        return dict(policy=net.state_dict(),optimizer=opt.state_dict(),update=step,
                    decisions=decisions,best=best,actions=ACTIONS)
    def audit():
        nonlocal best
        result=evaluate(dev,net,"greedy")
        base=json.loads(baseline_path.read_text())
        if any(base[k]["hash"]!=r["hash"] for k,r in result.items()):
            raise RuntimeError("Unpaired evaluation")
        grade=(sum(r["completed"] for r in result.values()),sum(sum(r["gates"]) for r in result.values()))
        row={"update":step,"decisions":decisions,"completed":grade[0],"total":96,"by_level":result}
        atomic_json(folder/"eval-latest.json",row)
        if grade>best:
            best=grade
            save(folder/"best.pt",checkpoint())
            atomic_json(folder/"eval-best.json",row)
        record={"update":step,"completed":grade[0],"gates":grade[1]}
        for k,r in result.items():
            rate=r["completed"]/len(r["gates"])
            record["rate_level_"+k]=rate
            tb.add_scalar("Validation/level_"+k+"_three_gate_rate",rate,step)
        tb.add_scalar("Validation/three_gate_rate",grade[0]/96,step)
        tb.add_scalar("Validation/mean_gates",grade[1]/96,step)
        tb.add_scalar("Validation/center_rule",sum(r["completed"] for r in base.values())/96,step)
        tb.flush()
        with (folder/"eval.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        return grade[0]
    if not (folder/"baseline.json").exists():
        atomic_json(folder/"baseline.json",evaluate(dev,net,"center"))
    if (folder/"eval.jsonl").exists() and not (folder/"curves.jsonl").exists():
        # Earlier versions preserved fixed evaluation history only.
        base=json.loads(baseline_path.read_text())
        for line in (folder/"eval.jsonl").read_text().splitlines():
            previous=json.loads(line)
            tb.add_scalar("Validation/three_gate_rate",previous["completed"]/96,previous["update"])
            tb.add_scalar("Validation/mean_gates",previous["gates"]/96,previous["update"])
            tb.add_scalar("Validation/center_rule",sum(r["completed"] for r in base.values())/96,previous["update"])
        tb.flush()
    if not last.exists():
        audit()
        save(last,checkpoint())
    while not stopping and not (folder/"STOP").exists() and (not args.updates or step<args.updates):
        start=time.monotonic()
        step+=1
        rows,gates,_=collect(tr,net,2+(step%2),1000000+step,"sample")
        n,metrics=update(net,opt,rows)
        decisions+=n
        status={"update":step,"decisions":decisions,"batch":args.batch,"gpu":torch.cuda.get_device_name(),
                "train_completed":gates.count(3),"level":2+step%2,
                "seconds":round(time.monotonic()-start,2),"pid":os.getpid()}
        valid_ticks=tr.flight_ticks.sum().clamp_min(1)
        metrics.update({"Train/three_gate_rate":gates.count(3)/args.batch,
                        "Train/mean_gates":sum(gates)/args.batch,
                        "Train/episode_length_ticks":float(tr.flight_ticks.float().mean()),
                        "Flight/speed_mps":float(tr.speed_sum.sum()/valid_ticks),
                        "Flight/horizontal_error_m":float(tr.xy_error_sum.sum()/valid_ticks),
                        "Perf/decisions_per_second":n/(time.monotonic()-start)})
        for code,name in enumerate(("timeout","collision","bounds","missed_gate","success")):
            metrics["Outcome/"+name]=float((tr.reason==code).float().mean())
        metrics["Train/level_"+str(status["level"])+"_three_gate_rate"]=metrics["Train/three_gate_rate"]
        for key,val in metrics.items(): tb.add_scalar(key,val,step)
        with (folder/"curves.jsonl").open("a") as stream:
            stream.write(json.dumps({"update":step,"level":status["level"],**metrics},allow_nan=False)+"\n")
        status["mean_episode_reward"]=metrics["Train/episode_reward"]
        if step==1 or step%args.eval_every==0:
            status["dev_completed"]=audit()
        if step==1 or step%5==0 or "dev_completed" in status: save(last,checkpoint())
        atomic_json(folder/"status.json",status)
    save(last,checkpoint())
    tb.close()
    print("Stopped after update",step,flush=True)


if __name__=="__main__":
    main()

