"""Run the predeclared matched-geometry study and save all held-out branches."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from course import CANDIDATES, DT, OBS_NAMES, generate


def feature_view(data, variant):
    x = data["features"].clone()
    if variant == "D":
        x[..., OBS_NAMES.index("turn_sign")] = 0.
    return x


def gather(value, choice):
    return value.gather(1, choice[:, None]).squeeze(1)


def metrics(data, choice):
    gates = gather(data["gates"], choice)
    completed = gates == 3
    feasible = data["gates"].amax(1) == 3
    return dict(complete_rate=completed.float().mean().item(),
                choice_success_given_feasible=(completed.sum() / feasible.sum().clamp_min(1)).item(),
                mean_gates=gates.float().mean().item(),
                short_pass=gather(data["short"], choice).float().mean().item(),
                mean_seconds_on_completions=gather(data["time"], choice)[completed].mean().item()
                    if completed.any() else None)


def scores(output, variant):
    if variant == "B":
        return output[..., 0].sigmoid() + 0.02 * output[..., 1].clamp(-2, 2)
    # Expected ordered gate count, with short clearance used only as a small tie-break.
    return output[..., 2:5].sigmoid().sum(-1) + 0.002 * output[..., 1].clamp(-2, 2)


def make_model():
    return nn.Sequential(nn.Linear(len(OBS_NAMES), 64), nn.SiLU(), nn.Linear(64, 64),
                         nn.SiLU(), nn.Linear(64, 5))


def train(train_set, dev, test, variant, seed, epochs, out):
    torch.manual_seed(seed)
    x = feature_view(train_set, variant).flatten(0, 1)
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-5)
    x = (x - mean) / scale
    dy = (feature_view(dev, variant) - mean) / scale
    ty = (feature_view(test, variant) - mean) / scale
    model = make_model().to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    progress = train_set["gates"].flatten()[:, None] >= torch.arange(1, 4, device=x.device)
    short = train_set["short"].flatten().float()
    clearance = train_set["clearance"].flatten().clamp(-2, 2)
    best = None
    for epoch in range(epochs):
        for idx in torch.randperm(len(x), device=x.device).split(1024):
            pred = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(pred[:, 0], short[idx])
            loss = loss + 0.3 * F.smooth_l1_loss(pred[:, 1], clearance[idx])
            if variant != "B":
                loss = loss + F.binary_cross_entropy_with_logits(pred[:, 2:5], progress[idx].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            dev_choice = scores(model(dy), variant).argmax(1)
            # Each selector is chosen on development scenes by the same final task metric.
            key = (gather(dev["gates"], dev_choice) == 3).sum().item()
            if best is None or key > best[0]:
                best = (key, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    torch.save(dict(state_dict=best[2], mean=mean.cpu(), scale=scale.cpu(), variant=variant,
                    feature_names=OBS_NAMES, seed=seed, epoch=best[1]), out / f"{variant}-{seed}.pt")
    with torch.no_grad():
        return scores(model(ty), variant).argmax(1), best[1]


def paired_diff(data, first, second):
    # Re-sample entire geometry families; each family has four correlated scenes.
    def outcomes(choices):
        return np.stack([(gather(data["gates"], c) == 3).cpu().numpy() for c in choices]).mean(0)
    delta = (outcomes(first) - outcomes(second)).reshape(-1, 4).mean(1)
    rng = np.random.default_rng(2026)
    samples = delta[rng.integers(0, len(delta), (3000, len(delta)))].mean(1)
    return dict(delta_pp=float(100 * delta.mean()), family_bootstrap_95_pp=(100*np.quantile(samples, [.025,.975])).tolist())


def validate(data):
    x = data["features"]
    assert torch.isfinite(x).all() and torch.isfinite(data["clearance"]).all()
    assert ((data["gates"] >= 0) & (data["gates"] <= 3)).all()
    assert ((data["gates"] == 0) | data["short"]).all()
    # Reversing the future turn cannot change any pre-branch label.
    assert torch.equal(data["short"][::4], data["short"][1::4])
    # Mirror every lateral coordinate and reverse the candidate side within each speed block.
    n = len(CANDIDATES) // 2
    mirrored = torch.arange(len(CANDIDATES),device=x.device).reshape(2,n).flip(1).flatten()
    assert torch.equal(data["gates"][::4], data["gates"][2::4][:,mirrored])
    assert torch.equal(data["gates"][1::4], data["gates"][3::4][:,mirrored])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--families", type=int, nargs=3, default=[800,200,400])
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[7,23,61])
    a = p.parse_args()
    if min(*a.families, a.epochs) < 1: p.error("counts must be positive")
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    split_seeds = [11003,22003,33013]
    sets = [generate(seed,n,device) for n,seed in zip(a.families, split_seeds)]
    for data in sets: validate(data)
    tr, dev, test = sets
    choices = {"A": test["rule"].argmax(1), "oracle": (test["gates"].float() - 0.01*test["time"]).argmax(1)}
    report = dict(scope="Planar synthetic full-state proxy; one candidate decision per episode, shared continuation controller.",
                  families=a.families, split_seeds=split_seeds, network_seeds=a.seeds, epochs=a.epochs,
                  device=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
                  torch_version=str(torch.__version__), python=sys.version.split()[0], platform=platform.platform(),
                  dt=DT, candidates=CANDIDATES, metrics={}, selected_epochs={},
                  sources={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ("course.py","experiment.py")})
    for k in choices: report["metrics"][k] = metrics(test,choices[k])
    print("Held-out oracle and rule",report["metrics"],flush=True)
    for variant in ("B","C","D"):
        for seed in a.seeds:
            name=f"{variant}-{seed}"
            choices[name], epoch = train(tr,dev,test,variant,seed,a.epochs,a.out)
            report["metrics"][name] = metrics(test,choices[name])
            report["selected_epochs"][name] = epoch
            print(name,report["metrics"][name],flush=True)
    def group(k):
        return [choices[k]] if k in choices else [choices[f"{k}-{s}"] for s in a.seeds]
    report["paired"]={f"{i}-{j}":paired_diff(test,group(i),group(j)) for i,j in
                      [("oracle","A"),("C","A"),("C","B"),("C","D"),("oracle","C")]}
    # Ceiling when the turn bit is unavailable: best common action for each opposite-turn pair.
    success = (test["gates"] == 3).reshape(-1, 2, len(CANDIDATES))
    report["turn_blind_oracle_rate"] = success.float().mean(1).amax(1).mean().item()
    arrays = {k:v.cpu().numpy() for k,v in test.items()}
    np.savez_compressed(a.out/"holdout-branches.npz",**arrays)
    chosen = {k:v.cpu().tolist() for k,v in choices.items()}
    with (a.out/"holdout-choices.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["family","member",*chosen])
        for i in range(len(test["gates"])):
            w.writerow([i//4,i%4,*(col[i] for col in chosen.values())])
    (a.out/"report.json").write_text(json.dumps(report, indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Done",a.out,flush=True)

if __name__ == "__main__":
    main()

