"""Independent confirmation on new layouts with a direction-aware geometric baseline."""
import argparse
import json
from pathlib import Path

import torch
from ruma_ppo.hitsz.course import generate
from scripts.legacy.experiment import feature_view, make_model, metrics, paired_diff, scores, validate


def direction_rule(data):
    ox, oy, ov, radius, speed, sign, amplitude, gx, offset, factor = data["features"].unbind(-1)
    eta = ox / (speed * factor)
    clearance = (offset - oy - ov * eta).abs() - radius - 0.32
    # Static analytic prediction of lateral reach during the remaining time to gate two.
    remaining_time = (gx - 14.) / speed
    lateral_reach = 0.5 * 8. * remaining_time.square()
    bend_margin = 1.5 + lateral_reach - (sign * amplitude - offset).abs()
    bottleneck = torch.minimum(clearance, bend_margin)
    return (bottleneck + 0.01 * factor).argmax(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    a=p.parse_args()
    header=json.loads((a.results/"report.json").read_text())
    device=torch.device("cuda")
    test=generate(44017,400,device)
    validate(test)
    chosen={"A":test["rule"].argmax(1),"geometry_turn":direction_rule(test),
            "oracle":(test["gates"].float()-0.01*test["time"]).argmax(1)}
    with torch.no_grad():
        for seed in header["network_seeds"]:
            name=f"C-{seed}"
            ck=torch.load(a.results/(name+".pt"),map_location=device,weights_only=True)
            model=make_model().to(device)
            model.load_state_dict(ck["state_dict"])
            obs=(feature_view(test,"C")-ck["mean"])/ck["scale"]
            chosen[name]=scores(model(obs),"C").argmax(1)
    cs=[chosen[f"C-{s}"] for s in header["network_seeds"]]
    report={"split_seed":44017,"families":400,"training":"frozen",
            "metrics":{k:metrics(test,v) for k,v in chosen.items()},
            "paired":{"C-geometry_turn":paired_diff(test,cs,[chosen["geometry_turn"]]),
                      "geometry_turn-A":paired_diff(test,[chosen["geometry_turn"]],[chosen["A"]])}}
    (a.results/"direction-control.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["paired"]))

if __name__=="__main__":
    main()

