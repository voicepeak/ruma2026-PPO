from collections import Counter, defaultdict
from math import comb

import numpy as np

def metrics(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    times = [row["flight_s"] for row in rows if row["outcome"] == "success"]
    report = {"cases": len(rows), "outcomes": dict(sorted(outcomes.items())),
              "three_gate_rate": outcomes["success"] / len(rows),
              "mean_gates": sum(r["gates"] for r in rows) / len(rows),
              "near_miss_count": sum(r["near_miss"] for r in rows),
              "mean_success_s": float(np.mean(times)) if times else None,
              "minimum_clearance_m": min(r["min_clearance_m"] for r in rows)}
    if "brake_s" in rows[0]:
        interventions = [row["brake_s"] for row in rows if row["brake_s"] > 0]
        report["braked_cases"] = len(interventions)
        report["mean_brake_s_when_active"] = float(np.mean(interventions)) if interventions else None
    if "detour_s" in rows[0]:
        report["detour_cases"] = sum(row["detour_s"] > 0 for row in rows)
    return report


def paired_statistics(cases):
    categories = Counter()
    groups = defaultdict(list)
    for case in cases:
        ours = case["candidate"]["outcome"] == "success"
        theirs = case["v12"]["outcome"] == "success"
        categories[ours, theirs] += 1
        groups[case["route"], case["level"], case["seed"]].append(int(ours) - int(theirs))
    wins, losses = categories[True, False], categories[False, True]
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n))

    rng = np.random.default_rng(20261121)
    strata = defaultdict(list)
    for (route, level, _), changes in sorted(groups.items()):
        strata[route, level].append(sum(changes))
    sample = np.zeros(10_000)
    for totals in strata.values():
        sample += rng.choice(totals, size=(len(sample), len(totals)), replace=True).sum(axis=1)
    ci = np.percentile(sample / len(cases), [2.5, 97.5]).tolist()
    return {"candidate_only": wins, "v12_only": losses, "both": categories[True, True],
            "neither": categories[False, False], "mcnemar_exact_p": p,
            "gain": (wins - losses) / len(cases), "group_bootstrap_95_ci": ci}

