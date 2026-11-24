"""Summarize equal-budget comparisons on the independent A/B scenarios."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from ruma_ppo.evaluation.metrics import paired_statistics


NAMES = ("ab-start-u2501", "ab-control-u2601", "ab-alignment-u2601", "ab-control-u2701", "ab-alignment-u2701")


def load(root, name):
    with (root / name / "scenes.csv").open() as stream:
        rows = {row["scene_sha256"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 384
    return rows


def counts(rows):
    all_rows = list(rows.values())
    misses = [r for r in all_rows if r["outcome"] == "missed_gate"]
    crossing = [r for r in all_rows if r["final_crossing_gate_id"] == "40"]
    return {
        "cases": len(all_rows),
        "passed": sum(r["outcome"] == "success" for r in all_rows),
        "missed_gates": len(misses),
        "first_contact_at_spawn": sum(r["initial_contact"] == "True" for r in all_rows),
        "flight_collisions": sum(r["initial_contact"] == "False" for r in all_rows),
        "below_gate_40_misses": sum(r["miss_side"] == "below" and r["missed_gate_id"] == "40" for r in misses),
        "median_gate_40_lower_margin_m": median(float(r["final_crossing_lower_margin_m"]) for r in crossing),
    }


def compare(a, b):
    assert a.keys() == b.keys()
    assert all(row["initial_contact"] == b[key]["initial_contact"] or
               {row["initial_contact"], b[key]["initial_contact"]} <= {"", "False"} for key, row in a.items())
    cases = [{**row, "candidate": row, "v12": a[key]} for key, row in b.items()]
    stats = paired_statistics(cases)
    stats["reference_only"] = stats.pop("v12_only")
    common_passes = [key for key, row in b.items() if row["outcome"] == a[key]["outcome"] == "success"]
    stats["both_passed_mean_time_delta_s"] = mean(float(b[key]["flight_s"]) - float(a[key]["flight_s"])
                                                  for key in common_passes) if common_passes else None
    stats["outcome_changes"] = dict(Counter(
        f"{a[key]['outcome']} -> {row['outcome']}" for key, row in b.items() if a[key]["outcome"] != row["outcome"]
    ))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    root = parser.parse_args().report_dir
    rows = {name: load(root, name) for name in NAMES if (root / name / "scenes.csv").is_file()}
    report = {"seeds": [62021, 62022, 62023, 62024], "obstacle_levels": [2, 3],
              "results": {name: counts(data) for name, data in rows.items()}, "comparisons": {}}
    for update in (2601, 2701):
        experiment = f"ab-alignment-u{update}"
        if experiment in rows:
            for reference in (f"ab-control-u{update}", "ab-start-u2501"):
                report["comparisons"][f"{experiment} vs {reference}"] = compare(rows[reference], rows[experiment])
    output = json.dumps(report, indent=2) + "\n"
    print(output, end="")


if __name__ == "__main__":
    main()
