"""Frozen-weight evaluation on unseen HITSZ gate windows and routes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.bypass_learning import BypassActorCritic, BypassLearningEnv
from evaluation.harness import BATCH
from hitsz_branches import NEXT_GATE, ROUTE, UP, station
from hitsz_ppo import Course, DEVICE, atomic_json
from maps.gate_graph import GateGraph
from scripts.train_bypass_ppo import episode, gate_graph, summarize


SEEDS = (20261121, 20261122, 20261123)
WINDOWS = (
    ("r1-early", ROUTE, 10),
    ("r1-late", ROUTE, 51),
    ("r2-climb", "path_point/3-5_1_training.yaml", 31),
    ("r3-descent", "path_point/5-8_2_training.yaml", 14),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_geometry(route, first):
    graph = GateGraph.from_hitsz(UP / route).select(range(first, first + 3))
    start = graph.gates[0].station_m - 8.0
    points = np.asarray(graph.waypoints_ned)
    stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(start, graph.gates[-1].station_m, 31)
    xyz = np.stack([np.interp(samples, stations, points[:, axis]) for axis in range(3)], axis=-1)
    dx = np.diff(xyz, axis=0)
    yaw = np.unwrap(np.arctan2(dx[:, 1], dx[:, 0]))
    return graph, start, float(np.degrees(np.ptp(yaw))), float(xyz[0, 2] - xyz[-1, 2])


def freeze(out, model_root):
    out.mkdir(parents=True, exist_ok=False)
    segments = []
    for index, (name, route, first) in enumerate(WINDOWS):
        graph, start, turn, climb = segment_geometry(route, first)
        segments.append({"id": name, "route": route, "route_sha256": sha(UP / route),
                         "gate_graph_sha256": graph.sha256, "start_m": start,
                         "gate_ids": [g.id for g in graph.gates], "gate_stations_m": [g.station_m for g in graph.gates],
                         "heading_span_deg": round(turn, 3), "climb_m": round(climb, 3),
                         "groups": [{"level": level, "seed": 83001 + 2 * index + j} for j, level in enumerate((2, 3))]})
    weights = []
    for seed in SEEDS:
        path = model_root / f"wait-{seed}" / "latest.pt"
        weights.append({"seed": seed, "path": str(path.resolve()), "sha256": sha(path), "update": 24})
    atomic_json(out / "protocol.json", {"batch_per_group": BATCH, "unique_scenes": len(segments) * 2 * BATCH,
                                         "segments": segments, "weights": weights})


def rule_policy(features, prior):
    # Greedy evaluation of the original candidate scoring rule.
    return prior, features.new_zeros(len(features))


def verify_frames(course, graph, start):
    e = course.e
    indices = torch.tensor([6 * gate.id for gate in graph.gates], device=e.device)
    centers = (e.base_voxels_gate[0, indices, :3] + e.base_voxels_gate[0, indices + 1, :3]) / 2
    torch.testing.assert_close(centers, e.p.new_tensor([g.center_ned for g in graph.gates]), rtol=0, atol=5e-4)
    torch.testing.assert_close(e.base_voxels_gate_R[0, indices], e.R.new_tensor([g.basis_ned for g in graph.gates]),
                               rtol=0, atol=1e-4)
    torch.testing.assert_close(station(e), e.p.new_full((e.batch_size,), start), rtol=0, atol=5e-4)


def anchor(baseline):
    graph = gate_graph()
    course = Course(BATCH)
    env = BypassLearningEnv(course, graph, allow_wait=False)
    _, rows, fingerprint, _ = episode(env, rule_policy, 2, 61001, sample=False)
    with baseline.open(newline="", encoding="utf-8") as stream:
        archived = {int(r["row"]): r for r in csv.DictReader(stream) if r["model"] == "pwm_lidar_bypass"
                    and r["level"] == "2" and r["seed"] == "61001"}
    assert len(archived) == len(rows) == BATCH
    for i, row in enumerate(rows):
        expected = archived[i]
        assert expected["batch_sha256"] == fingerprint
        assert expected["outcome"] == row["outcome"] and int(expected["gates"]) == row["gates"]
        assert abs(float(expected["flight_s"]) - row["flight_s"]) <= 1e-6
    print("TRANSFER_ANCHOR_OK", summarize(rows), flush=True)


def evaluate_segment(out, segment_id):
    protocol = json.loads((out / "protocol.json").read_text())
    segment, = [s for s in protocol["segments"] if s["id"] == segment_id]
    route, gate_ids = segment["route"], segment["gate_ids"]
    graph, start, _, _ = segment_geometry(route, gate_ids[0])
    if sha(UP / route) != segment["route_sha256"] or graph.sha256 != segment["gate_graph_sha256"] or start != segment["start_m"]:
        raise ValueError("Route geometry differs from the frozen protocol")

    course = Course(BATCH, route=route, start=start, first_gate=gate_ids[0])
    rule_env = BypassLearningEnv(course, graph, allow_wait=False)
    policy_env = BypassLearningEnv(course, graph, allow_wait=True)
    roster = [("rule", rule_env, rule_policy)]
    training_hash = gate_graph().sha256
    for weight in protocol["weights"]:
        path = Path(weight["path"])
        if sha(path) != weight["sha256"]:
            raise ValueError("Checkpoint changed after the protocol was frozen")
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        if (checkpoint["config"]["graph_sha256"] != training_hash or checkpoint["update"] != weight["update"]
                or checkpoint["config"]["seed"] != weight["seed"] or not checkpoint["config"]["allow_wait"]):
            raise ValueError("Unexpected policy architecture, seed, training graph or checkpoint update")
        policy = BypassActorCritic(True).to(DEVICE).eval()
        policy.load_state_dict(checkpoint["policy"])
        roster.append((f"ppo-{weight['seed']}", policy_env, policy))

    rows = []
    for group in segment["groups"]:
        level, scene_seed = group["level"], group["seed"]
        expected = None
        for name, env, policy in roster:
            _, scores, fingerprint, _ = episode(env, policy, level, scene_seed, sample=False)
            if name == "rule":
                expected = fingerprint
                starting_contact = [r["initial_contact"] for r in scores]
                env.reset(level, scene_seed)
                verify_frames(course, graph, start)
            if fingerprint != expected or [r["initial_contact"] for r in scores] != starting_contact:
                raise RuntimeError(f"Unpaired scene initialization: {segment_id}, {level}, {scene_seed}")
            for row, score in enumerate(scores):
                key = [segment_id, route, gate_ids, start, level, scene_seed, row, fingerprint]
                scene_hash = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
                rows.append({"segment": segment_id, "route": route, "level": level, "scene_seed": scene_seed,
                             "row": row, "scene_sha256": scene_hash, "batch_sha256": fingerprint, "model": name, **score})
            print("TRANSFER_GROUP", segment_id, level, name, summarize(scores), flush=True)

    destination = out / f"{segment_id}.csv"
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combine(out):
    protocol = json.loads((out / "protocol.json").read_text())
    models = ["rule", *(f"ppo-{seed}" for seed in SEEDS)]
    totals, segments = defaultdict(list), {}
    for segment in protocol["segments"]:
        by_model = defaultdict(dict)
        with (out / f"{segment['id']}.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["initial_contact"] = row["initial_contact"] == "True"
                row["wait_s"] = float(row["wait_s"])
                by_model[row["model"]][row["scene_sha256"]] = row
        rule = by_model["rule"]
        paired = {}
        for model in models:
            rows = by_model[model]
            if len(rows) != 96 or rows.keys() != rule.keys():
                raise RuntimeError("Incomplete scene pairing")
            totals[model].extend(rows.values())
            stats = summarize(list(rows.values()))
            if model != "rule":
                transitions = Counter((rule[k]["outcome"], r["outcome"]) for k, r in rows.items() if not r["initial_contact"])
                stats["vs_rule"] = {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())}
            paired[model] = stats
        segments[segment["id"]] = paired
    result = {"unique_scenes": protocol["unique_scenes"], "segments": segments,
              "aggregate": {model: summarize(rows) for model, rows in totals.items()}}
    atomic_json(out / "result.json", result)
    print("TRANSFER_DONE", result["aggregate"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("freeze")
    prep.add_argument("--models", required=True, type=Path)
    check = sub.add_parser("anchor")
    check.add_argument("--baseline", required=True, type=Path)
    segment = sub.add_parser("segment")
    segment.add_argument("--id", choices=[segment[0] for segment in WINDOWS], required=True)
    combined = sub.add_parser("combine")
    for command in (prep, segment, combined):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out, args.models)
    elif args.command == "anchor":
        anchor(args.baseline)
    elif args.command == "segment":
        evaluate_segment(args.out, args.id)
    else:
        combine(args.out)


if __name__ == "__main__":
    main()
