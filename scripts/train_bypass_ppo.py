"""Train high-level PPO over the LiDAR candidates and evaluate frozen R1 scenes."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ruma_ppo.evaluation.bypass_learning import BypassActorCritic, BypassLearningEnv, Rewards, RULE_PRIOR_SCALE, WAIT
from ruma_ppo.evaluation.harness import BATCH, SUITES
from ruma_ppo.evaluation.rollout import COLLISION
from hitsz_branches import NEXT_GATE, ROUTE, UP, seed
from hitsz_ppo import Course, DEVICE, atomic_json, save, update
from ruma_ppo.env.maps.gate_graph import GateGraph


LEARNING_RATE = 3e-4
PPO = {"gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "entropy_coefficient": 0.01, "minibatch": 512}


def gate_graph():
    return GateGraph.from_hitsz(UP / ROUTE).select(range(NEXT_GATE, NEXT_GATE + 3))


@torch.no_grad()
def episode(env, policy, level, scene_seed, *, sample):
    fingerprint = env.reset(level, scene_seed)
    progress = env.progress
    starting_contact = (progress.outcome == COLLISION).tolist()
    active_now = ~progress.done
    records = []
    decision_count = torch.zeros_like(progress.gates)
    wait_seconds = progress.flight_s.clone()
    action_counts = torch.zeros(WAIT + 1, dtype=torch.long, device=DEVICE)
    while not progress.done.all():
        features, prior = env.observe()
        logits, values = policy(features, prior)
        if not logits.isfinite().all() or not values.isfinite().all():
            raise RuntimeError("Nonfinite policy output")
        distribution = Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(dim=-1)
        before = progress.flight_s.clone()
        reward = env.step(action)
        action_counts += torch.bincount(action[active_now], minlength=WAIT + 1)
        decision_count += active_now.long()
        wait_seconds += (action == WAIT) * (progress.flight_s - before)
        continues = ~progress.done
        if sample:
            records.append((features, prior, action, distribution.log_prob(action), values, reward,
                            continues.float(), active_now))
        active_now = continues
    scores = list(progress.rows())
    for row, contact, decisions, waiting in zip(scores, starting_contact,
                                               decision_count.tolist(), wait_seconds.tolist()):
        row.update(initial_contact=contact, decisions=decisions, wait_s=round(waiting, 6))
    return records, scores, fingerprint, action_counts.tolist()


def summarize(scores):
    valid = [row for row in scores if not row["initial_contact"]]
    return {"cases": len(scores), "initial_contact": len(scores) - len(valid),
            "valid_start_outcomes": dict(sorted(Counter(row["outcome"] for row in valid).items())),
            "waiting_cases": sum(row["wait_s"] > 0 for row in scores)}


def train(args):
    graph = gate_graph()
    config = {"seed": args.seed, "batch": args.batch, "allow_wait": args.wait,
              "graph_sha256": graph.sha256, "rewards": asdict(Rewards()),
              "ppo": {"learning_rate": LEARNING_RATE, **PPO}, "rule_prior_scale": RULE_PRIOR_SCALE}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    with (args.out / "trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed(args.seed)
        env = BypassLearningEnv(Course(args.batch), graph, allow_wait=args.wait)
        policy = BypassActorCritic(args.wait).to(DEVICE)
        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        completed = decisions = 0
        path = args.out / "latest.pt"
        if args.resume:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
            if checkpoint["config"] != config:
                raise ValueError("Resume settings differ from the checkpoint")
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            completed, decisions = checkpoint["update"], checkpoint["decisions"]
        atomic_json(args.out / "config.json", config)
        atomic_json(args.out / "status.json", {"update": completed, "updates": args.updates, "decisions": decisions})

        for number in range(completed + 1, args.updates + 1):
            level = 2 + (number - 1) % 2
            # Independent training seeds above the numeric range of every frozen suite.
            scene_seed = 1_000_000 + int(np.random.SeedSequence([args.seed, number]).generate_state(1)[0]) % 1_000_000_000
            records, scores, fingerprint, actions = episode(env, policy, level, scene_seed, sample=True)
            if not records:
                raise RuntimeError("Every reset state was in contact; no PPO transitions collected")
            count, statistics = update(policy, optimizer, records, **PPO)
            decisions += count
            details = {"update": number, "updates": args.updates, "decisions": decisions,
                       "level": level, "scene_seed": scene_seed, "batch_sha256": fingerprint,
                       "action_counts": actions, **summarize(scores), **statistics}
            save(path, {"config": config, "update": number, "decisions": decisions,
                        "policy": policy.state_dict(), "optimizer": optimizer.state_dict()})
            with (args.out / "curves.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(details, allow_nan=False) + "\n")
            atomic_json(args.out / "status.json", details)
            print(json.dumps(details, allow_nan=False), flush=True)


def evaluate(args):
    graph = gate_graph()
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    config = checkpoint["config"]
    if config["graph_sha256"] != graph.sha256:
        raise ValueError("Checkpoint and validation gate graph differ")
    policy = BypassActorCritic(config["allow_wait"]).to(DEVICE).eval()
    policy.load_state_dict(checkpoint["policy"])
    env = BypassLearningEnv(Course(BATCH), graph, allow_wait=config["allow_wait"])
    with args.baseline.open(newline="", encoding="utf-8") as stream:
        baseline = {(int(r["level"]), int(r["seed"]), int(r["row"])): r for r in csv.DictReader(stream)
                    if r["model"] == "pwm_lidar_bypass"}
    if len(baseline) != 192:
        raise ValueError("Expected the 192 archived rule-bypass episodes")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, reference, pair = [], [], Counter()
    for level in SUITES["val192"][0]:
        for scene_seed in SUITES["val192"][1]:
            _, scores, fingerprint, actions = episode(env, policy, level, scene_seed, sample=False)
            for number, score in enumerate(scores):
                old = baseline[level, scene_seed, number]
                if old["batch_sha256"] != fingerprint:
                    raise RuntimeError(f"Different initial conditions at {(level, scene_seed, number)}")
                old_initial = old["outcome"] == "collision" and float(old["flight_s"]) == 0
                if old_initial != score["initial_contact"]:
                    raise RuntimeError("A policy changed the initial contact count")
                reference.append({"outcome": old["outcome"], "initial_contact": old_initial, "wait_s": 0.0})
                if not old_initial:
                    pair[old["outcome"], score["outcome"]] += 1
                rows.append({"level": level, "seed": scene_seed, "row": number, "scene_sha256": old["scene_sha256"],
                             "batch_sha256": fingerprint, "rule_outcome": old["outcome"], **score})
            print("val192", level, scene_seed, summarize(scores), "actions", actions, flush=True)
    with (args.out / "paired.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"suite": "val192", "config": config, "checkpoint_update": checkpoint["update"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "ppo": summarize(rows), "rule": summarize(reference),
              "paired_transitions": {f"{old}->{new}": n for (old, new), n in sorted(pair.items())}}
    atomic_json(args.out / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--batch", type=int, default=32)
    training.add_argument("--updates", type=int, required=True)
    training.add_argument("--wait", action="store_true")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--out", type=Path, required=True)
    validation = sub.add_parser("eval")
    validation.add_argument("--checkpoint", type=Path, required=True)
    validation.add_argument("--baseline", type=Path, required=True)
    validation.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.batch < 2 or args.updates < 1 or not 0 <= args.seed < 2**32:
            parser.error("Use a batch of at least 2, a positive update count, and a uint32 seed")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
