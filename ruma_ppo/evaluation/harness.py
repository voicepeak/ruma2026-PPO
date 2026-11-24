"""Run frozen R1 scene suites with per-scene pairing and grouped confidence intervals."""

import argparse
import csv
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ruma_ppo.evaluation.metrics import metrics, paired_statistics
from ruma_ppo.hitsz.v12 import V12Course
from ruma_ppo.evaluation.lidar_bypass import LidarBypassFollower
from ruma_ppo.hitsz.branches import NEXT_GATE, ROUTE, UP, station
from ruma_ppo.hitsz.legacy_policy import ActorCritic, Course, DEVICE, HOLD_TICKS
from ruma_ppo.evaluation.rollout import LidarBrakeFollower, Progress, PWMRouteFollower
from ruma_ppo.env.maps.gate_graph import GateGraph
from ruma_ppo.env.sim.pwm_env import CONTROL_HZ


BATCH = 48
SUITES = {
    "val192": ((2, 3), (61001, 61002)),
    "test384": ((2, 3), (71001, 71002, 71003, 71004)),
    "hard96": ((4,), (72001, 72002)),
}


class LegacyPolicy:
    def __init__(self, course, network=None):
        self.course, self.network = course, network
        run_native = course.e._run

        def record_elapsed(acceleration, yaw, ctl_dt):
            self.dt = ctl_dt
            run_native(acceleration, yaw, ctl_dt)

        course.e._run = record_elapsed

    def step(self, tick, progress):
        course = self.course
        if self.network is None:
            velocity = course.velocity(None)
        else:
            course.gates = progress.gates
            if tick % HOLD_TICKS == 0:
                course.previous = self.network(*course.observe())[0].argmax(-1)
            velocity = course.velocity(course.previous)
        course.obs, _, _, _ = course.env.step(velocity)
        if self.network is not None:
            course.s = station(course.e)
        return self.dt


class NativePPOPolicy(LegacyPolicy):
    def __init__(self, course, policy, gate_input=None, future_input=None):
        super().__init__(course)
        self.policy = policy
        self.gate_input = gate_input
        self.future_input = future_input

    def step(self, tick, progress):
        if tick == 0:
            self.policy.reset()
        course = self.course
        obs = course.obs
        if self.gate_input is not None:
            index = progress.gates.clamp_max(len(progress.center) - 1)
            center, basis, active = progress.center[index], progress.basis[index], ~progress.done
            obs["gate"] = self.gate_input(
                course.e.p, course.env._current_R_yaw, center, basis, active,
            )
            if self.future_input is not None:
                obs["gate_future"] = self.future_input(
                    course.e.p, course.e.v, center, basis, progress.half_aperture[index], active,
                )
        course.obs, _, _, _ = course.env.step(self.command(obs, progress))
        return self.dt

    def command(self, obs, progress):
        return self.policy.act_inference(obs)


class NativeResidualPolicy(NativePPOPolicy):
    def step(self, tick, progress):
        if tick == 0:
            self.policy.base_policy.reset()
            self.steps = torch.zeros_like(progress.gates)
            self.saturated = torch.zeros_like(progress.gates)
            self.abs_delta_sum = torch.zeros_like(progress.flight_s)
        return super().step(tick, progress)

    def command(self, obs, progress):
        from ruma_ppo.training.residual import SATURATION_FRACTION, apply_vertical_delta

        obs["base_action"] = self.policy.base_policy.act_inference(obs)
        delta = self.policy.act_inference(obs)
        abs_delta = delta[:, 0].abs()
        active = ~progress.done
        self.steps += active.long()
        self.saturated += (active & (abs_delta >= self.policy.bound_mps * SATURATION_FRACTION)).long()
        self.abs_delta_sum += torch.where(active, abs_delta, 0.0)
        return apply_vertical_delta(obs["base_action"], delta)

    def record_residual(self, rows):
        for row, steps, saturated, delta_sum in zip(
            rows, self.steps.tolist(), self.saturated.tolist(), self.abs_delta_sum.tolist(), strict=True,
        ):
            row.update(residual_steps=steps, residual_saturated_steps=saturated, residual_abs_delta_sum_mps=delta_sum)


def load_native_ppo(path, course):
    from rsl_rl.modules import CustomVisionActorCritic
    from ruma_ppo.cli.setup_hitsz import git
    from ruma_ppo.training.hitsz_joint import (
        GateActorCritic, GateFutureConfig, OccupancyConfig, gate_features, gate_future_features,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    recipe = checkpoint["recipe"]
    if recipe["hitsz_commit"] != git(UP, "rev-parse", "HEAD"):
        raise ValueError("Native PPO checkpoint and evaluation HITSZ commits differ")
    cfg = recipe["train"]["policy"].copy()
    cfg.pop("class_name")
    variant = recipe["variant"]
    if variant != "residual":
        cfg["base_model_path"] = None
    future = GateFutureConfig(**recipe["additions"].get("gate_future", {}))
    if variant == "hitsz":
        model = CustomVisionActorCritic(None, recipe["train"]["obs_groups"], course.env.num_actions, **cfg)
    elif variant == "residual":
        from ruma_ppo.training.residual import ResidualGateActorCritic, make_frozen_base

        model = ResidualGateActorCritic(make_frozen_base(recipe["residual_base"]), **cfg)
    else:
        occupancy = OccupancyConfig(**recipe["additions"]["occupancy"]) if variant == "joint" else None
        model = GateActorCritic(None, recipe["train"]["obs_groups"], course.env.num_actions,
                                occupancy=occupancy, gate_future=future, **cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()
    policy_cls = NativeResidualPolicy if variant == "residual" else NativePPOPolicy
    policy = policy_cls(
        course, model, gate_input=None if variant == "hitsz" else gate_features,
        future_input=partial(gate_future_features, min_closing_mps=future.min_closing_mps)
        if future.enabled else None,
    )
    return policy, {"name": f"native_{variant}_{checkpoint['update']}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_suite(suite, graph, candidate, reference, pwm):
    cases = []
    graph_hash = graph.sha256
    levels, seeds = SUITES[suite]
    for level in levels:
        for seed in seeds:
            batch = {}
            for name, policy in (("v12", reference), ("candidate", candidate)):
                course = policy.course if isinstance(policy, LegacyPolicy) else pwm
                fingerprint = course.start(level, seed, False)
                progress = Progress(course.e, graph)
                braking = isinstance(policy, LidarBrakeFollower)
                bypassing = isinstance(policy, LidarBypassFollower)
                if braking:
                    policy.reset(seed)
                    brake_ticks = torch.zeros_like(progress.gates)
                if bypassing:
                    detour_ticks = torch.zeros_like(progress.gates)
                tick = 0
                while not progress.done.all():
                    before = course.e.p.clone()
                    if isinstance(policy, LegacyPolicy):
                        dt = policy.step(tick, progress)
                    else:
                        dt = policy.step(policy.command())
                        if braking:
                            brake_ticks += (~progress.done & (policy.speed_fraction < 1)).long()
                        if bypassing:
                            detour_ticks += (~progress.done & policy.detouring).long()
                    if not np.isfinite(dt) or dt <= 0:
                        raise RuntimeError(f"Invalid simulator tick {dt}")
                    progress.advance(before, dt)
                    tick += 1
                scores = list(progress.rows())
                if braking:
                    for score, pulses in zip(scores, brake_ticks.tolist()):
                        score["brake_s"] = pulses / CONTROL_HZ
                if bypassing:
                    for score, pulses in zip(scores, detour_ticks.tolist()):
                        score["detour_s"] = pulses / CONTROL_HZ
                if isinstance(policy, NativeResidualPolicy):
                    policy.record_residual(scores)
                batch[name] = (fingerprint, scores)
            expected, reference_rows = batch["v12"]
            actual, candidate_rows = batch["candidate"]
            if expected != actual or len(reference_rows) != BATCH or len(candidate_rows) != BATCH:
                raise RuntimeError(f"Unpaired scenes at level {level}, seed {seed}")
            for row, (ours, theirs) in enumerate(zip(candidate_rows, reference_rows)):
                identity = {"route": graph.route, "level": level, "seed": seed, "row": row}
                scene_key = [*identity.values(), graph_hash, expected]
                scene_hash = hashlib.sha256(json.dumps(scene_key, separators=(",", ":")).encode()).hexdigest()
                cases.append({**identity, "scene_sha256": scene_hash, "batch_sha256": expected,
                              "candidate": ours, "v12": theirs})
            print(f"{suite} level={level} seed={seed} cases={len(cases)}", flush=True)
    return cases


def save(out, suite, graph, cases, names, verified):
    with (out / "paired.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = ["route", "level", "seed", "row", "scene_sha256", "batch_sha256"]
        scores = list(cases[0]["candidate"])
        writer = csv.DictWriter(stream, fieldnames=[*fields, "model", *scores])
        writer.writeheader()
        for case in cases:
            for key in ("candidate", "v12"):
                writer.writerow({**{field: case[field] for field in fields}, "model": names[key]["name"], **case[key]})
    manifest = [{field: case[field] for field in fields} for case in cases]
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
    (out / "scenes.json").write_bytes(manifest_bytes)
    report = {"suite": suite, "route": graph.route, "gate_ids": [g.id for g in graph.gates],
              "gate_graph_sha256": graph.sha256, "scenes_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "models": names, "repeat_verified": verified,
              "candidate": metrics([case["candidate"] for case in cases]),
              "v12": metrics([case["v12"] for case in cases]), "paired": paired_statistics(cases)}
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--v12", type=Path, required=True)
    candidate_args = parser.add_mutually_exclusive_group()
    candidate_args.add_argument("--ppo", type=Path, help="use a frozen legacy PPO candidate instead of the PWM follower")
    candidate_args.add_argument("--native-ppo", type=Path, help="score a native joint or residual checkpoint")
    candidate_args.add_argument("--lidar-brake", action="store_true", help="apply the LiDAR stopping-distance rule")
    candidate_args.add_argument("--lidar-bypass", action="store_true", help="score lateral paths and brake using LiDAR")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-repeat", action="store_true", help="require identical per-scene results after reset")
    args = parser.parse_args()
    if args.verify_repeat and args.suite == "test384":
        parser.error("Verify repeatability on val192 or hard96 and reserve test384 for one evaluation")
    graph = GateGraph.from_hitsz(UP / ROUTE).select(range(NEXT_GATE, NEXT_GATE + 3))
    names = {"v12": {"name": "v12", "sha256": hashlib.sha256(args.v12.read_bytes()).hexdigest()}}
    reference = LegacyPolicy(V12Course(BATCH, args.v12))
    course = Course(BATCH)
    if args.ppo:
        checkpoint = torch.load(args.ppo, map_location=DEVICE, weights_only=True)
        network = ActorCritic().to(DEVICE).eval()
        network.load_state_dict(checkpoint["policy"])
        candidate = LegacyPolicy(course, network)
        names["candidate"] = {"name": f"ppo_{checkpoint['update']}",
                              "sha256": hashlib.sha256(args.ppo.read_bytes()).hexdigest()}
    elif args.native_ppo:
        candidate, names["candidate"] = load_native_ppo(args.native_ppo, course)
    elif args.lidar_brake or args.lidar_bypass:
        candidate = LidarBypassFollower(course.e) if args.lidar_bypass else LidarBrakeFollower(course.e)
        name = "pwm_lidar_bypass" if args.lidar_bypass else "pwm_lidar_brake"
        names["candidate"] = {"name": name, "speed_mps": candidate.speed_mps,
                              "brake": asdict(candidate.brake), "sensors": asdict(candidate.sensor.config)}
        if args.lidar_bypass:
            names["candidate"]["bypass"] = asdict(candidate.bypass)
    else:
        candidate = PWMRouteFollower(course.e)
        names["candidate"] = {"name": "pwm_geometric", "speed_mps": candidate.speed_mps}
    args.out.mkdir(parents=True, exist_ok=False)
    cases = run_suite(args.suite, graph, candidate, reference, course)
    if args.verify_repeat and cases != run_suite(args.suite, graph, candidate, reference, course):
        raise RuntimeError("Evaluation changed after resetting the same policies and scenes")
    save(args.out, args.suite, graph, cases, names, args.verify_repeat)


if __name__ == "__main__":
    main()
