"""Train the native HITSZ PPO, with optional gates and vehicle occupancy."""

import argparse
from copy import deepcopy
import fcntl
from hashlib import file_digest, sha256
import os
from pathlib import Path
import random
import sys

from ruma_ppo.cli.setup_hitsz import CACHE, ROOT, git, verify_checkout


DEFAULT_CONFIG = ROOT / "configs" / "train_joint.yaml"


def train(args, hitsz):
    import numpy as np
    import torch
    from Env.utils import load_yaml, make_env
    from RL.rl_env_wrapper import RLQuadEnv
    from ruma_ppo.hitsz.legacy_policy import atomic_json
    from ruma_ppo.training.hitsz_joint import GateConfig, GateEnv, GateFutureConfig, HitszRunner, OccupancyConfig
    from ruma_ppo.training.residual import (
        RESIDUAL_GROUPS, RESIDUAL_POLICY, ResidualGateEnv, ResidualRunner, make_frozen_base,
    )

    if not torch.cuda.is_available():
        raise ValueError("HITSZ training requires a working CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("This training entry runs one process on a single GPU")

    residual = args.variant == "residual"
    if args.residual_bc_steps is not None and (
        not residual or args.base_checkpoint is None or args.residual_bc_steps < 1
    ):
        raise ValueError("--residual-bc-steps needs a positive count and a new residual run with --base-checkpoint")
    checkpoint = args.out / "latest.pt" if args.resume else args.resume_from
    for path in (checkpoint, args.base_checkpoint):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    addition = load_yaml(args.config)
    future_settings = addition.pop("gate_future", {})
    if args.gate_future or residual:
        future_settings["enabled"] = True
    if args.gate_future_unfreeze_actor_head:
        future_settings["unfreeze_actor_head"] = True
    future = GateFutureConfig(**future_settings)
    if future.unfreeze_actor_head and args.variant != "joint":
        raise ValueError("A0b requires the joint variant")
    if future.enabled:
        if args.variant == "hitsz":
            raise ValueError("Gate future features require the gate or joint variant")
        if args.base_checkpoint is None and not (args.resume or args.resume_from):
            raise ValueError("A new gate future run requires --base-checkpoint")
        addition["gate_future"] = future_settings
    if args.gate_height_alignment_weight is not None:
        if args.variant == "hitsz":
            raise ValueError("Gate height alignment requires the gate or joint variant")
        addition["gate"]["height_alignment_weight"] = args.gate_height_alignment_weight
    env_cfg = load_yaml(hitsz / "RL" / "configs" / "config_env.yaml")
    train_cfg = load_yaml(hitsz / "RL" / "configs" / "config_rl.yaml")
    if residual:
        if args.lr_schedule == "stable":
            raise ValueError("Residual PPO supports fixed or adaptive learning rates")
        train_cfg["obs_groups"] = deepcopy(RESIDUAL_GROUPS)
        train_cfg["policy"] = RESIDUAL_POLICY.copy()
    train_cfg["save_interval"] = int(addition["checkpoint_interval"])
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.envs is not None:
        env_cfg["env"]["batch_size"] = args.envs
    if args.num_steps_per_env is not None:
        train_cfg["num_steps_per_env"] = args.num_steps_per_env
    if args.num_mini_batches is not None:
        train_cfg["algorithm"]["num_mini_batches"] = args.num_mini_batches
    if args.lr_schedule is not None:
        train_cfg["algorithm"]["schedule"] = args.lr_schedule
    if args.learning_rate is not None:
        train_cfg["algorithm"]["learning_rate"] = args.learning_rate
    if train_cfg["num_steps_per_env"] < 1 or train_cfg["algorithm"]["num_mini_batches"] < 1:
        raise ValueError("Use positive rollout step and mini-batch counts")
    if train_cfg["algorithm"]["learning_rate"] <= 0:
        raise ValueError("The learning rate must be positive")
    updates = args.updates if args.updates is not None else train_cfg["max_iterations"]
    envs, seed = env_cfg["env"]["batch_size"], train_cfg["seed"]
    if updates < 1 or train_cfg["save_interval"] < 1 or not 0 <= seed < 2**32:
        raise ValueError("Use positive update and save counts and a uint32 seed")
    if args.target_env_steps is not None and args.target_env_steps < 1:
        raise ValueError("The environment step target must be positive")
    if args.grad_accum_steps is not None and args.grad_accum_steps < 1:
        raise ValueError("The gradient accumulation count must be positive")
    minibatches = train_cfg["algorithm"]["num_mini_batches"]
    if envs < minibatches or envs % minibatches:
        raise ValueError(f"Environment count must be a positive multiple of {minibatches}")

    gate = GateConfig(**addition["gate"]) if args.variant != "hitsz" else None
    occupancy = OccupancyConfig(**addition["occupancy"]) if args.variant == "joint" else None
    recipe = {
        "variant": args.variant,
        "hitsz_commit": git(hitsz, "rev-parse", "HEAD"),
        "environment": deepcopy(env_cfg),
        "train": deepcopy(train_cfg),
        "additions": addition,
    }
    if residual:
        source = torch.load(args.base_checkpoint or checkpoint, map_location="cpu", weights_only=True)
        if checkpoint is None:
            recipe["residual_base"] = source["recipe"]
            base_weights = source["model_state_dict"]
            if args.residual_bc_steps is not None:
                recipe["residual_bc"] = {"teacher": "geometric_height", "steps": args.residual_bc_steps}
        else:
            recipe["residual_base"] = source["recipe"]["residual_base"]
            if "residual_bc" in source["recipe"]:
                recipe["residual_bc"] = source["recipe"]["residual_bc"]
            base_weights = {key.removeprefix("base_policy."): value
                            for key, value in source["model_state_dict"].items() if key.startswith("base_policy.")}
        if recipe["residual_base"]["hitsz_commit"] != recipe["hitsz_commit"]:
            raise ValueError("Residual base checkpoint and training HITSZ commits differ")

    device = f"cuda:{env_cfg['runtime']['gpu_id']}"
    args.out.mkdir(parents=True, exist_ok=args.resume or args.resume_from is not None)
    with (args.out / "trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        sim = make_env(env_cfg, device=torch.device(device))
        cfg = {**train_cfg, "env": env_cfg["env"]}
        actual_cfg = deepcopy(train_cfg)
        if residual:
            base_policy = make_frozen_base(recipe["residual_base"])
            base_policy.load_state_dict(base_weights)
            env = ResidualGateEnv(cfg, sim, gate, future, base_policy.to(device), RESIDUAL_POLICY["bound_mps"])
            runner = ResidualRunner(env, actual_cfg, args.out, device, recipe, None)
        else:
            env = RLQuadEnv(cfg, sim) if gate is None else GateEnv(cfg, sim, gate, occupancy, future)
            actual_cfg["policy"]["base_model_path"] = (
                None if args.base_checkpoint is not None or future.enabled
                else str(hitsz / train_cfg["policy"]["base_model_path"])
            )
            runner = HitszRunner(env, actual_cfg, args.out, device, recipe, occupancy, future)
        if args.grad_accum_steps is not None:
            if args.variant != "joint":
                raise ValueError("Gradient accumulation is only implemented for the joint variant")
            runner.alg.accum_steps = args.grad_accum_steps
        if checkpoint is not None:
            runner.restore(
                checkpoint,
                allow_optimization_change=args.resume_from is not None,
                allow_gate_height_alignment_change=(
                    args.resume_from is not None and args.gate_height_alignment_weight is not None
                ),
            )
        elif args.base_checkpoint is not None:
            if residual:
                runner.restore_curriculum(source["curriculum"])
            else:
                runner.load_base(args.base_checkpoint)
        if args.learning_rate is not None:
            runner.alg.learning_rate = args.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate * group.get("lr_scale", 1.0)
        steps_per_update = envs * train_cfg["num_steps_per_env"]
        steps_done = runner.environment_steps(runner.current_learning_iteration)
        if args.target_env_steps is not None:
            steps_left = max(0, args.target_env_steps - steps_done)
            updates = runner.current_learning_iteration + (steps_left + steps_per_update - 1) // steps_per_update
        remaining = updates - runner.current_learning_iteration
        if remaining <= 0:
            print(f"Target already reached at update {runner.current_learning_iteration}", flush=True)
            return

        import quadsim_cuda

        if not args.resume:
            atomic_json(args.out / "config.json", recipe)
            environment = {
                "repo_commit": git(ROOT, "rev-parse", "HEAD"),
                "repo_status": git(ROOT, "status", "--porcelain").splitlines(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
                "quadsim_cuda_sha256": sha256(Path(quadsim_cuda.__file__).read_bytes()).hexdigest(),
                "steps_per_update": steps_per_update,
                "resume_from": str(checkpoint.resolve()) if checkpoint else None,
                "initial_environment_steps": steps_done,
            }
            if args.base_checkpoint is not None:
                with args.base_checkpoint.open("rb") as stream:
                    environment["base_checkpoint_sha256"] = file_digest(stream, "sha256").hexdigest()
                environment["base_checkpoint"] = str(args.base_checkpoint.resolve())
            atomic_json(args.out / "environment.json", environment)
        if args.residual_bc_steps is not None:
            from ruma_ppo.training.residual_bc import behavior_clone

            behavior_clone(runner, args.residual_bc_steps)
            runner.restore_curriculum(source["curriculum"])
        print(f"Training {args.variant}: {envs} environments, updates "
              f"{runner.current_learning_iteration + 1}–{updates}, seed {seed}", flush=True)
        runner.learn(remaining, init_at_random_ep_len=checkpoint is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("hitsz", "gate", "joint", "residual"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--envs", type=int, help="Override the upstream 4096 environments")
    parser.add_argument("--num-steps-per-env", type=int, help="Override rollout steps per update")
    parser.add_argument("--num-mini-batches", type=int, help="Override PPO mini-batches per epoch")
    parser.add_argument("--grad-accum-steps", type=int, help="Accumulate this many micro-batches per optimizer step")
    parser.add_argument("--lr-schedule", choices=("adaptive", "fixed", "stable"), help="Override the PPO learning-rate schedule")
    parser.add_argument("--learning-rate", type=float, help="Override the initial learning rate")
    parser.add_argument("--gate-height-alignment-weight", type=float,
                        help="Add a bounded penalty for approaching below the next gate")
    parser.add_argument("--gate-future", action="store_true",
                        help="Add explicit future gate crossing features (A0)")
    parser.add_argument("--gate-future-unfreeze-actor-head", action="store_true",
                        help="Also fit the joint actor output head at one tenth of the base learning rate (A0b)")
    parser.add_argument("--residual-bc-steps", type=int,
                        help="Warm-start the residual actor with this many geometric teacher rollout steps")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--updates", type=int, help="Total updates including any checkpoint progress")
    budget.add_argument("--target-env-steps", type=int, help="Total environment steps across all devices")
    parser.add_argument("--seed", type=int, help="Override the upstream training seed")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", help="Continue in the same output directory")
    resume.add_argument("--resume-from", type=Path, help="Continue from a checkpoint in a new directory")
    resume.add_argument("--base-checkpoint", type=Path,
                        help="Start a new run with the checkpoint model and curriculum")
    args = parser.parse_args()
    hitsz = Path(os.environ.get("HITSZ_ROOT", CACHE / "rmua_rl")).resolve()
    verify_checkout(hitsz)
    os.environ["HITSZ_ROOT"] = str(hitsz)
    sys.path[:0] = [str(hitsz), str(hitsz / "RL" / "rsl_rl")]
    train(args, hitsz)


if __name__ == "__main__":
    main()
