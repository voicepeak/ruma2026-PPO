"""Train the native HITSZ PPO, with optional gates and vehicle occupancy."""

import argparse
from copy import deepcopy
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import random
import sys

from scripts.hitsz_repro import CACHE, ROOT, git, verify_checkout


DEFAULT_CONFIG = ROOT / "configs" / "hitsz_joint.yaml"


def train(args, hitsz):
    import numpy as np
    import torch
    from Env.utils import load_yaml, make_env
    from RL.rl_env_wrapper import RLQuadEnv
    from hitsz_ppo import atomic_json
    from training.hitsz_joint import GateConfig, GateEnv, HitszRunner, OccupancyConfig

    if not torch.cuda.is_available():
        raise ValueError("HITSZ training requires a working CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("This training entry runs one process on a single GPU")

    addition = load_yaml(args.config)
    env_cfg = load_yaml(hitsz / "RL" / "configs" / "config_env.yaml")
    train_cfg = load_yaml(hitsz / "RL" / "configs" / "config_rl.yaml")
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
    device = f"cuda:{env_cfg['runtime']['gpu_id']}"
    checkpoint = args.out / "latest.pt" if args.resume else args.resume_from
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
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
        env = RLQuadEnv(cfg, sim) if gate is None else GateEnv(cfg, sim, gate, occupancy)
        actual_cfg = deepcopy(train_cfg)
        actual_cfg["policy"]["base_model_path"] = str(hitsz / train_cfg["policy"]["base_model_path"])
        runner = HitszRunner(env, actual_cfg, args.out, device, recipe, occupancy)
        if args.grad_accum_steps is not None:
            if args.variant != "joint":
                raise ValueError("Gradient accumulation is only implemented for the joint variant")
            runner.alg.accum_steps = args.grad_accum_steps
        if checkpoint is not None:
            runner.restore(checkpoint, allow_optimization_change=args.resume_from is not None)
        if args.learning_rate is not None:
            runner.alg.learning_rate = args.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate
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
            atomic_json(args.out / "environment.json", {
                "repo_commit": git(ROOT, "rev-parse", "HEAD"),
                "repo_status": git(ROOT, "status", "--porcelain").splitlines(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
                "quadsim_cuda_sha256": sha256(Path(quadsim_cuda.__file__).read_bytes()).hexdigest(),
                "steps_per_update": steps_per_update,
                "resume_from": str(checkpoint.resolve()) if checkpoint else None,
                "initial_environment_steps": steps_done,
            })
        print(f"Training {args.variant}: {envs} environments, updates "
              f"{runner.current_learning_iteration + 1}–{updates}, seed {seed}", flush=True)
        runner.learn(remaining, init_at_random_ep_len=checkpoint is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("hitsz", "gate", "joint"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--envs", type=int, help="Override the upstream 4096 environments")
    parser.add_argument("--num-steps-per-env", type=int, help="Override rollout steps per update")
    parser.add_argument("--num-mini-batches", type=int, help="Override PPO mini-batches per epoch")
    parser.add_argument("--grad-accum-steps", type=int, help="Accumulate this many micro-batches per optimizer step")
    parser.add_argument("--lr-schedule", choices=("adaptive", "fixed", "stable"), help="Override the PPO learning-rate schedule")
    parser.add_argument("--learning-rate", type=float, help="Override the initial learning rate")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--updates", type=int, help="Total updates including any checkpoint progress")
    budget.add_argument("--target-env-steps", type=int, help="Total environment steps across all devices")
    parser.add_argument("--seed", type=int, help="Override the upstream training seed")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", help="Continue in the same output directory")
    resume.add_argument("--resume-from", type=Path, help="Continue from a checkpoint in a new directory")
    args = parser.parse_args()
    hitsz = Path(os.environ.get("HITSZ_ROOT", CACHE / "rmua_rl")).resolve()
    verify_checkout(hitsz)
    os.environ["HITSZ_ROOT"] = str(hitsz)
    sys.path[:0] = [str(hitsz), str(hitsz / "RL" / "rsl_rl")]
    train(args, hitsz)


if __name__ == "__main__":
    main()
