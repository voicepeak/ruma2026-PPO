"""Train the native HITSZ PPO, with optional gate conditioning."""

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
    from training.hitsz_joint import GateConfig, GateEnv, HitszRunner

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
    updates = args.updates if args.updates is not None else train_cfg["max_iterations"]
    envs, seed = env_cfg["env"]["batch_size"], train_cfg["seed"]
    if updates < 1 or train_cfg["save_interval"] < 1 or not 0 <= seed < 2**32:
        raise ValueError("Use positive update and save counts and a uint32 seed")
    minibatches = train_cfg["algorithm"]["num_mini_batches"]
    if envs < minibatches or envs % minibatches:
        raise ValueError(f"Environment count must be a positive multiple of {minibatches}")

    gate = GateConfig(**addition["gate"]) if args.variant != "hitsz" else None
    recipe = {
        "variant": args.variant,
        "hitsz_commit": git(hitsz, "rev-parse", "HEAD"),
        "environment": deepcopy(env_cfg),
        "train": deepcopy(train_cfg),
        "additions": addition,
    }
    device = f"cuda:{env_cfg['runtime']['gpu_id']}"
    args.out.mkdir(parents=True)
    with (args.out / "trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        sim = make_env(env_cfg, device=torch.device(device))
        cfg = {**train_cfg, "env": env_cfg["env"]}
        env = RLQuadEnv(cfg, sim) if gate is None else GateEnv(cfg, sim, gate)
        actual_cfg = deepcopy(train_cfg)
        actual_cfg["policy"]["base_model_path"] = str(hitsz / train_cfg["policy"]["base_model_path"])
        runner = HitszRunner(env, actual_cfg, args.out, device, recipe)
        import quadsim_cuda

        atomic_json(args.out / "config.json", recipe)
        atomic_json(args.out / "environment.json", {
            "repo_commit": git(ROOT, "rev-parse", "HEAD"),
            "repo_status": git(ROOT, "status", "--porcelain").splitlines(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "quadsim_cuda_sha256": sha256(Path(quadsim_cuda.__file__).read_bytes()).hexdigest(),
            "steps_per_update": envs * train_cfg["num_steps_per_env"],
        })
        print(f"Training {args.variant}: {envs} environments, {updates} updates, seed {seed}", flush=True)
        runner.learn(updates, init_at_random_ep_len=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("hitsz", "gate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--envs", type=int, help="Override the upstream 4096 environments")
    parser.add_argument("--updates", type=int, help="Total PPO updates")
    parser.add_argument("--seed", type=int, help="Override the upstream training seed")
    args = parser.parse_args()
    hitsz = Path(os.environ.get("HITSZ_ROOT", CACHE / "rmua_rl")).resolve()
    verify_checkout(hitsz)
    os.environ["HITSZ_ROOT"] = str(hitsz)
    sys.path[:0] = [str(hitsz), str(hitsz / "RL" / "rsl_rl")]
    train(args, hitsz)


if __name__ == "__main__":
    main()
