"""Set up and reproduce the archived HITSZ PPO baseline on a Linux GPU host."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".hitsz-repro"
UPSTREAM = "https://github.com/HITSZcjh/rmua_rl.git"
REVISION = "9ac9657334817ad0bac6db3520bca59e5f864bf3"
V12_BLOB = f"{REVISION}:deployments/rmua_ws/src/basic/src/v12.onnx"
V12 = CACHE / "v12.onnx"
PROFILE = {"smoke": (8, 1), "baseline": (256, 3000)}


def git(folder, *args):
    return subprocess.check_output(["git", "-C", folder, *args], text=True).strip()


def verify_checkout(hitsz):
    if not (hitsz / ".git").exists():
        raise ValueError(f"HITSZ checkout missing at {hitsz}. Run setup first.")
    if git(hitsz, "rev-parse", "HEAD") != REVISION or git(
        hitsz, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError(f"Use a clean HITSZ checkout at {REVISION}: {hitsz}")


def setup(hitsz):
    toolkit = os.environ.get("CUDA_HOME")
    nvcc = str(Path(toolkit) / "bin/nvcc") if toolkit else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise ValueError("CUDA Toolkit 11.8 is required. Set CUDA_HOME or put nvcc on PATH.")
    nvcc = Path(nvcc).resolve()
    if not re.search(r"release 11\.8,", subprocess.check_output([nvcc, "--version"], text=True)):
        raise ValueError(f"Expected CUDA Toolkit 11.8 at {nvcc}.")
    os.environ["CUDA_HOME"] = str(nvcc.parent.parent)

    if not hitsz.exists():
        hitsz.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, hitsz], check=True)
        subprocess.run(["git", "-C", hitsz, "checkout", "--detach", REVISION], check=True)
    verify_checkout(hitsz)

    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run([*pip, "--upgrade", "pip", "setuptools==81.0.0", "wheel", "ninja"], check=True)
    subprocess.run([
        *pip, "--index-url", "https://download.pytorch.org/whl/cu118",
        "torch==2.6.0+cu118", "torchvision==0.21.0+cu118", "torchaudio==2.6.0+cu118",
    ], check=True)
    subprocess.run([
        *pip, "torch==2.6.0+cu118", "torchvision==0.21.0+cu118",
        "numpy==1.26.4", "tensordict==0.7.2", "tensorboard", "tqdm", "pyyaml",
        "onnxruntime", "matplotlib", "-e", hitsz / "RL/rsl_rl",
    ], check=True)
    subprocess.run([*pip, "--no-build-isolation", "-e", hitsz / "Env/src"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as model:
        temp_path = Path(model.name)
        try:
            subprocess.run(["git", "-C", hitsz, "show", V12_BLOB], stdout=model, check=True)
            model.close()
            temp_path.replace(V12)
        finally:
            temp_path.unlink(missing_ok=True)


def check(hitsz):
    verify_checkout(hitsz)
    if not V12.is_file() or git(hitsz, "hash-object", "--no-filters", V12) != git(
        hitsz, "rev-parse", V12_BLOB
    ):
        raise ValueError("v12.onnx is missing or differs from the pinned Git blob. Run setup.")

    import numpy as np
    import onnxruntime as ort
    import torch

    if torch.__version__ != "2.6.0+cu118" or not torch.cuda.is_available():
        raise ValueError("PyTorch 2.6.0+cu118 with a working NVIDIA GPU is required.")
    torch.ones(1, device="cuda").cpu()

    # Match the imports performed by the original trainer and simulator.
    sys.path.insert(0, str(ROOT))
    from hitsz_ppo import ROUTE
    import quadsim_cuda

    if not (hitsz / ROUTE).is_file():
        raise ValueError(f"Missing baseline route at {hitsz / ROUTE}.")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(V12), opts, providers=["CPUExecutionProvider"])
    outputs = session.run(["act", "next_hx"], {
        "depth": np.zeros((1, 1, 20, 80), np.float32),
        "state": np.zeros((1, 14), np.float32),
        "hx": np.zeros((1, 192), np.float32),
    })
    if [x.shape for x in outputs] != [(1, 3), (1, 192)] or any(
        not np.isfinite(x).all() for x in outputs
    ):
        raise ValueError("Unexpected v12 output shapes or values.")

    return {
        "repo_commit": git(ROOT, "rev-parse", "HEAD"),
        "repo_status": git(ROOT, "status", "--porcelain").splitlines(),
        "hitsz_commit": REVISION,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "gpu": torch.cuda.get_device_name(),
        "quadsim_cuda_sha256": hashlib.sha256(Path(quadsim_cuda.__file__).read_bytes()).hexdigest(),
        "v12_sha256": hashlib.sha256(V12.read_bytes()).hexdigest(),
    }


def run_profile(profile, out, environment):
    batch, updates = PROFILE[profile]
    out.mkdir(parents=True, exist_ok=False)
    (out / "environment.json").write_text(json.dumps({
        **environment, "profile": profile, "batch": batch, "updates": updates, "eval_every": 25,
    }, indent=2) + "\n", encoding="utf-8")
    python = [sys.executable, "-u"]
    if profile == "baseline":
        subprocess.run([
            *python, ROOT / "hitsz_branches.py", "--out", out / "bend", "--batch", "48",
        ], cwd=ROOT, check=True)
        subprocess.run([
            *python, ROOT / "hitsz_amplitude.py", "--out", out / "amplitude",
        ], cwd=ROOT, check=True)
    subprocess.run([
        *python, ROOT / "hitsz_ppo.py", "--out", out / "ppo",
        "--batch", str(batch), "--updates", str(updates), "--eval-every", "25",
    ], cwd=ROOT, check=True)
    status = json.loads((out / "ppo/status.json").read_text(encoding="utf-8"))
    if status["update"] != updates:
        raise ValueError(f"PPO stopped at update {status['update']} of {updates}: {out / 'ppo'}")
    checkpoint = "best.pt" if profile == "baseline" else "latest.pt"
    paired = out / "paired"
    paired.mkdir()
    subprocess.run([
        *python, ROOT / "compare_hitsz_v12.py", V12, out / "ppo" / checkpoint,
    ], cwd=paired, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("setup", help="install dependencies, build CUDA extension, extract v12")
    phases.add_parser("check", help="verify the pinned simulator, CUDA, and v12 inference")
    for name, (_, updates) in PROFILE.items():
        phase = phases.add_parser(name, help=f"run {updates} PPO update(s) and compare with v12")
        phase.add_argument("--out", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 11):
        parser.error("Run on the Linux training server with Python 3.11.")

    hitsz = Path(os.environ.get("HITSZ_ROOT") or CACHE / "rmua_rl").resolve()
    os.environ["HITSZ_ROOT"] = str(hitsz)
    try:
        if args.phase == "setup":
            setup(hitsz)
        environment = check(hitsz)
        if args.phase in PROFILE:
            run_profile(args.phase, args.out.resolve(), environment)
        else:
            print(json.dumps(environment, indent=2))
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
