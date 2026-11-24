#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export HITSZ_ROOT="${HITSZ_ROOT:-/home/featurize/work/projects/rmua-branch-research/hitsz-bend-02/hitsz}"
python_bin="${RMUA_PYTHON:-/home/featurize/work/envs/py311_rmua/bin/python}"
mkdir -p training
exec "$python_bin" -u -m ruma_ppo.hitsz.legacy_policy --out training "$@"
