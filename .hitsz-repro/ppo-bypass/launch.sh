#!/usr/bin/env bash
set -euo pipefail
source "$HOME/work/projects/ruma2026-PPO/.hitsz-repro/activate.sh"
cd "$RMUA_REPO"
export PYTHONPATH="$RMUA_REPO:$PYTHONPATH"
root=.hitsz-repro/ppo-bypass
python -u "$root/smoke-check.py"
python -u -m scripts.train_bypass_ppo train --seed 20261120 --batch 8 --updates 1 --out "$root/smoke-five"
python -u -m scripts.train_bypass_ppo train --seed 20261120 --batch 8 --updates 1 --wait --out "$root/smoke-wait"
python - <<'CHECK_LOAD'
from pathlib import Path
import torch
from evaluation.bypass_learning import BypassActorCritic, OBS_SIZE
from hitsz_ppo import DEVICE
for suffix, actions in (('five',5),('wait',6)):
    checkpoint=torch.load(Path('.hitsz-repro/ppo-bypass')/f'smoke-{suffix}'/'latest.pt',
                          weights_only=True,map_location=DEVICE)
    assert checkpoint['update']==1 and checkpoint['decisions']>0
    model=BypassActorCritic(actions==6).to(DEVICE)
    model.load_state_dict(checkpoint['policy'])
    logits, values=model(torch.zeros(2,OBS_SIZE,device=DEVICE), torch.zeros(2,actions,device=DEVICE))
    assert logits.shape==(2,actions) and logits.isfinite().all() and values.isfinite().all()
print('PPO_RELOAD_OK')
CHECK_LOAD
printf 'PPO_SMOKES_OK\n'
sha256sum control/lidar_bypass.py evaluation/lidar_bypass.py evaluation/rollout.py evaluation/bypass_learning.py hitsz_ppo.py hitsz_branches.py scripts/train_bypass_ppo.py > "$root/provenance.txt"
for seed in 20261121 20261122 20261123; do
  for mode in five wait; do
    run="$root/$mode-$seed"
    options=()
    if [[ "$mode" == wait ]]; then options+=(--wait); fi
    printf 'PPO_TRIAL_START %s\n' "$run"
    python -u -m scripts.train_bypass_ppo train --seed "$seed" --batch 32 --updates 24 "${options[@]}" --out "$run"
    python -u -m scripts.train_bypass_ppo eval --checkpoint "$run/latest.pt" \
      --baseline .hitsz-repro/pwm-lidar-bypass-val192-01/paired.csv --out "$run/val192"
    printf 'PPO_TRIAL_DONE %s\n' "$run"
  done
done
printf 'PPO_QUEUE_DONE\n'
