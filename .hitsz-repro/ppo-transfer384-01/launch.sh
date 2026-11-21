#!/usr/bin/env bash
set -euo pipefail
source "$HOME/work/projects/ruma2026-PPO/.hitsz-repro/activate.sh"
cd "$RMUA_REPO"
out=.hitsz-repro/ppo-transfer384-01
python -u -m scripts.eval_bypass_transfer anchor --baseline .hitsz-repro/pwm-lidar-bypass-val192-01/paired.csv
run_pair() {
  python -u -m scripts.eval_bypass_transfer segment --id "$1" --out "$out" > "$out/$1.log" 2>&1 &
  first=$!
  python -u -m scripts.eval_bypass_transfer segment --id "$2" --out "$out" > "$out/$2.log" 2>&1 &
  second=$!
  wait "$first"
  wait "$second"
}
run_pair r1-early r1-late
run_pair r2-climb r3-descent
python -u -m scripts.eval_bypass_transfer combine --out "$out"
