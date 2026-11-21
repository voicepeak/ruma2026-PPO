set -euo pipefail
source "$HOME/work/projects/ruma2026-PPO/.hitsz-repro/activate.sh"
cd "$RMUA_REPO"
export PYTHONPATH="$RMUA_REPO:$PYTHONPATH"
out=.hitsz-repro/pwm-entry-speed-ab96-01
python -u "$out/bench.py" r1-late > "$out/r1-late.log" 2>&1 &
x=$!
python -u "$out/bench.py" r2-climb > "$out/r2-climb.log" 2>&1 &
y=$!
wait "$x"
wait "$y"
python -u "$out/bench.py" summary
