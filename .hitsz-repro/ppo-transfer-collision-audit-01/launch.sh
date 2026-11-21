set -euo pipefail
source "$HOME/work/projects/ruma2026-PPO/.hitsz-repro/activate.sh"
cd "$RMUA_REPO"
export PYTHONPATH="$RMUA_REPO:$PYTHONPATH"
in=.hitsz-repro/ppo-transfer384-01
out=.hitsz-repro/ppo-transfer-collision-audit-01
pair() {
  python -u "$RMUA_RUNTIME/diagnose-transfer-collisions.py" "$in" "$out" "$1" > "$out/$1.log" 2>&1 &
  a=$!
  python -u "$RMUA_RUNTIME/diagnose-transfer-collisions.py" "$in" "$out" "$2" > "$out/$2.log" 2>&1 &
  b=$!
  wait "$a"
  wait "$b"
}
pair r1-early r1-late
pair r2-climb r3-descent
printf 'COLLISION_REPLAY_DONE\n'
