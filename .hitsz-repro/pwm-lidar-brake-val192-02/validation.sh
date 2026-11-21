set -euo pipefail
source "$HOME/work/projects/ruma2026-PPO/.hitsz-repro/activate.sh"
cd "$RMUA_REPO"
python - <<'BRAKE_TEST'
import torch
from control.lidar_brake import first_path_return, lidar_confirms_clear, stopping_speed
path = torch.tensor([[[0.,0,0],[2,0,0],[2,2,0]]]).expand(4,-1,-1)
points = torch.tensor([[[1.,1,0],[2,1,0]], [[1,0.5,0],[2,1,0]], [[1,0,0],[2,1,0]], [[4.,4,0],[-9,0,0]]])
valid = torch.tensor([[True,True],[True,True],[False,False],[True,True]])
d,surface = first_path_return(points,valid,path,torch.tensor([0.,2,4]),0.72)
torch.testing.assert_close(d, torch.tensor([3.,1.,float('inf'),float('inf')]))
torch.testing.assert_close(surface[:2], torch.tensor([[2.,1,0],[1,.5,0]]))
cloud = torch.zeros(3, 20_000, 3)
valid = torch.zeros(3, 20_000, dtype=torch.bool)
cloud[2,13500] = torch.tensor([.8,0,0]); valid[2,13500] = True
target = torch.tensor([[1.,0,0],[1.,0,2],[1.,0,0]])
assert lidar_confirms_clear(cloud, valid, target, (20,1000), 50).tolist() == [True,False,False]
v = stopping_speed(torch.tensor([0.,1.,5.,10.]), 2.5, 0.25)
torch.testing.assert_close(v * 0.25 + v.square() / 5, torch.tensor([0.,1.,5.,10.]))
print('BRAKE_VISIBILITY_OK')
BRAKE_TEST
python -u -m evaluation.harness --suite val192 --lidar-brake --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-lidar-brake-val192-02
python -u - <<'BRAKE_OPEN'
import csv
from pathlib import Path
import torch
from evaluation.rollout import LidarBrakeFollower, Progress
from hitsz_branches import NEXT_GATE, ROUTE, UP
from hitsz_ppo import Course
from maps.gate_graph import GateGraph
with (Path('.hitsz-repro/pwm-clear-val192-01/cases.csv')).open() as stream:
    expected = [x['flight_s'] for x in csv.DictReader(stream) if (x['level'],x['seed'])==('2','61001')]
with torch.inference_mode():
    batch=48
    course=Course(batch)
    e=course.e
    course.start(2,61001,False)
    offset=e.num_voxels+e.num_bg_voxels+e.num_bottom_voxels
    e.static_voxels[:,:offset,:3]=1e6
    e.static_voxels[:,:offset,3:]=0
    e.active_num_cars=0
    e._update_cars_idx()
    graph=GateGraph.from_hitsz(UP/ROUTE).select(range(NEXT_GATE,NEXT_GATE+3))
    policy=LidarBrakeFollower(e)
    policy.reset(61001)
    progress=Progress(e,graph)
    limited=torch.zeros(batch,dtype=torch.bool,device=e.device)
    while not progress.done.all():
        prior=e.p.clone()
        pwm=policy.command()
        limited |= ~progress.done & (policy.speed_fraction < 1)
        progress.advance(prior,policy.step(pwm))
    rows=list(progress.rows())
    assert not limited.any()
    assert all(x['outcome']=='success' for x in rows)
    assert [r['flight_s'] for r in rows]==[float(x) for x in expected]
    print('BRAKE_OPEN_OK 48/48 identical finish times; no braking')
BRAKE_OPEN
printf 'BRAKE_FIX_OK\n'
