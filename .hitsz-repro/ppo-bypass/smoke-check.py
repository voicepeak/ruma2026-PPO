from pathlib import Path
import torch
from evaluation.bypass_learning import BypassActorCritic, BypassLearningEnv, Rewards, OBS_SIZE, WAIT, decision_reward
from evaluation.rollout import COLLISION, SUCCESS, TIMEOUT, Progress
from hitsz_ppo import Course, DEVICE
from scripts.train_bypass_ppo import gate_graph

@torch.inference_mode()
def main():
    zeros = torch.zeros(3, device=DEVICE)
    terminal = torch.tensor([SUCCESS,COLLISION,TIMEOUT], device=DEVICE)
    reward = decision_reward(Rewards(), zeros, zeros, torch.full_like(zeros,.1),
                             torch.full_like(zeros,2), terminal)
    torch.testing.assert_close(reward, torch.tensor([5.99,-8.01,-4.01],device=DEVICE))

    graph = gate_graph()
    env = BypassLearningEnv(Course(4), graph, allow_wait=True)
    e = env.course.e
    offsets = e.num_voxels + e.num_bg_voxels + e.num_bottom_voxels

    def reset_clear():
        env.reset(1,50731)
        e.static_voxels[:, :offsets, :3] = 1e6
        e.static_voxels[:, :offsets, 3:] = 0
        e.active_num_cars = 0
        e._update_cars_idx()
        env.progress = Progress(e,graph)

    def advance(action, count):
        for _ in range(count):
            features, prior = env.observe()
            assert features.shape == (4,OBS_SIZE) and prior.shape == (4,6)
            env.step(torch.full_like(env.last_action, action))

    reset_clear()
    advance(1, 14)
    left = e.p.clone()
    reset_clear()
    advance(2, 14)
    difference = (e.p-left).norm(dim=-1)
    assert not env.progress.done.any() and (difference > .4).all(), difference
    reset_clear()
    advance(WAIT, 30)
    assert env.hold_locked.all() and not env.progress.done.any(), env.waiting_s
    assert (e.v.norm(dim=-1)<.3).all(), e.v
    at_rest=e.p.clone()
    advance(WAIT, 10)
    drift=(e.p-at_rest).norm(dim=-1).max().item()
    assert drift < .10, drift
    print('PPO_MECHANISM_OK', 'min_left_right_separation_m',float(difference.min()),
          'max_wait_drift_m_over_1s',drift)

if __name__ == '__main__':
    main()
