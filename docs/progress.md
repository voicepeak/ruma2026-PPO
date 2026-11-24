# 当前进度（2026-11-24，交给 4090 接手）

## TL;DR

- 代码收敛在单分支 `feat/ppo-fullbatch-stable`（已包含功能、重构、评测和进度提交），已推送到 GitHub。
- 当前最好权重是 3060 全样本续训的 `u2501`，不是公开的 `u1338`。
- 在 4090 上从 `u2501` 续训，恢复 4096×20 的原 4090 吞吐；推荐原始 `adaptive` LR，起始 0.0038。
- 先搬 `checkpoint_u2501.pt` 到 4090，跑 smoke，再启动长训练。

## 分支与提交

- `feat/ppo-fullbatch-stable`：唯一工作分支，包含功能、重构、评测和进度提交；目录已整理为 `ruma_ppo/{env,training,evaluation,hitsz,cli}`，测试 2 passed。
- `main`：默认分支；尚未合并本分支。
- 旧临时分支和 `refactor/repo-layout` 已删除，远端只保留 `main` 和 `feat/ppo-fullbatch-stable`。

## 当前权重

| 权重 | 路径/来源 | sha256 | 状态 |
|---|---|---|---|
| u2501（最好） | 本地 `runs/native-joint-20261121-local-fullbatch-1024x80-ga8-stable/backups/checkpoint_u2501.pt` | `30a169914591566dffc3ca163dd3350b816b5906010cf9c9b2596878e0997ec7` | 需要 scp/上传，尚未进 HF |
| u1338（4090 原始） | HF `joint-4090-u1338.pt` | `2575179ac7d5fb446742173435fff6ac65459d30ec78dee5f8693ea516fb631e` | 公开，可作为回退 |
| u2301（3060 seed） | 本地 `...-stable/seed_u2301.pt` | `7eeefb2877a3309eb1a68c4050c18ca1f124502b1c765993572426d1dc6ccb99` | 可选保留 |

`u2501` 记录：update 2501，累计环境步数 204,881,920；训练配方 1024 envs × 80 steps、32 micro-batch、8 步梯度累积、stable LR；课程 phase 1，speed 3/3，obstacle 3/4，success EMA 0.551。

传输：

```bash
scp runs/native-joint-20261121-local-fullbatch-1024x80-ga8-stable/backups/checkpoint_u2501.pt \
  <user>@<4090-host>:/data/checkpoints/
sha256sum /data/checkpoints/checkpoint_u2501.pt
# 期望 30a169914591566dffc3ca163dd3350b816b5906010cf9c9b2596878e0997ec7
```

## 4090 上准备

```bash
git clone git@github.com:voicepeak/ruma2026-PPO.git
cd ruma2026-PPO
git checkout feat/ppo-fullbatch-stable

conda create -n py311_rmua python=3.11 -y
conda activate py311_rmua
export HITSZ_ROOT=/data/hitsz/rmua_rl-native   # 固定 9ac9657，可让 setup 自己 clone
python -m scripts.hitsz_repro setup
python -m scripts.hitsz_repro check
```

`setup` 会装 PyTorch 2.6.0+cu118、编译 quadsim，并从固定提交导出 `.hitsz-repro/v12.onnx`。HITSZ 版本固定为 `9ac9657334817ad0bac6db3520bca59e5f864bf3`，v12 sha256 `8e997b5215152c8f63797ba1319b8ad8abd7d7bd5e4e57b9e667d71e9604a86d`。

## 先 smoke

```bash
python -m scripts.train_joint --variant joint --seed 20261121 \
  --envs 4096 --updates 2 --out /data/runs/smoke-4090-u2501
```

只跑 2 轮，确认 4096×20 的显存和速度。原始 4090 配方峰值约 22.8GB；如果 OOM，退到 `--envs 2048 --num-steps-per-env 40 --num-mini-batches 4`（每次仍是 81,920 样本）。

## 从 u2501 继续（推荐）

```bash
python -m scripts.train_joint --variant joint --seed 20261121 \
  --envs 4096 --num-steps-per-env 20 --num-mini-batches 4 \
  --grad-accum-steps 1 --lr-schedule adaptive --learning-rate 0.0038 \
  --updates 100000000 \
  --resume-from /data/checkpoints/checkpoint_u2501.pt \
  --out /data/runs/native-joint-20261124-4090-u2501
```

说明：

- 4096×20 = 81,920 样本/轮，和 u2501 的 1024×80 有效 batch 相同；`num_mini_batches=4` 每次优化器步 20,480 样本，回到 4090 原始节奏。
- 推荐用原始 `adaptive`（4090 的 minibatch 大，KL 稳），起始 LR 0.0038；`--resume-from` 会放宽 batch/steps/minibatches/schedule/LR 的 recipe 校验。
- 如果坚持沿用 u2501 的 stable schedule，把 `--lr-schedule adaptive --learning-rate 0.0038` 换成 `--lr-schedule stable --learning-rate 0.003`；但 stable 在 0.006 上限下曾让熵/碰撞回升，4090 上不如原 adaptive。
- 从 u2501 续训会先把所有环境 reset，前 20~40 轮 success 会掉一截，属正常瞬态。
- 训练器每 100 轮保存 `latest.pt`/`status.json`；4090 上建议用 tmux/nohup，自己加定期备份。本地的 `supervise.sh`/`battery_guard.sh` 不在 Git 里，4090 不需要电池 guard。

## 评测

```bash
python -m ruma_ppo.evaluation.harness --suite val192 \
  --native-ppo /data/checkpoints/checkpoint_u2501.pt \
  --v12 .hitsz-repro/v12.onnx \
  --out /data/reports/val192-u2501
```

已提交的结果在 `reports/2026-11-24-u2501-val192/`：

- success **121/192 = 63.0%**，collision **40/192 = 20.8%**
- HITSZ v12：success 61/192 = 31.8%，collision 52/192 = 27.1%
- paired：candidate_only 62、v12_only 2，McNemar p=2.3e-16，gain +31.25pp

`val192` 是验证集；`test384` 保持 held-out，不要用于调参。`hard96` 是 obstacle 4，当前还没到，等课程解锁后再评。

## 当前课程与下一步

- obstacle 3/4，phase 1；success EMA 0.551；`good_success_epochs=0`。
- 解锁 obstacle 4 需要 success EMA > 0.7 连续 50 个 epoch；4090 上每轮样本恢复后可能更快。
- 主要瓶颈是过门精度（pass/miss/three），不是碰撞；碰撞率已横在 0.0043 左右。
- `training/hitsz_joint.py` 仍集中 gates/occupancy/models/PPO/runner；`docs/architecture.md` 写了后续拆分计划，不影响本次训练。

## 注意

- 本地 3060 训练已停止，u2501 已保存；u2501 之后的 2502–2559 未存盘，已丢失。
- u2501 没有上传 HuggingFace；请先 scp 或用自己的 token 上传。
- 旧 768 run（u2101）、fixed 1024 run（u2301）、原 4090 u1338 和公开权重都保留，可按需回退。
- 新服务器首次训练前先跑 `python -m scripts.hitsz_repro check` 和 2 轮 smoke。
