# 当前进度（2026-11-24 夜，4090 交接完成）

> 最新交接报告：[4090 交接：门前高度对准 A/B 与避障回放](../reports/2026-11-24-4090-handover/README.md)。本文件保留下面的 4090 准备步骤和原始配方；已过期的“交给 4090 接手”描述以最新报告为准。

## TL;DR

- 4090 两个作业已正常结束：`u2501` 续训到 `u3839`，以及带门前高度对准奖励的 200 轮 A/B。远端 GPU 已空闲，最终权重、状态和日志已取回本地并通过 SHA-256 校验，可以关闭实例。
- 原配方 `u3839` 相对 `u2501` 基本持平：384 局 237 vs 230 通过，`val192` 123 vs 121；漏门仍集中在门下沿。
- 下一轮优先候选是高度对准 `u2601`：384 局 243 通过（`u2501` 230、同轮对照 195），漏门 62→46；但飞行中碰撞 40→43，干净起点子集又少 7 次。需要新种子复核。
- 最大的未闭环问题：训练和评测的过门口径不同；车辆占位预测有风险信号，但当前融合反而给前向加速。下一步先做小改动验证，不要先加训练量。
- `val192` 是开发验证集；`test384` 已经用过一次。最终结论要用新种子或预注册留出协议。

## 分支与提交

- `feat/ppo-fullbatch-stable`：唯一工作分支，包含 2026-11-24 的代码、评测和交接提交；目录已整理为 `ruma_ppo/{env,training,evaluation,hitsz,cli}`。
- `main`：默认分支；尚未合并本分支。
- 旧临时分支和 `refactor/repo-layout` 已删除，远端只保留 `main` 和 `feat/ppo-fullbatch-stable`。
- 本机没有安装 pytest；交接时直接调用 `tests/test_metrics.py` 的两项检查，并跑了高度 shaping 的边界检查，均通过。新机器在开始训练前补跑 `python -m pytest -q`。

## 当前权重

| 权重 | 路径/来源 | sha256 | 状态 |
|---|---|---|---|
| u2501（基线） | HF `joint-3060-u2501.pt`（本地备份 `runs/native-joint-20261121-local-fullbatch-1024x80-ga8-stable/backups/checkpoint_u2501.pt`） | `30a169914591566dffc3ca163dd3350b816b5906010cf9c9b2596878e0997ec7` | 已发布；本轮所有续训的起点 |
| **高度对准 u2601（下一轮候选）** | 本地 `runs/native-joint-20261124-failure-audit/alignment-u2601.pt` | `151fdd00eb749932ad11718aa1f7897161d0690cb6b3c48f11a620118ab892e6` | 384 局 243/384；需新种子复核 |
| 高度对准 u2701 | 本地 `runs/handover-2026-11-24-4090/height-alignment/latest.pt` | `d051a09d87073692fa6346b301085930fde02850ea89119dd6cda32cab991af2` | 200 轮；过门收益回落，碰撞较低 |
| u3839（4090 原配方完训） | 本地 `runs/handover-2026-11-24-4090/original-4090/latest.pt`（同 `runs/native-joint-20261124-failure-audit/checkpoint_u3839.pt`） | `cf5d40e8b174e90a499087d91f6136cb1bf9415bc2ee50394bcda97925873815` | 相对 u2501 基本持平，不作突破 |
| u1338（4090 原始） | HF `joint-4090-u1338.pt` | `2575179ac7d5fb446742173435fff6ac65459d30ec78dee5f8693ea516fb631e` | 公开，可作为回退 |
| u2301（3060 seed） | 本地 `...-stable/seed_u2301.pt` | `7eeefb2877a3309eb1a68c4050c18ca1f124502b1c765993572426d1dc6ccb99` | 可选保留 |

`u2501` 记录：update 2501，累计环境步数 204,881,920；训练配方 1024 envs × 80 steps、32 micro-batch、8 步梯度累积、stable LR；课程 phase 1，speed 3/3，obstacle 3/4，success EMA 0.551。

下载：

```bash
curl -L --fail -o /data/checkpoints/joint-3060-u2501.pt \
  https://huggingface.co/Vociepeak/ruma2026-hitsz-joint-ppo/resolve/main/joint-3060-u2501.pt
sha256sum /data/checkpoints/joint-3060-u2501.pt
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

## 从 u2501 继续（2026-11-24 实际配方）

```bash
python -m scripts.train_joint --variant joint --seed 20261121 \
  --envs 4096 --num-steps-per-env 20 --num-mini-batches 4 \
  --grad-accum-steps 1 --lr-schedule adaptive --learning-rate 0.001 \
  --updates 3839 \
  --resume-from /data/checkpoints/joint-3060-u2501.pt \
  --out /data/runs/joint-4090-u2501-to-u3839

python -m scripts.train_joint --variant joint --seed 20261121 \
  --envs 4096 --num-steps-per-env 20 --num-mini-batches 4 \
  --grad-accum-steps 1 --lr-schedule adaptive --learning-rate 0.001 \
  --gate-height-alignment-weight 0.5 --updates 2701 \
  --resume-from /data/checkpoints/joint-3060-u2501.pt \
  --out /data/runs/height-alignment
```

说明：

- 4096×20 = 81,920 样本/轮，和 u2501 的 1024×80 有效 batch 相同；`num_mini_batches=4` 每次优化器步 20,480 样本，回到 4090 原始节奏。
- **实际作业使用原始 `adaptive`、起始 LR 0.001**，不是本文早先建议的 0.0038。两条作业的 `config.json` 都记录 0.001；复现时不要改。
- `--resume-from` 会放宽 batch/steps/minibatches/schedule/LR 的 recipe 校验。高度对准只允许 `gate`/`joint`，并且只能在 `--resume-from` 时改变该权重。
- 如果坚持沿用 u2501 的 stable schedule，把 `--lr-schedule adaptive --learning-rate 0.001` 换成 `--lr-schedule stable --learning-rate 0.003`；但 stable 在 0.006 上限下曾让熵/碰撞回升，4090 上不如原 adaptive。本轮没有跑 stable 对照。
- 从 u2501 续训会先把所有环境 reset，前 20~40 轮 success 会掉一截，属正常瞬态。
- 训练器每 100 轮保存 `latest.pt`/`status.json`；4090 上建议用 tmux/nohup，自己加定期备份。本地的 `supervise.sh`/`battery_guard.sh` 不在 Git 里，4090 不需要电池 guard。

## 评测

```bash
python -m ruma_ppo.evaluation.harness --suite val192 \
  --native-ppo /data/checkpoints/joint-3060-u2501.pt \
  --v12 .hitsz-repro/v12.onnx \
  --out /data/reports/val192-u2501
```

已提交的结果在 `reports/2026-11-24-u2501-val192/`：

- success **121/192 = 63.0%**，collision **40/192 = 20.8%**
- HITSZ v12：success 61/192 = 31.8%，collision 52/192 = 27.1%
- paired：candidate_only 62、v12_only 2，McNemar p=2.3e-16，gain +31.25pp

2026-11-24 夜新增的 384 局独立场景（种子 `62021–62024`，障碍 2–3 档，52 局起点接触单列）：

| 权重 | 通过 | 漏门 | 飞行中碰撞 |
|---|---:|---:|---:|
| `u2501` | 230 | 62 | 40 |
| 原配方 `u2601` | 195 | 95 | 42 |
| **高度对准 `u2601`** | **243** | **46** | **43** |
| 原配方 `u2701` | 125 | 166 | 41 |
| 高度对准 `u2701` | 228 | 70 | 34 |
| `u3839` | 237 | 57 | 38 |

逐局 CSV、配对统计和 282 局干净起点压力测试都在 `reports/2026-11-24-4090-handover/`。结论和注意事项以该目录的 `README.md` 为准。

`val192` 是验证集；`test384` 保持 held-out，不要用于调参。`hard96` 是 obstacle 4，当前还没到，等课程解锁后再评。

## 2026-11-24 夜下一步

1. 用新种子复核高度对准 `u2601`，同时对照 `u2501`、原配方 `u2601` 和 `u3839`；预注册门平面下沿余量、飞行中碰撞和用时的判定。
2. 统一训练和三门评测的过门口径：共用门平面交点、门洞和门序判定；课程升档看按序过门率。整段任务保留回门重试，成功触发当前门才推进正式计分索引。
3. 从 `u2601` 验证车辆预测如何影响动作：限制高风险栅格内的正向加速，或显式给最小净空/最早冲突时间；不要先扩大预测网络。
4. 用官方 ROS 记录标定观测误差、延迟和丢帧；补静态细线障碍；最后到官方模拟器核对计分、时钟和控制接口。

## 历史：u2501 课程状态（2026-11-24 早）

- obstacle 3/4，phase 1；success EMA 0.551；`good_success_epochs=0`。
- 解锁 obstacle 4 需要 success EMA > 0.7 连续 50 个 epoch；4090 上每轮样本恢复后可能更快。
- 主要瓶颈是过门精度（pass/miss/three），不是碰撞；碰撞率已横在 0.0043 左右。
- `training/hitsz_joint.py` 仍集中 gates/occupancy/models/PPO/runner；`docs/architecture.md` 写了后续拆分计划，不影响本次训练。

## 注意

- 本地 3060 训练已停止，u2501 已保存；u2501 之后的 2502–2559 未存盘，已丢失。
- u2501 已发布到 HuggingFace（`joint-3060-u2501.pt`），`SHA256SUMS` 和 `HANDOVER-u2501.md` 已更新；4090 可直接 curl 下载。
- 4090 交接包在本地 `runs/handover-2026-11-24-4090/`（`runs/` 不进 Git），包含 u3839 与高度对准检查点、日志、TensorBoard event、配置、补丁和 `SHA256SUMS`。远端实例可以关闭。
- 旧 768 run（u2101）、fixed 1024 run（u2301）、原 4090 u1338 和公开权重都保留，可按需回退。
- 新服务器首次训练前先跑 `python -m scripts.hitsz_repro check` 和 2 轮 smoke。
