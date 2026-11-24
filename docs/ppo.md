# HITSZ 三维候选策略 · 持续 PPO 训练

## 原生连续速度 PPO 与车辆预测

新入口从固定版本的 HITSZ 读取原始网络、PPO 参数、六条训练路线、速度和障碍课程。默认每轮 4096 个环境各采样 20 步，每局最多 150 步，训练上限 15000 轮。三组实验沿用 HITSZ 的三轴连续速度输出和速度 PI。

| `--variant` | 训练内容 |
|---|---|
| `hitsz` | 原始 HITSZ PPO 配方 |
| `gate` | 增加下一门信息。每次按序过门 +5，连续过三门额外 +15 |
| `joint` | 使用 `gate` 的飞行奖励，增加 0.5 s 和 1.0 s 的车辆占位预测。预测特征进入 Actor |

新增奖励和预测参数在 [共用策略配置](configs/train_joint.yaml)。当前门奖励使用 6 × 3 m 研究孔径。训练仍按 HITSZ 的碰撞、越界和回合时限终止，穿越窗口外会清掉连续过门计数，飞机继续训练后续路段。车辆标签由当前场景按仿真车速推演，栅格采用当前航向坐标系下的车辆包围盒。Actor 在训练和评测都读取预测头输出。

先按下节准备 Python 3.11、CUDA 11.8 和固定 HITSZ 签出。环境检查后，用 4096 环境跑一个独立的短作业，按日志里的 `Steps per second` 和 `status.json` 里的峰值显存量核对机器余量。

```bash
python scripts/hitsz_repro.py check
python -m scripts.train_hitsz_joint --variant joint --updates 5 \
  --out runs/native-ppo-4090-probe
```

正式作业用不同的新目录，默认从上游基础检查点初始化。同一轮对照的 `hitsz`、`gate`、`joint` 使用相同的种子和环境步数预算。

```bash
python -m scripts.train_hitsz_joint --variant joint --seed 20261121 \
  --envs 4096 --updates 1338 --out runs/native-ppo-joint-20261121
```

检查点和 `status.json` 会记录累计更新轮数、环境步数和当前显存峰值。续训使用相同的目录和参数并加 `--resume`。训练器会恢复模型、优化器、学习率和课程进度，从新的回合继续采样。`--updates` 表示包含之前进度的总轮数。

以上命令复现 RTX 4090 归档的 1,338 次更新，共 109,608,960 个环境步。下面的 RTX 3060 命令用 768 个并行环境追加 400 次更新，将累计预算扩到 115,752,960 步。

公开仓库提供两个完整检查点 `joint-4090-u1338.pt` 和 `joint-3060-u1738.pt`。下载后用 `sha256sum -c SHA256SUMS` 核对文件，将所选路径传给 `--native-ppo` 评测或 `--resume-from` 续训。实验对照和指标记录在 GitHub Issue。 [Hugging Face 权重与校验文件](https://huggingface.co/Vociepeak/ruma2026-hitsz-joint-ppo) · [GitHub Issue #2](https://github.com/voicepeak/ruma2026-PPO/issues/2)

换机器时，用 `--resume-from` 指向已保存的检查点，并指定新输出目录和目标机器能容纳的 `--envs`。用 `--target-env-steps` 保持原总采样预算。训练器会按检查点记录的既有环境步数和新并行规模计算剩余更新轮数。

```bash
python -m scripts.train_hitsz_joint --variant joint --seed 20261121 \
  --envs 768 --target-env-steps 115752960 \
  --resume-from runs/native-ppo-joint-20261121/latest.pt \
  --out runs/native-ppo-joint-20261121-local
```

## 全样本续训与稳定自适应 LR

显存较小的机器不必降低每轮 PPO 的有效样本数：保持 1024 个环境，把每个环境 rollout 提到 80 步，使用 32 个 micro-batch，并用 `--grad-accum-steps 8` 累积梯度。这样每次更新仍是 81,920 个样本，优化器每 epoch 走 4 步、共 20 步，与 4090 的 4096×20 配置对齐；代价是每轮耗时约 31~40 秒。

```bash
python -m scripts.train_hitsz_joint --variant joint --seed 20261121 \
  --envs 1024 --num-steps-per-env 80 --num-mini-batches 32 \
  --grad-accum-steps 8 --lr-schedule stable --learning-rate 0.003 \
  --resume-from <checkpoint> --out runs/<new-run>
```

`--lr-schedule stable` 是按整轮 KL 的 EMA 调整学习率的保守自适应版本：死区 0.006~0.018，调整倍率 1.2，上限 0.006。它不是 4090 的逐 minibatch adaptive，而是为了降低小 micro-batch 的 KL 噪声；需要精确复刻 4090 时仍用 `--lr-schedule adaptive`。

本地候选 checkpoint（sha256 `30a169914591566dffc3ca163dd3350b816b5906010cf9c9b2596878e0997ec7`）在 `val192` 上对 HITSZ v12：success 63.0% vs 31.8%，collision 20.8% vs 27.1%，McNemar p=2.3e-16。`val192` 是验证集，最终对外结论仍需使用未参与调参的 held-out 协议。

用训练机本地复跑的开发场景做对照，先用 `--lidar-bypass` 记录本机规则结果，再评新权重。两个输出目录的 `scenes.json` 可用来核对场景哈希。做策略消融时，分别以 `hitsz`、`gate` 和 `joint` 训练，从同一预算取 checkpoint，逐个用 `--native-ppo` 在同一批场景评测。原冻结 `test384` 已用于 P0 的一次性评测。

```bash
python -m ruma_ppo.evaluation.harness --suite val192 --lidar-bypass \
  --v12 .hitsz-repro/v12.onnx --out runs/native-eval-rule
python -m ruma_ppo.evaluation.harness --suite val192 \
  --native-ppo runs/native-ppo-joint-20261121/latest.pt \
  --v12 .hitsz-repro/v12.onnx --out runs/native-eval-joint-20261121
```

## 新服务器复现

使用 Ubuntu 22.04、NVIDIA 驱动 ≥535、RTX 4090 和 CUDA Toolkit 11.8。需要能执行 `nvidia-smi`、`nvcc --version` 和 `g++ --version`。将本目录同步到训练机后，在仓库根目录运行以下命令。在装有 Conda 的机器上首次建立 Python 环境可运行 `conda create -n py311_rmua python=3.11 -y`。

```bash
conda activate py311_rmua
python scripts/hitsz_repro.py setup
python scripts/hitsz_repro.py smoke --out .hitsz-repro/smoke-01
```

`setup` 安装 PyTorch 2.6.0 + cu118，默认在 `.hitsz-repro/rmua_rl` 签出固定版本的 HITSZ，编译 `quadsim_cuda`，并从同一提交提取 v12。可用 `HITSZ_ROOT` 指定已有的固定版本签出，用 `CUDA_HOME` 指定 Toolkit 11.8。`python scripts/hitsz_repro.py check` 可单独复检 GPU、扩展导入和 v12 CPU 推理。冒烟检查执行 1 轮 PPO 并使用新权重与 v12 配对评测。 [Issue #1 §6–§7](https://github.com/voicepeak/ruma2026-PPO/issues/1)

冒烟通过后可在 SSH 会话中启动完整复现。输出目录每次需用新名称。

```bash
nohup python -u scripts/hitsz_repro.py baseline --out .hitsz-repro/baseline-01 \
  > .hitsz-repro/baseline-01.log 2>&1 </dev/null &
```

`environment.json` 记录环境、提交与权重来源。`bend/report.json` 对应历史候选实验的 44/96，`amplitude/report.json` 对应原幅与半幅的 50/96 和 59/96。`ppo/` 保存 3000 轮复现的训练日志、权重和验证结果，启动后可读 `ppo/status.json` 跟踪进度。`paired/paired.csv` 和 `paired/result.json` 保存旧口径的 dev96 与 fresh96 逐局对照。Issue #1 的 `test384` 属于后续评测协议。 [Issue #1 §6](https://github.com/voicepeak/ruma2026-PPO/issues/1)

## P0 仿真与评测

激活训练环境并设置 `HITSZ_ROOT` 后，在仓库根目录执行。门图基于 `path_point` 的 HITSZ 合成门，采用 NED 世界坐标和 6×3 m 研究判定孔径。相机提供有遮挡关系的合成 RGB-D，LiDAR 提供最多 20,000 个有序射线样本和有效返回标记。两者共享采样时刻，可配置帧延迟、测距噪声和丢点。PWM 以 HITSZ 电机顺序输入四个 0–1 值，每条命令推进 10 ms。 [Issue #1 §5](https://github.com/voicepeak/ruma2026-PPO/issues/1)

```bash
python -m maps.gate_graph "$HITSZ_ROOT/path_point/1-3_2_training.yaml" --out .hitsz-repro/gates-r1.json
python -m scripts.smoke_pwm --out .hitsz-repro/pwm-smoke-01
python -m ruma_ppo.evaluation.harness --suite val192 --verify-repeat \
  --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-val192-01
```

冒烟保存第一帧传感器数据和逐拍 PWM、位姿。脚本从保存的 PWM 回放，逐拍验证位姿与结果一致。评测器默认对照状态反馈 PWM 控制器与 v12，传入 `--ppo <权重>` 可换成旧 PPO。每局记录场景哈希、过门数、首次失败位置、用时和最小净空。净空是当前体素姿态下沿该控制步取三个点计算的到实体盒面距离，扣除机体余量。越界单列。配对报告使用精确 McNemar 检验及按难度分层的场景组 bootstrap。完成 `val192` 复跑后，最终候选使用未用过的留出协议（新种子或 Issue #1 的 `test384-MR`）评测；P0 阶段已经用过一次 `test384`。

### 雷达前视制动

`--lidar-brake` 用 10 Hz 的合成 LiDAR 量测前方 16 m 路线上的障碍，按反应时间和制动距离计算速度上限，再把速度和前视距离交给 100 Hz PWM 控制器。恢复速度前，要求连续两帧观测到原障碍点附近的射线已经穿过原占据位置。逐局 CSV 的 `brake_s` 记录速度上限生效的时长。这一版沿原门序和中心线行驶，等待堵塞解除。

```bash
python -m ruma_ppo.evaluation.harness --suite val192 --lidar-brake \
  --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-lidar-brake-val192-01
```

### 雷达局部绕行

`--lidar-bypass` 每帧比较中心线及左右各两档横移的路线，用点云估计每条路线能安全前进多远，再兼顾偏移幅度和切换代价。横移参考随前向距离逐步变化，偏移上限按当前三门场景设为 2.05 m。速度仍由选中路线上的第一处障碍约束，逐局 CSV 同时记录 `brake_s` 和 `detour_s`。

```bash
python -m ruma_ppo.evaluation.harness --suite val192 --lidar-bypass \
  --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-lidar-bypass-val192-01
```

### PPO 高层选路

每 0.1 秒选一次绕行轨迹。开启 `--wait` 时，第六个动作先把速度降到零，并在速度低于 0.3 m/s 后锁定停靠位置。网络读取三帧五候选净空、当前速度、横向偏差、门位姿、剩余时间和上一动作。策略从现有规则排序出发，学习修正各动作分数。限速与 PWM 跟踪沿用 P0 逻辑。训练种子与验证场景分开。每轮保存 `latest.pt` 和 `status.json`，重启时可用相同参数和 `--resume` 续训。

```bash
python -m scripts.train_bypass_ppo train --seed 20261121 --batch 32 --updates 24 \
  --out .hitsz-repro/ppo-bypass/five-20261121
python -m scripts.train_bypass_ppo eval \
  --checkpoint .hitsz-repro/ppo-bypass/five-20261121/latest.pt \
  --baseline .hitsz-repro/pwm-lidar-bypass-val192-01/paired.csv \
  --out .hitsz-repro/ppo-bypass/five-20261121/val192
```

### 冻结权重跨赛段评测

`scripts.eval_bypass_transfer` 的 `freeze` 会记录四个 HITSZ 门组、八组场景种子和三个 24 轮六动作检查点的哈希。每个门组跑两档难度，每档 48 局。`anchor` 可用原 `val192` 配对表复核原弯道的 48 局。随后用 `segment` 分别评测四段，最后用 `combine` 汇总逐段结果。所有策略共用 10 秒时限和同一批初始场景，起点接触单列。

### PWM 风场回归

`PWMEnv` 的可选 `WindField` 以 NED 坐标提供常值风、平滑阵风和有时间相关性的随机湍流。刚体积分沿用 HITSZ 形式，阻力按相对风速计算。`python -m scripts.eval_pwm_wind --out .hitsz-repro/pwm-wind-new` 同场景对照无风、6 m/s 持续横风和带阵风的横风，保留门框以隔离轨迹跟踪，并校验随机风和飞行结果可复现。

## 历史 featurize 作业

原服务器作业目录为 `/home/featurize/work/projects/rmua-branch-research/hitsz-ppo-01`。下文所用的 `training/` 是该作业的输出目录。[runs/hitsz-ppo-01](runs/hitsz-ppo-01) 保存取回时的快照。

## 当前策略

每次决策输入一帧 20×80 深度、HITSZ 状态向量、下一道合成门的相对位置、已过门数和上一个动作等。策略从 5 个横移目标和 2 个速度比例组成的 10 个动作中采样。每个动作保持 4 个控制步，名义约 0.4 秒，到下一次决策时可以改选。共享跟踪器使用仿真路线和机体状态。相对门位置和路线在实机上需要独立估计。

网络随机初始化。每局内容是按顺序过三道研究用虚拟门；穿门奖励 2 分，第三门额外奖励 5 分，碰撞、越界或漏门扣 5 分，另有沿程进度奖励和时间代价。最多 100 个底层控制步，超时仍未完成会扣分。记录的三门通过率按每次出发统计，包含失败局。

每轮有 256 个新场景；一半训练固定弯道，一半训练同一路线的随机门段，障碍等级 2、3 按轮交替。从这些新回合的动作、回报和旧动作概率计算 GAE 后做 PPO 更新。学习率 0.0003，裁剪比 0.2，GAE 系数 0.95，折扣 0.99，每批数据更新 4 遍。代码入口是 [hitsz_ppo.py](hitsz_ppo.py)。

## 如何看进度

在服务器上的作业目录执行：

```bash
cat training/status.json
tail -n 8 training/eval.jsonl
```

status.json 每轮更新，包含累计策略决策数、本轮难度、训练成功数和训练进程 PID。在初始化、更新 1 及之后每 25 轮，策略用确定性动作评测固定 96 局（两种障碍等级各 48 局）。baseline.json 是相同场景的居中规则结果，eval-latest.json 和 eval-best.json 保存最新及最优模型的逐场结果。固定验证集用于选择模型，要验证泛化需使用新的场景种子和路线。

latest.pt 至少每 5 轮原子替换一次，保存策略、价值网络、优化器与进度；best.pt 先按三门通过数、再按总过门数选出。作业脱离 SSH 会话运行，一直训练至收到停止请求、进程退出或云主机关闭。主机关机后的恢复需要重新启动作业。

## 停止与恢复

在服务器作业目录执行 `touch training/STOP`。训练会在轮次边界写入断点退出。恢复时确认旧进程已经退出，然后：

```bash
rm -f training/STOP
nohup bash run_ppo.sh >> trainer.log 2>&1 </dev/null &
```

同一目录使用文件锁防止多个训练器同时写入权重。

## 训练曲线

逐轮曲线保存于服务器 training/curves.jsonl 和 training/tensorboard/。按训练难度分别读取平均回报、三门通过率与结束原因。横向路线误差和速度只统计回合有效的控制步。PPO 同时记录样本加权的策略损失、价值损失、动作熵、近似 KL 和裁剪比例。验证指标使用固定 96 局，按轮关联到同一条训练时间轴。前 76 轮保留的历史验证记录已导入 TensorBoard，训练回报等详细曲线从第 77 轮开始。

在训练机作业目录可用 `/home/featurize/work/envs/py311_rmua/bin/tensorboard --logdir training/tensorboard --host 127.0.0.1 --port 6006` 打开曲线服务。桌面快照可以通过 `python plot_ppo_training.py runs/hitsz-ppo-01` 从已取回的记录重新绘制。
