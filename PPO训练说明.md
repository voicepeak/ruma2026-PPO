# HITSZ 三维候选策略 · 持续 PPO 训练

服务器作业目录为 `/home/featurize/work/projects/rmua-branch-research/hitsz-ppo-01`。训练文件在该目录的 `training/` 下。桌面 runs/hitsz-ppo-01 下的文件是取回时的快照。

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
