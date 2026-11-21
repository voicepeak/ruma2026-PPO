# 交接文档：RMUA 2026 P0/P1 现状与下一步

> 交接时间：2026-11-21 夜（本地）。上一手会话因额度中断，本文档记录当前代码、结果、已知坑和下一步。
> 新接手请先完整读 [Issue #1](https://github.com/voicepeak/ruma2026-PPO/issues/1) 的 §0、§3、§5、§6、§7、§11，再读本文件和 [PPO训练说明.md](PPO训练说明.md)。

## 1. 速览

**已完成**

- P0-1 PWM 接口：HITSZ 电机顺序、四路 0–1 指令、100 Hz 闭环。
- P0-2 传感器渲染：4 路合成 RGB-D + 最多 20,000 点 LiDAR，帧对齐、延迟/噪声/丢点可配置。
- P0-3 规范化门图：从 HITSZ `path_point` 提取门 ID、位姿、法向、孔径、顺序和稳定哈希。
- P0-4 评测 harness：`val192` / `test384` / `hard96`，逐局配对、McNemar、分层 bootstrap、失败分类、净空与超时。
- P0-5 100 Hz 几何控制器：位置/速度外环 + SO(3) 内环 + 混控 + PWM 限幅；悬停、阶跃、制动、首门回放通过。
- P0-7 风场：常值风 + 阵风 + 有色湍流，可复现；48 局回归全部通过。
- P1 探索：雷达前视制动、五路局部绕行、PPO 高层选路、跨赛段冻结评测、碰撞审计、入口速度消融。

**未完成 / 当前瓶颈**

- **P0-6 细线障碍未实现**：没有 `sim/obstacles.py`，没有细圆柱/线段碰撞体与渲染。Issue #1 的 P0-6 仍是缺口。
- **飞行中撞击的根本原因未闭环**：最强规则在 `val192` 仍有 53 次飞行中撞击；跨赛段 384 局里 Rule 与 3 个 PPO 种子**共同失败 109 局**，说明瓶颈在两者共用的候选/避障层，而不是 PPO 选路。
- 候选轨迹可行性诊断尚未做（下一步第一件事）。

**当前最好成绩（Rule = 雷达制动 + 局部绕行）**

| 集合 | 策略 | 三门通过 | 飞行中撞击 | 超时 | 起点接触（单列） |
|---|---|---:|---:|---:|---:|
| `val192` | 原几何 PWM | 66 | 90 | 0 | 36 |
| `val192` | 雷达制动 | 46 | 65 | 45 | 36 |
| `val192` | **雷达制动 + 绕行** | **97** | **53** | **6** | 36 |
| `test384`（只评一次） | 原几何 PWM | 128 | 197 | 0 | 59 |
| `test384`（只评一次） | v12 | 129 | 31 | 0 | 59 |

> `test384` 是冻结测试集，交接后**不要**用它调参；新开发用 `val192` 或新种子。

## 2. 仓库、远端与运行环境

- 仓库：<https://github.com/voicepeak/ruma2026-PPO>，分支 `main`。
- 交接前的历史提交：`b9ba996`（`origin/main`）。本次交接提交之后，以 `git log` 为准。
- 所有复现产物（权重、CSV、报告、日志、`v12.onnx`）已放在 `.hitsz-repro/` 并纳入版本控制，约 20 MB。**新实例不需要旧服务器即可继续**。
- 旧服务器：Featurize，`featurize@workspace.featurize.cn`，交接时端口 `14409`，密钥 `~/.ssh/id_ed25519_featurize`（密钥不在仓库内）。实例可能已被关闭；如果还能登录，路径为 `~/work/projects/ruma2026-PPO`。
- 本地 Windows 只能读代码/数据，不能跑 HITSZ 仿真。训练/评测必须在 Linux + NVIDIA GPU 上。

### 2.1 新实例配置

按 Issue #1 §7 和 [PPO训练说明.md](PPO训练说明.md) 执行：

```bash
conda create -n py311_rmua python=3.11 -y
conda activate py311_rmua
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install tensorboard tqdm pyyaml numpy==1.26.4 onnxruntime matplotlib

git clone https://github.com/voicepeak/ruma2026-PPO.git
cd ruma2026-PPO
python scripts/hitsz_repro.py setup
python scripts/hitsz_repro.py check
```

`setup` 会签出固定 HITSZ 提交 `9ac9657334817ad0bac6db3520bca59e5f864bf3`、编译 `quadsim_cuda`、提取 v12。已有固定签出时用 `HITSZ_ROOT`；已有 CUDA 11.8 时用 `CUDA_HOME`。

环境自检：

```bash
python -c "import torch, quadsim_cuda; print(torch.cuda.is_available())"
sha256sum .hitsz-repro/v12.onnx
# 期望 8e997b5215152c8f63797ba1319b8ad8abd7d7bd5e4e57b9e667d71e9604a86d
```

## 3. 代码地图

新增模块（相对仓库根目录）：

| 文件 | 作用 |
|---|---|
| `maps/gate_graph.py` | 从 HITSZ `path_point` 提取门图；门选择、NED 位姿、孔径、路线、稳定哈希 |
| `sim/pwm_env.py` | HITSZ 动力学的 PWM 封装；10 ms 控制步；可选风场 |
| `sim/render.py` | 4 路相机 + LiDAR 合成渲染，角度/延迟/噪声/丢点可配置 |
| `sim/wind.py` | 常值风、平滑阵风、有时间相关性的湍流 |
| `control/geometric.py` | 串级几何控制器：位置/速度 PD → 姿态 → 力矩 → 混控 → PWM |
| `control/lidar_brake.py` | 前视 16 m 制动距离规则与“确认清空”释放逻辑 |
| `control/lidar_bypass.py` | 中心线 + 左右各两档横移的候选评分 |
| `evaluation/rollout.py` | 门序、净空、越界、超时判定；PWM 跟随器与 LiDAR 制动跟随器 |
| `evaluation/lidar_bypass.py` | 五路候选的点云前向净空估计与横移参考 |
| `evaluation/harness.py` | 冻结场景配对评测；与 v12 同场对比；CSV/JSON 输出 |
| `evaluation/bypass_learning.py` | 10 Hz PPO 适配层、奖励、观测、等待/停靠逻辑 |
| `scripts/smoke_pwm.py` | PWM 悬停/阶跃/制动/首门冒烟，保存首帧传感器与逐拍 PWM 回放 |
| `scripts/train_bypass_ppo.py` | PPO 高层选路训练与 `val192` 规则对照 |
| `scripts/eval_bypass_transfer.py` | 冻结权重跨赛段评测（freeze/anchor/segment/combine） |
| `scripts/eval_pwm_wind.py` | 风场回归（无风 / 持续横风 / 阵风+湍流） |
| `scripts/hitsz_repro.py` | 新机器环境自举、旧基线复现入口 |

根目录修改：

- `hitsz_branches.py`：`reset(..., start=...)`、`station(..., position=...)`，用于自定义赛段起点。
- `hitsz_ppo.py`：`Course(..., route, start, first_gate)`；PPO 超参改为显式参数，默认值与旧行为一致。
- `.gitignore`：不再忽略 `.hitsz-repro/`，复现产物纳入仓库。

## 4. 关键结果

### 4.1 环境与旧权重快对照（fresh96）

| 策略 | 三门通过 | 撞击 | 漏门 |
|---|---:|---:|---:|
| `best.pt`（2750 轮） | 45/96 | 51 | 0 |
| `latest.pt`（3313 轮） | 42/96 | 54 | 0 |
| v12 | 33/96 | 11 | 52 |

数据：[paired.csv](.hitsz-repro/p0-results/archived-fresh96/paired.csv)、[result.json](.hitsz-repro/p0-results/archived-fresh96/result.json)。量级与 Issue #1 一致。

本轮按用户决定**没有完整复现** 3000 轮旧基线和 44/50/59 弯道实验；只做了 1 轮环境健康检查和上面这次 fresh96 快对照。需要时运行 `python scripts/hitsz_repro.py baseline`。

### 4.2 P0 PWM 几何控制器

- 冒烟：悬停漂移 0 m；单电机推力阶跃 2.36 N；1 s 制动后 0.31 m/s；逐拍 PWM 回放位姿完全一致；首门 2.23 s。
- `val192`：三门 66/192，撞击 126（36 起点接触 + 90 飞行中）；v12 三门 64/192，撞击 50（36 + 14），漏门 78。
- `test384`（冻结，仅此一次）：三门 128/384（撞击 256，59 起点接触 + 197 飞行中），v12 129/384（撞击 90，59 + 31，漏门 165）。通过率持平，v12 的差距主要在“保命”。**这一次评测已经用掉了 test384 的“一次性”额度；后续最终策略应预注册新的留出集。**
- 净空对照：移除障碍和车辆、保留门框后 `val192` 192/192 过三门，用时 5.37–5.95 s，最大水平偏离 0.36 m。**底层跟踪没问题，问题在上层避障**。

报告：[pwm-smoke-02](.hitsz-repro/p0-results/pwm-smoke-02/report.json)、[val192](.hitsz-repro/p0-results/pwm-val192-01/result.json)、[test384](.hitsz-repro/p0-results/pwm-test384-01/result.json)、[净空对照](.hitsz-repro/pwm-clear-val192-01/report.json)。

### 4.3 撞击审计（原几何控制器）

`val192` 的 36 次起点接触单列后，90 次飞行中撞击：

| 对象 | 次数 | 关键观察 |
|---|---:|---|
| 固定障碍 | 53 | 撞击时间中位数 0.54 s；撞击速度中位数 5.6 m/s |
| 移动车辆 | 37 | 撞击时车辆中位速度 7.8 m/s、飞机 4.9 m/s；有 2 局飞机几乎停住仍被车撞 |

- 62/90 局：即使把机体放到撞击点对应的路线中心，净空仍低于机体余量，说明**中心线本身已被堵死**。
- 51/90 发生在过完第二门、到第三门之前。
- 偏离路线中位数仅 0.10 m。

数据：[collisions.csv](.hitsz-repro/pwm-collision-audit-01/collisions.csv)、[代表轨迹与报告](.hitsz-repro/pwm-collision-audit-01.tar.gz)。

### 4.4 雷达制动与局部绕行

| 方案 | 三门通过 | 飞行中撞击 | 超时 |
|---|---:|---:|---:|
| 原几何 | 66 | 90 | 0 |
| 前视制动 | 46 | 65 | 45 |
| **制动 + 五路绕行** | **97** | **53** | **6** |
| 制动 + 绕行，横移限速 3.6 m/s | 91 | 58 | 7 |

- 单纯制动把 45 局变成超时；加绕行后，45 个超时里 30 个改为通过。
- 绕行使配对增益 +17.2 pp（McNemar p≈1.1e-4，95% CI 10.4–24.0 pp）。
- 横移限速降到 3.6 m/s 反而更差，保留第一版。
- 48 局净空场景全部通过，用时与无绕行时逐局一致。
- **规则绕行还没有在冻结 `test384` 上评测**。交接后不要随手补跑，等候选改进定型、预注册新协议后再评一次。

报告：[制动](.hitsz-repro/pwm-lidar-brake-val192-01/result.json)、[绕行](.hitsz-repro/pwm-lidar-bypass-val192-01/result.json)、[横移限速对照](.hitsz-repro/pwm-lidar-bypass-val192-02/result.json)。

### 4.5 PPO 高层选路

- 10 Hz 选一次；动作 = 5 条横移候选（0, ±1.1, ±2.05 m），`--wait` 时加第 6 个“等待/停靠”。
- 观测：3 帧 × 5 候选净空、速度、横向偏差、门位姿、剩余时间、上一动作、等待时长。
- 奖励：进度 0.08/m、过门 1、成功 6、时间 -0.1/s、near-miss 小惩罚、撞击/越界 -8、漏门 -5、超时 -4。
- 24 轮检查点，每轮 32 局；三个种子 × 五动作/六动作两套，共 6 个作业。每作业约 3 万决策。
- 48 轮续训（`ppo-bypass-cont48-01`）只带来小幅提升，仍未改变结论。

`val192`（**样本内，只作参考**；Rule 97/53/6）：

| 检查点 | 三门通过 | 飞行中撞击 | 超时 |
|---|---:|---:|---:|
| five-20261121 | 102 | 48 | 6 |
| five-20261122 | 97 | 54 | 5 |
| five-20261123 | 93 | 55 | 8 |
| wait-20261121 | 101 | 48 | 7 |
| wait-20261122 | 101 | 49 | 6 |
| wait-20261123 | 100 | 49 | 7 |
| cont48 wait-20261121 | 101 | 48 | 7 |
| cont48 wait-20261122 | 98 | 52 | 6 |
| cont48 wait-20261123 | 99 | 51 | 6 |

跨赛段 384（4 个新门组 × 96 局，冻结权重；这是真正的外推结果）：

| 策略 | 三门通过 | 飞行中撞击 | 超时 | 主动等待 |
|---|---:|---:|---:|---:|
| 规则绕行 | 201 | 116 | 15 | 0 |
| PPO 20261121 | 200 | 118 | 14 | 0 |
| PPO 20261122 | 204 | 113 | 15 | 0 |
| PPO 20261123 | 202 | 116 | 14 | 0 |

- PPO 平均只多过 1 局；三个权重都**没有选择等待**。
- 分赛段：PPO 在 `r2-climb` 上平均多过约 2 局，`r3-descent` 基本持平，其余两段略降。
- 训练信号很弱：`policy_loss` 约 ±1e-4、`approx_KL` 约 1e-4、`clip_fraction` 常为 0、熵仍在 1.0 左右。**不要把这些结果解读为“PPO 学到了避障”**；它主要是在规则的候选排序上做小幅修正。

报告：[val192 与训练队列](.hitsz-repro/ppo-bypass/queue.log)、[跨赛段](.hitsz-repro/ppo-transfer384-01/result.json)、[冻结协议](.hitsz-repro/ppo-transfer384-01/protocol.json)。

### 4.6 共同失败审计

跨赛段里 Rule 与 PPO 共同失败 **109 局**：

- 55 局撞移动车辆，54 局撞固定障碍；全是飞行中撞击。
- 固定障碍：撞击时间中位数 0.535 s，撞击速度中位数 5.59 m/s；36 局中心线已被堵死；54 局里 49 局在撞前半秒已触发限速（仍有 5 局全程没有限速）。
- 移动车辆：撞击时车辆中位速度 7.76 m/s、飞机 4.92 m/s；38 局中心线已被堵死；55 局里 13 局在最后半秒没有限速。
- 用 Rule 与 PPO-20261122 复跑同一批 109 局：**102 局撞同一物体、同一 10 ms 控制拍**。

结论：两者共用同一套候选与避障层，PPO 只能改“选哪条”，改不了“有没有可行的路”。

数据：[summarize.py](.hitsz-repro/ppo-transfer-collision-audit-01/summarize.py)、[summary.json](.hitsz-repro/ppo-transfer-collision-audit-01/summary.json)、[逐局 CSV](.hitsz-repro/ppo-transfer-collision-audit-01/)。

### 4.7 入口速度消融

在跨赛段的 `r1-late` + `r2-climb` 共 96 局上，把起飞速度直接压到 4 m/s：

| 方案 | 三门通过 | 撞击 | 超时 | 起飞后 1 s 内撞击 |
|---|---:|---:|---:|---:|
| 原始 | 59 | 17 | 5 | 6 |
| 入口限速 4 m/s | 55 | 20 | 6 | 3 |

这 96 局中有 15 局起点接触已单列，表中撞击为有效出发后的飞行中撞击。入口限速把“第一秒撞击”减半，但把移动车辆撞击从 9 增到 16，净收益为负。**不要简单限速**；应结合提前横移和车辆预测。

数据：[result.json](.hitsz-repro/pwm-entry-speed-ab96-01/result.json)、[bench.py](.hitsz-repro/pwm-entry-speed-ab96-01/bench.py)。

### 4.8 风场回归

48 局（保留门框、移除随机障碍，隔离轨迹跟踪）在三种风况下：

| 风况 | 通过 | 平均用时 | 横向峰值 p50/p95/max | 最小门框净空 |
|---|---:|---:|---|---:|
| 无风 | 48/48 | 5.737 s | 0.119 / 0.333 / 0.365 m | 2.629 m |
| 6 m/s 持续横风 | 48/48 | 5.765 s | 0.211 / 0.337 / 0.414 m | 2.609 m |
| 阵风 + 湍流 | 48/48 | 5.771 s | 0.337 / 0.416 / 0.461 m | 2.613 m |

随机风与飞行结果可复现（`wind_trace_reproducible`、`gust_rollout_reproducible` 均为 true）。注意：这是**保留门框、移除障碍**的隔离测试，不等于在完整障碍场景里的抗风结论。

数据：[result.json](.hitsz-repro/pwm-wind-ab48-01/result.json)、[provenance.txt](.hitsz-repro/pwm-wind-ab48-01/provenance.txt)。

## 5. 关键判断

1. **底层 PWM 跟踪已足够好**：净空对照 192/192，用时一致，偏离 ≤0.36 m。当前损失几乎都来自上层避障。
2. **规则绕行是目前最强的可用策略**；PPO 在样本内 +4~5 局，跨赛段只 +1 局，且从未选择等待。24/48 轮远未到“学会避障”的规模。
3. **共同失败 109 局**是当前最值得投入的方向；它们不是 PPO 选路错误，而是候选集/评分/预测的共同缺陷。相对 Issue #1 的 P1 目标（通过 ≥70%、撞击 ≤20%，见 §5），当前有效出发 97/156≈62%、53/156≈34%，仍有差距。
4. **固定障碍**多数在起飞后 0.5 s 内撞上，且中心线已被堵死；需要更早横移或更早减速，而不是最后一刻刹车。
5. **移动车辆**需要 1–2 s 的短时预测和安全停靠点；当前只按“当前位置”做净空判断，导致飞机停住后仍被车撞（2 局）。
6. **入口限速**单独使用无效；**风场**目前不是瓶颈；**P0-6 细线障碍**仍是空白，不能拿现有结果外推到细线。

## 6. 下一步建议（按优先级）

### 第 1 步：候选可行性诊断（最先做）

目标：回答“109 局共同失败里，换一个动作能不能过”。

1. 用 `ppo-transfer-collision-audit-01` 的逐局记录定位撞击前 1–2 s 的仿真状态。
2. 从同一状态分别试跑：5 条现有候选、仅制动、短暂等待（以及必要的组合）。
3. 把每局归类为：
   - **A 选错路**：某个现有候选能安全通过；
   - **B 无可行候选**：所有候选都过不去，需要新增轨迹；
   - **C 需要预测**：只有考虑车辆未来 1–2 s 的占位才能选对；
   - **D 不可能**：初始状态/门序本身不可行。
4. 交付一张逐局表 + 汇总比例。**先不要改训练代码**，这份诊断决定下一步改哪一层。

可直接复用的入口：`ppo-transfer-collision-audit-01/diagnose.py`、`Course(route, start, first_gate)`、`BypassLearningEnv`、`evaluation/harness.py` 的 `Progress`。

### 第 2 步：改共用避障层（按诊断结果选一项）

- **移动车辆**：常速度/占位预测（1–2 s），按“到达时间 × 占用概率”给候选打分；停靠点必须在车辆未来路径之外；不要把“停住”当作安全。
- **固定障碍**：把横移决策提前到起飞后 0–0.5 s；检查当前 `scoring_horizon_m=9`、`merge_distance_m=3.1`、偏移上限 2.05 m 是否足够；必要时增加更早的横向候选或调整代价。
- **速度**：入口硬限速已验证无效，优先改“何时开始横移/减速”，而不是统一降速。

### 第 3 步：补 P0-6 细线障碍

Issue #1 要求 `sim/obstacles.py`：线段/细圆柱碰撞体、直径 1–5 cm、垂度/反光随机、LiDAR/相机渲染、near-miss 统计。当前完全没有；在细线场景上，现有 HITSZ/v12/规则结果都不能作为证据。

### 第 4 步：候选改进后再回到 PPO

- 训练赛段从单弯扩到 4 段（参考 `scripts/eval_bypass_transfer.py` 的 `WINDOWS`）。
- 至少 100–300 轮、3 个种子；每轮保存 `status.json`；用 `val192` 选模。
- 冻结后使用**未用过的留出集**（新种子或 Issue #1 的 `test384-MR`）评一次；不要复用已用于 P0 的 `test384`。报告三门通过、飞行中撞击、超时、等待次数。
- 如果候选层没有明显改善，**不要**指望加大 PPO 训练量解决 109 局共同失败。

### 第 5 步：官方仿真校准

按 Issue #1 的 P3/P4 做；本地 6×3 m 研究门和官方裁判分数必须分开报告。

## 7. 已知坑

- `val192` 是开发集；`test384` 只评一次。交接后不要用 `test384` 调参。
- **起点接触必须单列**：`val192` 有 36 局、跨赛段 384 有 52 局。报告通过率时同时给“含起点”和“有效出发”两个口径。
- `.hitsz-repro/ppo-transfer384-01/protocol.json` 里的权重路径是旧服务器绝对路径。新实例上要么把权重放回对应路径，要么重新运行 `freeze --models .hitsz-repro/ppo-bypass` 生成新协议；校验哈希后再评。
- `scripts/train_bypass_ppo.py` 使用 `fcntl`，只能在 Linux 跑。
- `Course()` 的默认参数保持旧行为；自定义赛段时显式传 `route/start/first_gate`。
- 旧服务器 `activate.sh` 里是 Featurize 的 `$HOME` 路径，不要直接用于新机器。
- 新输出目录用新名字；harness/训练/评测都要求输出目录不存在。
- `.hitsz-repro` 已纳入仓库；以后的新实验可以继续放这里，但注意仓库体积，避免提交大体积原始视频/点云。

## 8. 权重与校验

跨赛段使用的 24 轮六动作权重（`.hitsz-repro/ppo-bypass/wait-*/latest.pt`）：

```
1b694c085fe51f0ea68da6d29ef06b6a174d26eb4bf3e9d41df130f1e2d10c5d  wait-20261121
a43904707c3ec1dd9ee9ce4266371841cab6daf38d5a1ccf12b249ed739bdcda  wait-20261122
51166aa6c5b6714d5bcb98f199f0d302d758ba858cdbc16a0a47aa3bd3f06f98  wait-20261123
```

48 轮续训权重（`.hitsz-repro/ppo-bypass-cont48-01/wait-*/latest.pt`）：

```
56ee103b3eba5f74fda83f817546ac5fa4ff23c97ef3e70f98563cd36efb8a2a  wait-20261121
3a8756693ea997a8067609f49a90aaf674c6a2915bdb38c2297c0520fb040743  wait-20261122
9cf137fd217827fc48d667b4681558e994ae33896c7cfa4e5098ace785bb54ad  wait-20261123
```

v12 基线：

```
8e997b5215152c8f63797ba1319b8ad8abd7d7bd5e4e57b9e667d71e9604a86d  .hitsz-repro/v12.onnx
```

`scripts/eval_bypass_transfer.py` 的 `freeze`/`segment` 会再次校验这些哈希；不匹配会直接报错。

## 9. 快速验收（新实例）

```bash
# 1. 环境
python scripts/hitsz_repro.py check

# 2. PWM 冒烟
python -m scripts.smoke_pwm --out .hitsz-repro/pwm-smoke-new
# 期望 replay_identical=true，首门约 2.23 s

# 3. val192 原几何
python -m evaluation.harness --suite val192 --verify-repeat \
  --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-val192-new
# 期望 66 success / 126 collision（36 起点接触），重跑逐局一致

# 4. val192 规则绕行
python -m evaluation.harness --suite val192 --lidar-bypass \
  --v12 .hitsz-repro/v12.onnx --out .hitsz-repro/pwm-lidar-bypass-new
# 期望 97 success / 89 collision（含 36 起点接触）/ 6 timeout

# 5. 风场回归
python -m scripts.eval_pwm_wind --out .hitsz-repro/pwm-wind-new
# 期望 48/48 三种风况全部通过
```

如果第 3 步数字有差异，先查源码哈希、`HITSZ_ROOT` 提交和 `v12.onnx` 哈希，不要先怀疑物理参数。

## 10. 数据索引

| 路径 | 内容 |
|---|---|
| `.hitsz-repro/p0-results/` | P0 冒烟、val192、test384、旧权重 fresh96 |
| `.hitsz-repro/pwm-clear-val192-01/` | 净空对照（移除障碍，192/192） |
| `.hitsz-repro/pwm-collision-audit-01/` | 原几何控制器 90 次飞行中撞击回放 |
| `.hitsz-repro/pwm-lidar-brake-val192-01/` | 前视制动首版 |
| `.hitsz-repro/pwm-lidar-bypass-val192-01/` | 五路绕行主结果（97/53/6） |
| `.hitsz-repro/pwm-lidar-bypass-val192-02/` | 横移限速 3.6 m/s 对照 |
| `.hitsz-repro/ppo-bypass/` | 五动作/六动作 × 3 种子 24 轮训练、`val192` 对照、`queue.log` |
| `.hitsz-repro/ppo-bypass-cont48-01/` | 24→48 轮续训与吞吐探测 |
| `.hitsz-repro/ppo-transfer384-01/` | 四赛段冻结评测、协议、逐段 CSV |
| `.hitsz-repro/ppo-transfer-collision-audit-01/` | 109 局共同失败回放与汇总 |
| `.hitsz-repro/pwm-entry-speed-ab96-01/` | 入口速度消融 |
| `.hitsz-repro/pwm-wind-ab48-01/` | 风场回归 |
| `.hitsz-repro/v12.onnx` | v12 基线（CPU ONNX Runtime） |
| `.hitsz-repro/gates-r1.json` | R1 门图快照 |
| `.hitsz-repro/*.tar.gz` | 各阶段源码/报告快照（历史追溯） |

## 11. 交接清单

- [ ] 读完 Issue #1 和本文件。
- [ ] 新实例完成 `setup` + `check`。
- [ ] 校验 `v12.onnx` 与 3 个 24 轮权重哈希。
- [ ] 跑通 §9 的 5 条验收命令。
- [ ] 读 `ppo-transfer-collision-audit-01/summary.json` 和 `diagnose.py`。
- [ ] 完成第 6 节第 1 步“候选可行性诊断”，产出逐局分类表。
- [ ] 根据诊断决定：改候选/改预测/改 PPO 训练分布；**不要先盲目加训练量**。
- [ ] 更新 Issue #1 的 checklist 和结果链接。
