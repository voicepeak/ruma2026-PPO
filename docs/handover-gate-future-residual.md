# 交接文档：第四栏 A0/A1 实施

> 读者：接手实现的人或 agent。设计背景见 [plan-gate-future-residual.md](plan-gate-future-residual.md)；本文只讲“怎么改、怎么跑、怎么验收”。

## 0. 先做什么

第一个 PR 只做 **A0：23 维输入，不改动作，不做残差**。

- A0 的目的：验证“未来穿门点”这三个特征到底有没有用。
- A0 不改变 3 维速度输出，PPO 更新逻辑不需要重写。
- A0 通过后再做 **A1：冻结 base + 1 维 `Δv_z` 残差**。
- B（候选风险）和 C（观测质量）在 A 通过后再做。

不要把 A0/A1/B/C 塞进一个 PR。每步都要能单独回滚和配对评测。

## 1. 仓库与环境

- 仓库：`git@github.com:voicepeak/ruma2026-PPO.git`
- 分支：`main`（远端只保留 `main`；`gh-pages` 仅用于报告站）
- 当前方案提交：`645ceca`（以 `git rev-parse main` 为准）
- HITSZ 固定版本：`9ac9657`
- Python：`py311_rmua`，PyTorch `2.6.0+cu118`，Ubuntu 22.04
- 训练 GPU：RTX 4090；评测/回放：RTX 3060 也可
- 训练前必须编译 `quadsim_cuda`

```bash
cd /home/tianbot/ruma-ppo
git pull --ff-only origin main
conda activate py311_rmua
export HITSZ_ROOT=/home/tianbot/ruma-train/rmua_rl-native
python -m scripts.hitsz_repro check
```

`HITSZ_ROOT` 必须指向固定 `9ac9657` 的签出。`check` 失败时先修环境，不要继续写代码。

## 2. 基座模型

默认用 **alignment-u2601**（当前候选，fresh384 243/384）：

```bash
BASE_CKPT=runs/native-joint-20261124-failure-audit/alignment-u2601.pt
sha256sum "$BASE_CKPT"
# 151fdd00eb749932ad11718aa1f7897161d0690cb6b3c48f11a620118ab892e6
```

备选：

| 名称 | 路径 | sha256 | 说明 |
|---|---|---|---|
| alignment-u2601 | `runs/native-joint-20261124-failure-audit/alignment-u2601.pt` | `151fdd...ab892e6` | 默认 |
| u3839 | `runs/handover-2026-11-24-4090/original-4090/latest.pt` | `cf5d40...873815` | 稳定基座 |
| u2501 | `runs/handover-2026-11-24-4090/reference/joint-3060-u2501.pt` | `30a169...997ec7` | 已发布回退 |
| alignment-u2701 | `runs/handover-2026-11-24-4090/height-alignment/latest.pt` | `d051a0...991af2` | 时间最后，但 fresh384 只有 228/384 |

规则：

- 同一组 A/B/C 只用一个基座。
- 如果用 u2701，A0 必须同时跑 u2601 对照；否则分不清基座退化和升维收益。
- 基座参数默认全部冻结；只训练新增投影/动作头/残差头。

## 3. A0 输入定义

A0 = 现有 20 维 + 3 维：

| 维度 | 名称 | 定义 | 归一化 / 哨兵 |
|---|---|---|---|
| 1 | `t_gate` | 到门平面预计时间 | `clip(t,0,2)/2`；无门/闭合速度 ≤0.5 → 1.0 |
| 1 | `z_cross` | 预计穿门点垂直偏差（门系，正=偏下） | 除以半高 1.5 m，clip ±1.5；无门 → 0 |
| 1 | `lower_margin` | 预计点到门下沿余量 | `(half_height - z_cross_raw)/half_height`，clip [-1,1]；无门 → 0 |

门坐标系和 `height_alignment_penalty` 保持一致：x=法向，y=右，z=下。**不要另写一套符号约定。**

关键公式：

```text
rel_p = (position - center) @ basis          # [B,3]
rel_v = velocity @ basis                    # [B,3]
ahead = -rel_p[:, 0]
t = ahead / rel_v[:, 0].clamp_min(0.5)      # 无门/远离时置 1.0
z_cross = rel_p[:, 2] + t * rel_v[:, 2]
lower = (half_height - z_cross) / half_height
t_norm = clip(t, 0, 2) / 2
z_norm = clip(z_cross / half_height, -1.5, 1.5)
lower_norm = clip(lower, -1, 1)
```

`half_aperture` 是 `[2]` 或 `[B,2]` 都支持；用 `half_aperture[..., 0]` 取半宽、`[..., 1]` 取半高。

## 4. A0 代码改动

### 4.1 `ruma_ppo/training/hitsz_joint.py`

**(a) 新增配置**

```python
@dataclass(frozen=True)
class GateFutureConfig:
    enabled: bool = False
    min_closing_mps: float = 0.5
```

**(b) 新增特征函数**

```python
def gate_future_features(position, velocity, center, basis, half_aperture, active,
                         min_closing_mps=0.5):
    rel_p = ((position - center)[:, None] @ basis).squeeze(1)
    rel_v = (velocity[:, None] @ basis).squeeze(1)
    half_h = half_aperture[..., 1]
    ahead = -rel_p[:, 0]
    closing = rel_v[:, 0]
    t = (ahead / closing.clamp_min(min_closing_mps)).clamp(0, 2)
    z = rel_p[:, 2] + t * rel_v[:, 2]
    lower = (half_h - z) / half_h
    moving = active & (closing > min_closing_mps)
    t = torch.where(moving, t / 2.0, torch.ones_like(t))
    z = torch.where(active, z / half_h, torch.zeros_like(z))
    lower = torch.where(active, lower.clamp(-1, 1), torch.zeros_like(lower))
    return torch.stack([t, z, lower], dim=-1)
```

要求：

- 不产生 NaN；`closing` 可能为 0。
- `active` 是 `[B]` bool。
- 无门时 `t=1.0, z=0, lower=0`；base `gate` 全零作为 mask。

**(c) `GateEnv`**

```python
class GateEnv(RLQuadEnv):
    def __init__(self, cfg, env_cuda, gate, occupancy=None, future=None):
        super().__init__(cfg, env_cuda)
        self.future = future
        ...
    def get_observations(self):
        obs = super().get_observations()
        obs["gate"] = self.gates.features(self._current_R_yaw)
        if self.future is not None and self.future.enabled:
            center, basis, active = self.gates.current()
            obs["gate_future"] = gate_future_features(
                self.env.p, self.env.v, center, basis, self.gates.half_aperture, active,
                self.future.min_closing_mps,
            )
        if self.occupancy is not None:
            obs["vehicle_target"] = self.occupancy.targets(self._current_R_yaw)
        return obs
```

**(d) `GateActorCritic`**

- 构造函数增加 `gate_future=None`：
  - 若启用，新增 `self.actor_future = zero_projection(3, self.actor_encoder.output_dim)`；
  - 新增 `self.critic_future = zero_projection(3, self.critic_encoder.output_dim)`。
- `encode` 增加 `future` 参数，注入 `+ future(obs["gate_future"])`。
- `actor_outputs`/`evaluate` 传入对应投影。
- `std`、`occupancy`、`forecast` 路径先不改。
- 保持 3 维动作；A0 不引入 residual。

A0 的两种冻结方式：

- **A0a（先做）**：所有 base 参数 `requires_grad=False`，只训 `actor_future` / `critic_future`。
- **A0b（备选）**：如果 A0a 的新投影权重仍接近 0、动作几乎不变，解冻 `actor_head`（单独 param group，小 LR）；stem/SRU/forecast 仍冻结。

```python
def encode(self, obs, encoder, gate, future):
    depth = obs["depth"].reshape(-1, 1, encoder.img_height, encoder.img_width)
    visual = encoder.stem(depth).reshape(*obs.batch_size, encoder.output_dim)
    out = visual + encoder.v_proj(obs["state"]) + gate(obs["gate"])
    if future is not None:
        out = out + future(obs["gate_future"])
    return encoder.act(out)
```

**(e) `HitszRunner`**

- `__init__` 增加 `future=None`，保存；`_construct_algorithm` 把 `gate_future=self.future` 传给 `GateActorCritic`。
- 新增 `load_base(path)`：
  - `torch.load(..., weights_only=True)`；
  - `self.alg.policy.load_state_dict(ckpt["model_state_dict"], strict=False)`；
  - 缺失 key 只允许是新投影（`actor_future`/`critic_future`，后续 `risk`/`quality`/`residual`）；
  - `unexpected_keys` 必须为空；
  - 恢复 `ckpt["curriculum"]` 到 env，避免从低速/低障碍重新开始；
  - **不要**恢复 optimizer、RNG、`current_learning_iteration`；这是新 run，不是 resume；
  - 最后 `self.env._apply_obstacle_level()` + `self.env._reset_envs(...)`。

### 4.2 `ruma_ppo/cli/train_joint.py`

新增参数：

```python
parser.add_argument("--gate-future", action="store_true",
                    help="Add explicit future gate crossing features (A0)")
parser.add_argument("--base-checkpoint", type=Path,
                    help="Load model_state_dict from a joint checkpoint; new keys are zero-initialized")
```

互斥：`--base-checkpoint` 不能和 `--resume`/`--resume-from` 同时用。

`train()`：

- `future = GateFutureConfig(enabled=args.gate_future)`；
- 传入 `GateEnv(..., future=future)` 和 `HitszRunner(..., future=future)`；
- `recipe["additions"]["gate_future"] = asdict(future)`；
- 如果 `args.base_checkpoint`：
  - `args.out.mkdir(parents=True, exist_ok=False)`；
  - `runner.load_base(args.base_checkpoint)`；
  - 不调用 `runner.restore`；
  - `checkpoint = None`，`remaining = updates`。

### 4.3 `configs/train_joint.yaml`

增加：

```yaml
gate_future:
  enabled: false
  min_closing_mps: 0.5
```

A0 可用 CLI `--gate-future` 打开；也可以直接把 `enabled` 改成 `true`。不要改 `gate`/`occupancy` 的默认值。

### 4.4 `ruma_ppo/evaluation/harness.py`

`load_native_ppo()`：

- 从 `recipe["additions"]` 读 `gate_future`；
- 构造 `GateActorCritic(..., gate_future=FutureGateConfig(**cfg))`；
- 旧 checkpoint 没有该字段时默认 `enabled=False`。

`NativePPOPolicy`：

- 增加 `future_enabled`；
- `step()` 里在 `course.env.step(...)` 之前设置 `obs["gate_future"]`：

```python
if self.future_enabled:
    index = progress.gates.clamp_max(len(progress.center) - 1)
    obs["gate_future"] = gate_future_features(
        course.e.p, course.e.v, progress.center[index], progress.basis[index],
        progress.half_aperture[index], ~progress.done,
        self.future_config.min_closing_mps,
    )
```

- 复用同一个 `gate_future_features`，不要在 eval 里复制公式。

### 4.5 测试

新增 `tests/test_gate_future.py`：

1. **特征范围**：
   - 门在正前方 5 m，速度 10 m/s：`t≈0.25`，`z=0`，`lower=1.0`；
   - 偏下 0.15 m：`z≈0.1`，`lower≈0.9`；
   - 无门：`t=1.0, z=0, lower=0`；
   - 闭合速度 0：不产生 NaN。
2. **第 0 步等价性**：
   - 构造 base 模型（`gate_future.enabled=False`）和 new 模型（`enabled=True`）；
   - 复制 base `state_dict` 到 new，`strict=False`；
   - 固定 `TensorDict`（depth 全 0、state 全 0、gate 全 0、gate_future 全 0）；
   - `act_inference` 输出差 `<1e-6`。
3. **checkpoint 加载**：
   - `load_state_dict(..., strict=False)` 的 `unexpected_keys` 为空；
   - `missing_keys` 只包含新投影。

运行：

```bash
python -m pytest -q tests/test_gate_future.py
```

本机若没装 pytest，先在 `py311_rmua` 里安装；不要用系统 Python。

## 5. A0 训练与评测命令

先 2 轮 smoke：

```bash
python -m scripts.train_joint --variant joint \
  --out runs/gate-future-a0-smoke \
  --base-checkpoint "$BASE_CKPT" \
  --gate-future --updates 2 --envs 4096 --num-steps-per-env 20 \
  --lr-schedule fixed --learning-rate 0.001 --seed 20261125
```

正式 50 轮：

```bash
python -m scripts.train_joint --variant joint \
  --out runs/gate-future-a0-50 \
  --base-checkpoint "$BASE_CKPT" \
  --gate-future --updates 50 --envs 4096 --num-steps-per-env 20 \
  --lr-schedule fixed --learning-rate 0.001 --seed 20261125
```

3060 配方：

```bash
python -m scripts.train_joint --variant joint \
  --out runs/gate-future-a0-50-3060 \
  --base-checkpoint "$BASE_CKPT" \
  --gate-future --updates 50 --envs 1024 --num-steps-per-env 80 \
  --num-mini-batches 32 --grad-accum-steps 8 \
  --lr-schedule stable --learning-rate 0.003 --seed 20261125
```

评测：

```bash
python -m ruma_ppo.evaluation.harness --suite val192 \
  --native-ppo runs/gate-future-a0-50/latest.pt \
  --v12 .hitsz-repro/v12.onnx \
  --out runs/gate-future-a0-val192 --verify-repeat
```

A0 验收：

- smoke 2 轮无 NaN、无显存爆炸；
- 第 0 步等价性通过；
- val192 配对：漏门下降，飞行中碰撞不上升；
- 如果方向不对，停止，不要加 B/C。

## 6. A1：残差头

A0 有信号后再做，单独分支/PR。

### 6.1 结构

```text
base = 冻结的 joint（3 维速度输出）
residual = MLP([gate_future, base_action, hidden?]) → bound * tanh(.)
v_cmd = clamp(base_action + [0,0,Δv_z], limits)
```

- 首版只输出 `Δv_z`，`|Δv_z| ≤ 0.8 m/s`。
- 残差头最后一层零初始化；第 0 步 `v_cmd == base_action`。
- 基座参数 `requires_grad=False`。

### 6.2 代码建议

- 新建 `ruma_ppo/training/residual.py`：
  - `ResidualGateActorCritic`：`act()` 只采样 `Δv`；`act_inference()` 返回 `Δv` 均值。
  - `ResidualGateEnv`：在 `step(delta)` 里计算 base action，执行 `base + delta`，返回 obs/reward/done/extras。
  - `ResidualPPO`：不要复用 `OccupancyPPO.update`，因为它的动作维度、forecast 路径和 `vehicle_target` 假设都是 3 维 base 动作。写一个只处理 1 维动作的 PPO 更新，或先用 `--variant gate` + frozen base。
- 观测：`ResidualGateEnv` 把 `base_action`（3 维）放进 `obs["base_action"]`，残差头读它；`gate_future` 复用 A0 helper。
- 评测：`harness` 需要支持 `residual` checkpoint；A1 前先加一个 `--native-residual` 或单独的 eval 入口。

A1 验收：

- Δv 饱和率、check 干预频率、门平面下沿余量、漏门→通过、通过→碰撞；
- 和固定高度探针、几何门中心控制器对照；
- 残差收益必须超过 A0，否则回退。

## 7. B/C 后续

B（37 维）需要：

- `lat_cross`、`side_margin`；
- 4 候选 × 3 风险；
- 候选前推用同一个速度 PI、同一加速度限制、1.0–1.5 s；
- `risk` 组独立零初始化投影；
- 先冻结 base，残差 3 维；再考虑解冻最后一层。

C（40 维）需要：

- `gate_age`、`unknown_fraction`、`visual_confidence`；
- 官方 ROS 记录标定延迟/丢帧；
- actor 看估计输入，critic 可看真值；
- 官方模拟器核对门数、时钟、控制模式。

## 8. 常见坑

1. **不要用 `--resume` 加载架构变化**：`HitszRunner.restore` 会比较 recipe；新增 obs group 会报错。用 `load_base`。
2. **不要关掉 occupancy**：基座 checkpoint 有 `forecast`/`forecast_velocity`；关掉后第 0 步动作会变，无法做等价性验证。
3. **eval 必须也加 `gate_future`**：`load_native_ppo` 和 `NativePPOPolicy` 不改，模型会因为缺 key 或特征全零而表现异常。
4. **符号约定**：复用 `height_alignment_penalty` 的门系；不要自己定义正负。
5. **无门哨兵**：`t=1.0`，`z=0`，`lower=0`；不要用 `t=0` 表示无门。
6. **`half_aperture` 形状**：训练是 `[2]`，eval 是 `[B,2]`；helper 用 `[..., 0]`/`[..., 1]`。
7. **`TensorDict` batch size**：新增 key 必须是 `[B,3]`；不要加 `[3]`。
8. **`test384` 不能再调参**：A 阶段用 val192；最终用新种子或预注册留出。
9. **`HITSZ_ROOT`**：必须固定 `9ac9657`；否则 checkpoint 加载会被 `load_native_ppo` 拒绝。
10. **显存**：4090 用 4096×20；3060 用 1024×80 + grad accumulation；先 smoke 2 轮。

## 9. Agent 执行清单

- [ ] `git pull --ff-only origin main`，确认 head。
- [ ] `conda activate py311_rmua`，`export HITSZ_ROOT=...`，`python -m scripts.hitsz_repro check`。
- [ ] 核对基座 sha256。
- [ ] 实现 `GateFutureConfig`、`gate_future_features`、`GateEnv`、`GateActorCritic`、`HitszRunner.load_base`。
- [ ] 实现 CLI `--gate-future` / `--base-checkpoint`。
- [ ] 改 `configs/train_joint.yaml`。
- [ ] 改 `harness.py` 的模型构造和 `NativePPOPolicy`。
- [ ] 写 `tests/test_gate_future.py`。
- [ ] 第 0 步等价性通过。
- [ ] 2 轮 smoke 通过。
- [ ] 50 轮 A0，val192 配对。
- [ ] 写 `docs/progress.md`：基座、命令、结果、下一步。
- [ ] 如果 A0 通过，再开 A1；否则停止并记录失败原因。

**Definition of Done（A0）**：

- 代码在 `main` 上可复现；
- 测试通过；
- 第 0 步等价性 `<1e-6`；
- 50 轮训练日志和 val192 `result.json` 已归档；
- `docs/progress.md` 记录结论；
- 没有改 `main` 之外的远端分支。

## 10. 交接信息

- 远端：`main` + `gh-pages`（仅报告站）
- 方案：`docs/plan-gate-future-residual.md`
- 本文：`docs/handover-gate-future-residual.md`
- 基座：`alignment-u2601`（默认）/ `u3839` / `u2501` / `alignment-u2701`
- 关键源码：
  - `ruma_ppo/training/hitsz_joint.py`
  - `ruma_ppo/cli/train_joint.py`
  - `ruma_ppo/evaluation/harness.py`
  - `configs/train_joint.yaml`
- 关键数据：
  - `runs/native-joint-20261124-failure-audit/alignment-u2601.pt`
  - `runs/handover-2026-11-24-4090/`
  - `.hitsz-repro/v12.onnx`

如果只能做一件事：**先让 A0 的 23 维特征、第 0 步等价性和 val192 配对跑通，再谈残差和 B/C。**
