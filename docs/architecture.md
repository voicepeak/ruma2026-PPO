# 架构

目标结构：

```text
ruma_ppo/
├── env/              # sim、maps、control
├── training/         # hitsz_joint.py：门/占位包装、模型、PPO、累积、stable LR、runner
├── evaluation/       # harness、rollout、metrics、lidar
├── hitsz/            # 固定 HITSZ 适配层：branches、course、v12、legacy policy
└── cli/              # train_joint、setup_hitsz
configs/              # 训练/仿真/评测 YAML
scripts/              # 薄入口；旧命令保留 wrapper
tests/                # 纯逻辑单测和 smoke
docs/                 # 架构、训练、评测、复现、交接
reports/              # 小体积、可版本化的评测结果
runs/                 # 本地训练数据，gitignore
```

依赖方向：

```text
scripts -> ruma_ppo
training/evaluation -> env/hitsz
env/hitsz 不 import training/evaluation
任何代码不 import scripts
```

约定：

- 一个逻辑改动一个 commit；不要为一个同步拆多个分支。
- `env/` 不写 PPO/评测逻辑；`training/` 不写评测；`evaluation/` 不写训练。
- 新研究线先放 `scripts/legacy/` 或独立小目录，验证后再提升到 `ruma_ppo`。
- 权重、日志、TensorBoard、checkpoint 不进 Git；小评测结果进 `reports/`，大文件走 HuggingFace。
- 固定 HITSZ commit 和 `v12.onnx` 的 sha256 记录在 `docs/reproducibility.md`；新上游版本只改 `hitsz/` 适配层。
- `training/hitsz_joint.py` 当前集中门/占位、模型、PPO 和 runner；继续增长时再拆成 `env/gates.py`、`env/occupancy.py`、`training/models.py`、`training/ppo.py`、`training/runner.py`。
