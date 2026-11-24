# 评测报告

每个目录对应一次固定场景的评测，包含 `result.json`、`paired.csv`、`scenes.json` 和 `manifest.json`。`manifest.json` 记录 checkpoint、HITSZ v12 的 sha256、suite 和命令，便于复现。

| 目录 | suite | candidate | success | v12 |
|---|---|---|---|---|
| `2026-11-24-u2501-val192` | val192 | native_joint_2501 | 63.0% | 31.8% |

## 2026-11-24 4090 交接

`2026-11-24-4090-handover/` 保存原配方续训、门前高度对准 A/B、漏门/碰撞回放和车辆预报路径。它不是单一 suite：`replays/` 是 `val192` 和 192 局/384 局回放，`ab/` 是 384 局独立场景的等预算 A/B。最高候选是高度对准 `u2601`（243/384），起点 `u2501` 是 230/384，同轮原配方对照是 195/384。详细口径、配对统计、碰撞分解和限制见该目录的 `README.md`。

权重路径、完整评分和选择建议汇总在[根目录 README](../README.md)。
