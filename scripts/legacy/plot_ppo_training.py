"""Render a static research snapshot from the trainer's recorded scalar history."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    # A resumed job can emit the same iteration after recovery from a checkpoint.
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted({r["update"]:r for r in data}.values(), key=lambda r:r["update"])


def main(folder):
    train = rows(folder/"curves.jsonl")
    dev = rows(folder/"eval.jsonl")
    base = json.loads((folder/"baseline.json").read_text())
    fixed = 100*sum(r["completed"] for r in base.values())/sum(len(r["gates"]) for r in base.values())
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.4),layout="constrained")
    fig.suptitle(f"HITSZ 三维环境  /  PPO 训练至第 {train[-1]['update']} 轮",fontweight="bold")
    blue,orange,gray="#206799","#ae512d","#52606d"
    for level,color in [(2,blue),(3,orange)]:
        subset=[r for r in train if r["level"]==level]
        x=[r["update"] for r in subset]
        axes[0,0].plot(x,[r["Train/episode_reward"] for r in subset],color=color,label=f"障碍等级 {level}")
        axes[1,0].plot(x,[100*r["Outcome/collision"] for r in subset],color=color,label=f"障碍等级 {level}")
    axes[0,0].set(title="随机训练场景 · 平均每局奖励",ylabel="奖励")
    axes[1,0].set(title="随机训练场景 · 以碰撞结束的回合",ylabel="占比（%）")
    axes[0,1].plot([r["update"] for r in dev],[r["completed"]*100/96 for r in dev],
                   color=blue,marker="o",ms=4,label="PPO 固定 96 局")
    axes[0,1].axhline(fixed,color=gray,ls="--",label=f"居中规则 {fixed:.1f}%")
    axes[0,1].set(title="固定验证场景 · 连过三门",ylabel="通过率（%）")
    axes[1,1].plot([r["update"] for r in train],[r["PPO/entropy"] for r in train],color=blue)
    axes[1,1].set(title="策略探索 · 动作分布熵",ylabel="熵（自然对数）")
    for ax in axes.flat:
        ax.set_xlabel("PPO 更新轮数")
        ax.grid(alpha=.16)
        ax.spines[["top","right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8,frameon=False)
    fig.savefig(folder/"learning-curves.png",dpi=155)
    print("history",train[0]["update"],train[-1]["update"],"baseline",fixed,"last dev",dev[-1])
if __name__=="__main__":
    main(Path(sys.argv[1]))
