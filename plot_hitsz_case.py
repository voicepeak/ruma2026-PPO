"""Plot one recorded scene where only a narrower candidate passes all three gates."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    report=json.loads((root/"report.json").read_text())
    original={-3.6,-2.3,0,2.3,3.6}
    with (root/"branches.csv").open() as f:
        rows=list(csv.DictReader(f))
    for lvl in report["levels"]:
        level=lvl["level"]
        for case in range(lvl["n"]):
            group=[r for r in rows if int(r["level"])==level and int(r["case"])==case]
            passed=[r for r in group if int(r["gates"])==3]
            if passed and all(float(r["offset_m"]) not in original for r in passed):
                break
        else:
            continue
        break
    else:
        raise ValueError("No narrower-only case in this report.")
    choice=min(passed,key=lambda r:int(r["steps"]))
    center=next(r for r in group if float(r["offset_m"])==0 and float(r["factor"])==1)
    scene=np.load(root/f"scene-L{level}.npz")
    paths=np.load(root/f"tracks-L{level}.npz")
    origin=paths["c0"][0,case,:2]
    direction=scene["gate_basis"][0,:2,0].copy()
    direction/=np.linalg.norm(direction)
    axes=np.stack([direction,[-direction[1],direction[0]]],axis=1)
    def xy(x):return (x[...,:2]-origin)@axes
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,(track,depth)=plt.subplots(2,1,figsize=(9,6.2),gridspec_kw={"height_ratios":[1.3,1]},layout="constrained")
    fig.suptitle(f"HITSZ 三维弯道   /   障碍等级 {level}，场景 {case}",fontweight="bold")
    road=xy(scene["waypoints"])
    nearby=(road[:,0]>-5)&(road[:,0]<38)&(np.abs(road[:,1])<15)
    track.plot(*np.where(nearby[:,None],road,np.nan).T,ls=":",color=".5",label="路线中心")
    for i,(c,b) in enumerate(zip(scene["gate_centres"],scene["gate_basis"])):
        gate=xy(np.stack([c-b[:,1]*3,c+b[:,1]*3]))
        track.plot(*gate.T,color="#8ca3b8",lw=3)
        track.annotate(f"门{i+1}",gate[1],xytext=(1,4),textcoords="offset points",color="#476176")
    for row,col,name in [(center,"#ac4534","原速居中"),(choice,"#057864","缩小横移")]:
        path=xy(paths["c"+row["candidate"]][:int(row["steps"])+1,case])
        track.plot(*path.T,color=col,lw=2,label=f"{name}：{row['offset_m']} m，过 {row['gates']} 门")
        track.scatter(*path[-1],s=26,color=col)
    track.set(xlabel="沿起始门方向（米）",ylabel="横向（米）",aspect="equal")
    track.legend(frameon=False,loc="upper left",fontsize=9)
    track.grid(alpha=.15)
    im=depth.imshow(np.clip(scene["depth"][case]*30,0,30),aspect="auto",cmap="viridis",vmin=0,vmax=30,interpolation="nearest")
    depth.set(xlabel="全景深度图列",ylabel="行",title="两条候选出发时的同一帧深度观测")
    fig.colorbar(im,ax=depth,label="含噪深度（米）",shrink=.8)
    fig.savefig(root/"bend-example.png",dpi=160)
    print("example",level,case,choice["candidate"],center["gates"],choice["gates"])

if __name__=="__main__":
    main(Path(sys.argv[1]))
