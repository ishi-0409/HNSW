from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

from common import load_sift, brute_force, recall_at_k
from graph import build_knn_graph, greedy_search, search_layer, reachable_from

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def step3_12_curve(save:bool) -> pd.DataFrame:
    "(M, ef)ごとに平均し、距離計算回数とrecall@10曲線を描く"
    df = pd.read_csv(RESULTS / "step3_beam_M_ef.csv")

    # 検算用
    # sub = df[(df["M"] == 10) & (df["ef"] == 10)]
    # print(len(sub), sub["recall10"].mean(), sub["n_dist"].mean())

    summary = (
        df.groupby(["M", "ef"])
        .agg(
            recall10 = ("recall10", "mean"),
            recall1 = ("recall1", "mean"),
            n_dist = ("n_dist", "mean"),
            n_rows = ("recall10", "size"),
        )
        .reset_index()
    )
    summary["n_dist_ratio"] = summary["n_dist"] / 10000
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6,4))
    for M, g in summary.groupby("M"):
        ax.plot(g["n_dist"], g["recall10"], marker="o", label=f"M={M}")
        for _, r in g.iterrows():
            ax.annotate(f"{int(r["ef"])}", (r["n_dist"], r["recall10"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("distance computations (mean)")
    ax.set_ylabel("recall@10 (mean)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)        

    if save:
        summary.to_csv(RESULTS / "step3_12_summary.csv", index=False)
        fig.savefig(RESULTS / "step3_12_recall_vs_ndist.png", dpi=150, bbox_inches="tight")

    return summary

def step3_13_compare(save:bool) -> pd.DataFrame:
    "同じ距離計算回数ならMとefのどちらを増やすほうが得か"
    _, _, GT = load_sift()
    beam = pd.read_csv(RESULTS / "step3_beam_M_ef.csv")
    greedy = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")

    # Step 2 の recall@1 = 停止点が本当の1位か
    greedy["recall1"] = (
        greedy["stop"].to_numpy() == GT[greedy["query"].to_numpy(), 0]
    ).astype(float)

    s3 = (beam.groupby(["M", "ef"])
          .agg(recall1=("recall1", "mean"), recall10=("recall10", "mean"),
               n_dist=("n_dist", "mean"))
               .reset_index())
    s2 = (greedy.groupby("M")
          .agg(recall1=("recall1", "mean"), n_dist=("n_dist", "mean"))
          .reset_index())
    print("Step 2 (ef=1):")
    print(s2.to_string(index=False))

    # 図：縦軸　recall@1 
    fig, ax = plt.subplots(figsize=(6, 4))
    for M, g in s3.groupby("M"):
        g1 = s2[s2["M"] == M]
        xs = list(g1["n_dist"]) + list(g["n_dist"])
        ys = list(g1["recall1"]) + list(g["recall1"])
        ax.plot(xs, ys, marker="o", label=f"M={M} (ef=1, 10..256)")
    ax.plot(s2["n_dist"], s2["recall1"], "k--", marker="s", label="ef=1, M varies (Step 2)")
    ax.set_xscale("log")
    ax.set_xlabel("distance computations (mean)")
    ax.set_ylabel("recall@1 (mean)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # 表
    rows: list[dict] = []
    for target in [400, 600, 900]:
        for M, g in s3.groupby("M"):
            x = g["n_dist"].to_numpy()
            if x.min() <= target <= x.max():
                y = float(np.interp(np.log(target), np.log(x),g["recall10"].to_numpy()))
                rows.append({"n_dist": target, "M": M, "recall10": y})
    interp = pd.DataFrame(rows)
    print(interp.pivot(index="n_dist", columns="M", values="recall10").round(3))

    if save:
        interp.to_csv(RESULTS / "step3_13_interp.csv", index=False)
        fig.savefig(RESULTS / "step3_13_recall1_vs_ndist.png", dpi=150, bbox_inches="tight")

    return interp


if __name__ == "__main__":
    # step3_12_curve(save=True)
    step3_13_compare(save=True)