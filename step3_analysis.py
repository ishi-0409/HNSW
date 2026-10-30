from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

if __name__ == "__main__":
    step3_12_curve(save=True)