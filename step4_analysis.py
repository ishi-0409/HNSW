from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
RESULTS = BASE / "results"


def step4_8_summary(save: bool) -> pd.DataFrame:
    reach = pd.read_csv(RESULTS / "step4_8_reachable.csv")
    deg = pd.read_csv(RESULTS / "step4_8_degree.csv")

    rows = []
    for M, d in deg.groupby("M"):
        n = len(d)
        r = reach[reach["M"] == M]["n_reachable"]
        rows.append({
            "M": int(M),
            "reach_min": int(r.min()),
            "reach_max": int(r.max()),
            "mean_degree": d["degree"].mean(),
            "max_degree": int(d["degree"].max()),
            "frac_degree_eq_M": (d["degree"] == M).mean(),
            "mean_first1000": d[d["insert_pos"] < 1000]["degree"].mean(),
            "mean_last1000": d[d["insert_pos"] >= n - 1000]["degree"].mean(),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for M, d in deg.groupby("M"):
        axes[0].hist(d["degree"], bins=np.arange(M, d["degree"].max() + 2),
                     histtype="step", label=f"M={M}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("number of points")
    axes[0].legend()

    d10 = deg[deg["M"] == 10]
    axes[1].scatter(d10["insert_pos"], d10["degree"], s=2, alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("insert position (M=10)")
    axes[1].set_ylabel("degree")
    fig.tight_layout()

    if save:
        summary.to_csv(RESULTS / "step4_8_summary.csv", index=False)
        fig.savefig(RESULTS / "step4_8_degree.png", dpi=150)
    return summary


if __name__ == "__main__":
    step4_8_summary(save=True)