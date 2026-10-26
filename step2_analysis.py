from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_sift

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ---- 2-6: M ごとの平均・最小・最大と、距離計算回数の上限の検算 ----
def step2_6_summary() -> None:
    df = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
    print(len(df))

    cols = ["recall1", "recall10", "n_dist", "hops", "n_topk"]
    per_seed = df.groupby(["M", "seed"])[cols].mean()
    summary = per_seed.groupby("M").agg(["mean", "min", "max"])

    over = df["n_dist"] > 1 + (df["hops"] + 1) * df["M"]

    print(summary.round(3))
    print("上限超え:", over.sum())

    # 2段の列名を recall1_mean のように1段にして保存
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.to_csv(RESULTS / "step2_6_summary.csv")


# ---- 2-7: 順位や入次数の列を足して保存 ----
def step2_7_add_columns() -> pd.DataFrame:
    X, Q, GT = load_sift()
    df = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
    full = np.load(DATA / "knn_sift10k_M40.npy")

    # クエリごとに全点の距離計算
    D = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])

    # Mごとの入次数
    indeg = {M: np.bincount(full[:, :M].ravel(), minlength=len(X))
             for M in df["M"].unique()}

    ranks, true1_indeg = [], []
    for row in df.itertuples():
        d = D[row.query]
        rank = int((d < d[row.stop]).sum()) + 1
        ranks.append(rank)
        true1_indeg.append(int(indeg[row.M][GT[row.query, 0]]))

    df["rank"] = ranks
    df["true1_indeg"] = true1_indeg

    # 順位が1以上か
    print((df["rank"] >= 1).all())
    # 順位1位と recall@1 = 1 が一致するか
    print(((df["rank"] == 1) == (df["recall1"] == 1)).all())

    # 止まった点の入り次数を列に足す
    df["stop_indeg"] = [int(indeg[M][s]) for M, s in zip(df["M"], df["stop"])]

    # 順位で4種類に分類
    df["kind"] = pd.cut(df["rank"], bins=[0, 1, 10, 100, 10000],
                        labels=["hit", "near", "mid", "far"])

    # 真の1位が反ハブ(入次数0)か
    df["true1_anti"] = df["true1_indeg"] == 0

    # 入口の順位と、止まった点までに何倍よくなったか
    df["entry_rank"] = [int((D[qi] < D[qi, e]).sum()) + 1
                        for qi, e in zip(df["query"], df["entry"])]
    df["improve"] = df["entry_rank"] / df["rank"]

    # 真の1位の入次数を M ごとに4分位で分ける
    df["true1_q"] = df.groupby("M")["true1_indeg"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    )

    df.to_csv(RESULTS / "step2_local_minima.csv", index=False)
    return df


# ---- 2-7: 外れたときの止まった点の順位 ----
def step2_7_miss_summary(df: pd.DataFrame) -> None:
    miss = df[df["recall1"] == 0]
    m = miss.groupby("M")

    summary = pd.DataFrame({
        "外れ割合": 1 - df.groupby("M")["recall1"].mean(),
        "外れ件数": m.size(),
        "順位_25%": m["rank"].quantile(0.25),
        "順位_中央値": m["rank"].median(),
        "順位_75%": m["rank"].quantile(0.75),
        "10位以内": m["rank"].apply(lambda r: (r <= 10).mean()),
        "100位より外": m["rank"].apply(lambda r: (r > 100).mean()),
        "recall10>0": m["recall10"].apply(lambda r: (r > 0).mean()),
    })
    print(summary.round(3))
    summary.to_csv(RESULTS / "step2_7_miss_summary.csv")


# ---- 2-7: 外れたときの順位のヒストグラム ----
def step2_7_hist(df: pd.DataFrame) -> None:
    miss = df[df["recall1"] == 0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    bins = np.logspace(0, 4, 30)          # 1位〜10000位を対数で30区間
    for ax, M in zip(axes, sorted(miss["M"].unique())):
        ax.hist(miss.loc[miss["M"] == M, "rank"], bins=bins)
        ax.set_xscale("log")
        ax.set_title(f"M={M}")
        ax.set_xlabel("rank of stop point")
    axes[0].set_ylabel("n_miss")
    plt.tight_layout()
    plt.savefig(RESULTS / "step2_miss_rank_hist.png", dpi=150)
    plt.close(fig)


# ---- 2-7: 入次数との関係 ----
def step2_7_indeg(df: pd.DataFrame) -> None:
    full = np.load(DATA / "knn_sift10k_M40.npy")
    indeg = {M: np.bincount(full[:, :M].ravel(), minlength=len(full))
             for M in df["M"].unique()}

    # 種類ごとの止まった点の入次数
    a = df.groupby(["M", "kind"], observed=True)["stop_indeg"].agg(["size", "median", "mean"])
    print(a.round(1))
    a.to_csv(RESULTS / "step2_7_stop_indeg_by_kind.csv")

    # データ全体の入次数
    rows = []
    for M in sorted(indeg):
        print(M, "全体の中央値:", np.median(indeg[M]), "全体の平均:", indeg[M].mean())
        rows.append({"M": M, "indeg_median": np.median(indeg[M]), "indeg_mean": indeg[M].mean()})
    pd.DataFrame(rows).to_csv(RESULTS / "step2_7_indeg_overall.csv", index=False)

    # 真の1位が反ハブかで当たる割合を比べる
    b = df.groupby(["M", "true1_anti"])["recall1"].agg(["size", "mean"])
    print(b.round(3))
    b.to_csv(RESULTS / "step2_7_true1_anti.csv")

    # 真の1位の入次数の4分位ごとに当たる割合を比べる
    c = df.groupby(["M", "true1_q"], observed=True)["recall1"].agg(["size", "mean"])
    print(c.round(3))
    c.to_csv(RESULTS / "step2_7_true1_quartile.csv")


# ---- 2-7: 入口からどれだけ近づいたか ----
def step2_7_entry(df: pd.DataFrame) -> None:
    e = df.groupby(["M", "kind"], observed=True)[["entry_rank", "rank", "improve"]].median()
    print(e.round(1))
    e.to_csv(RESULTS / "step2_7_entry_rank.csv")


if __name__ == "__main__":
    step2_6_summary()
    df = step2_7_add_columns()
    step2_7_miss_summary(df)
    step2_7_hist(df)
    step2_7_indeg(df)
    step2_7_entry(df)
