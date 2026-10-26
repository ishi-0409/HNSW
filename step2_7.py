import pandas as pd
from pathlib import Path
import numpy as np
from common import read_fvecs, read_ivecs
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
SIFT = ROOT / "siftsmall"
DATA = ROOT / "data"

X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

df = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
full = np.load(DATA / "knn_sift10k_M40.npy")

#　クエリごとに全点の距離計算
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
# df.to_csv(RESULTS / "step2_local_minima.csv", index=False)

# print((df["rank"] >= 1).all())                           # True のはず
# print(((df["rank"] == 1) == (df["recall1"] == 1)).all())  # True のはず

# miss = df[df["recall1"] == 0]
# m = miss.groupby("M")

# summary = pd.DataFrame({
#     "外れ割合": 1 - df.groupby("M")["recall1"].mean(),
#     "外れ件数": m.size(),
#     "順位_25%": m["rank"].quantile(0.25),
#     "順位_中央値": m["rank"].median(),
#     "順位_75%": m["rank"].quantile(0.75),
#     "10位以内": m["rank"].apply(lambda r: (r <= 10).mean()),
#     "100位より外": m["rank"].apply(lambda r: (r > 100).mean()),
#     "recall10>0": m["recall10"].apply(lambda r: (r > 0).mean()),
# })
# print(summary.round(3))


# fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
# bins = np.logspace(0, 4, 30)          # 1位〜10000位を対数で30区間
# for ax, M in zip(axes, sorted(miss["M"].unique())):
#     ax.hist(miss.loc[miss["M"] == M, "rank"], bins=bins)
#     ax.set_xscale("log")
#     ax.set_title(f"M={M}")
#     ax.set_xlabel("rank of stop point")
# axes[0].set_ylabel("n_miss")
# plt.tight_layout()
# plt.savefig(RESULTS / "step2_miss_rank_hist.png", dpi=150)

# 止まった点の入り次数を列に足す
df["stop_indeg"] = [int(indeg[M][s]) for M, s in zip(df["M"], df["stop"])]

# 順位で4種類に分類
df["kind"] = pd.cut(df["rank"], bins=[0, 1, 10, 100, 10000],
                    labels=["hit", "near", "mid", "far"])

# 種類ごとの止まった点の入次数
a = df.groupby(["M", "kind"], observed=True)["stop_indeg"].agg(["size", "median", "mean"])
print(a.round(1))

# データ全体の入次数
for M in sorted(indeg):
    print(M, "全体の中央値:", np.median(indeg[M]), "全体の平均:", indeg[M].mean())

# 真の1位が反ハブかで当たる割合を比べる
df["true1_anti"] = df["true1_indeg"] == 0
b = df.groupby(["M", "true1_anti"])["recall1"].agg(["size", "mean"])
print(b.round(3))

df["entry_rank"] = [int((D[qi] < D[qi, e]).sum()) + 1
                    for qi, e in zip(df["query"], df["entry"])]
df["improve"] = df["entry_rank"] / df["rank"]

print(df.groupby(["M", "kind"], observed=True)[["entry_rank", "rank", "improve"]].median().round(1))

df["true1_q"] = df.groupby("M")["true1_indeg"].transform(
    lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
)
print(df.groupby(["M", "true1_q"], observed=True)["recall1"].agg(["size", "mean"]).round(3))

df.to_csv(RESULTS / "step2_local_minima.csv", index=False)

