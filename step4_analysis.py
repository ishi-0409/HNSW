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

def interp_recall(sub: pd.DataFrame, x: float) -> float:
    """1本の曲線(1つの graph・M)を横軸対数で直線補間し、距離計算回数 x での recall@10 を返す。測った範囲の外は nan"""
    sub = sub.sort_values("n_dist")
    xs = np.log(sub["n_dist"].to_numpy())
    ys = sub["recall10"].to_numpy()
    lx = np.log(x)
    if lx < xs.min() or lx > xs.max():
        return float("nan")
    return float(np.interp(lx, xs, ys))


def step4_10_curve(save: bool = True) -> pd.DataFrame:
    knn = pd.read_csv(RESULTS / "step3_beam_M_ef.csv").assign(graph="knn")
    nsw = pd.read_csv(RESULTS / "step4_9_beam_M_ef.csv").assign(graph="nsw")
    df = pd.concat([knn, nsw], ignore_index=True)

    # 集計表:graph × M × ef ごとの平均
    summary = (df.groupby(["graph", "M", "ef"])
                 .agg(recall10=("recall10", "mean"),
                      recall1=("recall1", "mean"),
                      n_dist=("n_dist", "mean"),
                      n_rows=("query", "size"))
                 .reset_index())
    summary["n_dist_ratio"] = summary["n_dist"] / 10000

    # 補間の表:同じ距離計算回数での recall@10
    interp_rows = []
    for (g, M), sub in summary.groupby(["graph", "M"]):
        for x in (400, 600, 900):
            interp_rows.append({"graph": g, "M": M, "n_dist": x,
                                "recall10": interp_recall(sub, x)})
    interp = pd.DataFrame(interp_rows)

    # 図:色 = M、実線 = NSW、破線 = kNN
    colors = {10: "tab:blue", 20: "tab:orange", 40: "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for (g, M), sub in summary.groupby(["graph", "M"]):
        sub = sub.sort_values("ef")
        ax.plot(sub["n_dist"], sub["recall10"], marker="o",
                linestyle="-" if g == "nsw" else "--",
                color=colors[M], label=f"{g} M={M}")
    ax.set_xscale("log")
    ax.set_xlabel("distance computations (log)")
    ax.set_ylabel("recall@10")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    print(summary.to_string())
    print(interp.pivot(index="n_dist", columns=["graph", "M"], values="recall10").round(3))

    if save:
        summary.to_csv(RESULTS / "step4_10_summary.csv", index=False)
        interp.to_csv(RESULTS / "step4_10_interp.csv", index=False)
        fig.savefig(RESULTS / "step4_10_recall_vs_ndist.png", dpi=150)
    plt.close(fig)
 
    return summary

def step4_10_check(summary: pd.DataFrame) -> None:
    # 確認1:形
    assert len(summary) == 36
    assert (summary["n_rows"] == 500).all()

    # 確認2:kNN 側が Step 3 の集計と一致
    s3 = pd.read_csv(RESULTS / "step3_12_summary.csv")
    knn = summary[summary["graph"] == "knn"].merge(s3, on=["M", "ef"], suffixes=("", "_3"))
    assert len(knn) == 18
    for col in ["recall10", "recall1", "n_dist"]:
        assert np.allclose(knn[col], knn[col + "_3"])

    # 確認3:NSW の1行を別に計算
    raw = pd.read_csv(RESULTS / "step4_9_beam_M_ef.csv")
    one = raw[(raw["M"] == 10) & (raw["ef"] == 10)]
    row = summary[(summary["graph"] == "nsw") & (summary["M"] == 10) & (summary["ef"] == 10)].iloc[0]
    assert len(one) == 500
    assert np.isclose(one["recall10"].mean(), row["recall10"])
    assert np.isclose(one["n_dist"].mean(), row["n_dist"])

    # 確認4:ef が大きいほど n_dist が増える
    for (g, M), sub in summary.groupby(["graph", "M"]):
        assert sub.sort_values("ef")["n_dist"].is_monotonic_increasing

    # 確認5:補間に測った点を入れると、測った recall が返る
    sub = summary[(summary["graph"] == "knn") & (summary["M"] == 10)]
    r = sub[sub["ef"] == 64].iloc[0]
    assert np.isclose(interp_recall(sub, r["n_dist"]), r["recall10"])

    print("4-10 の確認:すべて通過")

def step4_11_review(save: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4-11: NSW で far の割合と入口シード間の recall の幅が ef でどう変わるか"""
    df = pd.read_csv(RESULTS / "step4_9_beam_M_ef.csv")  # 変更:NSW の csv を読む

    # 1. 一番近い点の本当の順位で4種類に分ける(Step 2-7・3-14 と同じ区切り)
    df["kind"] = pd.cut(df["rank_top1"], bins=[0, 1, 10, 100, np.inf],
                        labels=["hit", "near", "mid", "far"])
    kinds = (pd.crosstab([df["M"], df["ef"]], df["kind"])
             .reindex(columns=["hit", "near", "mid", "far"], fill_value=0)
             .reset_index())
    print(kinds.to_string(index=False))

    # 2. シードごとに100クエリ平均 → 5シードの最小〜最大と幅
    per_seed = (df.groupby(["M", "ef", "seed"])
                .agg(recall1=("recall1", "mean"), recall10=("recall10", "mean"))
                .reset_index())
    width = (per_seed.groupby(["M", "ef"])
             .agg(r1_min=("recall1", "min"), r1_max=("recall1", "max"),
                  r10_min=("recall10", "min"), r10_max=("recall10", "max"))
             .reset_index())
    width["r1_width"] = width["r1_max"] - width["r1_min"]
    width["r10_width"] = width["r10_max"] - width["r10_min"]
    print(width.round(3).to_string(index=False))

    if save:
        kinds.to_csv(RESULTS / "step4_11_kinds.csv", index=False)  # 変更:保存先
        width.to_csv(RESULTS / "step4_11_seed_width.csv", index=False)  # 変更:保存先
    return kinds, width

def step4_11_hard_queries(save: bool) -> pd.DataFrame:
    """4-11 NSW でクエリごとに、5入口中何回 far だったかを数える"""
    df = pd.read_csv(RESULTS / "step4_9_beam_M_ef.csv")  # 変更:NSW の csv を読む
    df["far"] = df["rank_top1"] > 100

    # (M, ef, クエリ) ごとに far の回数(0〜5)
    per_q = (df.groupby(["M", "ef", "query"])["far"].sum()
             .reset_index(name="n_far"))

    # (M, ef) ごとに、far 0回・1回・…・5回のクエリがそれぞれ何個あるか
    dist = (pd.crosstab([per_q["M"], per_q["ef"]], per_q["n_far"])
            .reindex(columns=range(6), fill_value=0)
            .reset_index())
    print(dist.to_string(index=False))

    if save:
        per_q.to_csv(RESULTS / "step4_11_far_per_query.csv", index=False)  # 変更:保存先
        dist.to_csv(RESULTS / "step4_11_far_dist.csv", index=False)  # 変更:保存先
    return dist

def step4_11_check(kinds: pd.DataFrame, dist: pd.DataFrame) -> None:
    s = pd.read_csv(RESULTS / "step4_10_summary.csv")
    s = s[s["graph"] == "nsw"]

    # 確認1:各 (M, ef) で4種類の合計が 500
    assert len(kinds) == 18
    assert (kinds[["hit", "near", "mid", "far"]].sum(axis=1) == 500).all()

    # 確認2:hit ÷ 500 が 4-10 の recall@1 と一致
    m = kinds.merge(s, on=["M", "ef"])
    assert len(m) == 18
    assert np.allclose(m["hit"] / 500, m["recall1"])

    # 確認3:クエリごとの表は各行の合計が 100、far の回数 × 個数の合計が far の総数
    cols = list(range(6))
    assert len(dist) == 18
    assert (dist[cols].sum(axis=1) == 100).all()
    m2 = dist.merge(kinds[["M", "ef", "far"]], on=["M", "ef"])
    assert ((m2[cols] * np.arange(6)).sum(axis=1) == m2["far"]).all()

    # 確認4:NSW M=10・ef=256 はすべて hit
    row = kinds[(kinds["M"] == 10) & (kinds["ef"] == 256)].iloc[0]
    assert row["hit"] == 500

    print("4-11 の確認:すべて通過")


if __name__ == "__main__":
    kinds, width = step4_11_review(save=True)
    dist = step4_11_hard_queries(save=True)