import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import result_path

GRAPH_NAMES = ["nsw", "cap", "layered", "hnsw"]  # グラフ1〜4
GRAPH_LABELS = {"nsw": "1 NSW", "cap": "2 上限だけ", "layered": "3 階層あり", "hnsw": "4 HNSW"}


def step5_10_summary(save: bool = True) -> pd.DataFrame:
    """5-10 の csv を読み、グラフ・M ごとの層0の次数と届く点の数をまとめる。探索はしない"""
    deg = pd.read_csv(result_path("step5_10_degree.csv"))
    reach = pd.concat([pd.read_csv(result_path(f"step5_10_reachable_M{M}.csv")) for M in (10, 20, 40)])

    rows = []
    for (M, name), d in deg.groupby(["M", "graph"]):
        r = reach[(reach["M"] == M) & (reach["graph"] == name)]
        rand = r[r["entry_kind"] == "random"]["n_reachable"]
        rows.append({
            "graph": name,
            "M": int(M),
            "deg_min": int(d["degree"].min()),
            "deg_mean": float(d["degree"].mean()),
            "deg_max": int(d["degree"].max()),
            "at_cap_ratio": float((d["degree"] == 2 * M).mean()) if name != "nsw" else np.nan,  # 上限 2M ちょうどの点の割合
            "indeg0": int((d["indegree"] == 0).sum()),  # 誰からも指されていない点
            "reach_fixed": int(r[r["entry_kind"] == "fixed"]["n_reachable"].iloc[0]),
            "reach_random_min": int(rand.min()),
            "reach_random_max": int(rand.max()),
            "random_all_ratio": float((rand == len(d)).mean()),  # ランダムな入口のうち全点に届く割合
        })
    table = pd.DataFrame(rows)
    table["graph"] = pd.Categorical(table["graph"], GRAPH_NAMES)
    table = table.sort_values(["M", "graph"]).reset_index(drop=True)
    step5_10_summary_check(table, deg)
    print(table.to_string())

    if save:
        table.to_csv(result_path("step5_10_summary.csv"), index=False)
        # 層0の次数のヒストグラム(M=10、グラフ1 は 2M=20 を超える点もあるので 60 で切る)
        plt.rcParams["font.family"] = ["Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
        for ax, name in zip(axes, GRAPH_NAMES):
            d = deg[(deg["M"] == 10) & (deg["graph"] == name)]["degree"]
            ax.hist(d.clip(upper=60), bins=np.arange(0, 62), color="gray")
            ax.axvline(20, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"{GRAPH_LABELS[name]}(M=10)")
            ax.set_xlabel("層0の次数(60以上は60)")
        axes[0].set_ylabel("点の数")
        fig.tight_layout()
        fig.savefig(result_path("step5_10_degree_hist.png"), dpi=120)
    return table


def step5_10_summary_check(table: pd.DataFrame, deg: pd.DataFrame) -> None:
    for r in table.itertuples():
        if r.graph != "nsw":
            assert r.deg_max <= 2 * r.M
        assert r.reach_random_max <= 10000 and r.reach_fixed <= 10000
        # 被次数0の点が k 個あれば、その点以外から入ると最大でも 10000 - k + 1 点にしか届かない
        assert r.reach_random_min <= 10000 - r.indeg0 + (1 if r.indeg0 else 0)
    # 平均次数は 5-8 の layers.csv(層0)と一致する
    lay = pd.read_csv(result_path("step5_8_layers.csv"))
    lay0 = lay[lay["layer"] == 0]
    for r in lay0.itertuples():
        t = table[(table["graph"] == r.graph) & (table["M"] == r.M)]
        assert np.isclose(t["deg_mean"].iloc[0], r.avg_deg)
    print("5-10 集計の確認: OK")


if __name__ == "__main__":
    step5_10_summary()
