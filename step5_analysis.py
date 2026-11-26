import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import result_path
from step4_analysis import interp_recall  # 横軸対数の直線補間(4-10 と同じ)

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


CURVE_KEYS = ["graph", "mode", "M", "ef"]
TARGETS = (400, 600, 800)  # 4-10 と同じ 400・600 に、Q12 の 800 を足す


def load_step5_11() -> pd.DataFrame:
    """5-11 の csv(グラフ2〜4)と 4-9 の csv(グラフ1 NSW)を1つにする"""
    df = pd.concat([pd.read_csv(result_path(f"step5_11_search_{n}_M{M}.csv"))
                    for n in ("cap", "layered", "hnsw") for M in (10, 20, 40)])
    nsw = pd.read_csv(result_path("step4_9_beam_M_ef.csv")).assign(graph="nsw", mode="random")
    return pd.concat([nsw, df], ignore_index=True)


def step5_12_curves(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同じ距離計算回数での recall@10 を、グラフ1〜4・探し方ごとに比べる。探索はしない"""
    df = load_step5_11()
    summary = (df.groupby(CURVE_KEYS)
                 .agg(recall10=("recall10", "mean"), recall1=("recall1", "mean"),
                      n_dist=("n_dist", "mean"), n_rows=("recall10", "size"))
                 .reset_index())

    interp_rows = []
    for (g, mode, M), sub in summary.groupby(["graph", "mode", "M"]):
        for x in TARGETS:
            interp_rows.append({"graph": g, "mode": mode, "M": M, "n_dist": x,
                                "recall10": interp_recall(sub, x)})
    interp = pd.DataFrame(interp_rows)

    # 4-12 の「最初に入れた5点」を入口にした NSW(ef=10 の1点だけ)
    first = pd.read_csv(result_path("step4_12_hub_entry.csv"))
    first = (first[(first["group"] == "first") & (first["ef"] == 10)]
             .groupby("M")[["recall10", "n_dist"]].mean().reset_index())

    step5_12_check(summary, interp)
    print(interp.pivot(index=["graph", "mode"], columns=["M", "n_dist"], values="recall10").round(3).to_string())
    print("4-12 最初の5点(NSW、ef=10):")
    print(first.round(3).to_string())

    if save:
        summary.to_csv(result_path("step5_12_summary.csv"), index=False)
        interp.to_csv(result_path("step5_12_interp.csv"), index=False)
        plt.rcParams["font.family"] = ["Yu Gothic", "Noto Sans CJK JP", "sans-serif"]
        styles = {("nsw", "random"): ("1 NSW", "-", "o"), ("cap", "random"): ("2 上限だけ", "-", "s"),
                  ("layered", "fixed"): ("3 階層あり(A)", "-", "^"), ("hnsw", "fixed"): ("4 HNSW(A)", "-", "D"),
                  ("layered", "layer0"): ("3 層0だけ(D)", "--", "^"), ("hnsw", "layer0"): ("4 層0だけ(D)", "--", "D")}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, M in zip(axes, (10, 20, 40)):
            for (g, mode), (label, ls, mk) in styles.items():
                sub = summary[(summary["graph"] == g) & (summary["mode"] == mode) & (summary["M"] == M)]
                ax.plot(sub["n_dist"], sub["recall10"], linestyle=ls, marker=mk, label=label)
            f = first[first["M"] == M]
            ax.plot(f["n_dist"], f["recall10"], "k*", markersize=10, label="1 NSW 最初の5点(ef=10)")
            ax.set_xscale("log")
            ax.set_title(f"M={M}")
            ax.set_xlabel("距離計算回数(平均、対数)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("recall@10(平均)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=9)  # 線と重ならないよう図の下に置く
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(result_path("step5_12_recall_vs_ndist.png"), dpi=120)
    return summary, interp


def step5_12_check(summary: pd.DataFrame, interp: pd.DataFrame) -> None:
    # NSW の平均は 4-10 の集計と一致する
    s4 = pd.read_csv(result_path("step4_10_summary.csv"))
    s4 = s4[s4["graph"] == "nsw"]
    m = summary[summary["graph"] == "nsw"].merge(s4, on=["M", "ef"], suffixes=("", "_4"))
    assert len(m) == 18
    for col in ["recall10", "recall1", "n_dist"]:
        assert np.allclose(m[col], m[col + "_4"])
    # 行数:NSW・グラフ2・D は 500、A は 100
    assert (summary[summary["mode"] == "fixed"]["n_rows"] == 100).all()
    assert (summary[summary["mode"] != "fixed"]["n_rows"] == 500).all()
    # ef が大きいほど n_dist が増える
    for _, sub in summary.groupby(["graph", "mode", "M"]):
        assert sub.sort_values("ef")["n_dist"].is_monotonic_increasing
    # 補間した値は両隣の ef の値のあいだにある
    for r in interp.dropna().itertuples():
        sub = summary[(summary["graph"] == r.graph) & (summary["mode"] == r.mode) & (summary["M"] == r.M)].sort_values("n_dist")
        lo = sub[sub["n_dist"] <= r.n_dist].iloc[-1]
        hi = sub[sub["n_dist"] >= r.n_dist].iloc[0]
        assert min(lo["recall10"], hi["recall10"]) - 1e-12 <= r.recall10 <= max(lo["recall10"], hi["recall10"]) + 1e-12
    print("5-12 の確認: OK")


if __name__ == "__main__":
    step5_10_summary()
    step5_12_curves()
