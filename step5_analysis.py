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


KINDS = ["hit", "near", "mid", "far"]


def add_kind(df: pd.DataFrame) -> pd.DataFrame:
    """一番近い点の本当の順位で4種類に分ける(2-7・3-14・4-11 と同じ区切り)"""
    df = df.copy()
    df["kind"] = pd.cut(df["rank_top1"], bins=[0, 1, 10, 100, np.inf], labels=KINDS)
    return df


def step5_13_kinds(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """グラフ・探し方・M・ef ごとの hit/near/mid/far と、同じ距離計算回数での far の割合"""
    df = add_kind(load_step5_11())
    kinds = (pd.crosstab([df["graph"], df["mode"], df["M"], df["ef"]], df["kind"])
             .reindex(columns=KINDS, fill_value=0).reset_index())
    kinds.columns.name = None
    kinds["n_rows"] = kinds[KINDS].sum(axis=1)
    kinds["far_ratio"] = kinds["far"] / kinds["n_rows"]
    nd = df.groupby(CURVE_KEYS)["n_dist"].mean().reset_index()
    kinds = kinds.merge(nd, on=CURVE_KEYS)

    interp_rows = []
    for (g, mode, M), sub in kinds.groupby(["graph", "mode", "M"]):
        sub = sub.rename(columns={"far_ratio": "recall10"})  # interp_recall は recall10 列を補間する
        for x in TARGETS:
            interp_rows.append({"graph": g, "mode": mode, "M": M, "n_dist": x,
                                "far_ratio": interp_recall(sub, x)})
    interp = pd.DataFrame(interp_rows)

    step5_13_kinds_check(kinds)
    print(interp.pivot(index=["graph", "mode"], columns=["M", "n_dist"], values="far_ratio").round(3).to_string())
    if save:
        kinds.to_csv(result_path("step5_13_kinds.csv"), index=False)
        interp.to_csv(result_path("step5_13_far_interp.csv"), index=False)
    return kinds, interp


def step5_13_kinds_check(kinds: pd.DataFrame) -> None:
    # hit の割合は 5-12 の recall@1 と一致する
    s12 = pd.read_csv(result_path("step5_12_summary.csv"))
    m = kinds.merge(s12, on=CURVE_KEYS, suffixes=("", "_12"))
    assert len(m) == len(kinds) == len(s12)
    assert (m["n_rows"] == m["n_rows_12"]).all()
    assert np.allclose(m["hit"] / m["n_rows"], m["recall1"])
    assert np.allclose(m["n_dist"], m["n_dist_12"])
    # NSW は 4-11 と一致する
    k4 = pd.read_csv(result_path("step4_11_kinds.csv"))
    n = kinds[kinds["graph"] == "nsw"].merge(k4, on=["M", "ef"], suffixes=("", "_4"))
    assert len(n) == 18
    for c in KINDS:
        assert (n[c] == n[c + "_4"]).all()
    print("5-13 種類の確認: OK")


def step5_13_per_query(save: bool = True) -> pd.DataFrame:
    """入口が5つある探し方(NSW・グラフ2・D)で、クエリごとに far が何回だったか。
    全入口で far(5回)と入口しだい(1〜4回)のクエリ数を数える"""
    df = load_step5_11()
    df = df[df["mode"] != "fixed"].copy()
    df["far"] = df["rank_top1"] > 100
    per_q = df.groupby(["graph", "mode", "M", "ef", "query"])["far"].sum().reset_index(name="n_far")
    dist = (pd.crosstab([per_q["graph"], per_q["mode"], per_q["M"], per_q["ef"]], per_q["n_far"])
            .reindex(columns=range(6), fill_value=0).reset_index())
    dist.columns = [str(c) for c in dist.columns]
    dist["all_far"] = dist["5"]  # 全入口で far
    dist["depends"] = dist[["1", "2", "3", "4"]].sum(axis=1)  # 入口しだい

    # A(固定の入口、1クエリ1回)は far のクエリ数だけ
    a = load_step5_11()
    a = a[a["mode"] == "fixed"]
    fixed = (a.assign(far=a["rank_top1"] > 100).groupby(["graph", "mode", "M", "ef"])["far"].sum()
             .reset_index(name="far_queries"))

    # 確認:各行の合計が100クエリ。NSW は 4-11 と一致
    assert (dist[[str(i) for i in range(6)]].sum(axis=1) == 100).all()
    d4 = pd.read_csv(result_path("step4_11_far_dist.csv"))
    n = dist[dist["graph"] == "nsw"].merge(d4, on=["M", "ef"], suffixes=("", "_4"))
    assert len(n) == 18 and all((n[str(i)] == n[f"{i}_4"]).all() for i in range(6))
    print("5-13 クエリごとの確認: OK")

    cols = ["graph", "mode", "M", "ef", "all_far", "depends"]
    print(dist[cols].to_string(index=False))
    print(fixed.to_string(index=False))
    if save:
        dist.to_csv(result_path("step5_13_far_dist.csv"), index=False)
        fixed.to_csv(result_path("step5_13_far_fixed.csv"), index=False)
    return dist


def step5_13_closed_set_summary(M: int = 10, save: bool = True) -> pd.DataFrame:
    """グラフ3 M の A を「降りた場所が S の中か」×「本当の1位が S の中か」に分け、far の割合を出す"""
    pts = pd.read_csv(result_path(f"step5_13_closed_set_points_M{M}.csv"))
    qs = pd.read_csv(result_path(f"step5_13_closed_set_queries_M{M}.csv"))
    df = pd.read_csv(result_path(f"step5_11_search_layered_M{M}.csv"))
    df = df[df["mode"] == "fixed"].merge(qs, on="query")
    df["start_in_S"] = pts.set_index("point").loc[df["start0"], "in_S"].to_numpy()
    df["far"] = df["rank_top1"] > 100
    # S の中に降りると S の外には出られないので、recall@10 の上限は「本当の上位10点のうち S の中の割合」
    df["ceiling"] = np.where(df["start_in_S"] == 1, df["gt10_in_S"] / 10, 1.0)
    assert (df["recall10"] <= df["ceiling"] + 1e-12).all()
    table = (df.groupby(["ef", "start_in_S", "gt1_in_S"])
               .agg(n_queries=("query", "size"), far=("far", "sum"), recall10=("recall10", "mean"),
                    ceiling=("ceiling", "mean"))
               .reset_index())
    # 確認:各 ef で4通りの合計が100クエリ
    assert (table.groupby("ef")["n_queries"].sum() == 100).all()
    print(table.round(3).to_string(index=False))
    if save:
        table.to_csv(result_path(f"step5_13_closed_set_M{M}.csv"), index=False)
    return table


def step5_14_decompose(save: bool = True) -> pd.DataFrame:
    """層0で1点を調べるごとに新しく測った数を deg_mean × new_ratio に分ける(4-15 と同じ)。探索はしない"""
    df = pd.concat([pd.read_csv(result_path(f"step5_11_search_{n}_M{M}.csv"))
                    for n in ("cap", "layered", "hnsw") for M in (10, 20, 40)])
    nsw = pd.read_csv(result_path("step4_15_deg_sum.csv"))
    nsw = nsw[nsw["graph"] == "nsw"].assign(mode="random", n_dist0=lambda d: d["n_dist"])  # NSW は層0だけ
    df = pd.concat([nsw, df], ignore_index=True)

    t = (df.groupby(CURVE_KEYS)
           .agg(n_dist=("n_dist", "mean"), n_dist0=("n_dist0", "mean"),
                n_expand=("n_expand", "mean"), deg_sum=("deg_sum", "mean"))
           .reset_index())
    t["upper"] = t["n_dist"] - t["n_dist0"]  # 上の層での距離計算回数(A だけ 0 でない)
    t["per_expand"] = (t["n_dist0"] - 1) / t["n_expand"]
    t["deg_mean"] = t["deg_sum"] / t["n_expand"]
    t["new_ratio"] = (t["n_dist0"] - 1) / t["deg_sum"]
    # 層0の平均次数(5-10)と、それに対する deg_mean の比
    s10 = pd.read_csv(result_path("step5_10_summary.csv"))[["graph", "M", "deg_mean", "deg_max"]]
    t = t.merge(s10.rename(columns={"deg_mean": "avg_deg"}), on=["graph", "M"], how="left")
    t["deg_over_avg"] = t["deg_mean"] / t["avg_deg"]

    step5_14_check(t)
    cols = ["graph", "mode", "M", "ef", "n_dist", "upper", "n_expand", "per_expand", "deg_mean", "avg_deg", "new_ratio"]
    print(t[t["ef"] == 64][cols].round(3).to_string(index=False))
    if save:
        t.to_csv(result_path("step5_14_decompose.csv"), index=False)
    return t


def step5_14_check(t: pd.DataFrame) -> None:
    assert np.allclose(t["per_expand"], t["deg_mean"] * t["new_ratio"])
    assert (t[t["graph"] != "nsw"]["deg_mean"] <= 2 * t[t["graph"] != "nsw"]["M"]).all()
    assert (t[t["mode"] != "fixed"]["upper"] == 0).all()
    # NSW は 4-15 と一致
    s4 = pd.read_csv(result_path("step4_15_summary.csv"))
    s4 = s4[s4["graph"] == "nsw"].iloc[0]
    n = t[t["graph"] == "nsw"].iloc[0]
    assert np.isclose(n["deg_mean"], s4["deg_mean"]) and np.isclose(n["new_ratio"], s4["new_ratio"])
    # n_dist は 5-12 と一致
    s12 = pd.read_csv(result_path("step5_12_summary.csv"))
    m = t.merge(s12, on=CURVE_KEYS, suffixes=("", "_12"))
    assert len(m) == len(t) and np.allclose(m["n_dist"], m["n_dist_12"])
    print("5-14 の確認: OK")


if __name__ == "__main__":
    step5_10_summary()
    step5_12_curves()
    step5_13_kinds()
    step5_13_per_query()
    step5_13_closed_set_summary()
    step5_14_decompose()
