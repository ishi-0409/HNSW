import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_sift, brute_force, recall_at_k
from graph import build_knn_graph, greedy_search

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

M_MAX = 40
GRAPH_PATH = DATA / f"knn_sift10k_M{M_MAX}.npy"


# ---- 2-2: kNN グラフを作って確認 ----
def step2_2_build_graph() -> None:
    X, Q, GT = load_sift()
    N = len(X)

    # knnグラフがなければ作成する
    if GRAPH_PATH.exists():
        graph = np.load(GRAPH_PATH)
    else:
        graph = build_knn_graph(X, M_MAX)
        np.save(GRAPH_PATH, graph)

    # 形と型が合っているか
    print(graph.shape == (N, M_MAX), graph.dtype == np.int32)

    # 点0の近所が全探索と一致するか
    expected = brute_force(X, X[0], 11)
    expected = expected[expected != 0][:10]
    print(np.array_equal(graph[0, :10], expected))

    # 自分自身を含まないか
    print(not np.any(graph == np.arange(N)[:, None]))


# ---- 2-2: 入次数の csv ----
def step2_2_indegree() -> None:
    X, Q, GT = load_sift()
    N = len(X)
    graph = np.load(GRAPH_PATH)

    #入次数0の点を数える
    rows = []
    for M in [5, 10, 20, 40]:
        indeg = np.bincount(graph[:, :M].ravel(), minlength=N)
        rows.append([M, (indeg == 0).sum(), int(indeg.max()), indeg.max() / M])

    # csvで保存
    with open(RESULTS / "step2_indegree.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "indeg_zero", "indeg_max", "max_over_M"])
        w.writerows(rows)


# ---- 2-3, 2-4: 貪欲法の試運転と経路表示 ----
def step2_3_4_greedy_trial() -> None:
    X, Q, GT = load_sift()
    M = 10
    g = np.load(GRAPH_PATH)[:, :M]

    # 貪欲法のテスト
    r = greedy_search(X, g, Q[0], entry=0)
    s = r["stop"]
    print("止まった点:", s, " 真の1位:", GT[0][0])
    print("ホップ数:", r["hops"], " 距離計算:", r["n_dist"])

    d_all = np.linalg.norm(X - Q[0], axis=1)
    order = np.argsort(d_all)
    print("止まった点の距離:", d_all[s], " 真の1位の距離:", d_all[GT[0][0]])
    print("止まった点の近所の最小距離:", d_all[g[s]].min())
    print("止まった点は全体で何位か:", np.where(order == s)[0][0] + 1)

    # 貪欲法の経路表示
    for p, d in r["path"]:
        print(p, d)


# ---- 2-5: M=10、入口0で100クエリ ----
def step2_5_greedy_entry0() -> None:
    X, Q, GT = load_sift()
    M = 10
    g = np.load(GRAPH_PATH)[:, :M]

    rows = []
    for qi in range(len(Q)):
        res = greedy_search(X, g, Q[qi], entry=0)
        r1 = recall_at_k([res["stop"]], GT[qi], 1)
        r10 = recall_at_k(res["topk"], GT[qi], 10)
        rows.append([qi, r1, r10, res["n_dist"], res["hops"], len(res["topk"])])

    arr = np.array(rows)          # (100, 6) の表にする
    print("recall@1 :", arr[:, 1].mean())
    print("recall@10:", arr[:, 2].mean())
    print("距離計算 :", arr[:, 3].mean())
    print("ホップ数 :", arr[:, 4].mean())

    with open(RESULTS / "step2_greedy_M10_entry0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "recall1", "recall10", "n_dist", "hops", "n_topk"])
        w.writerows(rows)


# ---- 2-6: M と入口を変えて計測 ----
def step2_6_greedy_M_entry() -> None:
    X, Q, GT = load_sift()

    Ms = [5, 10, 20, 40]
    seeds = range(5)
    full = np.load(GRAPH_PATH)
    entries = np.stack([np.random.default_rng(s).integers(0, len(X), size=len(Q))
                        for s in seeds])

    rows = []
    for M in Ms:
        graph = full[:, :M]
        for s in seeds:
            for i, q in enumerate(Q):
                entry = int(entries[s, i])
                r = greedy_search(X, graph, q, entry)
                rows.append({
                    "M": M, "seed": s, "query": i, "entry": entry,
                    "stop": r["stop"], "hops": r["hops"], "n_dist": r["n_dist"],
                    "n_topk": len(r["topk"]),
                    "recall1": recall_at_k([r["stop"]], GT[i], 1),
                    "recall10": recall_at_k(r["topk"], GT[i], 10),
                    "d_stop": np.linalg.norm(X[r["stop"]] - q),
                    "d_true1": np.linalg.norm(X[GT[i, 0]] - q),
                })

    pd.DataFrame(rows).to_csv(RESULTS / "step2_greedy_M_entry.csv", index=False)


if __name__ == "__main__":
    step2_2_build_graph()
    # step2_2_indegree()
    # step2_3_4_greedy_trial()
    # step2_5_greedy_entry0()
    # step2_6_greedy_M_entry()
