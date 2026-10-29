import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_sift, brute_force, recall_at_k
from graph import build_knn_graph, greedy_search, search_layer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def step3_1_check_init() -> None:
    X, Q, GT = load_sift()
    graph = np.load(DATA / "knn_sift10k_M40.npy")[:, :10]
    for ef in [1, 3]:
        res = search_layer(X, graph, Q[0], [0], ef=ef)
        print(ef, res)
    print(np.linalg.norm(X[0] - Q[0]))

# 貪欲法とef=1のビーム探索の出力を比較
def step3_6_compare_greedy() -> None:
    X, Q, GT = load_sift()
    graph = np.load(DATA / "knn_sift10k_M40.npy")[:, :10]
    q, entry = Q[0], 0

    g = greedy_search(X, graph, q, entry)
    s = search_layer(X, graph, q, [entry], ef=1)

    print("経路", [p for p, _ in g["path"]])
    print("停止点", g["stop"], s["topk"][0], g["stop"] == s["topk"][0])
    print("n_dist", g["n_dist"], s["n_dist"], g["n_dist"] == s["n_dist"])
    print("ホップ", g["hops"], s["n_expand"] - 1, g["hops"] == s["n_expand"] - 1)

# 2000クエリで貪欲法をef=1のビーム探索を比較
def step3_7_check_all_ef1() -> None:
    X, Q, GT = load_sift()
    full = np.load(DATA / "knn_sift10k_M40.npy")
    df2 = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")

    rows = []
    for r in df2.itertuples():
        s = search_layer(X, full[:, :r.M], Q[r.query], [int(r.entry)], ef=1)
        rows.append({
            "M": r.M, "seed": r.seed, "query": r.query, "entry": r.entry,
            "stop2": r.stop, "stop3": s["topk"][0],
            "n_dist2": r.n_dist, "n_dist3": s["n_dist"],
            "hops2": r.hops, "hops3": s["n_expand"] - 1,
        })
    df = pd.DataFrame(rows)
    df["match"] = ((df["stop2"] == df["stop3"])
                   & (df["n_dist2"] == df["n_dist3"])
                   & (df["hops2"] == df["hops3"]))
    df.to_csv(RESULTS / "step3_7_ef1_vs_greedy.csv", index=False)

    print("行", len(df), "一致", int(df["match"].sum()))
    print(df[-df["match"]])


if __name__ == "__main__":
    # step3_1_check_init()
    # step3_6_compare_greedy()
    step3_7_check_all_ef1()

