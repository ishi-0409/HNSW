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

if __name__ == "__main__":
    # step3_1_check_init()
    step3_6_compare_greedy()

