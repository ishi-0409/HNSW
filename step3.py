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

if __name__ == "__main__":
    step3_1_check_init()
