import numpy as np
from common import brute_force

# knnグラフ構築
def build_knn_graph(X: np.ndarray, M: int) -> np.ndarray:
    N = len(X)
    graph = np.empty((N, M), dtype=np.int32)
    for i in range(N):
        idx = brute_force(X, X[i], M + 1)
        graph[i] = idx[idx != i][:M]
        if i % 1000 == 0:
            print(i)
    return graph

# 貪欲法
def greedy_search(X: np.ndarray, graph: np.ndarray, q: np.ndarray, entry: int, k: int = 10) -> dict:
    cur = entry
    cur_d = np.linalg.norm(X[cur] - q)
    n_dist = 1
    seen = {cur: cur_d}
    hops = 0
    path = [(cur, cur_d)]
    while True:
        ids = [i for i in graph[cur] if i not in seen]
        if len(ids) == 0:
            break
        dists = np.linalg.norm(X[ids] - q, axis=1)
        n_dist += len(ids)
        seen.update(zip(ids, dists))
        best = np.argmin(dists)
        best_id = ids[best]
        best_d = dists[best]
        if best_d < cur_d:
            cur, cur_d = best_id, best_d
            path.append((cur, cur_d))
            hops += 1
        else:
            break
    pairs = sorted(seen.items(), key=lambda x: x[1])
    pairs = pairs[:k]
    topk = [int(j) for j, d in pairs]
    return {"stop": cur, "hops": hops, "n_dist": n_dist, "path": path, "topk": topk}



