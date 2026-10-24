import numpy as np
from common import brute_force

# knnグラフ構築
def build_knn_graph(X, M):
    N = len(X)
    graph = np.empty((N, M), dtype=np.int32)
    for i in range(N):
        idx = brute_force(X, X[i], M + 1)
        graph[i] = idx[idx != i][:M]
        if i % 1000 == 0:
            print(i)
    return graph

# 貪欲法
def greedy_search(X, graph, q, entry):
    cur = entry
    cur_d = np.linalg.norm(X[cur] - q)
    n_dist = 1
    visited = {cur}
    hops = 0
    while True:
        ids = [i for i in graph[cur] if i not in visited]
        if len(ids) == 0:
            break
        dists = np.linalg.norm(X[ids] - q, axis=1)
        n_dist += len(ids)
        visited.update(ids)
        best = np.argmin(dists)
        best_id = ids[best]
        best_d = dists[best]
        if best_d < cur_d:
            cur, cur_d = best_id, best_d
            hops += 1
        else:
            break
    return cur, hops, n_dist


