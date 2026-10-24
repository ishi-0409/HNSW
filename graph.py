import numpy as np
from common import brute_force


def build_knn_graph(X, k):
    N = len(X)
    graph = np.empty((N, k), dtype=np.int32)
    for i in range(N):
        idx = brute_force(X, X[i], k + 1)
        graph[i] = idx[idx != i][:k]
        if i % 1000 == 0:
            print(i)
    return graph

