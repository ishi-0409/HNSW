import numpy as np
from common import brute_force


def build_knn_graph(X, M):
    N = len(X)
    graph = np.empty((N, M), dtype=np.int32)
    for i in range(N):
        idx = brute_force(X, X[i], M + 1)
        graph[i] = idx[idx != i][:M]
        if i % 1000 == 0:
            print(i)
    return graph

