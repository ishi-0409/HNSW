import numpy as np
from common import brute_force
import heapq

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

# ビーム探索
def search_layer(X, graph, q, entries, ef) -> dict:
    """
    1つの層でビーム探索をして、クエリに近い点を最大ef個返す。
    ef=1のときは貪欲法と同じになる。

    Args:
        X: データ(N, d)
        graph: 近傍リスト(N, M)
        q: クエリ(d,)
        entries: 入口の点番号
        ef: 保持する候補の数
    Returns:
        topk: クエリに近い順の点番号(list[int])
        n_dist: 距離計算回数(int)
        n_expand: 近所を調べた点の数(int)
    """
    
    d0 = np.linalg.norm(X[entries] - q, axis=1)
    n_dist = len(entries)
    visited = set(entries)  #　距離を測った点
    C = [(d, e) for d, e in zip(d0, entries)]   # 候補   
    W = [(-d, e) for d, e in zip(d0, entries)]  # 結果
    heapq.heapify(C)
    heapq.heapify(W)

    # ループの停止判断
    n_expand = 0
    while C:
        d_c, c = heapq.heappop(C)
        d_f = -W[0][0]
        if d_c > d_f:
            break
        n_expand += 1

        # cの近所のうち未計測の点だけ距離を計算する
        ids = [int(e) for e in graph[c] if e not in visited]
        visited.update(ids)
        if not ids:
            continue
        dists = np.linalg.norm(X[ids] - q, axis=1)
        n_dist += len(ids)
        print(c, ids, dists)


    return {"topk": [], "n_dist": n_dist, "n_expand": n_expand}

