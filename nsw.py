import numpy as np
from pathlib import Path
from graph import SearchResult, search_layer

def init_nsw(n: int, seed: int) -> tuple[list[list[int]], np.ndarray]:
    "空の近所リストとランダムな挿入順を返す"
    graph: list[list[int]] = [[] for _ in range(n)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return graph, order

def add_edge(graph: list[list[int]], a: int, b:int):
    "点aと点bを双方向につなぐ"
    graph[a].append(int(b))
    graph[b].append(int(a))


def insert_first_two(graph: list[list[int]], order: np.ndarray):
    "最初の2点を挿入し、探索の入口を返す"
    first = int(order[0])
    second = int(order[1])
    add_edge(graph, first, second)
    return first

def insert_point(
        X: np.ndarray,
        graph: list[list[int]],
        idx: int,
        entry: int,
        M: int,
        ef_construction: int,
        ) -> int:
    "点idxをグラフに入れ、距離計算回数を返す。"
    q: np.ndarray = X[idx]
    res: SearchResult = search_layer(X, graph, q, [entry], ef_construction)
    neighbors: list[int] = res["topk"][:M]

    for nb in neighbors:
        add_edge(graph, idx, nb)

    return res["n_dist"]

def build_nsw(
    X: np.ndarray,
    M: int,
    ef_construction: int,
    seed: int,
) -> tuple[list[list[int]], np.ndarray, int]:
    "Xの全点をseedでシャッフルした順に入れ、NSWを作る"
    graph , order = init_nsw(len(X),seed)
    entry: int = insert_first_two(graph, order)
    n_dist_total: int = 0
    for idx in order[2:]:
        n_dist_total += insert_point(X, graph, int(idx), entry, M, ef_construction)

    return graph, order, n_dist_total

def save_nsw(path: Path, graph: list[list[int]], order: np.ndarray) -> None:
    """グラフを CSR 形式(offsets, neighbors)にして、order と一緒に .npz に保存する。"""
    degs = np.array([len(nb) for nb in graph], dtype=np.int64)
    offsets = np.zeros(len(graph) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(degs)
    neighbors = np.array([j for nb in graph for j in nb], dtype=np.int32)
    np.savez(path, offsets=offsets, neighbors=neighbors, order=order)


def load_nsw(path: Path) -> tuple[list[list[int]], np.ndarray]:
    """save_nsw で保存したファイルから (グラフ, 挿入順) を読み戻す。"""
    data = np.load(path)
    offsets = data["offsets"]
    neighbors = data["neighbors"]
    graph = [neighbors[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    return graph, data["order"]