import numpy as np

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

