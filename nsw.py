import numpy as np

def init_nsw(n: int, seed: int) -> tuple[list[list[int]], np.ndarray]:
    "空の近所リストとランダムな挿入順を返す"
    graph: list[list[int]] = [[] for _ in range(n)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return graph, order

