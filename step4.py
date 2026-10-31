import numpy as np
from common import load_sift
from nsw import init_nsw

def step4_1_check_init():
    X, Q, GT = load_sift()
    n = len(X)
    graph, order = init_nsw(n, seed=0)

    #　器: n個のリスト
    assert len(graph) == n
    assert all(len(nb) == 0 for nb in graph)
    assert len({id(nb) for nb in graph}) == n

    # 挿入順
    assert len(order) == n
    assert np.array_equal(np.sort(order), np.arange(n))
    assert not np.array_equal(order, np.arange(n))

    # 再現性
    _, order2 = init_nsw(n, seed=0)
    assert np.array_equal(order, order2)

    print("orderの先頭5つ", order[:5])
    print("OK")