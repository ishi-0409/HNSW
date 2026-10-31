import numpy as np
from common import load_sift
from nsw import init_nsw, add_edge, insert_first_two

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

def step4_2_check_first_two():
    graph: list[list[int]] = [[] for _ in range(4)]
    order = np.array([3, 0, 2, 1])
    entry = insert_first_two(graph, order)
    expected = [[3], [], [], [0]]

    assert graph == expected, graph
    assert entry == 3

    X, Q, GT = load_sift()
    graph, order = init_nsw(len(X), seed=0)
    entry = insert_first_two(graph, order)
    o0, o1 = int(order[0]), int(order[1])

    assert entry == o0
    assert graph[o0] == [o1] and graph[o1] == [o0]
    assert sum(len(nb) for nb in graph) == 2
    assert all(type(v) is int for nb in graph for v in nb)

    print("入口", entry, "2点目", o1)
    print("OK")

    