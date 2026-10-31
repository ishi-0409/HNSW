import numpy as np
from common import load_sift
from nsw import init_nsw, add_edge, insert_first_two, insert_point

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

def step4_3_check_line():
    X = np.array([[0.0], [10.0], [4.0], [7.0], [2.0]])
    names = ["A", "B", "C", "D", "E"]
    order = np.array([0, 1, 2, 3, 4])
    M = 2
    ef_construction = 10

    graph: list[list[int]] = [[] for _ in range(len(X))]
    entry = insert_first_two(graph, order)   # A と B
    for idx in order[2:]:                    # C, D, E
        insert_point(X, graph, int(idx), entry, M, ef_construction)

    for i, nb in enumerate(graph):
        print(names[i], "->", [names[j] for j in sorted(nb)])

    expected: list[list[int]] = [
        [1, 2, 4],  # A
        [0, 2, 3],  # B
        [0, 1, 3, 4],  # C
        [1, 2],  # D
        [0, 2],  # E
    ]
    assert [sorted(nb) for nb in graph] == expected
    print("4-3 OK")

        # 2点目を一般の手順で入れても insert_first_two と同じになるか
    g1: list[list[int]] = [[] for _ in range(len(X))]
    insert_first_two(g1, order)
    g2: list[list[int]] = [[] for _ in range(len(X))]
    insert_point(X, g2, int(order[1]), int(order[0]), M, ef_construction)
    assert g1 == g2
    print("2点目の検算 OK")