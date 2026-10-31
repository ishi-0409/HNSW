import numpy as np
from common import load_sift
from pathlib import Path
from nsw import init_nsw, add_edge, insert_first_two, insert_point, build_nsw, save_nsw, load_nsw
import time
import pandas as pd

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"

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

def check_nsw_graph(graph: list[list[int]]):
    "NSWのグラフが満たす4つの性質を確かめる"
    n = len(graph)
    indeg = np.zeros(n, dtype=int)
    for i, nb in enumerate(graph):
        assert i not in nb , f"自分への辺: {i}"
        assert len(nb) == len(set(nb)), f"重複した辺: {i}"
        for j in nb:
            assert i in graph[j], f"一方通行の辺: {i} -> {j}"
            indeg[j] += 1
    assert (indeg > 0).all(), f"入ってくる辺が0本の辺: {int((indeg == 0).sum())}"

def step4_4_check_build():
    X = np.array([[0.0], [10.0], [4.0], [7.0], [2.0]])
    graph: list[list[int]] = [[] for _ in range(len(X))]
    order = np.array([0, 1, 2, 3, 4])
    entry = insert_first_two(graph, order)
    for idx in order[2:]:
        insert_point(X, graph, int(idx), entry, 2, 10)
    check_nsw_graph(graph)
    print("OK")

    X, Q, GT = load_sift()
    X_small = X[:1000]
    graph, order, n_dist_total = build_nsw(X_small, M=10, ef_construction=100, seed=0)
    check_nsw_graph(graph)
    degs = [len(nb) for nb in graph]
    print("OK")
    print("平均次数:", np.mean(degs), "最大次数", max(degs))
    print("構築の距離計算回数", n_dist_total, "全点ペア", 1000 * 999 // 2 )
    print("OK")

def expected_degree_sum(n: int, M: int) -> int:
    """k 点目は min(M, k-1) 本の辺を張る。辺1本で次数は2増える。"""
    return 2 * sum(min(M, k - 1) for k in range(2, n + 1))


def step4_5_build_sift(
    Ms: tuple[int, ...] = (10, 20, 40),
    ef_construction: int = 100,
    seed: int = 0,
    save: bool = True,
) -> None:
    X, Q, GT = load_sift()
    n = len(X)
    rows = []
    for M in Ms:
        t0 = time.perf_counter()
        graph, order, n_dist = build_nsw(X, M, ef_construction, seed)
        build_sec = time.perf_counter() - t0

        check_nsw_graph(graph)
        degs = np.array([len(nb) for nb in graph])
        assert degs.sum() == expected_degree_sum(n, M)

        row = {
            "M": M,
            "ef_construction": ef_construction,
            "seed": seed,
            "n_dist_build": n_dist,
            "ratio_pairs": n_dist / (n * (n - 1) / 2),   # 全点ペア(全探索で挿入)との比
            "ratio_full": n_dist / (n * n),              # 全点 × 全点との比
            "avg_deg": float(degs.mean()),
            "max_deg": int(degs.max()),
            "build_sec": build_sec,                      # 参考(1回測っただけ)
        }
        print(row)
        rows.append(row)

        if save:
            path = DATA / f"nsw_sift10k_M{M}_efc{ef_construction}_seed{seed}.npz"
            save_nsw(path, graph, order)
            graph2, order2 = load_nsw(path)
            assert graph2 == graph and np.array_equal(order2, order)

    if save:
        pd.DataFrame(rows).to_csv(RESULTS / "step4_5_build.csv", index=False)

def step4_6_check_trick(M: int = 10, ef_construction: int = 100, seed: int = 0, t: int = 100) -> None:
    """graph[p] の先頭 M 個 = p が挿入時に選んだ M 点、を1点で確かめる。"""
    X, Q, GT = load_sift()
    graph_final, order = load_nsw(DATA / f"nsw_sift10k_M{M}_efc{ef_construction}_seed{seed}.npz")

    # t 番目の点まで入れたところで止めたグラフを作り直す
    graph_t, order_t = init_nsw(len(X), seed)
    assert np.array_equal(order_t, order)
    entry = insert_first_two(graph_t, order_t)
    for idx in order_t[2 : t + 1]:
        insert_point(X, graph_t, int(idx), entry, M, ef_construction)

    p = int(order[t])
    print("挿入直後の近所:", graph_t[p])
    print("保存したグラフの先頭 M 個:", graph_final[p][:M])
    assert graph_t[p] == graph_final[p][:M]
    print("4-6 の工夫 OK")


def step4_6_insert_recall(
    Ms: tuple[int, ...] = (10, 20, 40),
    ef_construction: int = 100,
    seed: int = 0,
    n_points: int = 100,
    save: bool = True,
) -> pd.DataFrame:
    X, Q, GT = load_sift()
    n = len(X)
    positions = np.linspace(100, n - 1, n_points).astype(int)   # 何番目に入れた点か
    rows = []
    for M in Ms:
        graph, order = load_nsw(DATA / f"nsw_sift10k_M{M}_efc{ef_construction}_seed{seed}.npz")
        for t in positions:
            p = int(order[t])
            before = order[:t]                                   # p より前に入っていた点
            d = np.linalg.norm(X[before] - X[p], axis=1)
            true_M = set(before[np.argsort(d)[:M]].tolist())     # その時点の本当の近い M 点
            found_M = set(graph[p][:M])                          # 挿入時に選んだ M 点
            rows.append({
                "M": M,
                "t": int(t),
                "point": p,
                "recall_M": len(true_M & found_M) / M,
            })
    df = pd.DataFrame(rows)
    print(df.groupby("M")["recall_M"].agg(["mean", "min", "count"]))
    if save:
        df.to_csv(RESULTS / "step4_6_insert_recall.csv", index=False)
    return df


def insert_recall_rows(
    X: np.ndarray, graph: list[list[int]], order: np.ndarray, M: int, positions: np.ndarray
) -> list[dict]:
    """positions 番目に入れた各点について、挿入時の recall_M を計算する。"""
    rows = []
    for t in positions:
        p = int(order[t])
        before = order[:t]
        d = np.linalg.norm(X[before] - X[p], axis=1)
        true_M = set(before[np.argsort(d)[:M]].tolist())
        found_M = set(graph[p][:M])
        rows.append({"M": M, "t": int(t), "point": p, "recall_M": len(true_M & found_M) / M})
    return rows


def step4_6_check_small_ef(M: int = 10, ef_construction: int = 10, seed: int = 0) -> None:
    """ef_construction を小さくすると recall_M が 1.0 未満になるか(測り方の確認)。保存しない。"""
    X, Q, GT = load_sift()
    graph, order, _ = build_nsw(X, M, ef_construction, seed)
    positions = np.linspace(100, len(X) - 1, 100).astype(int)
    df = pd.DataFrame(insert_recall_rows(X, graph, order, M, positions))
    print(df["recall_M"].agg(["mean", "min", "count"]))
    df.to_csv(RESULTS / "step4_6_check_small_ef.csv", index=False)

if __name__ == "__main__":
    step4_5_build_sift()