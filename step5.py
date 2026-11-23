import numpy as np
import csv

from common import result_path
from hnsw import assign_levels, insert_point_hnsw, select_neighbors_simple, shrink_neighbors


def step5_1_check_levels():
    n, M = 10000, 10
    mL = 1 / np.log(M)
    levels = assign_levels(n, mL, seed=0)

    print(f"n={n}, M={M}, mL={mL:.4f}")
    rows = []
    for k in range(1, levels.max() + 1):
        count = int((levels >= k).sum())
        expected = n / M**k
        rows.append({"k": k, "count": count, "expected": expected})
        print(f"  レベル>={k}: {count:5d} 点(期待値 {n / M**k:.1f})")
    print(f"  最大レベル: {levels.max()}")

    path = result_path("step5_1_levels.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "count", "expected"])
        writer.writeheader()
        writer.writerows(rows)
    
    # mL=0 なら全点がレベル0
    zero = assign_levels(n, 0.0, seed=0)
    assert (zero == 0).all()

    # 同じ seed なら同じ結果
    assert np.array_equal(levels, assign_levels(n, mL, seed=0))
    print("mL=0 と再現性の確認: OK")


def step5_2_check_select():
    # 数直線の5点。番号 0〜4 が x = 0, 1, 3, 4, 10
    X = np.array([[0], [1], [3], [4], [10]], dtype=np.float32)

    # 選ぶ:候補は近い順に並んでいる前提
    assert select_neighbors_simple([3, 1, 0, 4], 2) == [3, 1]
    assert select_neighbors_simple([3, 1, 0, 4], 10) == [3, 1, 0, 4]

    # 削る:点2(x=3)から 0,1,3,4 への距離は 3,2,1,7 → 近い2本は 3,1
    graph = [[] for _ in range(5)]
    graph[2] = [0, 1, 3, 4]
    n_dist = shrink_neighbors(X, graph, 2, 2)
    print(f"削った後: graph[2]={graph[2]}, n_dist={n_dist}")
    assert graph[2] == [3, 1]
    assert n_dist == 4

    # 上限以下なら何もしない
    graph[2] = [1, 3]
    n_dist = shrink_neighbors(X, graph, 2, 2)
    assert graph[2] == [1, 3]
    assert n_dist == 0
    print("5-2 の確認: OK")


def step5_3_check_insert():
    # 数直線の5点。番号 0〜4 が x = 0, 1, 3, 4, 10。レベルは手で決める
    X = np.array([[0], [1], [3], [4], [10]], dtype=np.float32)
    levels = np.array([1, 0, 0, 2, 0])

    # (Mmax0, 期待する層0, 期待する層1, 1点ごとの距離計算回数)。M=1、ef_construction=2
    cases = [
        (2, [[1], [0, 2], [1, 3], [2, 4], [3]], [[3], [], [], [0], []], [2, 3, 4, 6]),
        (1, [[1], [0], [1], [1], [3]], [[3], [], [], [0], []], [2, 5, 5, 8]),
    ]
    for Mmax0, want0, want1, want_n in cases:
        layers = [[[] for _ in range(5)] for _ in range(levels.max() + 1)]
        entry = 0  # 最初の点は入れるだけ(辺なし)
        n_list = []
        for idx in range(1, 5):
            entry, n_dist = insert_point_hnsw(X, layers, levels, idx, entry, 1, Mmax0, 2)
            n_list.append(n_dist)
        print(f"Mmax0={Mmax0}: 層0={layers[0]}, 層1={layers[1]}, 入口={entry}, n_dist={n_list}")
        assert layers[0] == want0
        assert layers[1] == want1
        assert layers[2] == [[] for _ in range(5)]
        assert entry == 3
        assert n_list == want_n
    print("5-3 の確認: OK")


if __name__ == "__main__":
    step5_1_check_levels()