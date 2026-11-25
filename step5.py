import numpy as np
import csv

from common import brute_force, make_uniform, recall_at_k, result_path
from hnsw import (
    assign_levels,
    build_hnsw,
    insert_point_hnsw,
    search_hnsw,
    select_neighbors_simple,
    shrink_neighbors,
)


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


def step5_4_check_build():
    n, d, M = 1000, 8, 5
    X = make_uniform(n, d, np.random.default_rng(0))
    mL = 1 / np.log(M)
    layers, levels, entry, n_dist = build_hnsw(X, M, 50, mL, 2 * M, seed=0)
    print(f"層の数={len(layers)}, 入口={entry}(レベル{levels[entry]}), 構築の距離計算回数={n_dist}")

    for l, graph in enumerate(layers):
        Mmax = 2 * M if l == 0 else M
        degs = [len(nb) for nb in graph]
        print(f"  層{l}: 点数={int((levels >= l).sum())}, 最大次数={max(degs)}(上限 {Mmax})")
        pointed = set()
        for i, nb in enumerate(graph):
            assert i not in nb             # 自分への辺がない
            assert len(nb) == len(set(nb))  # 重複がない
            assert len(nb) <= Mmax          # 次数が上限以下
            if levels[i] < l:
                assert nb == []             # 層 l にいない点は近所が空
            pointed.update(nb)
        assert all(levels[j] >= l for j in pointed)  # 層 l の辺はレベル l 以上の点にしか向かない

    assert levels[entry] == levels.max()  # 入口がいちばん高い

    # 同じ seed なら同じ結果
    layers2, _, entry2, n_dist2 = build_hnsw(X, M, 50, mL, 2 * M, seed=0)
    assert layers2 == layers and entry2 == entry and n_dist2 == n_dist
    print("5-4 の確認: OK")


def step5_5_check_search():
    # 手計算:5-3 の Mmax0=2 のグラフ(数直線の5点、入口は点3)で q=9 を探す
    X = np.array([[0], [1], [3], [4], [10]], dtype=np.float32)
    levels = np.array([1, 0, 0, 2, 0])
    layers = [[[] for _ in range(5)] for _ in range(3)]
    entry = 0
    for idx in range(1, 5):
        entry, _ = insert_point_hnsw(X, layers, levels, idx, entry, 1, 2, 2)
    res = search_hnsw(X, layers, levels, entry, np.array([9], dtype=np.float32), k=2, ef=2)
    print(f"手計算の例: {res}")
    assert res == {"topk": [4, 3], "n_dist": 6, "n_dist0": 3, "n_expand": 2, "deg_sum": 3}

    # 動作確認:5-4 と同じ1000点で、別に作った100クエリの recall@10 を見る
    n, d, M, k = 1000, 8, 5, 10
    X = make_uniform(n, d, np.random.default_rng(0))
    Q = make_uniform(100, d, np.random.default_rng(1))
    layers, levels, entry, _ = build_hnsw(X, M, 50, 1 / np.log(M), 2 * M, seed=0)
    recalls = []
    for q in Q:
        res = search_hnsw(X, layers, levels, entry, q, k, ef=100)
        recalls.append(recall_at_k(res["topk"], brute_force(X, q, k), k))
    print(f"ランダム1000点・100クエリ(ef=100)の recall@10 平均: {np.mean(recalls):.3f}")
    assert np.mean(recalls) > 0.9
    print("5-5 の確認: OK")


if __name__ == "__main__":
    step5_1_check_levels()