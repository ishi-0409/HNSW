import numpy as np
import csv
from pathlib import Path

from common import brute_force, load_sift, make_uniform, recall_at_k, result_path
from hnsw import (
    assign_levels,
    build_hnsw,
    insert_point_hnsw,
    load_hnsw,
    save_hnsw,
    search_hnsw,
    select_neighbors_heuristic,
    select_neighbors_simple,
    shrink_neighbors,
)
from nsw import load_nsw

DATA = Path(__file__).parent / "data"


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


def step5_6_check_heuristic():
    # 手計算:q=(0,0)、近い塊 A1・A2・A3、離れた点 B(論文 図2 のような配置)
    X = np.array([[0, 0], [2, 0], [2.2, 0.3], [2.1, -0.3], [-3, 0]])  # q, A1, A2, A3, B
    cands = [1, 3, 2, 4]  # q に近い順(A1, A3, A2, B)
    dists = [float(np.linalg.norm(X[c] - X[0])) for c in cands]
    assert dists == sorted(dists)

    assert select_neighbors_simple(cands, 2) == [1, 3]  # Alg. 3 は塊から2点
    # Alg. 4:A1 を選ぶ → A3・A2 は A1 のほうが近いので落とす → B を選ぶ。距離計算は A3-A1, A2-A1, B-A1 の3回
    assert select_neighbors_heuristic(X, X[0], cands, dists, 2) == ([1, 4], 3)
    # M を増やしても A3・A2 は落ちたまま(M個より少なくなる)
    assert select_neighbors_heuristic(X, X[0], cands, dists, 10) == ([1, 4], 3)

    # 削る:点0 の近所 [2, 4, 1, 3] を Mmax=2 に。e からの距離 4回 + 選ぶときの 3回
    graph = [[2, 4, 1, 3], [], [], [], []]
    assert shrink_neighbors(X, graph, 0, 2, heuristic=True) == 7
    assert graph[0] == [1, 4]
    graph = [[2, 4, 1, 3], [], [], [], []]
    assert shrink_neighbors(X, graph, 0, 2) == 4  # 既定(Alg. 3)は今までどおり
    assert graph[0] == [1, 3]
    print("手計算の例: OK")

    # 5-4 と同じ1000点で heuristic=True のグラフを作り、同じ形の確認をする
    n, d, M = 1000, 8, 5
    X = make_uniform(n, d, np.random.default_rng(0))
    mL = 1 / np.log(M)
    layers, levels, entry, n_dist = build_hnsw(X, M, 50, mL, 2 * M, seed=0, heuristic=True)
    _, _, _, n_dist_simple = build_hnsw(X, M, 50, mL, 2 * M, seed=0)
    print(f"構築の距離計算回数: Alg. 4={n_dist}, Alg. 3={n_dist_simple}")
    for l, graph in enumerate(layers):
        Mmax = 2 * M if l == 0 else M
        degs = [len(nb) for nb in graph if nb]
        print(f"  層{l}: 点数={int((levels >= l).sum())}, 最大次数={max(degs, default=0)}(上限 {Mmax}), 平均次数={np.mean(degs) if degs else 0:.2f}")
        pointed = set()
        for i, nb in enumerate(graph):
            assert i not in nb
            assert len(nb) == len(set(nb))
            assert len(nb) <= Mmax
            if levels[i] < l:
                assert nb == []
            pointed.update(nb)
        assert all(levels[j] >= l for j in pointed)
    assert levels[entry] == levels.max()
    layers2, _, entry2, n_dist2 = build_hnsw(X, M, 50, mL, 2 * M, seed=0, heuristic=True)
    assert layers2 == layers and entry2 == entry and n_dist2 == n_dist
    print("5-6 の確認: OK")


def step5_7_check_vs_nsw(Ms: tuple[int, ...] = (10, 20, 40), ef_construction: int = 100, seed: int = 0):
    # mL=0・層0の上限なし・Alg. 3・NSW と同じ順番で作ると、Step 4 の NSW と辺まで一致するはず
    X, _, _ = load_sift()
    with open(result_path("step4_5_build.csv"), encoding="utf-8") as f:
        nsw_n_dist = {int(r["M"]): int(r["n_dist_build"]) for r in csv.DictReader(f)}

    rows = []
    for M in Ms:
        nsw_graph, order = load_nsw(DATA / f"nsw_sift10k_M{M}_efc{ef_construction}_seed{seed}.npz")
        layers, levels, entry, n_dist = build_hnsw(X, M, ef_construction, 0.0, None, seed=seed, order=order)
        edges_equal = len(layers) == 1 and layers[0] == nsw_graph  # 近所リストの並び順まで比べる
        row = {
            "M": M,
            "edges_equal": edges_equal,
            "n_dist_hnsw": n_dist,
            "n_dist_nsw": nsw_n_dist[M],
            "diff": n_dist - nsw_n_dist[M],
        }
        print(row)
        rows.append(row)
        assert (levels == 0).all() and entry == int(order[0])  # 全点がレベル0、入口は最初の点のまま
        assert edges_equal
        assert row["diff"] == 1  # NSW は2点目を探さずにつなぐ(insert_first_two)ので、その1回だけ少ない

    with open(result_path("step5_7_vs_nsw.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("5-7 の確認: OK")


# 5-8 で作るグラフ:名前 → (mL を使うか, heuristic)。上限はどれも層0が 2M、層1以上が M
HNSW_VARIANTS = {
    "cap": (False, False),      # グラフ2:上限だけ(mL=0)
    "layered": (True, False),   # グラフ3:上限+層
    "hnsw": (True, True),       # グラフ4:上限+層+ヒューリスティック
}


def hnsw_path(name: str, M: int, ef_construction: int = 100) -> Path:
    return DATA / f"hnsw_sift10k_{name}_M{M}_efc{ef_construction}.npz"


def check_hnsw(layers: list[list[list[int]]], levels: np.ndarray, entry: int, M: int, Mmax0: int) -> None:
    """5-4 と同じ形の確認(自分への辺・重複なし、次数が上限以下、層の中身、入口のレベル)"""
    for l, graph in enumerate(layers):
        Mmax = Mmax0 if l == 0 else M
        pointed = set()
        for i, nb in enumerate(graph):
            assert i not in nb
            assert len(nb) == len(set(nb))
            assert len(nb) <= Mmax
            if levels[i] < l:
                assert nb == []
            pointed.update(nb)
        assert all(levels[j] >= l for j in pointed)
    assert levels[entry] == levels.max()


def step5_8_build_sift(
    Ms: tuple[int, ...] = (10, 20, 40),
    names: tuple[str, ...] = tuple(HNSW_VARIANTS),
    ef_construction: int = 100,
    seed: int = 0,
):
    """グラフ2〜4 を作って data/ に保存する。構築の距離計算回数もファイルに入れる(csv は step5_8_summary で出す)"""
    import time

    X, _, _ = load_sift()
    for M in Ms:
        for name in names:
            use_mL, heuristic = HNSW_VARIANTS[name]
            mL = 1 / np.log(M) if use_mL else 0.0
            t0 = time.perf_counter()
            layers, levels, entry, n_dist = build_hnsw(
                X, M, ef_construction, mL, 2 * M, seed=seed, heuristic=heuristic
            )
            build_sec = time.perf_counter() - t0  # 参考(画面に出すだけ)

            check_hnsw(layers, levels, entry, M, 2 * M)
            path = hnsw_path(name, M, ef_construction)
            save_hnsw(path, layers, levels, entry)
            np.save(path.with_suffix(".n_dist.npy"), np.array(n_dist))  # 構築の距離計算回数
            layers2, levels2, entry2 = load_hnsw(path)
            assert layers2 == layers and np.array_equal(levels2, levels) and entry2 == entry
            print(f"{name} M={M}: 層={len(layers)}, 入口={entry}, n_dist={n_dist}, {build_sec:.1f}秒", flush=True)


def step5_8_summary(Ms: tuple[int, ...] = (10, 20, 40), ef_construction: int = 100):
    """保存したグラフ2〜4 を読み、構築の距離計算回数と層ごとの点数・次数を csv に出す"""
    with open(result_path("step4_5_build.csv"), encoding="utf-8") as f:
        nsw_n_dist = {int(r["M"]): int(r["n_dist_build"]) for r in csv.DictReader(f)}

    build_rows, layer_rows = [], []
    for M in Ms:
        levels_by_name = {}
        for name in HNSW_VARIANTS:
            path = hnsw_path(name, M, ef_construction)
            layers, levels, entry = load_hnsw(path)
            check_hnsw(layers, levels, entry, M, 2 * M)
            n_dist = int(np.load(path.with_suffix(".n_dist.npy")))
            levels_by_name[name] = levels
            build_rows.append({
                "graph": name,
                "M": M,
                "ef_construction": ef_construction,
                "n_layers": len(layers),
                "entry": entry,
                "n_dist_build": n_dist,
                "ratio_nsw": n_dist / nsw_n_dist[M],  # NSW(4-5)の構築との比
            })
            for l, graph in enumerate(layers):
                degs = np.array([len(graph[i]) for i in np.flatnonzero(levels >= l)])
                layer_rows.append({
                    "graph": name,
                    "M": M,
                    "layer": l,
                    "n_points": len(degs),
                    "n_edges": int(degs.sum()),  # 片方向ずつ数える(近所リストの長さの合計)
                    "avg_deg": float(degs.mean()),
                    "max_deg": int(degs.max()),
                })

        # 層ありの2つはレベルが同じ(差はヒューリスティックだけ)。cap は全点がレベル0
        assert np.array_equal(levels_by_name["layered"], levels_by_name["hnsw"])
        assert (levels_by_name["cap"] == 0).all()

    for name, rows in [("step5_8_build.csv", build_rows), ("step5_8_layers.csv", layer_rows)]:
        with open(result_path(name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            print(r)
    print("5-8: OK")


if __name__ == "__main__":
    step5_1_check_levels()