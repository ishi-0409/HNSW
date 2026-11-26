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
from graph import reachable_from, search_layer
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
    assert res == {"topk": [4, 3], "n_dist": 6, "n_dist0": 3, "n_expand": 2, "deg_sum": 3, "start0": 3}  # 層1 で点3 から動かずに降りる

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


def layer0_in_out(graph0: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """層0の各点の次数(出る辺の数)と被次数(向かってくる辺の数)"""
    n = len(graph0)
    out_deg = np.array([len(nb) for nb in graph0])
    in_deg = np.zeros(n, dtype=int)
    for nb in graph0:
        in_deg[nb] += 1  # 近所に重複はない(check_hnsw で確認済み)
    return out_deg, in_deg


def step5_10_check_small():
    # 5-3 の Mmax0=1 のグラフ(数直線 x = 0, 1, 3, 4, 10)。層0 は [[1], [0], [1], [1], [3]]、入口は点3
    X = np.array([[0], [1], [3], [4], [10]], dtype=np.float32)
    levels = np.array([1, 0, 0, 2, 0])
    layers = [[[] for _ in range(5)] for _ in range(levels.max() + 1)]
    entry = 0
    for idx in range(1, 5):
        entry, _ = insert_point_hnsw(X, layers, levels, idx, entry, 1, 1, 2)
    assert layers[0] == [[1], [0], [1], [1], [3]] and entry == 3

    out_deg, in_deg = layer0_in_out(layers[0])
    print(f"次数={out_deg.tolist()}, 被次数={in_deg.tolist()}")
    assert out_deg.tolist() == [1, 1, 1, 1, 1]
    assert in_deg.tolist() == [1, 3, 0, 1, 0]  # 点2・点4 には誰も向かっていない
    assert out_deg.sum() == in_deg.sum()

    # 入口(点3)から届くのは 3 → 1 → 0 の3点。点2・点4 には届かない
    r = reachable_from(layers[0], [entry])
    print(f"点{entry} から届く点: {np.flatnonzero(r).tolist()}")
    assert np.flatnonzero(r).tolist() == [0, 1, 3]
    # 点4 から入ると 4 → 3 → 1 → 0 で4点(点2 だけは誰からも届かない)
    assert reachable_from(layers[0], [4]).sum() == 4
    print("5-10 の数直線の確認: OK")


def load_layer0(name: str, M: int) -> tuple[list[list[int]], np.ndarray, int]:
    """グラフ1(nsw)〜4 の層0、各点のレベル、入口を読む。NSW は層が1つで入口は最初に入れた点"""
    if name == "nsw":
        graph, order = load_nsw(DATA / f"nsw_sift10k_M{M}_efc100_seed0.npz")
        return graph, np.zeros(len(graph), dtype=int), int(order[0])
    layers, levels, entry = load_hnsw(hnsw_path(name, M))
    return layers[0], levels, entry


GRAPH_NAMES = ("nsw",) + tuple(HNSW_VARIANTS)  # グラフ1〜4


def step5_10_degree(Ms: tuple[int, ...] = (10, 20, 40)):
    """グラフ1〜4 の層0について、各点の次数・被次数・レベルを csv に出す"""
    rows = []
    for M in Ms:
        for name in GRAPH_NAMES:
            graph0, levels, entry = load_layer0(name, M)
            out_deg, in_deg = layer0_in_out(graph0)
            assert out_deg.sum() == in_deg.sum()  # どちらも辺の数(片方向ずつ)
            if name != "nsw":
                assert out_deg.max() <= 2 * M
            for i in range(len(graph0)):
                rows.append({"graph": name, "M": M, "point": i, "level": int(levels[i]),
                             "degree": int(out_deg[i]), "indegree": int(in_deg[i])})
            print(f"{name} M={M}: 次数 最大={out_deg.max()}, 平均={out_deg.mean():.2f}, "
                  f"被次数0の点={int((in_deg == 0).sum())}", flush=True)
    with open(result_path("step5_10_degree.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("5-10 次数: OK")


def step5_10_reachable(Ms: tuple[int, ...] = (10, 20, 40)):
    """層0で入口から届く点の数を csv に出す。入口は固定の入口(fixed)と、2-6 と同じランダムな 500 組(random)。
    M ごとに別の csv にする(全部まとめると実行に時間がかかるため)"""
    with open(result_path("step2_greedy_M_entry.csv"), encoding="utf-8") as f:
        pairs = [(int(r["seed"]), int(r["query"]), int(r["entry"])) for r in csv.DictReader(f) if int(r["M"]) == 10]
    assert len(pairs) == 500

    for M in Ms:
        rows = []
        for name in GRAPH_NAMES:
            graph0, _, entry = load_layer0(name, M)
            _, in_deg = layer0_in_out(graph0)
            cache: dict[int, int] = {}  # 同じ入口は1回だけ数える

            def count(e: int) -> int:
                if e not in cache:
                    cache[e] = int(reachable_from(graph0, [e]).sum())
                    # 被次数0の点には、その点自身から入らない限り届かない
                    assert cache[e] <= len(graph0) - int(((in_deg == 0) & (np.arange(len(graph0)) != e)).sum())
                return cache[e]

            rows.append({"graph": name, "M": M, "entry_kind": "fixed", "seed": -1, "query": -1,
                         "entry": entry, "n_reachable": count(entry)})
            for seed, query, e in pairs:
                rows.append({"graph": name, "M": M, "entry_kind": "random", "seed": seed, "query": query,
                             "entry": e, "n_reachable": count(e)})
            vals = [r["n_reachable"] for r in rows if r["graph"] == name]
            print(f"{name} M={M}: 固定の入口から {rows[-501]['n_reachable']}、全入口で {min(vals)}〜{max(vals)}", flush=True)
        with open(result_path(f"step5_10_reachable_M{M}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("5-10 届く点: OK")


def step5_10_layer0_start(M: int = 10, ef_construction: int = 100, seed: int = 0):
    """追加調査:グラフ3(階層あり)を、層0の探索だけ毎回点0から始めて作り直し、届く点を数える。
    グラフ3 にだけ閉じた集まりができたのは、層0の探索を新しい点の近くから始めるからか、を確かめる。
    グラフは保存しない"""
    import time

    X, _, _ = load_sift()
    t0 = time.perf_counter()
    layers, levels, entry, n_dist = build_hnsw(
        X, M, ef_construction, 1 / np.log(M), 2 * M, seed=seed, layer0_start=0
    )
    print(f"構築 {time.perf_counter() - t0:.1f}秒, n_dist={n_dist}", flush=True)
    check_hnsw(layers, levels, entry, M, 2 * M)
    # レベルは元のグラフ3 と同じ(同じシード)。違うのは層0の探索の始め方だけ
    _, levels3, entry3 = load_hnsw(hnsw_path("layered", M, ef_construction))
    assert np.array_equal(levels, levels3) and entry == entry3

    with open(result_path("step2_greedy_M_entry.csv"), encoding="utf-8") as f:
        pairs = [(int(r["seed"]), int(r["query"]), int(r["entry"])) for r in csv.DictReader(f) if int(r["M"]) == 10]
    graph0 = layers[0]
    out_deg, in_deg = layer0_in_out(graph0)
    cache: dict[int, int] = {}

    def count(e: int) -> int:
        if e not in cache:
            cache[e] = int(reachable_from(graph0, [e]).sum())
        return cache[e]

    rows = [{"graph": "layered_start0", "M": M, "entry_kind": "fixed", "seed": -1, "query": -1,
             "entry": entry, "n_reachable": count(entry)}]
    for s, qi, e in pairs:
        rows.append({"graph": "layered_start0", "M": M, "entry_kind": "random", "seed": s, "query": qi,
                     "entry": e, "n_reachable": count(e)})
    vals = [r["n_reachable"] for r in rows[1:]]
    print(f"次数 平均={out_deg.mean():.2f} 最大={out_deg.max()}, 被次数0の点={int((in_deg == 0).sum())}")
    print(f"固定の入口から {rows[0]['n_reachable']}、ランダムな入口で {min(vals)}〜{max(vals)}")
    with open(result_path(f"step5_10_layer0_start_M{M}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(result_path(f"step5_10_layer0_start_M{M}_degree.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["M", "n_dist_build", "deg_mean", "deg_max", "indeg0"])
        writer.writeheader()
        writer.writerow({"M": M, "n_dist_build": n_dist, "deg_mean": float(out_deg.mean()),
                         "deg_max": int(out_deg.max()), "indeg0": int((in_deg == 0).sum())})


EFS = (10, 16, 32, 64, 128, 256)  # 4-9 と同じ


def step5_11_search(
    names: tuple[str, ...] = tuple(HNSW_VARIANTS),
    Ms: tuple[int, ...] = (10, 20, 40),
    efs: tuple[int, ...] = EFS,
):
    """グラフ2〜4 で100クエリを探索し、1回ごとの結果を (グラフ, M) ごとの csv に出す。
    mode: random = 層0をランダムな入口 500 組から(グラフ2)、fixed = 固定の入口から上の層を降りる本来の探索(A)、
    layer0 = 層0だけをランダムな入口 500 組から(D)。ランダムな入口は 2-6・4-9 と同じ"""
    import time

    X, Q, GT = load_sift()
    with open(result_path("step2_greedy_M_entry.csv"), encoding="utf-8") as f:
        pairs = [(int(r["seed"]), int(r["query"]), int(r["entry"])) for r in csv.DictReader(f) if int(r["M"]) == 10]
    d_all = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])

    for M in Ms:
        for name in names:
            layers, levels, entry = load_hnsw(hnsw_path(name, M))
            modes = ("random",) if name == "cap" else ("fixed", "layer0")
            rows = []
            t0 = time.perf_counter()
            for mode in modes:
                for ef in efs:
                    if mode == "fixed":
                        runs = [(-1, qi, entry) for qi in range(len(Q))]
                    else:
                        runs = pairs
                    for seed, qi, e in runs:
                        if mode == "fixed":
                            s = search_hnsw(X, layers, levels, entry, Q[qi], ef, ef)
                        else:
                            s = search_layer(X, layers[0], Q[qi], [e], ef)
                            s = {**s, "n_dist0": s["n_dist"], "start0": e}
                        top1 = s["topk"][0]
                        rows.append({
                            "graph": name, "M": M, "mode": mode, "ef": ef, "seed": seed, "query": qi,
                            "entry": e, "start0": s["start0"], "top1": top1,
                            "rank_top1": int((d_all[qi] < d_all[qi, top1]).sum()) + 1,
                            "recall1": recall_at_k([top1], GT[qi], 1),
                            "recall10": recall_at_k(s["topk"][:10], GT[qi], 10),
                            "n_dist": s["n_dist"], "n_dist0": s["n_dist0"], "n_expand": s["n_expand"],
                            "deg_sum": s["deg_sum"], "n_topk": len(s["topk"]),
                        })
            with open(result_path(f"step5_11_search_{name}_M{M}.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{name} M={M}: {len(rows)} 行, {time.perf_counter() - t0:.1f}秒", flush=True)


def step5_11_check(Ms: tuple[int, ...] = (10, 20, 40)):
    """5-11 の csv の確認"""
    import pandas as pd

    nsw = pd.read_csv(result_path("step4_9_beam_M_ef.csv"))
    X, Q, GT = load_sift()
    for M in Ms:
        for name in HNSW_VARIANTS:
            df = pd.read_csv(result_path(f"step5_11_search_{name}_M{M}.csv"))
            if name == "cap":
                assert len(df) == 500 * len(EFS) and set(df["mode"]) == {"random"}
            else:
                assert len(df) == 600 * len(EFS) and set(df["mode"]) == {"fixed", "layer0"}
            assert (df["n_topk"] == df["ef"]).all()  # W は ef 件で満杯
            assert (df["n_dist0"] <= 10000).all()  # 層0 で同じ点は2回測らない
            r = df[df["mode"] != "fixed"]
            assert (r["n_dist"] == r["n_dist0"]).all() and (r["start0"] == r["entry"]).all()
            # ランダムな入口は 4-9 と同じ
            keys = ["ef", "seed", "query"]
            m = r.merge(nsw[nsw["M"] == M][keys + ["entry"]], on=keys, suffixes=("", "_nsw"))
            assert len(m) == len(r) and (m["entry"] == m["entry_nsw"]).all()
            # 1行を測り直す(各 mode の ef=10、最初の行)
            layers, levels, entry = load_hnsw(hnsw_path(name, M))
            for mode, d in df[df["ef"] == 10].groupby("mode"):
                row = d.iloc[0]
                q = Q[int(row["query"])]
                if mode == "fixed":
                    assert (d["entry"] == entry).all()
                    s = search_hnsw(X, layers, levels, entry, q, 10, 10)
                    assert s["start0"] == row["start0"] and s["n_dist"] > s["n_dist0"] - 1
                else:
                    s = search_layer(X, layers[0], q, [int(row["entry"])], 10)
                assert s["topk"][0] == row["top1"] and s["n_dist"] == row["n_dist"]
    print("5-11 の確認: OK")


def step5_13_closed_set(M: int = 10):
    """グラフ3(階層あり)M の層0の閉じた集まり S を求めて csv に出す。
    S = 5-10 で届く点が一番少なかった入口から届く点。S から外へ出る辺は 0 本のはず"""
    import pandas as pd

    X, Q, GT = load_sift()
    graph0, _, _ = load_layer0("layered", M)
    reach = pd.read_csv(result_path(f"step5_10_reachable_M{M}.csv"))
    r = reach[reach["graph"] == "layered"]
    e = int(r.loc[r["n_reachable"].idxmin(), "entry"])
    S = reachable_from(graph0, [e])
    n_out = sum(1 for i in np.flatnonzero(S) for j in graph0[i] if not S[j])
    print(f"入口 {e} から届く S: {int(S.sum())} 点、S から外へ出る辺: {n_out} 本")
    assert int(S.sum()) == int(r["n_reachable"].min()) and n_out == 0

    with open(result_path(f"step5_13_closed_set_points_M{M}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point", "in_S"])
        writer.writerows([[i, int(S[i])] for i in range(len(S))])
    with open(result_path(f"step5_13_closed_set_queries_M{M}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "gt1", "gt1_in_S", "gt10_in_S"])  # gt10_in_S:本当の上位10点のうち S の中の数
        for qi in range(len(Q)):
            writer.writerow([qi, int(GT[qi][0]), int(S[GT[qi][0]]), int(S[GT[qi][:10]].sum())])
    print("5-13 S: OK")


# 5-15 のデータ:名前 → (次元, M)。M は論文の図に合わせる
SYNTH = {
    "uniform4": (4, 6),     # 図 12(a):4次元の一様乱数、M=6
    "cluster10": (10, 16),  # 図 7:10次元、互いに離れた100個の塊、M=16
}
N_CLUSTERS, CLUSTER_SIGMA = 100, 0.01  # 塊の中心は [0,1]^10 に一様、点は中心のまわりに標準偏差 0.01 の正規分布


def make_clustered(n: int, d: int, centers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """塊の中心から1つを選び、そのまわりに正規分布で点を置く。n 点"""
    c = rng.integers(0, len(centers), size=n)
    return (centers[c] + rng.normal(0, CLUSTER_SIGMA, size=(n, d))).astype(np.float32)


def synth_path(name: str, what: str) -> Path:
    return DATA / f"step5_15_{name}_{what}.npy"


def step5_15_make_data(n: int = 10000, n_q: int = 100):
    """5-15 のデータ(X, Q, GT)を作って data/ に保存する。クエリはデータとは別に同じ作り方で作る"""
    for name, (d, _) in SYNTH.items():
        if name == "uniform4":
            X = make_uniform(n, d, np.random.default_rng(0))
            Q = make_uniform(n_q, d, np.random.default_rng(1))
        else:
            centers = make_uniform(N_CLUSTERS, d, np.random.default_rng(2))
            X = make_clustered(n, d, centers, np.random.default_rng(0))
            Q = make_clustered(n_q, d, centers, np.random.default_rng(1))
            # 塊が離れているか:中心どうしの最短距離と、塊の広がり(中心からの距離の 99% 点)
            dc = np.linalg.norm(centers[:, None] - centers[None], axis=2)
            np.fill_diagonal(dc, np.inf)
            spread = np.quantile(np.linalg.norm(X - centers[np.argmin(np.linalg.norm(X[:, None] - centers[None], axis=2), axis=1)], axis=1), 0.99)
            print(f"中心どうしの最短距離={dc.min():.3f}、塊の広がり(99%)={spread:.3f}")
            assert dc.min() > 4 * spread
        GT = np.stack([brute_force(X, q, 100) for q in Q])
        for what, arr in (("X", X), ("Q", Q), ("GT", GT)):
            np.save(synth_path(name, what), arr)
        print(f"{name}: X={X.shape}, Q={Q.shape}, GT={GT.shape}")


def load_synth(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(synth_path(name, w)) for w in ("X", "Q", "GT"))


def synth_graph_path(name: str, graph: str) -> Path:
    return DATA / f"step5_15_{name}_{graph}.npz"


def step5_15_build(names: tuple[str, ...] = tuple(SYNTH), ef_construction: int = 100, seed: int = 0):
    """5-15 のデータでグラフ1〜4 を作って保存する(ef_construction・シードは SIFT と同じ)"""
    import time
    from nsw import build_nsw, save_nsw

    for name in names:
        X, _, _ = load_synth(name)
        _, M = SYNTH[name]
        t0 = time.perf_counter()
        graph, order, n_dist = build_nsw(X, M, ef_construction, seed)
        save_nsw(synth_graph_path(name, "nsw"), graph, order)
        np.save(synth_graph_path(name, "nsw").with_suffix(".n_dist.npy"), np.array(n_dist))
        print(f"{name} nsw: n_dist={n_dist}, {time.perf_counter() - t0:.1f}秒", flush=True)
        for g, (use_mL, heuristic) in HNSW_VARIANTS.items():
            t0 = time.perf_counter()
            mL = 1 / np.log(M) if use_mL else 0.0
            layers, levels, entry, n_dist = build_hnsw(X, M, ef_construction, mL, 2 * M, seed=seed, heuristic=heuristic)
            check_hnsw(layers, levels, entry, M, 2 * M)
            save_hnsw(synth_graph_path(name, g), layers, levels, entry)
            np.save(synth_graph_path(name, g).with_suffix(".n_dist.npy"), np.array(n_dist))
            print(f"{name} {g}: 層={len(layers)}, n_dist={n_dist}, {time.perf_counter() - t0:.1f}秒", flush=True)


def step5_15_search(names: tuple[str, ...] = tuple(SYNTH), efs: tuple[int, ...] = EFS, n_seeds: int = 5):
    """5-15 のデータで、グラフ1〜4 の層0の届く点と探索を csv に出す。
    NSW・グラフ2 はランダムな入口(シード0〜4、クエリごと。2-6 と同じ作り方)、グラフ3・4 は固定の入口から降りる(A)"""
    import time

    for name in names:
        X, Q, GT = load_synth(name)
        n = len(X)
        d_all = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])
        entries = np.stack([np.random.default_rng(s).integers(0, n, size=len(Q)) for s in range(n_seeds)])
        rows, reach_rows = [], []
        t0 = time.perf_counter()
        for g in GRAPH_NAMES:
            if g == "nsw":
                graph, order = load_nsw(synth_graph_path(name, g))
                layers, levels, entry = [graph], np.zeros(n, dtype=int), int(order[0])
            else:
                layers, levels, entry = load_hnsw(synth_graph_path(name, g))
            # 層0で届く点(固定の入口と、ランダムな入口)
            _, in_deg = layer0_in_out(layers[0])
            cache: dict[int, int] = {}
            for e in [entry] + sorted(set(entries.ravel().tolist())):
                cache[e] = int(reachable_from(layers[0], [e]).sum())
            reach_rows.append({"data": name, "graph": g, "indeg0": int((in_deg == 0).sum()),
                               "reach_fixed": cache[entry],
                               "reach_random_min": min(cache[int(e)] for e in entries.ravel()),
                               "reach_random_max": max(cache[int(e)] for e in entries.ravel())})
            mode = "fixed" if g in ("layered", "hnsw") else "random"
            for ef in efs:
                runs = ([(-1, qi, entry) for qi in range(len(Q))] if mode == "fixed"
                        else [(s, qi, int(entries[s, qi])) for s in range(n_seeds) for qi in range(len(Q))])
                for s, qi, e in runs:
                    if mode == "fixed":
                        r = search_hnsw(X, layers, levels, entry, Q[qi], ef, ef)
                    else:
                        r = search_layer(X, layers[0], Q[qi], [e], ef)
                    top1 = r["topk"][0]
                    rows.append({"data": name, "graph": g, "mode": mode, "ef": ef, "seed": s, "query": qi,
                                 "entry": e, "top1": top1,
                                 "rank_top1": int((d_all[qi] < d_all[qi, top1]).sum()) + 1,
                                 "recall1": recall_at_k([top1], GT[qi], 1),
                                 "recall10": recall_at_k(r["topk"][:10], GT[qi], 10),
                                 "n_dist": r["n_dist"], "n_topk": len(r["topk"])})
            print(f"{name} {g}: 届く点 {reach_rows[-1]}, {time.perf_counter() - t0:.1f}秒", flush=True)
        for fname, rs in ((f"step5_15_search_{name}.csv", rows), (f"step5_15_reachable_{name}.csv", reach_rows)):
            with open(result_path(fname), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rs[0]))
                writer.writeheader()
                writer.writerows(rs)


if __name__ == "__main__":
    step5_1_check_levels()