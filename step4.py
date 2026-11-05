import numpy as np
from common import load_sift, recall_at_k
from pathlib import Path
from graph import reachable_from, search_layer
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

def step4_8_check_small() -> None:
    # 4-3 の数直線5点(A=0, B=10, C=4, D=7, E=2, M=2)の NSW
    graph = [[1, 2, 4], [0, 2, 3], [0, 1, 3, 4], [1, 2], [0, 2]]
    for e in range(5):
        r = reachable_from(graph, [e])
        assert r.shape == (5,)
        assert r.all(), f"入口 {e} から届かない点がある: {r}"
    print("4-8 小さいグラフ: どの入口からも5点すべてに届く OK")

def step4_8_reachable(Ms: list[int], save: bool) -> pd.DataFrame:
    base = Path(__file__).parent
    df2 = pd.read_csv(base / "results" / "step2_greedy_M_entry.csv")
    entries = df2[df2["M"] == 10][["seed", "query", "entry"]]  # ← (1) query を残し、drop_duplicates を消す
    print(f"入口の組: {len(entries)} 行")

    rows = []
    for M in Ms:
        graph, order = load_nsw(base / "data" / f"nsw_sift10k_M{M}_efc100_seed0.npz")
        for seed, query, entry in entries.itertuples(index=False):  # ← (2) query も受け取る
            r = reachable_from(graph, [int(entry)])
            rows.append({"M": M, "seed": int(seed), "query": int(query),
                         "entry": int(entry), "n_reachable": int(r.sum())})
    df = pd.DataFrame(rows)
    print(df.groupby("M")["n_reachable"].agg(["min", "max", "count"]))  # ← (3) 全行を要約して表示
    if save:
        df.to_csv(base / "results" / "step4_8_reachable.csv", index=False)
    return df

def step4_8_degree(Ms: list[int], save: bool) -> pd.DataFrame:
    base = Path(__file__).parent
    rows = []
    for M in Ms:
        graph, order = load_nsw(base / "data" / f"nsw_sift10k_M{M}_efc100_seed0.npz")
        n = len(graph)
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)  # pos[i] = 点 i を何番目に入れたか(0始まり)
        for i in range(n):
            rows.append({"M": M, "point": i, "insert_pos": int(pos[i]),
                         "degree": len(graph[i])})
    df = pd.DataFrame(rows)
    step4_8_check_degree(df, Ms)
    if save:
        df.to_csv(base / "results" / "step4_8_degree.csv", index=False)
    return df

def step4_8_check_degree(df: pd.DataFrame, Ms: list[int]) -> None:
    for M in Ms:
        d = df[df["M"] == M]
        n = len(d)
        assert n == 10000
        assert np.array_equal(np.sort(d["insert_pos"].to_numpy()), np.arange(n))
        assert d["degree"].sum() == expected_degree_sum(n, M)
        # 最後に入れた点は、自分で張った M 本だけ
        assert d.loc[d["insert_pos"] == n - 1, "degree"].item() == M
        # k 番目(0始まり)に入れた点は、自分で min(M, k) 本張っている
        assert (d["degree"] >= np.minimum(M, d["insert_pos"])).all()
        print(f"M={M}: 次数の検算 OK")

def step4_9_beam_M_ef(
        Ms: tuple[int, ...] = (10, 20, 40),
        efs: tuple[int, ...] = (10, 16, 32, 64, 128, 256),
        save: bool = True,
        ) -> pd.DataFrame:  # 変更:確認関数に渡すため df を返す
    X, Q, GT = load_sift()
    # 変更:kNN グラフ(full)の読み込みは削除

    df2 = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
    entries = df2[df2["M"] == 10][["seed", "query", "entry"]]

    d_all = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])

    rows = []
    t0 = time.perf_counter()
    for M in Ms:
        # 変更:M ごとに別の NSW グラフを読む(order は使わない)
        graph, _ = load_nsw(DATA / f"nsw_sift10k_M{M}_efc100_seed0.npz")
        for ef in efs:
            for r in entries.itertuples():
                s = search_layer(X, graph, Q[r.query], [int(r.entry)], ef=ef)
                top1 = s["topk"][0]
                rows.append({
                    "M": M, "ef": ef, "seed": r.seed, "query": r.query, "entry": r.entry,
                    "top1": top1,
                    "rank_top1": int((d_all[r.query] < d_all[r.query, top1]).sum()) + 1,
                    "recall1": recall_at_k([top1], GT[r.query], 1),
                    "recall10": recall_at_k(s["topk"][:10], GT[r.query], 10),
                    "n_dist": s["n_dist"], "n_expand": s["n_expand"],
                    "n_topk": len(s["topk"]),
                })
            print(f"M={M} ef={ef} 経過 {time.perf_counter() - t0:.1f} 秒")

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(RESULTS / "step4_9_beam_M_ef.csv", index=False)  # 変更:保存先
    else:
        print(df.groupby(["M", "ef"])[["recall1", "recall10", "n_dist"]].mean())
    return df  # 変更

def step4_9_check(df: pd.DataFrame) -> None:
    df3 = pd.read_csv(RESULTS / "step3_beam_M_ef.csv")

    # 確認2:形が Step 3 と同じ
    assert len(df) == 9000
    assert list(df.columns) == list(df3.columns)

    # 確認1:入口が Step 3 と同じ
    keys = ["M", "ef", "seed", "query"]
    m = df.merge(df3[keys + ["entry"]], on=keys, suffixes=("", "_3"))
    assert len(m) == 9000
    assert (m["entry"] == m["entry_3"]).all()

    # 確認3:W は必ず ef 件で満杯
    assert (df["n_topk"] == df["ef"]).all()

    # 確認4:同じ点は2回測らない
    assert (df["n_dist"] <= 10000).all()

    # 確認5:1行を測り直す
    X, Q, GT = load_sift()
    graph, _ = load_nsw(DATA / "nsw_sift10k_M10_efc100_seed0.npz")
    r = df[(df["M"] == 10) & (df["ef"] == 10) & (df["seed"] == 0) & (df["query"] == 0)].iloc[0]
    s = search_layer(X, graph, Q[0], [int(r["entry"])], ef=10)
    d = np.linalg.norm(X[s["topk"]] - Q[0], axis=1)
    assert np.all(np.diff(d) >= 0)    # 近い順に並んでいる
    assert s["topk"][0] == r["top1"]   # csv の top1 と一致
    assert s["n_dist"] == r["n_dist"]  # 同じ条件なら同じ回数

    print("4-9 の確認:すべて通過")

def step4_12_hub_entry(
        Ms: tuple[int, ...] = (10, 20, 40),
        efs: tuple[int, ...] = (1, 10),
        save: bool = True,
        ) -> pd.DataFrame:
    X, Q, GT = load_sift()
    df2 = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
    rand = df2[df2["M"] == 10][["seed", "query", "entry"]]
    d_all = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])
    nq = len(Q)

    rows = []
    for M in Ms:
        graph, order = load_nsw(DATA / f"nsw_sift10k_M{M}_efc100_seed0.npz")
        deg = np.array([len(nb) for nb in graph])
        pos = np.empty(len(order), dtype=int)
        pos[order] = np.arange(len(order))  # 点 → 何番目に入れたか

        # (group, k, query, entry) の一覧を作る
        jobs = [("random", int(r.seed), int(r.query), int(r.entry)) for r in rand.itertuples()]
        hubs = np.argsort(-deg, kind="stable")[:5]
        jobs += [("hub", k, q, int(h)) for k, h in enumerate(hubs) for q in range(nq)]
        jobs += [("first", k, q, int(o)) for k, o in enumerate(order[:5]) for q in range(nq)]

        for ef in efs:
            for group, k, q, entry in jobs:
                s = search_layer(X, graph, Q[q], [entry], ef=ef)
                top1 = s["topk"][0]
                rows.append({
                    "M": M, "ef": ef, "group": group, "k": k, "query": q, "entry": entry,
                    "entry_deg": int(deg[entry]), "entry_pos": int(pos[entry]),
                    "entry_rank": int((d_all[q] < d_all[q, entry]).sum()) + 1,
                    "top1": top1,
                    "rank_top1": int((d_all[q] < d_all[q, top1]).sum()) + 1,
                    "recall1": recall_at_k([top1], GT[q], 1),
                    "recall10": recall_at_k(s["topk"][:10], GT[q], 10) if ef >= 10 else np.nan,
                    "n_dist": s["n_dist"], "n_expand": s["n_expand"],
                    "stuck": top1 == entry,
                })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(RESULTS / "step4_12_hub_entry.csv", index=False)
    else:
        print(df.groupby(["M", "ef", "group"])[["recall1", "entry_rank", "stuck"]].mean())
    return df

def step4_12_check(df: pd.DataFrame) -> None:
    # 確認5:形
    assert len(df) == 9000
    assert (df.groupby(["M", "ef", "group"]).size() == 500).all()

    # 確認1:ランダム入口の ef=10 が 4-9 と一致
    d9 = pd.read_csv(RESULTS / "step4_9_beam_M_ef.csv")
    d9 = d9[d9["ef"] == 10]
    r = df[(df["group"] == "random") & (df["ef"] == 10)].rename(columns={"k": "seed"})
    m = r.merge(d9, on=["M", "seed", "query"], suffixes=("", "_9"))
    assert len(m) == 1500
    assert (m["entry"] == m["entry_9"]).all()
    assert (m["top1"] == m["top1_9"]).all()
    assert (m["n_dist"] == m["n_dist_9"]).all()

    # 確認2:k=0 のハブが最大次数の点
    max_deg = {10: 238, 20: 452, 40: 858}
    for M, d in max_deg.items():
        h = df[(df["M"] == M) & (df["group"] == "hub") & (df["k"] == 0)]
        assert (h["entry_deg"] == d).all()

    # 確認3:最初の5点は挿入順の先頭
    f = df[df["group"] == "first"]
    assert set(f["entry_pos"]) == {0, 1, 2, 3, 4}

    # 確認4:ef=1 では stuck と n_expand == 1 が同じ
    e1 = df[df["ef"] == 1]
    assert (e1["stuck"] == (e1["n_expand"] == 1)).all()

    print("4-12 の確認:すべて通過")

if __name__ == "__main__":
    df = step4_12_hub_entry()
    step4_12_check(df)