import numpy as np

from graph import search_layer
from nsw import add_edge


def assign_levels(n: int, mL: float, seed: int = 0) -> np.ndarray:
    """各点のレベル l = floor(-ln(U) * mL) を決める(論文 4.1節)。
    
    引数
        n: 点の数
        mL: レベルの出やすさ。1/ln(M)で上の層に約1/M残る。
        seed: レベル用の乱数シード
    戻り値
        長さnのint配列。levels[i]は点iのレベルで、点iは層0~levels[i]に存在する。    
    """
    rng = np.random.default_rng(seed)
    U = 1.0 - rng.random(n)  # (0, 1]。U=0 だと ln(0) = -inf になるため
    return np.floor(-np.log(U) * mL).astype(int)



def select_neighbors_simple(cands: list[int], M: int) -> list[int]:
    """近い順に並んだ候補から先頭のM個を返す(論文 Alg. 3)。

    引数
        cands: クエリに近い順の点番号(search_layer の topk)
        M: 選ぶ数
    戻り値
        近い順の点番号。候補がM個未満なら全部
    """
    return cands[:M]


def shrink_neighbors(X: np.ndarray, graph: list[list[int]], e: int, Mmax: int) -> int:
    """点eの近所がMmax本を超えていたら、eに近い順にMmax本だけ残す(論文 Alg. 1 の12〜16行目)。

    引数
        X: データ(N, d)
        graph: 1つの層の近所リスト。graph[e] を書き換える
        e: 削る点の番号
        Mmax: 次数の上限
    戻り値
        距離計算回数。上限以下なら0
    """
    nbs = graph[e]
    if len(nbs) <= Mmax:
        return 0
    dists = np.linalg.norm(X[nbs] - X[e], axis=1)  # 基準は新しい点ではなく e
    keep = np.argsort(dists, kind="stable")[:Mmax]  # 同じ距離でも順番が揺れないように stable
    graph[e] = select_neighbors_simple([nbs[i] for i in keep], Mmax)
    return len(nbs)


def insert_point_hnsw(
    X: np.ndarray,
    layers: list[list[list[int]]],
    levels: np.ndarray,
    idx: int,
    entry: int,
    M: int,
    Mmax0: int | None,
    ef_construction: int,
) -> tuple[int, int]:
    """点idxを上の層から順に入れる(論文 Alg. 1)。

    引数
        X: データ(N, d)
        layers: layers[l][i] が層 l での点 i の近所。書き換える
        levels: 各点のレベル(assign_levels の戻り値)
        idx: 入れる点の番号
        entry: 今の入口(すでに入っている点)
        M: 1点につなぐ数。層1以上の次数の上限も M
        Mmax0: 層0の次数の上限。None なら削らない(NSW との検算用)
        ef_construction: 辺を張る層で保持する候補の数
    戻り値
        (更新後の入口, 距離計算回数)。距離計算回数は探索と削る処理の合計
    """
    q = X[idx]
    l = int(levels[idx])
    L = int(levels[entry])
    n_dist = 0
    ep = [entry]

    # l より上の層:ef=1 で一番近い点を探して降りるだけ
    for lc in range(L, l, -1):
        res = search_layer(X, layers[lc], q, ep, 1)
        n_dist += res["n_dist"]
        ep = res["topk"][:1]

    # 層 min(L, l) 以下:近所を探して辺を張る
    for lc in range(min(L, l), -1, -1):
        res = search_layer(X, layers[lc], q, ep, ef_construction)
        n_dist += res["n_dist"]
        neighbors = select_neighbors_simple(res["topk"], M)
        for nb in neighbors:
            add_edge(layers[lc], idx, nb)

        Mmax = Mmax0 if lc == 0 else M
        if Mmax is not None:
            for nb in neighbors:
                n_dist += shrink_neighbors(X, layers[lc], nb, Mmax)

        ep = res["topk"]  # 論文どおり W をまるごと次の層の入口にする

    # 入れた点のレベルが一番上なら入口にする
    if l > L:
        entry = idx
    return entry, n_dist


def build_hnsw(
    X: np.ndarray,
    M: int,
    ef_construction: int,
    mL: float,
    Mmax0: int | None,
    seed: int = 0,
    order: np.ndarray | None = None,
) -> tuple[list[list[list[int]]], np.ndarray, int, int]:
    """Xの全点を順に入れてHNSWを作る(論文 Alg. 1 のくり返し)。

    引数
        X: データ(N, d)
        M, ef_construction, Mmax0: insert_point_hnsw と同じ
        mL: レベルの出やすさ(assign_levels に渡す)
        seed: レベル用の乱数シード
        order: 入れる順番。None なら番号順(5-7 では NSW と同じ順番を渡す)
    戻り値
        (layers, levels, 入口, 構築の距離計算回数の合計)
    """
    n = len(X)
    if order is None:
        order = np.arange(n)
    levels = assign_levels(n, mL, seed)
    layers: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(levels.max() + 1)]

    entry = int(order[0])  # 最初の点は入口にするだけ(辺なし)
    n_dist_total = 0
    for idx in order[1:]:
        entry, n_dist = insert_point_hnsw(X, layers, levels, int(idx), entry, M, Mmax0, ef_construction)
        n_dist_total += n_dist

    return layers, levels, entry, n_dist_total


def search_hnsw(
    X: np.ndarray,
    layers: list[list[list[int]]],
    levels: np.ndarray,
    entry: int,
    q: np.ndarray,
    k: int,
    ef: int,
) -> dict:
    """入口から上の層を ef=1 で降り、層0で ef のビーム探索をして近いk点を返す(論文 Alg. 5)。

    引数
        X, layers, levels, entry: build_hnsw の戻り値とデータ
        q: クエリ(d,)
        k: 返す数
        ef: 層0で保持する候補の数。k 以上にする(k 未満だと k 個に届かない)
    戻り値
        topk: クエリに近い順の点番号(最大k個)
        n_dist: 全層の距離計算回数
        n_dist0, n_expand, deg_sum: 層0だけの値(search_layer と同じ意味)
    """
    n_dist = 0
    ep = [entry]

    # 上の層:ef=1 で一番近い点を探して降りる
    for lc in range(int(levels[entry]), 0, -1):
        res = search_layer(X, layers[lc], q, ep, 1)
        n_dist += res["n_dist"]
        ep = res["topk"][:1]

    # 層0:ef でビーム探索
    res = search_layer(X, layers[0], q, ep, ef)
    n_dist += res["n_dist"]
    return {
        "topk": res["topk"][:k],
        "n_dist": n_dist,
        "n_dist0": res["n_dist"],
        "n_expand": res["n_expand"],
        "deg_sum": res["deg_sum"],
    }
