import numpy as np


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
