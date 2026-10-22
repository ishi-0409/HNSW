import numpy as np


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


# 一様乱数のデータ (n, d) を作る
def make_uniform(n, d, rng):
    return rng.random((n, d), dtype=np.float32)


# 全探索
def brute_force(X, q, k):
    dists = np.linalg.norm(X - q, axis=1)
    return np.argsort(dists)[:k]


def recall_at_k(pred, gt, k):
    """recall@k を計算する。

    pred : 探索で得た近傍の番号(上位 k 件)
    gt   : 正解の近傍の番号
    k    : 何件までで比べるか
    戻り値 : 正解の上位 k 件のうち、pred の上位 k 件に含まれていた割合 (0.0〜1.0)
    """
    raise NotImplementedError
