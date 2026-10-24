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

# 近似採点
def recall_at_k(pred, gt, k):
    gt_top   = gt[:k]
    pred_set = set(np.asarray(pred).tolist())
    gt_set   = set(gt_top.tolist())
    hits     = gt_set & pred_set
    return len(hits) / k


