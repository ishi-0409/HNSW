from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SIFT = ROOT / "siftsmall"


def read_fvecs(path: str | Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32)


def read_ivecs(path: str | Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


# SIFT10K のデータ・クエリ・正解を読む
def load_sift() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
    Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
    GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")
    return X, Q, GT


# 一様乱数のデータ (n, d) を作る
def make_uniform(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, d), dtype=np.float32)


# 全探索
def brute_force(X: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    dists = np.linalg.norm(X - q, axis=1)
    return np.argsort(dists)[:k]

# 近似採点
def recall_at_k(pred: np.ndarray | list[int], gt: np.ndarray, k: int) -> float:
    gt_top   = gt[:k]
    pred_set = set(np.asarray(pred).tolist())
    gt_set   = set(gt_top.tolist())
    hits     = gt_set & pred_set
    return len(hits) / k


