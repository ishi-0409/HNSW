import heapq
import time
from pathlib import Path

import numpy as np

from common import read_fvecs, read_ivecs, make_uniform

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SIFT = ROOT / "siftsmall"


# ---- 0-3, 0-4: 距離の計算と上位10件 ----
def step0_3_4_distance_topk() -> None:
    X = np.random.default_rng(0).random((100, 8), dtype = np.float32)

    a = X[0]
    b = X[1]
    d = np.linalg.norm(a - b)

    q = X[0]
    dist_loop = np.array([np.linalg.norm(X[i] - q) for i in range(len(X))])
    dists_vec = np.linalg.norm(X - q, axis=1)

    # for文とまとめて計算が一致するか
    print(np.allclose(dist_loop, dists_vec))
    # 2点間の距離が、まとめて計算の1番目と一致するか
    print(np.allclose(d, dists_vec[1]))

    idx = np.argsort(dists_vec)[:10]
    print(idx)
    print(dists_vec[idx])


# ---- 0-5: heapq の練習 ----
def step0_5_heapq() -> None:
    h = []
    heapq.heappush(h, (0.5, 3))
    heapq.heappush(h, (0.2, 7))
    heapq.heappush(h, (0.9, 1))

    # 小さい順に出るか
    print([heapq.heappop(h)[0] for _ in range(3)] == [0.2, 0.5, 0.9])

    w = []
    heapq.heappush(w, (-0.5, 3))
    heapq.heappush(w, (-0.2, 7))
    heapq.heappush(w, (-0.9, 1))

    # マイナスを付けると大きい順に出るか
    print([-heapq.heappop(w)[0] for _ in range(3)] == [0.9, 0.5, 0.2])

    dists = [0.8, 0.3, 0.9, 0.1, 0.5, 0.7, 0.2, 0.6]
    u = []

    for i, d in enumerate(dists):
        heapq.heappush(u, (-d, i))
        if len(u) > 5:
            heapq.heappop(u)

    top5 = sorted(-neg_d for neg_d, _ in u)
    print(top5)
    # 上位5件が、全部並べ替えた結果と一致するか
    print(top5 == sorted(dists)[:5])


# ---- 0-6: ランダムデータを作って保存 ----
def step0_6_make_random_data() -> None:
    rng = np.random.default_rng(42)
    X = make_uniform(10000, 32, rng)
    Q = make_uniform(100, 32, rng)
    print(X[0])
    print(X.shape)
    print(Q.shape)

    np.save(DATA / "X.npy", X)
    np.save(DATA / "Q.npy", Q)

    X2 = np.load(DATA / "X.npy")
    Q2 = np.load(DATA / "Q.npy")

    # 保存して読み込んだものが一致するか
    print(np.array_equal(X, X2))
    print(np.array_equal(Q, Q2))


# ---- 0-7, 0-8: SIFT を読み、for 文とまとめて計算の速さを比べる ----
def step0_7_8_sift_and_timing() -> None:
    X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
    Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
    GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

    # for文とまとめて計算が一致するか
    q = Q[0]
    d_loop = np.array([np.linalg.norm(X[i] - q) for i in range(len(X))])
    d_vec = np.linalg.norm(X - q, axis=1)
    print(np.allclose(d_loop, d_vec))

    t = time.perf_counter()
    for j in range(10):
        q = Q[j]
        d_loop = np.array([np.linalg.norm(X[i] - q) for i in range(len(X))])
    t_loop = (time.perf_counter() - t) / 10

    t = time.perf_counter()
    for j in range(100):
        q = Q[j]
        d_vec = np.linalg.norm(X - q, axis=1)
    t_vec = (time.perf_counter() - t) / 100

    print(f"for文: {t_loop*1000:.2f} ms/クエリ")
    print(f"まとめて: {t_vec*1000:.2f} ms/クエリ")
    print(f"{t_loop / t_vec:.0f} 倍")

    idx = np.argsort(np.linalg.norm(X - Q[0], axis=1))[:10]
    print(idx)
    print(GT[0][:10])
    # 全探索が GT と一致するか
    print(np.array_equal(idx, GT[0][:10]))


if __name__ == "__main__":
    # step0_3_4_distance_topk()
    # step0_5_heapq()
    step0_6_make_random_data()
    # step0_7_8_sift_and_timing()
