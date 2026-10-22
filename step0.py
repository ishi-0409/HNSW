import heapq
import time
from pathlib import Path

import numpy as np

from common import read_fvecs, read_ivecs, make_uniform

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SIFT = ROOT / "siftsmall"


# ---- 02: 距離の計算と上位10件 ----
x = np.random.default_rng(0).random((100, 8), dtype = np.float32)

# print(x.shape)
# print(x[50])
# print(x.dtype)


X = x[0]
Y = x[1]
d = np.linalg.norm(X - Y)

# print(d)


q = x[0]
dist_loop = np.array([np.linalg.norm(x[i] - q) for i in range(len(x))])

#print(dist_loop)


dists_vec = np.linalg.norm(x - q, axis=1)

# print(dists_vec)
# print(dists_vec.shape)
# print(np.allclose(dist_loop, dists_vec))

idx = np.argsort(dists_vec)[:10]
print(idx)
print(dists_vec[idx])


# ---- 04: heapq の練習 ----
h = []
heapq.heappush(h, (0.5, 3))
heapq.heappush(h, (0.2, 7))
heapq.heappush(h, (0.9, 1))

# print(heapq.heappop(h))
# print(heapq.heappop(h))
# print(heapq.heappop(h))

w = []
heapq.heappush(w, (-0.5, 3))
heapq.heappush(w, (-0.2, 7))
heapq.heappush(w, (-0.9, 1))

# print(heapq.heappop(w))
# print(heapq.heappop(w))
# print(heapq.heappop(w))


dists = [0.8, 0.3, 0.9, 0.1, 0.5, 0.7, 0.2, 0.6]
u = []

for i, d in enumerate(dists):
    heapq.heappush(u, (-d, i))
    if len(u) > 5:
        heapq.heappop(u)

print(sorted(-x for x, _ in u))


# ---- 06: ランダムデータを作って保存 ----
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

print(np.array_equal(X, X2))


# ---- 07: SIFT を読み、for 文とまとめて計算の速さを比べる ----
X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

# print(X.shape)
# print(Q.shape)
# print(GT.shape)
# print(X[0])

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

# print(f"for文: {t_loop*1000:.2f} ms/クエリ")
# print(f"まとめて: {t_vec*1000:.2f} ms/クエリ")
# print(f"{t_loop / t_vec:.0f} 倍")

idx = np.argsort(np.linalg.norm(X - Q[0], axis=1))[:10]
print(idx)
print(GT[0][:10])
