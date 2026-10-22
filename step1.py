from pathlib import Path

import numpy as np

from common import read_fvecs, read_ivecs, brute_force, recall_at_k

ROOT = Path(__file__).parent
SIFT = ROOT / "siftsmall"
DATA = ROOT / "data"

# ---- データ読み込み(SIFT10K) ----
X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

# print(X.shape, Q.shape, GT.shape)   # (10000, 128) (100, 128) (100, 100) になるはず


# ---- 1-1: brute_force の確認 ----
# print(len(brute_force(X, Q[0], 5)))                         # 5
# print(brute_force(X, Q[0], 10))
# print(np.array_equal(brute_force(X, Q[0], 10), GT[0][:10]))  # True


# ---- 1-2: recall_at_k の確認 ----
# print(recall_at_k(brute_force(X, Q[0], 10), GT[0], 10))   # 1.0
# print(recall_at_k(GT[0][:10], GT[0], 10))                 # 1.0
# print(recall_at_k(GT[0][5:15], GT[0], 10))                # 0.5


# ---- 1-3a: 物差しの確認(全探索 → 1.0)----
# recalls = []
# for i in range(len(Q)):
#     pred = brute_force(X, Q[i], 10)        # クエリ Q[i] の全探索の上位10件
#     r    = recall_at_k(pred, GT[i], 10)    # それを GT[i] で採点
#     recalls.append(r)
#
# print(np.mean(recalls), np.min(recalls))   # 1.0 1.0


# ---- 1-3b: 物差しの確認(ランダム → ほぼ0)----
# rng = np.random.default_rng(0)
# recalls_rand = []
# for i in range(len(Q)):
#     pred = rng.choice(len(X), 10, replace=False)
#     r    = recall_at_k(pred, GT[i], 10)
#     recalls_rand.append(r)

# print(np.mean(recalls_rand), max(recalls_rand), min(recalls_rand))   # 予想 0.001 → 結果 0.003 0.1 0.0


# ---- 1-4: ランダムデータの正解を作る ----
# X_u = np.load(DATA / "X.npy")
# Q_u = np.load(DATA / "Q.npy")
# print(X_u.shape, Q_u.shape)   # (10000, 32) (100, 32) になるはず


# gt_list = []
# for i in range(len(Q_u)):
#     top = brute_force(X_u, Q_u[i], 100)           # クエリ Q_u[i] の全探索の上位100件
#     gt_list.append(top)

# GT_u = np.array(gt_list)          # (100, 100) の配列にまとめる
# np.save(DATA / "GT.npy", GT_u)

# GT_u2 = np.load(DATA / "GT.npy")  # 読み直して確認
# print(GT_u2.shape)                # (100, 100) になるはず

# d = np.linalg.norm(X_u[GT_u2[0]] - Q_u[0], axis=1)
# print(np.all(np.diff(d) >= 0))   # True になるはず