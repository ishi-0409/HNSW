import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import load_sift, brute_force, recall_at_k

ROOT = Path(__file__).parent
DATA = ROOT / "data"


# ---- 1-1: brute_force の確認 ----
def step1_1_check_brute_force() -> None:
    X, Q, GT = load_sift()

    # 読み込んだ形が合っているか
    print(X.shape == (10000, 128), Q.shape == (100, 128), GT.shape == (100, 100))

    # k 件返ってくるか
    print(len(brute_force(X, Q[0], 5)) == 5)

    print(brute_force(X, Q[0], 10))
    # 全探索が GT と一致するか
    print(np.array_equal(brute_force(X, Q[0], 10), GT[0][:10]))


# ---- 1-2: recall_at_k の確認 ----
def step1_2_check_recall() -> None:
    X, Q, GT = load_sift()

    r1 = recall_at_k(brute_force(X, Q[0], 10), GT[0], 10)
    r2 = recall_at_k(GT[0][:10], GT[0], 10)
    r3 = recall_at_k(GT[0][5:15], GT[0], 10)   # 半分だけ当たる

    # 予想 1.0 1.0 0.5
    print(r1, r2, r3)
    print(np.allclose([r1, r2, r3], [1.0, 1.0, 0.5]))


# ---- 1-3: 物差しの確認 ----
def step1_3_ruler() -> None:
    X, Q, GT = load_sift()

    # 1-3a: 全探索 → 1.0
    recalls = []
    for i in range(len(Q)):
        pred = brute_force(X, Q[i], 10)        # クエリ Q[i] の全探索の上位10件
        r    = recall_at_k(pred, GT[i], 10)    # それを GT[i] で採点
        recalls.append(r)

    # 予想 1.0 1.0
    print(np.mean(recalls), np.min(recalls))
    print(np.allclose([np.mean(recalls), np.min(recalls)], [1.0, 1.0]))

    # 1-3b: ランダム → ほぼ0
    rng = np.random.default_rng(0)
    recalls_rand = []
    for i in range(len(Q)):
        pred = rng.choice(len(X), 10, replace=False)
        r    = recall_at_k(pred, GT[i], 10)
        recalls_rand.append(r)

    # 予想 0.001 → 結果 0.003 0.1 0.0
    print(np.mean(recalls_rand), max(recalls_rand), min(recalls_rand))
    print(np.allclose([np.mean(recalls_rand), max(recalls_rand), min(recalls_rand)], [0.003, 0.1, 0.0]))


# ---- 1-4: ランダムデータの正解を作る ----
def step1_4_make_uniform_gt() -> None:
    X_u = np.load(DATA / "X.npy")
    Q_u = np.load(DATA / "Q.npy")
    # 読み込んだ形が合っているか
    print(X_u.shape == (10000, 32), Q_u.shape == (100, 32))

    gt_list = []
    for i in range(len(Q_u)):
        top = brute_force(X_u, Q_u[i], 100)           # クエリ Q_u[i] の全探索の上位100件
        gt_list.append(top)

    GT_u = np.array(gt_list)          # (100, 100) の配列にまとめる
    np.save(DATA / "GT.npy", GT_u)

    GT_u2 = np.load(DATA / "GT.npy")  # 読み直して確認
    print(GT_u2.shape == (100, 100))

    # 距離が小さい順に並んでいるか
    d = np.linalg.norm(X_u[GT_u2[0]] - Q_u[0], axis=1)
    print(np.all(np.diff(d) >= 0))


# ---- 1-5: 全探索の速さを記録する ----
# ウォームアップ1回のあと5回測って中央値をとる
def step1_5_timing() -> None:
    X, Q, _ = load_sift()
    X_u = np.load(DATA / "X.npy")
    Q_u = np.load(DATA / "Q.npy")

    K = 10
    N_REPEATS = 5

    rows = []
    for name, Xd, Qd in [("sift10k", X, Q), ("uniform", X_u, Q_u)]:
        times = []
        for rep in range(1 + N_REPEATS):
            t0 = time.perf_counter()

            for i in range(len(Qd)):
                brute_force(Xd, Qd[i], K)

            t1 = time.perf_counter()

            if rep > 0:   # 1回目はウォームアップなので捨てる
                times.append((t1 - t0) / len(Qd))   # 1クエリあたりの時間(秒)

        ms = np.median(times) * 1000
        rows.append([datetime.now().isoformat(timespec="seconds"), name, Xd.shape[0], Xd.shape[1], K,
                     len(Qd), N_REPEATS, f"{ms:.3f}", f"{1000 / ms:.1f}"])
        print(name, f"{ms:.3f} ms/クエリ", f"{1000 / ms:.1f} QPS")

    with open(ROOT / "results" / "step1_brute_force.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "dataset", "n", "d", "k", "n_queries", "n_repeats", "median_ms_per_query", "qps"])
        w.writerows(rows)


if __name__ == "__main__":
    # step1_1_check_brute_force()
    # step1_2_check_recall()
    # step1_3_ruler()
    step1_4_make_uniform_gt()
    # step1_5_timing()
