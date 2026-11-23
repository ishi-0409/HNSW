import numpy as np
import csv

from common import result_path
from hnsw import assign_levels


def step5_1_check_levels():
    n, M = 10000, 10
    mL = 1 / np.log(M)
    levels = assign_levels(n, mL, seed=0)

    print(f"n={n}, M={M}, mL={mL:.4f}")
    rows = []
    for k in range(1, levels.max() + 1):
        count = int((levels >= k).sum())
        expected = n / M**k
        rows.append({"k": k, "count": count, "expected": expected})
        print(f"  レベル>={k}: {count:5d} 点(期待値 {n / M**k:.1f})")
    print(f"  最大レベル: {levels.max()}")

    path = result_path("step5_1_levels.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "count", "expected"])
        writer.writeheader()
        writer.writerows(rows)
    
    # mL=0 なら全点がレベル0
    zero = assign_levels(n, 0.0, seed=0)
    assert (zero == 0).all()

    # 同じ seed なら同じ結果
    assert np.array_equal(levels, assign_levels(n, mL, seed=0))
    print("mL=0 と再現性の確認: OK")


if __name__ == "__main__":
    step5_1_check_levels()