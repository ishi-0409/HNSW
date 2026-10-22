from pathlib import Path

import numpy as np

from common import read_fvecs, read_ivecs, brute_force

ROOT = Path(__file__).parent
SIFT = ROOT / "siftsmall"


X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

print(X.shape, Q.shape, GT.shape)   # (10000, 128) (100, 128) (100, 100) になるはず


print(len(brute_force(X, Q[0], 5)))
print(brute_force(X, Q[0], 10))
print(np.array_equal(brute_force(X, Q[0], 10), GT[0][:10]))   # True になるはず
