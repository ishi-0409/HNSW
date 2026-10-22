import numpy as np

def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


X  = read_fvecs(r"C:\mydev\hnsw-thesis\tests\siftsmall\siftsmall_base.fvecs")
Q  = read_fvecs(r"C:\mydev\hnsw-thesis\tests\siftsmall\siftsmall_query.fvecs")
GT = read_ivecs(r"C:\mydev\hnsw-thesis\tests\siftsmall\siftsmall_groundtruth.ivecs")

# 全探索
def brute_force(X, q, k):
    dists = np.linalg.norm(X - q, axis=1)
    return np.argsort(dists)[:k]


print(len(brute_force(X, Q[0], 5)))
print(brute_force(X, Q[0], 10))
print(np.array_equal(brute_force(X, Q[0], 10), GT[0][:10]))   # True になるはず

