import numpy as np
import time

def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

X  = read_fvecs("siftsmall/siftsmall_base.fvecs")
Q  = read_fvecs("siftsmall/siftsmall_query.fvecs")
GT = read_ivecs("siftsmall/siftsmall_groundtruth.ivecs")

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



