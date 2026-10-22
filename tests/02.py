import numpy as np

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
