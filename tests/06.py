import numpy as np

rng = np.random.default_rng(42)
X = rng.random((10000, 32), dtype=np.float32)
Q = rng.random((100, 32), dtype=np.float32)
print(X[0])
print(X.shape)
print(Q.shape)

np.save("X.npy", X)
np.save("Q.npy", Q)

X2 = np.load("X.npy")
Q2 = np.load("Q.npy")

print(np.array_equal(X, X2))


