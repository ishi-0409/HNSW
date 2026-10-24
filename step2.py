from pathlib import Path
import numpy as np
from common import read_fvecs, brute_force
from graph import build_knn_graph

ROOT = Path(__file__).parent
X = read_fvecs(ROOT / "siftsmall" / "siftsmall_base.fvecs")
N = len(X)

K_MAX = 40
path = ROOT / "data" / f"knn_sift10k_k{K_MAX}.npy"
if path.exists():
    graph = np.load(path)
else:
    graph = build_knn_graph(X, K_MAX)
    np.save(path, graph)

print(graph.shape, graph.dtype)

expected = brute_force(X, X[0], 11)
expected = expected[expected != 0][:10]
print(np.array_equal(graph[0, :10], expected))

print(np.any(graph == np.arange(N)[:, None]))
