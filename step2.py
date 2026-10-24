from pathlib import Path
import numpy as np
from common import read_fvecs, brute_force
from graph import build_knn_graph
import csv

ROOT = Path(__file__).parent
X = read_fvecs(ROOT / "siftsmall" / "siftsmall_base.fvecs")
N = len(X)

M_MAX = 40
path = ROOT / "data" / f"knn_sift10k_k{M_MAX}.npy"
# knnグラフがなければ作成する
if path.exists():
    graph = np.load(path)
else:
    graph = build_knn_graph(X, M_MAX)
    np.save(path, graph)

print(graph.shape, graph.dtype)

expected = brute_force(X, X[0], 11)
expected = expected[expected != 0][:10]
print(np.array_equal(graph[0, :10], expected))

print(np.any(graph == np.arange(N)[:, None]))

#入次数0の点を数える
rows = []
for M in [5, 10, 20, 40]:
    indeg = np.bincount(graph[:, :M].ravel(), minlength=N)
    rows.append([M, (indeg == 0).sum(), int(indeg.max()), indeg.max() / M])

# csvで保存
with open(ROOT / "results" / "step2_indegree.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["M", "indeg_zero", "indeg_max", "max_over_M"])
    w.writerows(rows)
