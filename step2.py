from pathlib import Path
import numpy as np
from common import read_fvecs, read_ivecs, brute_force, recall_at_k
from graph import build_knn_graph, greedy_search
import csv

ROOT = Path(__file__).parent
SIFT = ROOT / "siftsmall"

# ---- データ読み込み(SIFT10K) ----
X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")
N = len(X)

M_MAX = 40
path = ROOT / "data" / f"knn_sift10k_k{M_MAX}.npy"
# knnグラフがなければ作成する
if path.exists():
    graph = np.load(path)
else:
    graph = build_knn_graph(X, M_MAX)
    np.save(path, graph)

#print(graph.shape, graph.dtype)

expected = brute_force(X, X[0], 11)
expected = expected[expected != 0][:10]
#print(np.array_equal(graph[0, :10], expected))

#print(np.any(graph == np.arange(N)[:, None]))

#入次数0の点を数える
# rows = []
# for M in [5, 10, 20, 40]:
#     indeg = np.bincount(graph[:, :M].ravel(), minlength=N)
#     rows.append([M, (indeg == 0).sum(), int(indeg.max()), indeg.max() / M])

# # csvで保存
# with open(ROOT / "results" / "step2_indegree.csv", "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["M", "indeg_zero", "indeg_max", "max_over_M"])
#     w.writerows(rows)


# 貪欲法のテスト
M = 10
g = graph[:, :M]
# cur, hops, n_dist = greedy_search(X, g, Q[0], entry=0)
# print("止まった点:", cur, " 真の1位:", GT[0][0])
# print("ホップ数:", hops, " 距離計算:", n_dist)

# d_all = np.linalg.norm(X - Q[0], axis=1)
# order = np.argsort(d_all)
# print("点74の距離:", d_all[74], " 真の1位の距離:", d_all[GT[0][0]])
# print("点74の近所の最小距離:", d_all[g[74]].min())
# print("点74は全体で何位か:", np.where(order == 74)[0][0] + 1)

# 貪欲法の経路表示
# cur, hops, n_dist, path = greedy_search(X, g, Q[0], entry=0, return_path=True)
# for p, d in path:
#     print(p, d)

rows = []
for qi in range(len(Q)):
    res = greedy_search(X, g, Q[qi], entry=0)
    r1 = recall_at_k([res["stop"]], GT[qi], 1)
    r10 = recall_at_k(res["topk"], GT[qi], 10)
    rows.append([qi, r1, r10, res["n_dist"], res["hops"], len(res["topk"])])

arr = np.array(rows)          # (100, 6) の表にする
print("recall@1 :", arr[:, 1].mean())
print("recall@10:", arr[:, 2].mean())
print("距離計算 :", arr[:, 3].mean())
print("ホップ数 :", arr[:, 4].mean())

with open(ROOT / "results" / "step2_greedy_M10_entry0.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["query", "recall1", "recall10", "n_dist", "hops", "n_topk"])
    w.writerows(rows)