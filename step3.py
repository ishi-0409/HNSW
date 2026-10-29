import csv
from pathlib import Path
import time

import numpy as np
import pandas as pd

from common import load_sift, brute_force, recall_at_k
from graph import build_knn_graph, greedy_search, search_layer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def step3_1_check_init() -> None:
    X, Q, GT = load_sift()
    graph = np.load(DATA / "knn_sift10k_M40.npy")[:, :10]
    for ef in [1, 3]:
        res = search_layer(X, graph, Q[0], [0], ef=ef)
        print(ef, res)
    print(np.linalg.norm(X[0] - Q[0]))

# 貪欲法とef=1のビーム探索の出力を比較
def step3_6_compare_greedy() -> None:
    X, Q, GT = load_sift()
    graph = np.load(DATA / "knn_sift10k_M40.npy")[:, :10]
    q, entry = Q[0], 0

    g = greedy_search(X, graph, q, entry)
    s = search_layer(X, graph, q, [entry], ef=1)

    print("経路", [p for p, _ in g["path"]])
    print("停止点", g["stop"], s["topk"][0], g["stop"] == s["topk"][0])
    print("n_dist", g["n_dist"], s["n_dist"], g["n_dist"] == s["n_dist"])
    print("ホップ", g["hops"], s["n_expand"] - 1, g["hops"] == s["n_expand"] - 1)

# 2000クエリで貪欲法をef=1のビーム探索を比較
def step3_7_check_all_ef1() -> None:
    X, Q, GT = load_sift()
    full = np.load(DATA / "knn_sift10k_M40.npy")
    df2 = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")

    rows = []
    for r in df2.itertuples():
        s = search_layer(X, full[:, :r.M], Q[r.query], [int(r.entry)], ef=1)
        rows.append({
            "M": r.M, "seed": r.seed, "query": r.query, "entry": r.entry,
            "stop2": r.stop, "stop3": s["topk"][0],
            "n_dist2": r.n_dist, "n_dist3": s["n_dist"],
            "hops2": r.hops, "hops3": s["n_expand"] - 1,
        })
    df = pd.DataFrame(rows)
    df["match"] = ((df["stop2"] == df["stop3"])
                   & (df["n_dist2"] == df["n_dist3"])
                   & (df["hops2"] == df["hops3"]))
    df.to_csv(RESULTS / "step3_7_ef1_vs_greedy.csv", index=False)

    print("行", len(df), "一致", int(df["match"].sum()))
    print(df[-df["match"]])

def step3_8_trial_ef() -> None:
    X, Q, GT = load_sift()
    M = 10
    graph = np.load(DATA / "knn_sift10k_M40.npy")[:, :M]
    q = Q[0]
    d_all = np.linalg.norm(X - q, axis=1)

    for ef in [1, 3, 10, 32, 64]:
        s = search_layer(X, graph, q, [0], ef=ef)
        topk = s["topk"]

        ok_len = len(topk) == min(ef, s["n_dist"])
        ok_unique = len(set(topk)) == len(topk)
        ok_sorted = bool(np.all(np.diff(d_all[topk]) >= 0))
        ok_bound = s["n_dist"] <= 1 + s["n_expand"] * M

        rank1 = int((d_all < d_all[topk[0]]).sum()) + 1
        r10 = recall_at_k(topk[:10], GT[0], 10) if ef >= 10 else "-"

        print(ef, s["n_dist"], s["n_expand"], rank1, r10, "|",
              ok_len, ok_unique, ok_sorted, ok_bound)

def step3_10_beam_M_ef(
        Ms: tuple[int, ...] = (10, 20, 40),
        efs: tuple[int, ...] = (10, 16, 32, 64, 128, 256),
        save: bool = True,
        ) -> None:
    X, Q, GT = load_sift()
    full = np.load(DATA / "knn_sift10k_M40.npy")

    df2 = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
    entries = df2[df2["M"] == 10][["seed", "query", "entry"]]

    d_all = np.stack([np.linalg.norm(X - q, axis=1) for q in Q])

    rows = []
    t0 = time.perf_counter()
    for M in Ms:
        graph = full[:, :M]
        for ef in efs:
            for r in entries.itertuples():
                s = search_layer(X, graph, Q[r.query], [int(r.entry)], ef=ef)
                top1 = s["topk"][0]
                rows.append({
                    "M": M, "ef": ef, "seed": r.seed, "query": r.query, "entry": r.entry,
                    "top1": top1,
                    "rank_top1": int((d_all[r.query] < d_all[r.query, top1]).sum()) + 1,
                    "recall1": recall_at_k([top1], GT[r.query], 1),
                    "recall10": recall_at_k(s["topk"][:10], GT[r.query], 10),
                    "n_dist": s["n_dist"], "n_expand": s["n_expand"],
                    "n_topk": len(s["topk"]),
                })
            print(f"M={M} ef={ef} 経過 {time.perf_counter() - t0:.1f} 秒")

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(RESULTS / "step3_beam_M_ef.csv", index=False)
    else:
        print(df.groupby(["M", "ef"])[["recall1", "recall10", "n_dist"]].mean())


if __name__ == "__main__":
    # step3_1_check_init()
    # step3_6_compare_greedy()
    # step3_7_check_all_ef1()
    step3_10_beam_M_ef()

