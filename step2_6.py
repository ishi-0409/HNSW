import pandas as pd
from pathlib import Path
import numpy as np
from common import read_fvecs, read_ivecs

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
SIFT = ROOT / "siftsmall"
DATA = ROOT / "data"

X  = read_fvecs(SIFT / "siftsmall_base.fvecs")
Q  = read_fvecs(SIFT / "siftsmall_query.fvecs")
GT = read_ivecs(SIFT / "siftsmall_groundtruth.ivecs")

df = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
full = np.load(DATA / "knn_sift10k_M40.npy")
# print(len(df))

# cols = ["recall1", "recall10", "n_dist", "hops", "n_topk"]
# per_seed = df.groupby(["M", "seed"])[cols].mean()
# summary = per_seed.groupby("M").agg(["mean", "min", "max"])

# over = df["n_dist"] > 1 + (df["hops"] + 1) * df["M"]

# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 200)
# print(summary.round(3))
# print("上限超え:", over.sum())
