import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "step2_greedy_M_entry.csv")
# print(len(df))

cols = ["recall1", "recall10", "n_dist", "hops", "n_topk"]
per_seed = df.groupby(["M", "seed"])[cols].mean()
summary = per_seed.groupby("M").agg(["mean", "min", "max"])

over = df["n_dist"] > 1 + (df["hops"] + 1) * df["M"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(summary.round(3))
print("上限超え:", over.sum())
