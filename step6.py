import numpy as np

from common import load_sift, recall_at_k


def build_hnswlib(X: np.ndarray, M: int, ef_construction: int = 100, seed: int = 0):
    """hnswlib で番号順に全点を入れる。並列で順番が変わらないよう1スレッドで入れる"""
    import hnswlib

    index = hnswlib.Index(space="l2", dim=X.shape[1])
    index.init_index(max_elements=len(X), M=M, ef_construction=ef_construction, random_seed=seed)
    index.add_items(X, np.arange(len(X)), num_threads=1)
    return index


def step6_1_check(M: int = 10, ef: int = 256):
    """hnswlib が入っていて、SIFT10K で recall@10 が 1.0 に近いか"""
    import hnswlib

    print(f"hnswlib {hnswlib.__version__ if hasattr(hnswlib, '__version__') else '(バージョン不明)'}")
    X, Q, GT = load_sift()
    index = build_hnswlib(X, M)
    index.set_ef(ef)
    labels, _ = index.knn_query(Q, k=10, num_threads=1)
    r = np.mean([recall_at_k(labels[i], GT[i], 10) for i in range(len(Q))])
    print(f"SIFT10K M={M} ef={ef}: recall@10 = {r:.3f}")
    assert r > 0.95
    print("6-1 の確認: OK")


if __name__ == "__main__":
    step6_1_check()
