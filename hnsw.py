import numpy as np


def assign_levels(n: int, mL: float, seed: int = 0) -> np.ndarray:
    """各点のレベル l = floor(-ln(U) * mL) を決める(論文 4.1節)。
    
    引数
        n: 点の数
        mL: レベルの出やすさ。1/ln(M)で上の層に約1/M残る。
        seed: レベル用の乱数シード
    戻り値
        長さnのint配列。levels[i]は点iのレベルで、点iは層0~levels[i]に存在する。    
    """
    rng = np.random.default_rng(seed)
    U = 1.0 - rng.random(n)  # (0, 1]。U=0 だと ln(0) = -inf になるため
    return np.floor(-np.log(U) * mL).astype(int)

