# hnsw-thesis

HNSW（Hierarchical Navigable Small World）をはじめとする、グラフベースの
近似最近傍探索アルゴリズムの実装と実験。卒業論文用のコードと記録。

## 動かし方

依存関係の管理には [uv](https://docs.astral.sh/uv/) を使う。

```bash
# 依存関係のインストール
uv sync

# テストの実行（数直線の例など、答えが分かる小さな確認）
uv run pytest

# 実験の実行（experiments/ 以下の1ファイルが1つの実験）
uv run python experiments/<実験ファイル>.py
```

## ディレクトリ構成

```
hnsw-thesis/
├── pyproject.toml        # uv が管理
├── uv.lock
├── README.md             # 動かし方と実験の記録
├── src/hnsw_thesis/      # 自作の実装
│   ├── search.py         # アルゴリズム2と5
│   ├── insert.py         # アルゴリズム1
│   ├── select.py         # アルゴリズム3と4（後で NSG や Vamana の選び方も追加）
│   └── metrics.py        # 距離計算のカウント、Recall の計算
├── tests/                # 数直線の例のような、答えが分かる小さな確認
├── experiments/          # 実験スクリプト。1つの実験に1ファイル
├── data/                 # データセット（git に入れない）
├── results/              # 計測結果の CSV
└── figures/              # 卒論に貼るグラフ
```

## 実験の記録

| 日付 | 実験 | 概要 | 結果 |
|------|------|------|------|
|      |      |      |      |
