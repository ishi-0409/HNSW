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
├── siftsmall/            # ダウンロードした SIFT10K（中身は変更しない）
├── data/                 # 自分で生成したデータ・正解・グラフ（.npy、git に入れない）
├── results/              # 計測結果（csv）と図（png）
├── common.py             # 全 Step で使う関数（読み込み、データ生成、全探索、recall）
├── graph.py              # Step 2 以降用
├── nsw.py                # Step 2 以降用
├── hnsw.py               # Step 2 以降用
├── step0.py              # Step 0 の確認コード
├── step1.py              # Step 1 の確認コード
└── _old/                 # 整理前のファイル
```

## 実験の記録

| 日付 | 実験 | 概要 | 結果 |
|------|------|------|------|
|      |      |      |      |
