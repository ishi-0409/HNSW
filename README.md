# hnsw-thesis

HNSW（Hierarchical Navigable Small World）をはじめとする、グラフベースの
近似最近傍探索アルゴリズムの実装と実験。卒業論文用のコードと記録。

## 動かし方

依存関係の管理には [uv](https://docs.astral.sh/uv/) を使う。

```bash
# 依存関係のインストール
uv sync

# 各 Step の確認コードの実行
uv run python stepN.py
```

### SIFT10K

SIFT10K（`siftsmall/`）はリポジトリに含まれているので、ダウンロードは不要。
元データの入手方法は次のとおり（参考）。

```bash
curl.exe -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
```

Windows の PowerShell では `curl` が別コマンドの別名になっているため、`curl.exe` と書く。

### 自分で生成するデータ（data/）

`data/` の中身は git に入らないので、clone した直後は空。最初に次を実行するとデータと正解が作られる。

```bash
# 一様乱数のデータ data/X.npy, data/Q.npy を作る
uv run python step0.py

# 一様乱数の正解 data/GT.npy を作る（step1.py の 1-4 のコメントを外してから実行する）
uv run python step1.py
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
├── graph.py              # Step 2・3（kNN グラフ、貪欲法、ビーム探索）
├── nsw.py                # Step 4
├── hnsw.py               # Step 5
├── step0.py              # Step 0 の確認コード
├── step1.py              # Step 1 の確認コード
└── _old/                 # 整理前のファイル
```

## 実験の記録

| 日付 | 実験 | 概要 | 結果 |
|------|------|------|------|
| 2026-10-22 | Step 1 全探索の基準値 | SIFT10K と一様乱数(d=32)で全探索の速さを測定(k=10、5回の中央値) | SIFT10K 4.49 ms/クエリ(約223 QPS)、一様乱数 1.36 ms/クエリ(約734 QPS)。詳細は results/step1_brute_force.csv |

計測環境: 12th Gen Intel(R) Core(TM) i5-1235U / Windows 11 Pro (10.0.26300) / Python 3.14.7 / NumPy 2.5.3
