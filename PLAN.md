# HNSW 実装・評価 計画書

HNSW（論文: Malkov & Yashunin のアルゴリズム1〜5）を自分で実装し、
**データの種類（次元 D・分布）とデータ数 N を振って性能を評価する**ことが目的。

- 精度指標: **recall@k**
- 速度指標: **距離計算回数** および **検索時間**
- 正解(ground truth): **brute-force（総当たり）** の結果を基準にする

---

## リポジトリ構成

```
hnsw-thesis/
├── src/hnsw_thesis/
│   ├── search.py    # アルゴリズム2 (SEARCH-LAYER), 5 (K-NN-SEARCH)
│   ├── insert.py    # アルゴリズム1 (INSERT)
│   ├── select.py    # アルゴリズム3 (SELECT-NEIGHBORS-SIMPLE), 4 (HEURISTIC)
│   └── metrics.py   # 距離計算カウント, recall, brute-force
├── experiments/
│   └── nsw_step1.py # SEARCH-LAYER の学習実装（数直線の例・動作確認済み）
├── tests/           # 各アルゴリズムの答え合わせテスト
├── data/            # データセット（自作・標準データ）
├── results/         # 実験結果の CSV
└── figures/         # グラフ
```

---

## フェーズ1：HNSW 本体を完成させる（アルゴリズム1〜5）

`experiments/nsw_step1.py` で動いた SEARCH-LAYER を土台に `src/hnsw_thesis/` へ育てる。
**各ステップで小さい例のテストを書いてから次へ進む。**

| 順 | 実装するもの | 置き場所 | 内容 |
|----|------------|---------|------|
| 1 | dist + 距離カウンタ | `metrics.py` | 今の `dist` を移植。回数カウントを共通化 |
| 2 | SEARCH-LAYER (Alg2) | `search.py` | 学習実装を移植（`df` をループ内で更新する版に直す） |
| 3 | SELECT-NEIGHBORS (Alg3→4) | `select.py` | まず単純版(3)、動いたら発見的手法(4) |
| 4 | INSERT (Alg1) | `insert.py` | 点を1つずつ挿入して多層グラフを構築 |
| 5 | K-NN-SEARCH (Alg5) | `search.py` | 全層をたどって最終的な k 近傍を返す |

## フェーズ2：正しさの検証

- `metrics.py` に **brute-force**（全点と距離を測る総当たり）を実装 → 正解(ground truth)。
- **recall@k** を実装：HNSW の結果が正解を何割当てたか。
- 小〜中規模 (N=1000 程度) で recall が十分高いか確認。
  **ここが合格して初めて計測の意味が出る。**

## フェーズ3：評価の土台（データと実験ドライバ）

- **データ生成器**: N・D・分布（一様 / 正規 / クラスタ）を指定して点群を作る。
- **実験ドライバ**: パラメータを振って回し、結果を `results/` に CSV 保存。
  - 振る軸: `N`, `D`, 分布, `M`, `efConstruction`, `ef`
  - 記録する値: recall@k, 距離計算回数, 検索時間, 構築時間

## フェーズ4：計測と可視化（卒論の主役）

- CSV を読んで `figures/` にグラフ。
  - 例: recall vs 距離計算回数のトレードオフ曲線、N を増やしたときの検索時間
- 実データ（SIFT1M など）を足して「本物」での結果を出す。
- （任意）`hnswlib` など既存実装と比較。

---

## 評価で振るデータの軸

| 軸 | 例 | 何が分かるか |
|----|----|----|
| 次元数 D | 2, 10, 50, 128, 960 | 高次元での性能低下（次元の呪い） |
| データ数 N | 1千, 1万, 10万, 100万 | 規模が増えたときの速度・精度 |
| 分布 | 一様 / 正規 / クラスタ状 | データの偏りの影響 |
| 実データ | SIFT1M, GloVe, Fashion-MNIST | 本物での性能（卒論で説得力が出る） |

---

## 直近の 3 ステップ

1. 今日の SEARCH-LAYER を `src/hnsw_thesis/search.py` に移植し、
   `tests/test_search.py` に数直線の例をテストとして入れる。
2. `metrics.py` に dist + カウンタを移して共通化。
3. SELECT-NEIGHBORS-SIMPLE (Alg3) に進む。
