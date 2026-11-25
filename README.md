# hnsw-thesis

HNSW（Hierarchical Navigable Small World）をはじめとする、グラフベースの
近似最近傍探索アルゴリズムの実装と実験。卒業論文用のコードと記録。

## 動かし方

依存関係の管理には [uv](https://docs.astral.sh/uv/) を使う。

```bash
# 依存関係のインストール
uv sync

# stepN.py を実行すると、データやグラフを作る関数だけが動く
uv run python stepN.py

# 実験や確認は関数ごとに呼ぶ
uv run python -c "import step5; step5.step5_7_check_vs_nsw()"
```

`stepN.py` は実験(探索して生データの csv を出す)、`stepN_analysis.py` は分析(csv を読んで集計表と図を出す)。

### SIFT10K

SIFT10K（`siftsmall/`）はリポジトリに含まれているので、ダウンロードは不要。
元データの入手方法は次のとおり（参考）。

```bash
curl.exe -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
```

Windows の PowerShell では `curl` が別コマンドの別名になっているため、`curl.exe` と書く。

### 自分で生成するデータ（data/）

`data/` の中身は git に入らないので、clone した直後は空。次の順に実行すると作られる。シードを固定しているので、何度作り直しても同じ値になる。

```bash
# 一様乱数のデータ data/X.npy, data/Q.npy
uv run python step0.py

# 一様乱数の正解 data/GT.npy
uv run python step1.py

# SIFT10K の kNN グラフ data/knn_sift10k_M40.npy
uv run python step2.py

# NSW のグラフ data/nsw_sift10k_M{10,20,40}_efc100_seed0.npz
uv run python -c "import step4; step4.step4_5_build_sift()"

# HNSW のグラフ2〜4 data/hnsw_sift10k_{cap,layered,hnsw}_M{10,20,40}_efc100.npz
# (構築の距離計算回数は .n_dist.npy)。1つ数十秒かかるので M ごとに分けてもよい
uv run python -c "import step5; step5.step5_8_build_sift(Ms=(10,))"
```

## ディレクトリ構成

```
hnsw-thesis/
├── pyproject.toml        # uv が管理
├── uv.lock
├── README.md             # 動かし方と実験の記録
├── siftsmall/            # ダウンロードした SIFT10K（中身は変更しない）
├── data/                 # 自分で生成したデータ・正解・グラフ（.npy、git に入れない）
├── results/              # 計測結果（csv）と図（png）。Step ごとのフォルダ（step1/ 〜）に分ける
├── CLAUDE.md             # Claude Code に作業を頼むときのルール
├── common.py             # 全 Step で使う関数（読み込み、データ生成、全探索、recall、result_path）
├── graph.py              # Step 2・3（kNN グラフ、貪欲法、ビーム探索 search_layer）
├── nsw.py                # Step 4（NSW の挿入と保存）
├── hnsw.py               # Step 5（レベル、Alg. 1・3・4・5、保存）
├── step0.py 〜 step5.py  # 各 Step の実験・確認
└── step2_analysis.py 〜 step4_analysis.py  # 各 Step の分析(csv から表と図)
```

## 実験の記録

| 日付 | 実験 | 概要 | 結果 |
|------|------|------|------|
| 2026-10-22 | Step 1 全探索の基準値 | SIFT10K と一様乱数(d=32)で全探索の速さを測定(k=10、5回の中央値) | SIFT10K 4.49 ms/クエリ(約223 QPS)、一様乱数 1.36 ms/クエリ(約734 QPS)。詳細は results/step1/step1_brute_force.csv |
| 2026-11-25 | Step 5-7 HNSW と NSW の一致 | SIFT10K で mL=0・層0の上限なし・Alg. 3・NSW と同じ挿入順で build_hnsw を作り、Step 4 の NSW(M=10, 20, 40、efConstruction=100)と比較 | 3つとも辺が並び順まで一致。構築の距離計算回数は NSW よりちょうど1回多い(NSW は2点目を探索せずにつなぐため)。詳細は results/step5/step5_7_vs_nsw.csv |
| 2026-11-25 | Step 5-8 グラフ2〜4 の構築 | SIFT10K で上限だけ(cap)・階層あり(layered)・HNSW を M=10, 20, 40(efConstruction=100)で作成 | 構築の距離計算回数の NSW との比は、M=10 で cap 0.89・layered 0.81・HNSW 1.55、M=40 で 1.95・1.85・0.99。HNSW の層0の平均次数は M によらず 12〜16(上限 2M)。詳細は results/step5/step5_8_build.csv、step5_8_layers.csv |

計測環境: 12th Gen Intel(R) Core(TM) i5-1235U / Windows 11 Pro (10.0.26300) / Python 3.14.7 / NumPy 2.5.3
