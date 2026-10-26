# hnsw-thesis

卒論用のリポジトリ。HNSW を Python で自作し、hnswlib と同じデータで recall と距離計算回数の傾向が一致するかを確かめる。
全探索 → kNN グラフ + 貪欲法 → ビーム探索(ef) → NSW → HNSW の順に積み上げる(Step 0〜7)。

参考論文: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"(IEEE TPAMI 2020、arXiv 初版 2016)。コード中のアルゴリズム番号(Alg. 1〜5)は論文に合わせる。

## 実行

- 実行は `uv run python stepN.py`
- `data/` の .npy は git に入らない。`step0.py`(X.npy・Q.npy)→ `step1.py`(GT.npy)→ `step2.py`(kNN グラフ)の順に `uv run python` で実行すると作り直される(シード固定なので毎回同じ値)
- `siftsmall/`(SIFT10K)は配布データ。変更しない

## 構成

- `common.py`: 全 Step で使う道具(SIFT10K の読み込み `load_sift`、全探索、recall、距離計算のカウンタ)
- `graph.py`: Step 2・3(kNN グラフ、貪欲法、ビーム探索 `search_layer`)
- `nsw.py`: Step 4 / `hnsw.py`: Step 5
- `stepN.py`: 実験(探索を実行して生データの csv を出す)。実験ごとに関数に分け、`if __name__ == "__main__":` からはデータやグラフを作る関数だけを呼ぶ
- `stepN_analysis.py`: 分析(csv を読んで集計表と図を出す。探索はしない)
- `results/`: 計測結果の csv と図(png)
- `_old/`: 整理前のファイル。参照しない

## コードのルール

- モジュール(common.py など)には関数定義だけを書く。import しただけで処理が走らないようにする
- パスは `ROOT = Path(__file__).parent` を基準に組み立てる
- 乱数はシードを固定する。乱数を引く順番も変えない(順番が変わると同じシードでも値が変わる)。データ・正解・グラフは `data/` に保存して使い回す
- クエリはデータとは別に用意する(データから抜くと、クエリ自身が距離 0 で 1 位になる)
- 浮動小数点の比較は `==` ではなく `np.allclose`
- 行列は大文字(`X`, `Q`)、ベクトル 1 本は小文字(`x`, `q`)

## 計測のルール

- 自作の手法同士は「距離計算回数」と「recall@k」で比べる。hnswlib(C++)との実時間比較はしない
- 比較は「ある recall(0.9、0.95、0.99 など)での値」の形で揃える
- 時間を測るときは、ウォームアップ 1 回のあと複数回測って中央値を取る。同じ PC でも実行ごとに 2 割ほど揺れる
- 結果は `results/` に csv と png で保存する。時間計測のように実行ごとに値が変わる csv には `date` 列を付ける。シード固定で毎回同じ結果になる csv には付けない(git で変更が出ないことを、結果が変わっていない確認に使うため)
- README などに書く数値は、必ずコードを実行して csv に出した値にする

## 進め方

- ユーザーは学部生で、自分で説明できるようになることが目的。説明は日本語で、コードは Step の小項目単位で少しずつ書く
- 変更の前に方針を示す。依頼の前提と実際のコードが違っていたら、作業の前に報告する
- 既存のコードの書き方(短いコメント、変数名)に合わせる
- コミットや push の前に `git status` と差分を見せる