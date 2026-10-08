import heapq
import numpy as np

# ------------------------------------------------------------
# 距離計算（回数を数える。これが後で比較の指標になる）
# ------------------------------------------------------------
dist_count = 0


def dist(a, b):
    global dist_count
    dist_count += 1
    return float(np.linalg.norm(a - b))


# ------------------------------------------------------------
# アルゴリズム2：SEARCH-LAYER
#   q      : クエリのベクトル
#   ep     : 出発点の ID のリスト
#   ef     : 返す近傍の数（候補リストの大きさ）
#   graph  : {点ID: [隣の点IDのリスト]}
#   points : 全点のベクトル（points[i] が点 i）
#   戻り値 : [(距離, 点ID), ...] を距離の近い順に並べたもの
# ------------------------------------------------------------
def search_layer(q:np.ndarray,
                ep:list[int],
                ef:int,
                graph:dict[int, list[int]],
                points:np.ndarray,
                ) -> list[tuple[float, int]]:

    visited = set(ep)  # 論文の v
    C = []             # 論文の C。最小ヒープ (距離, 点ID)：一番近い点が先頭
    W = []             # 論文の W。最大ヒープ (-距離, 点ID)：一番遠い点が先頭


    # 初期化
    for e in ep:
        d = dist(q, points[e])
        heapq.heappush(C, (d, e))
        heapq.heappush(W, (-d, e))


    while C:
        dc , c = heapq.heappop(C) # Cから近い点を取り出す
        df = -W[0][0] # Wの中で一番遠い点fを見る
        
        if df < dc: # c が　f より遠い場合break
            break
        
        # Cの隣の点ｃを見る
        for e in graph[c]:
            if e in visited:
                continue
            
            # visitedに追加して距離計算
            visited.add(e)
            de = dist(q, points[e])
        
            # e が fより近い or W　がef 個未満
            if de < df or len(W) < ef:
                heapq.heappush(C, (de, e))
                heapq.heappush(W, (-de, e))

                if len(W) > ef:
                    heapq.heappop(W)

    return sorted((-d , e) for d , e in W)



# ------------------------------------------------------------
# 答え合わせ：前に手で追った数直線の例
#   A=0, B=3, D=5, E=8, F=10、辺は A-B, B-D, D-E, E-F
#   q=7、出発点 A、ef=2 → 答えは E（距離1）と D（距離2）
# ------------------------------------------------------------
if __name__ == "__main__":
    names = ["A", "B", "D", "E", "F"]
    points = np.array([[0.0], [3.0], [5.0], [8.0], [10.0]])
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
    q = np.array([7.0])

    result = search_layer(q, ep=[0], ef=2, graph=graph, points=points)
    print("結果:", [(names[e], d) for d, e in result])
    print("期待: [('E', 1.0), ('D', 2.0)]")
    print("距離計算の回数:", dist_count, "（期待: 5）")
