import heapq

h = []
heapq.heappush(h, (0.5, 3))
heapq.heappush(h, (0.2, 7))
heapq.heappush(h, (0.9, 1))

# print(heapq.heappop(h))
# print(heapq.heappop(h))
# print(heapq.heappop(h))

w = []
heapq.heappush(w, (-0.5, 3))
heapq.heappush(w, (-0.2, 7))
heapq.heappush(w, (-0.9, 1))

# print(heapq.heappop(w))
# print(heapq.heappop(w))
# print(heapq.heappop(w))


dists = [0.8, 0.3, 0.9, 0.1, 0.5, 0.7, 0.2, 0.6]
u = []

for i, d in enumerate(dists):
    heapq.heappush(u, (-d, i))
    if len(u) > 5:
        heapq.heappop(u)

print(sorted(-x for x, _ in u))
