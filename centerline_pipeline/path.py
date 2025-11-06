from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple

import numpy as np


CONNECTIVITY_OFFSETS = {
    4: (
        ((-1, 0), 1.0),
        ((1, 0), 1.0),
        ((0, -1), 1.0),
        ((0, 1), 1.0),
    ),
    8: (
        ((-1, 0), 1.0),
        ((1, 0), 1.0),
        ((0, -1), 1.0),
        ((0, 1), 1.0),
        ((-1, -1), np.sqrt(2.0)),
        ((-1, 1), np.sqrt(2.0)),
        ((1, -1), np.sqrt(2.0)),
        ((1, 1), np.sqrt(2.0)),
    ),
}


def shortest_path(
    cost: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    connectivity: int = 8,
) -> np.ndarray:
    """Run Dijkstra on the provided cost volume and return the optimal path."""

    if connectivity not in CONNECTIVITY_OFFSETS:
        raise ValueError(f"Unsupported connectivity: {connectivity}")

    rows, cols = cost.shape
    sr, sc = start
    er, ec = end
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("Start point lies outside the mask bounds")
    if not (0 <= er < rows and 0 <= ec < cols):
        raise ValueError("End point lies outside the mask bounds")
    if not np.isfinite(cost[sr, sc]) or not np.isfinite(cost[er, ec]):
        raise ValueError("Start or end resides in an impassable region (infinite cost)")

    dist = np.full((rows, cols), np.inf, dtype=float)
    prev = np.full((rows, cols, 2), -1, dtype=np.int32)
    dist[sr, sc] = 0.0
    queue: List[Tuple[float, int, int]] = []
    heapq.heappush(queue, (0.0, sr, sc))

    offsets = CONNECTIVITY_OFFSETS[connectivity]

    while queue:
        cur_cost, r, c = heapq.heappop(queue)
        if cur_cost > dist[r, c]:
            continue
        if r == er and c == ec:
            break

        for (dr, dc), step_len in offsets:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            edge_cost = 0.5 * (cost[r, c] + cost[nr, nc])
            if not np.isfinite(edge_cost):
                continue
            candidate = cur_cost + edge_cost * step_len
            if candidate < dist[nr, nc]:
                dist[nr, nc] = candidate
                prev[nr, nc, 0] = r
                prev[nr, nc, 1] = c
                heapq.heappush(queue, (candidate, nr, nc))

    if not np.isfinite(dist[er, ec]):
        raise RuntimeError("Failed to find a feasible path between start and end points")

    path: List[Tuple[int, int]] = []
    r, c = er, ec
    while r >= 0 and c >= 0:
        path.append((r, c))
        r, c = int(prev[r, c, 0]), int(prev[r, c, 1])
        if r == sr and c == sc:
            path.append((sr, sc))
            break
    path.reverse()
    return np.asarray(path, dtype=np.int32)
