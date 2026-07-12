"""NumPy prototype for task325: 4-connected component count -> cyan diagonal."""
from __future__ import annotations
import numpy as np
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.uint8) == 8
    seen = np.zeros_like(a, dtype=bool)
    count = 0
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if not a[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1] and a[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
    out = np.zeros((count, count), dtype=np.uint8)
    np.fill_diagonal(out, 8)
    return out.tolist()
