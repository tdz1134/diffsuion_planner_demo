# -*- coding: utf-8 -*-
"""混合 A* 的启发式。

只用**可采纳(admissible)下界**, 保证 A* 不因启发而漏解:
  h(pose) = max( 欧氏距离, 2D 有障碍 holonomic 场[cell] ) * weight
  - 欧氏距离: 任何路径 >= 直线距离。
  - holonomic 场: 从 goal 反向 Dijkstra(8 邻域)在占据图上的最短"无朝向约束"距离,
    忽略车辆运动学 => 也是下界, 且在泊车这种障碍主导场景里很紧。
说明: RS 曲线长度**不**用作启发(我们只实现 CSC 子集, 可能比真最优长 -> 不可采纳);
  RS 仅用于"解析扩展"(见 hybrid_a_star), 那不影响可采纳性, 只加速命中目标。
"""

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def holonomic_field(grid, goal_rc, res):
    """从 goal 反向 Dijkstra, 返回 (H,W) 米制距离场; 障碍/不可达 = inf。"""
    grid = np.asarray(grid)
    H, W = grid.shape
    dist = np.full((H, W), np.inf, dtype=np.float64)
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    gr = min(max(gr, 0), H - 1); gc = min(max(gc, 0), W - 1)
    dist[gr, gc] = 0.0
    pq = [(0.0, gr, gc)]
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c] + 1e-9:
            continue
        for dr, dc, w in _NB:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] <= 0:
                nd = d + w * res
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    return dist


class Heuristic:
    """h(x, y) 的 callable。预计算一次 holonomic 场。"""

    def __init__(self, grid, goal_pose, res, weight=1.0):
        self.res = res
        self.weight = weight
        self.gx, self.gy = float(goal_pose[0]), float(goal_pose[1])
        gc = int(round(self.gx / res)); gr = int(round(self.gy / res))
        self.field = holonomic_field(grid, (gr, gc), res)
        self.H, self.W = grid.shape

    def __call__(self, x, y):
        eu = math.hypot(x - self.gx, y - self.gy)
        c = int(round(x / self.res)); r = int(round(y / self.res))
        f = eu
        if 0 <= r < self.H and 0 <= c < self.W:
            fv = self.field[r, c]
            if np.isfinite(fv):
                f = max(eu, float(fv))
        return self.weight * f
