# -*- coding: utf-8 -*-
"""
占据栅格地图环境 + 专家路径搜索 (A* / 加权A* / 贪心)
=====================================================
为 diffusion path planner 生成训练数据:
  - 随机生成二值占据栅格: 1=障碍(渲染为黑), 0=空闲(渲染为白)
  - 在空闲区随机取 start / goal, 保证二者连通
  - 用 A*(w=1) / 加权A*(w>1) / 贪心(best-first) 求一条专家路径
  - 把路径按弧长重采样成定长 N 个航点, 坐标归一化到 [-1, 1]

坐标约定: 栅格 grid[r, c], r=行(y), c=列(x)。航点用 (x, y) = (c, r)。
"""

import heapq
import math
import numpy as np

SQRT2 = math.sqrt(2.0)

# 8 邻域: (dr, dc, cost)
NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
]


# --------------------------------------------------------------------------- #
# 地图生成
# --------------------------------------------------------------------------- #
def make_map(H=64, W=64, rng=None, n_blocks=14, block_max=10, border=True):
    """随机生成占据栅格。返回 float 数组 (H, W), 1=障碍, 0=空闲。"""
    rng = rng or np.random.default_rng()
    g = np.zeros((H, W), dtype=np.float32)
    # 随机矩形障碍块
    for _ in range(n_blocks):
        bh = rng.integers(3, block_max)
        bw = rng.integers(3, block_max)
        r = rng.integers(1, H - bh - 1)
        c = rng.integers(1, W - bw - 1)
        g[r:r + bh, c:c + bw] = 1.0
    # 少量随机散点障碍
    n_sp = int(H * W * 0.02)
    rs = rng.integers(1, H - 1, n_sp)
    cs = rng.integers(1, W - 1, n_sp)
    g[rs, cs] = 1.0
    if border:
        g[0, :] = 1.0; g[-1, :] = 1.0; g[:, 0] = 1.0; g[:, -1] = 1.0
    return g


def _bfs_reachable(grid, start):
    """从 start 出发 BFS, 返回可达集合(用于挑选连通的 goal)。"""
    H, W = grid.shape
    seen = {start}
    dq = [start]
    while dq:
        r, c = dq.pop()
        for dr, dc, _ in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                dq.append((nr, nc))
    return seen


def sample_start_goal(grid, rng, min_dist=16):
    """在空闲区随机取一对连通且距离足够的 (start, goal), 均为 (r, c)。"""
    H, W = grid.shape
    free = np.argwhere(grid == 0)
    for _ in range(200):
        i = rng.integers(len(free))
        start = tuple(free[i])
        reach = _bfs_reachable(grid, start)
        cand = [p for p in reach
                if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= min_dist]
        if cand:
            goal = cand[rng.integers(len(cand))]
            return start, goal
    return None


# --------------------------------------------------------------------------- #
# 搜索: A* / 加权A* / 贪心
# --------------------------------------------------------------------------- #
def _heuristic(a, b):
    # octile 距离(适配 8 邻域)
    dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2) * min(dr, dc)


def astar(grid, start, goal, w=1.0):
    """A*(w=1) / 加权A*(w>1)。w 越大越贪心。返回 [(r,c)...] 或 None。"""
    if grid[start] != 0 or grid[goal] != 0:
        return None
    open_heap = [(w * _heuristic(start, goal), 0.0, start)]
    came = {}
    gscore = {start: 0.0}
    cnt = 0
    while open_heap:
        f, _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        r, c = cur
        for dr, dc, cost in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if grid[nr, nc] != 0:
                continue
            # 禁止"切角": 对角移动时两个相邻正交格必须都空闲
            if dr != 0 and dc != 0 and (grid[r + dr, c] != 0 or grid[r, c + dc] != 0):
                continue
            ng = gscore[cur] + cost
            nxt = (nr, nc)
            if ng < gscore.get(nxt, float("inf")):
                gscore[nxt] = ng
                came[nxt] = cur
                cnt += 1
                heapq.heappush(open_heap, (ng + w * _heuristic(nxt, goal), cnt, nxt))
    return None


def greedy(grid, start, goal):
    """贪心 best-first: 优先级只用启发式 h, 不看已走代价。快但次优。"""
    return astar(grid, start, goal, w=1e6)


# --------------------------------------------------------------------------- #
# 路径重采样 -> 定长航点(归一化 [-1,1])
# --------------------------------------------------------------------------- #
def resample_path(path_rc, N=32, size=None):
    """把 [(r,c)...] 按弧长重采样为 (N,2) 的 (x,y) 航点, 归一化到 [-1,1]。"""
    pts = np.asarray(path_rc, dtype=np.float64)      # (L,2) = (r,c)
    xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)    # -> (x=c, y=r)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        xy_r = np.repeat(xy[:1], N, axis=0)
    else:
        targets = np.linspace(0, total, N)
        xs = np.interp(targets, cum, xy[:, 0])
        ys = np.interp(targets, cum, xy[:, 1])
        xy_r = np.stack([xs, ys], axis=1)
    if size is None:
        size = max(xy_r.max(), 1.0)
    # 归一化: grid 坐标 [0, size-1] -> [-1, 1]
    norm = xy_r / (size - 1) * 2.0 - 1.0
    return norm.astype(np.float32)


def path_length_grid(path_rc):
    pts = np.asarray(path_rc, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def distance_field(grid):
    """多源 BFS: 每个格到最近障碍的格数距离。障碍处为 0。返回 (H,W) float。"""
    from collections import deque
    H, W = grid.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    dq = deque()
    obs = np.argwhere(grid > 0)
    for r, c in obs:
        dist[r, c] = 0.0
        dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc, _ in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1.0
                dq.append((nr, nc))
    return dist


# --------------------------------------------------------------------------- #
# 一次性生成一条样本: (map, start, goal, path_norm)
# --------------------------------------------------------------------------- #
def make_sample(H=64, W=64, N=32, rng=None, planner="astar", w=1.0):
    rng = rng or np.random.default_rng()
    grid = make_map(H, W, rng)
    sg = sample_start_goal(grid, rng)
    if sg is None:
        return None
    start, goal = sg
    if planner == "greedy":
        path = greedy(grid, start, goal)
    else:
        path = astar(grid, start, goal, w=w)
    if path is None or len(path) < 2:
        return None
    pn = resample_path(path, N=N, size=H)
    return dict(grid=grid, start=start, goal=goal,
                path_rc=path, path_norm=pn)


if __name__ == "__main__":
    # 快速自检: 生成一张图 + 三种搜索路径, 打印长度
    rng = np.random.default_rng(0)
    s = make_sample(rng=rng)
    g, st, gl = s["grid"], s["start"], s["goal"]
    print("map shape:", g.shape, " occupied ratio:", g.mean().round(3))
    print("start:", st, " goal:", gl)
    for name, fn in [("A*", lambda: astar(g, st, gl, 1.0)),
                     ("wA* w=2", lambda: astar(g, st, gl, 2.0)),
                     ("greedy", lambda: greedy(g, st, gl))]:
        p = fn()
        print(f"  {name:8s} len={path_length_grid(p):7.2f}  pts={len(p)}")
    print("resampled waypoints:", s["path_norm"].shape,
          " range:", s["path_norm"].min().round(2), s["path_norm"].max().round(2))
