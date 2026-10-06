# -*- coding: utf-8 -*-
"""占据栅格工具: 米制<->栅格、C 空间膨胀、距离场/SDF、图元填充。

依赖 OpenCV(cv2, 环境已装)做形态学膨胀与欧氏距离变换; 不依赖 scipy/skimage。
栅格约定: grid[r, c], r=行(y), c=列(x); 1=障碍, 0=空闲; float32/uint8 皆可。
"""

import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- #
# 坐标变换
# --------------------------------------------------------------------------- #
def world_to_cell(x, y, res):
    return int(round(y / res)), int(round(x / res))     # (r, c)


def cell_to_world(r, c, res):
    return c * res, r * res                             # (x, y)


def in_bounds(r, c, H, W):
    return 0 <= r < H and 0 <= c < W


# --------------------------------------------------------------------------- #
# 图元填充(世界坐标矩形 -> 栅格)
# --------------------------------------------------------------------------- #
def fill_rect_world(grid, x0, y0, x1, y1, res, val=1.0):
    """把世界坐标轴对齐矩形 [x0,x1]x[y0,y1] 填成 val。就地修改并返回 grid。"""
    H, W = grid.shape
    r0, c0 = world_to_cell(x0, y0, res)
    r1, c1 = world_to_cell(x1, y1, res)
    r0, r1 = sorted((max(0, r0), min(H - 1, r1)))
    c0, c1 = sorted((max(0, c0), min(W - 1, c1)))
    grid[r0:r1 + 1, c0:c1 + 1] = val
    return grid


def fill_border(grid, thickness=1, val=1.0):
    t = max(1, int(thickness))
    grid[:t, :] = val; grid[-t:, :] = val
    grid[:, :t] = val; grid[:, -t:] = val
    return grid


# --------------------------------------------------------------------------- #
# C 空间膨胀 / 距离场
# --------------------------------------------------------------------------- #
def inflate(grid, radius_m, res):
    """按半径(米)对障碍做形态学膨胀, 返回 uint8 C 空间栅格(保守)。"""
    if cv2 is None:                     # 无 cv2 -> 简单方核退化
        k = max(1, int(round(radius_m / res)))
        return _dilate_box(np.asarray(grid) > 0, k)
    k = max(1, int(round(radius_m / res)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    obs = (np.asarray(grid) > 0).astype(np.uint8)
    return cv2.dilate(obs, kern).astype(np.uint8)


def _dilate_box(mask_bool, k):
    m = mask_bool.astype(np.uint8)
    out = m.copy()
    H, W = m.shape
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr * dr + dc * dc > k * k:
                continue
            out[max(0, dr):H + min(0, dr), max(0, dc):W + min(0, dc)] = np.maximum(
                out[max(0, dr):H + min(0, dr), max(0, dc):W + min(0, dc)],
                m[max(0, -dr):H + min(0, -dr), max(0, -dc):W + min(0, -dc)])
    return out


def distance_field_metric(grid, res, clip_m=None):
    """每个空闲格到最近障碍的欧氏距离(米); 障碍处为 0。

    用 cv2.distanceTransform(输入 = 空闲掩码, 输出 = 到最近 0 像素的距离)。
    """
    g = np.asarray(grid)
    free = (g <= 0).astype(np.uint8)        # 空闲=1, 障碍=0
    if cv2 is not None:
        dist_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    else:
        dist_px = _edt_fallback(free)
    dist_m = dist_px.astype(np.float32) * res
    if clip_m is not None:
        dist_m = np.clip(dist_m, 0.0, clip_m)
    return dist_m


def normalized_sdf(grid, res, clip_m):
    """归一化距离场 [0,1]: 0=贴障碍, 1=离障碍>=clip_m。"""
    d = distance_field_metric(grid, res, clip_m=clip_m)
    return (d / max(clip_m, 1e-6)).astype(np.float32)


def _edt_fallback(free):
    """无 cv2 时的近似: 多源 BFS(格数距离), 仅作降级。"""
    from collections import deque
    H, W = free.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    dq = deque()
    obs = np.argwhere(free == 0)
    for r, c in obs:
        dist[r, c] = 0.0
        dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and np.isinf(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1.0
                dq.append((nr, nc))
    return dist


if __name__ == "__main__":
    res = 0.2
    g = np.zeros((40, 60), dtype=np.float32)
    fill_border(g, 1)
    fill_rect_world(g, 4.0, 4.0, 6.0, 5.0, res)
    print("occupied ratio:", round(float(g.mean()), 3), " cv2:", cv2 is not None)
    d = distance_field_metric(g, res, clip_m=2.0)
    print("dist range:", round(float(d.min()), 2), round(float(d.max()), 2))
    gi = inflate(g, 0.9, res)
    print("inflate occupied ratio:", round(float(gi.mean()), 3))
