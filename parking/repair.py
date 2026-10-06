# -*- coding: utf-8 -*-
"""采样后修复(Phase 2 / M7 的"更强修复"组件): 把扩散原始输出修成可行/避障轨迹。

只做**后处理**, 不改训练损失(曲率/避障损失与采样引导留作进一步工作):
  1) SDF 梯度外推: 航点离障碍太近(< margin)时沿距离场梯度推离;
  2) 拉普拉斯平滑: 同时平滑 x, y 与 cos, sin(再归一化), 去掉抖动 -> 降曲率;
  3) 端点钉住: 首尾航点始终 = start / goal。

输入/输出均为世界坐标 (N,4)[x,y,cos,sin]。margin 取"中心到障碍的安全距离"
  (~车宽/2 + 余量); 矩形足迹的精确避障仍由 evaluate 的足迹检测把关。
"""

import numpy as np

from .occupancy import distance_field_metric


def repair_waypoints(traj4_world, grid, res, margin=1.1, push_iter=40,
                     smooth_iter=40, push_gain=0.6):
    wp = np.asarray(traj4_world, dtype=np.float64).copy()
    if wp.shape[0] < 3:
        return wp
    first, last = wp[0].copy(), wp[-1].copy()
    sdf = distance_field_metric(np.asarray(grid, dtype=np.float64), res)   # 米
    gr, gc = np.gradient(sdf)          # d(sdf)/d(row), d(sdf)/d(col)
    H, W = sdf.shape

    def _push():
        r = np.clip(np.round(wp[:, 1] / res).astype(int), 0, H - 1)
        c = np.clip(np.round(wp[:, 0] / res).astype(int), 0, W - 1)
        d = sdf[r, c]
        need = d < margin
        if not need.any():
            return False
        dx = gc[r, c]; dy = gr[r, c]          # 世界方向: x<-col, y<-row
        nrm = np.hypot(dx, dy) + 1e-8
        step = (margin - d) * push_gain
        wp[need, 0] += dx[need] / nrm[need] * step[need]
        wp[need, 1] += dy[need] / nrm[need] * step[need]
        return True

    for _ in range(push_iter):
        if not _push():
            break
    for _ in range(smooth_iter):
        wp[1:-1, 0] = 0.5 * wp[1:-1, 0] + 0.25 * (wp[:-2, 0] + wp[2:, 0])
        wp[1:-1, 1] = 0.5 * wp[1:-1, 1] + 0.25 * (wp[:-2, 1] + wp[2:, 1])
        wp[1:-1, 2] = 0.5 * wp[1:-1, 2] + 0.25 * (wp[:-2, 2] + wp[2:, 2])
        wp[1:-1, 3] = 0.5 * wp[1:-1, 3] + 0.25 * (wp[:-2, 3] + wp[2:, 3])
        nrm = np.hypot(wp[:, 2], wp[:, 3]) + 1e-9
        wp[:, 2] /= nrm; wp[:, 3] /= nrm
        _push()
    wp[0] = first; wp[-1] = last
    return wp


if __name__ == "__main__":
    # 自检: 造一条抖动+贴障碍的轨迹, 修复后应更平滑且离障碍更远
    from . import occupancy as occ
    res = 0.2
    g = np.zeros((40, 60), dtype=np.float64)
    occ.fill_border(g, 1)
    occ.fill_rect_world(g, 4.0, 3.0, 6.0, 4.0, res)
    rng = np.random.default_rng(0)
    N = 40
    t = np.linspace(0, 1, N)
    x = 1.0 + 9.0 * t + rng.normal(0, 0.3, N)
    y = 5.0 + 0.4 * np.sin(8 * t) + rng.normal(0, 0.3, N)   # 抖动且靠近障碍
    th = np.arctan2(np.gradient(y), np.gradient(x))
    t4 = np.stack([x, y, np.cos(th), np.sin(th)], axis=1)
    from .geometry import curvature_from_poses, cs_to_heading
    def kmax(t4):
        p = np.stack([t4[:, 0], t4[:, 1], cs_to_heading(t4[:, 2], t4[:, 3])], axis=1)
        return np.abs(curvature_from_poses(p)).max()
    sdf = distance_field_metric(g, res)
    def mindist(t4):
        r = np.clip(np.round(t4[:, 1] / res).astype(int), 0, g.shape[0] - 1)
        c = np.clip(np.round(t4[:, 0] / res).astype(int), 0, g.shape[1] - 1)
        return sdf[r, c].min()
    print("before: kappa_max=%.2f min_sdf=%.2f" % (kmax(t4), mindist(t4)))
    fx = repair_waypoints(t4, g, res, margin=1.2)
    print("after : kappa_max=%.2f min_sdf=%.2f" % (kmax(fx), mindist(fx)))
    print("endpoints pinned:", np.allclose(fx[0], t4[0]), np.allclose(fx[-1], t4[-1]))
