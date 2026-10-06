# -*- coding: utf-8 -*-
"""SE(2) 几何工具。所有函数输入/输出均为 numpy 数组, 与具体算法解耦。

位姿表示:
  - pose3  = (x, y, theta)                theta 弧度
  - traj4  = (x, y, cos(theta), sin(theta)) 扩散用(无角度环绕问题)
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(a):
    """把角度包到 [-pi, pi)。支持标量/数组。"""
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


def heading_to_cs(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return np.cos(theta), np.sin(theta)


def cs_to_heading(c, s):
    return np.arctan2(np.asarray(s), np.asarray(c))


def poses3_to_traj4(poses):
    """(L,3)[x,y,theta] -> (L,4)[x,y,cos,sin]。"""
    poses = np.asarray(poses, dtype=np.float64)
    c, s = heading_to_cs(poses[:, 2])
    return np.stack([poses[:, 0], poses[:, 1], c, s], axis=1)


def traj4_to_poses3(traj):
    """(L,4)[x,y,cos,sin] -> (L,3)[x,y,theta]。"""
    traj = np.asarray(traj, dtype=np.float64)
    th = cs_to_heading(traj[:, 2], traj[:, 3])
    return np.stack([traj[:, 0], traj[:, 1], th], axis=1)


def cum_arclen(xy):
    """累计弧长。xy:(L,2) -> cum:(L,) 从 0 开始。"""
    xy = np.asarray(xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_poses(poses, n, gear=None):
    """按累计弧长把密集位姿重采样为恰好 n 个。

    poses: (L,3)[x,y,theta]; gear: 可选 (L,) 每点档位(取最近邻)。
    返回 (n,3) 位姿; 若给 gear 再返回 (n,) 档位。
    x,y 线性插值; theta 先 unwrap 再插值; 端点严格对齐首尾。
    """
    poses = np.asarray(poses, dtype=np.float64)
    L = poses.shape[0]
    if L < 2:
        out = np.repeat(poses[:1], n, axis=0)
        return (out, np.zeros(n, dtype=np.int8)) if gear is not None else out

    cum = cum_arclen(poses[:, :2])
    total = cum[-1]
    if total <= 1e-9:
        targets = np.zeros(n)
    else:
        targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cum, poses[:, 0])
    ys = np.interp(targets, cum, poses[:, 1])
    th = np.interp(targets, cum, np.unwrap(poses[:, 2]))
    out = np.stack([xs, ys, th], axis=1)
    out[0] = poses[0]                    # 严格对齐起点
    out[-1] = poses[-1]                  # 严格对齐终点

    if gear is not None:
        gear = np.asarray(gear)
        idx = np.searchsorted(cum, targets, side="left")
        idx = np.clip(idx, 0, L - 1)
        g = gear[idx].astype(np.int8)
        g[0] = gear[0]; g[-1] = gear[-1]
        return out, g
    return out


def curvature_from_poses(poses):
    """由位姿序列估算逐点曲率 kappa = dtheta/ds (含符号)。poses:(L,3)。"""
    poses = np.asarray(poses, dtype=np.float64)
    dtheta = np.zeros(poses.shape[0])
    dtheta[1:-1] = wrap(poses[2:, 2] - poses[:-2, 2])
    dtheta[0] = wrap(poses[1, 2] - poses[0, 2])
    dtheta[-1] = wrap(poses[-1, 2] - poses[-2, 2])
    cum = cum_arclen(poses[:, :2])
    ds = np.gradient(cum)
    ds = np.where(np.abs(ds) < 1e-6, 1e-6, ds)
    return dtheta / ds


def path_length(poses):
    return float(cum_arclen(np.asarray(poses)[:, :2])[-1])


def normalize_xy(xy, x0, y0, x1, y1):
    """世界 xy -> [-1,1](按给定包围盒)。xy:(...,2)。"""
    xy = np.asarray(xy, dtype=np.float64)
    out = xy.copy()
    out[..., 0] = (xy[..., 0] - x0) / max(x1 - x0, 1e-9) * 2.0 - 1.0
    out[..., 1] = (xy[..., 1] - y0) / max(y1 - y0, 1e-9) * 2.0 - 1.0
    return out


def denormalize_xy(nxy, x0, y0, x1, y1):
    nxy = np.asarray(nxy, dtype=np.float64)
    out = nxy.copy()
    out[..., 0] = (nxy[..., 0] + 1.0) / 2.0 * (x1 - x0) + x0
    out[..., 1] = (nxy[..., 1] + 1.0) / 2.0 * (y1 - y0) + y0
    return out


def pose_error(a, b):
    """两姿态误差 -> (位置误差 m, 朝向误差 rad)。a,b=(x,y,theta)。"""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    pos = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    ang = float(abs(wrap(a[2] - b[2])))
    return pos, ang
