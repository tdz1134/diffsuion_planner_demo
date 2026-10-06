# -*- coding: utf-8 -*-
"""Reeds-Shepp 解析曲线(CSC 家族: LSL / RSR / LSR / RSL)。

用途: 混合 A* 的"解析扩展"(接近目标时一步连上), 加速强约束泊车搜索。

正确性策略(重要):
  闭式解对不确定的符号(如 LSR 的切线方向)**同时生成两个候选**, 再用自行车模型
  **数值积分**每条候选并**校验终点是否命中目标**(位置/朝向容差内)。只有校验通过的
  才可能被采用, 取弧长最短者。=> 即便某条闭式公式记错, 也只是被剔除, 绝不返回错路径。

范围: 仅 CSC(同向 LSL/RSR + 混合转向 LSR/RSL)及其"整体倒车"版本(通过交换起终点
  再反转得到)。混合档位的多次换挡由主搜索负责, 这里不做(Phase 2 可加 CCC/5 段族)。
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from .geometry import wrap
from .vehicle import kinematic_step_k

ZERO = 1e-6


def _mod2pi(a):
    return a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi))


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


# --------------------------------------------------------------------------- #
# 局部坐标系(把 goal 变到 start 系, 并按半径 rho 归一化)
# --------------------------------------------------------------------------- #
def _local(start, goal, rho):
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    lx = (c * dx + s * dy) / rho
    ly = (-s * dx + c * dy) / rho
    lphi = float(wrap(goal[2] - start[2]))
    return lx, ly, lphi


# --------------------------------------------------------------------------- #
# CSC 闭式解 -> 候选段序列 [(ctype, 归一化长度>=0), ...]
# --------------------------------------------------------------------------- #
def _csc_candidates(x, y, phi) -> List[List[Tuple[str, float]]]:
    cands: List[List[Tuple[str, float]]] = []

    # --- LSL: C1=(0,1), C2=(x-sin, y+cos) ---
    dx, dy = x - math.sin(phi), y - 1.0 + math.cos(phi)
    u, t = _polar(dx, dy)
    t = _mod2pi(t); v = _mod2pi(phi - t)
    if u >= -ZERO and t >= -ZERO and v >= -ZERO:
        cands.append([("L", t), ("S", u), ("L", v)])

    # --- RSR: C1=(0,-1), C2=(x+sin, y-cos) ---
    dx, dy = x + math.sin(phi), y - 1.0 - math.cos(phi) + 2.0   # C2-C1 = (x+sin, y-cos+1)
    u, alpha = _polar(dx, dy)
    t = _mod2pi(-alpha); v = _mod2pi(alpha - phi)
    if u >= -ZERO and t >= -ZERO and v >= -ZERO:
        cands.append([("R", t), ("S", u), ("R", v)])

    # --- LSR: C1=(0,1), C2=(x+sin, y-cos); 交叉切线, 两个符号候选 ---
    dx, dy = x + math.sin(phi), y - 1.0 - math.cos(phi)
    d, alpha = _polar(dx, dy)
    if d >= 2.0:
        u = math.sqrt(max(d * d - 4.0, 0.0))
        gamma = math.atan2(2.0, u)
        for psi in (alpha - gamma, alpha + gamma):
            t = _mod2pi(psi)                 # 首段左弧: 0 -> psi
            v = _mod2pi(psi - phi)           # 末段右弧: psi -> phi (递减)
            if u >= -ZERO and t >= -ZERO and v >= -ZERO:
                cands.append([("L", t), ("S", u), ("R", v)])

    # --- RSL: C1=(0,-1), C2=(x-sin, y+cos); 交叉切线, 两个符号候选 ---
    dx, dy = x - math.sin(phi), y + 1.0 + math.cos(phi)   # C2-C1
    d, alpha = _polar(dx, dy)
    if d >= 2.0:
        u = math.sqrt(max(d * d - 4.0, 0.0))
        gamma = math.atan2(2.0, u)
        for psi in (alpha - gamma, alpha + gamma):
            t = _mod2pi(-psi)                # 首段右弧: 0 -> psi (顺时针)
            v = _mod2pi(phi - psi)           # 末段左弧: psi -> phi (递增)
            if u >= -ZERO and t >= -ZERO and v >= -ZERO:
                cands.append([("R", t), ("S", u), ("L", v)])

    return cands


# --------------------------------------------------------------------------- #
# 段序列 -> 密集位姿(数值积分, 用于验证 + 碰撞检查)
# --------------------------------------------------------------------------- #
_STEER = {"L": 1.0, "R": -1.0, "S": 0.0}


def _integrate(start_pose, segs_norm, rho, wheelbase, max_steer, ds, gear):
    """按段序列从 start_pose 积分, 返回 (poses(L,3), gears(L,))。gear=±1 整体档位。"""
    x, y, th = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    poses = [(x, y, th)]
    gears = [gear]
    for ctype, L in segs_norm:
        meters = L * rho * gear              # gear=-1 -> 负弧长(倒车)
        steer = _STEER[ctype] * max_steer
        n = max(1, int(math.ceil(abs(meters) / ds)))
        step = meters / n
        for _ in range(n):
            x, y, th = kinematic_step_k(x, y, th, steer, step, wheelbase)
            poses.append((x, y, th))
            gears.append(gear)
    return np.asarray(poses, dtype=np.float64), np.asarray(gears, dtype=np.int8)


def _reaches(poses, goal, pos_tol, ang_tol):
    p = poses[-1]
    return (math.hypot(p[0] - goal[0], p[1] - goal[1]) <= pos_tol
            and abs(float(wrap(p[2] - goal[2]))) <= ang_tol)


# --------------------------------------------------------------------------- #
# 对外主函数
# --------------------------------------------------------------------------- #
def reeds_shepp_path(start, goal, rho, wheelbase, max_steer, ds=0.15,
                     pos_tol=0.30, ang_tol=0.20) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """求 start->goal 的一条 RS 曲线(前向或整体倒车), 返回 (poses(L,3), gears(L,), length) 或 None。

    rho: 最小转弯半径; 结果经数值积分验证端点命中。
    """
    start = tuple(float(v) for v in start)
    goal = tuple(float(v) for v in goal)
    best = None       # (length, poses, gears)

    # 前向: 在 start 系解 CSC, 从 start 积分, 校验到 goal
    for segs in _csc_candidates(*_local(start, goal, rho)):
        poses, gears = _integrate(start, segs, rho, wheelbase, max_steer, ds, +1)
        if _reaches(poses, goal, pos_tol, ang_tol):
            length = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
            if best is None or length < best[0]:
                best = (length, poses, gears)

    # 整体倒车: 在 goal 系解 CSC( goal->start ), 从 goal 积分校验到 start, 再反转
    for segs in _csc_candidates(*_local(goal, start, rho)):
        poses, gears = _integrate(goal, segs, rho, wheelbase, max_steer, ds, +1)
        if _reaches(poses, start, pos_tol, ang_tol):
            rposes = poses[::-1].copy()
            rgears = np.full(rposes.shape[0], -1, dtype=np.int8)   # 倒着走
            length = float(np.linalg.norm(np.diff(rposes[:, :2], axis=0), axis=1).sum())
            if best is None or length < best[0]:
                best = (length, rposes, rgears)

    if best is None:
        return None
    return best[1], best[2], best[0]


def rs_length(start, goal, rho, wheelbase, max_steer, ds=0.2):
    """只取 RS 长度(供调试/启发式参考; 不作为可采纳启发, 见 heuristics.py 说明)。"""
    r = reeds_shepp_path(start, goal, rho, wheelbase, max_steer, ds=ds)
    return None if r is None else r[2]


if __name__ == "__main__":
    rho, wb = 4.5, 2.7
    ms = math.atan(wb / rho)
    rng = np.random.default_rng(0)
    ok = 0
    trials = 200
    # 自检 1: 直线
    r = reeds_shepp_path((0, 0, 0), (5, 0, 0), rho, wb, ms)
    print("straight:", None if r is None else (r[0].shape, round(r[2], 3)))
    # 自检 2: 随机位姿命中率 + 端点误差
    worst = 0.0
    for _ in range(trials):
        s = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
        g = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
        res = reeds_shepp_path(s, g, rho, wb, ms)
        if res is not None:
            ok += 1
            p = res[0][-1]
            e = math.hypot(p[0] - g[0], p[1] - g[1]) + abs(float(wrap(p[2] - g[2])))
            worst = max(worst, e)
    print(f"RS hit-rate: {ok}/{trials}  worst endpoint err: {worst:.3f}")
