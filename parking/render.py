# -*- coding: utf-8 -*-
"""可视化: 车位 + 占据栅格 + 车辆足迹 + 轨迹(HA* 参考 / 扩散生成)。

约定(见 config.py): grid[r,c], r=y/res, c=x/res; 世界 x 向右, y 向下; 后轴位姿 (x,y,theta)。
渲染用 origin="lower" 让 y 轴朝上显示(纯展示, 不影响算法)。所有标签用英文。
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # 无头环境
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

FWD_COLOR = "#1f77b4"        # 前进: 蓝
REV_COLOR = "#d62728"        # 倒车: 红
START_FC = "#2ca02c"         # 起点车: 绿
GOAL_FC = "#9467bd"          # 终点车: 紫
TRAJ_COLOR = "#ff7f0e"       # 轨迹: 橙


def _fig_ax(world_w, world_h, figsize=None):
    if figsize is None:
        ar = world_h / max(world_w, 1e-6)
        figsize = (10.0, max(2.5, 10.0 * ar))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, world_w)
    ax.set_ylim(0, world_h)
    ax.set_aspect("equal")
    return fig, ax


def draw_occupancy(ax, grid, res):
    grid = np.asarray(grid)
    H, W = grid.shape
    ax.imshow(grid, origin="lower", extent=[0, W * res, 0, H * res],
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest", zorder=0)


def draw_footprint(ax, corners, facecolor="none", edgecolor="k", alpha=1.0,
                   lw=1.2, zorder=3, hatch=None):
    ax.add_patch(Polygon(np.asarray(corners), closed=True, facecolor=facecolor,
                         edgecolor=edgecolor, alpha=alpha, lw=lw, zorder=zorder,
                         hatch=hatch))


def draw_vehicle(ax, pose, vehicle, facecolor="none", edgecolor="k", alpha=1.0,
                 lw=1.2, zorder=3, heading=True, hatch=None):
    """画车辆足迹矩形(4 角点)+ 车头朝向线。pose=(x,y,theta) 后轴。"""
    corners = vehicle.corners_world(pose)
    draw_footprint(ax, corners, facecolor=facecolor, edgecolor=edgecolor,
                   alpha=alpha, lw=lw, zorder=zorder, hatch=hatch)
    if heading:
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        L = vehicle.cfg.wheelbase + vehicle.cfg.front_overhang
        ax.plot([x, x + L * np.cos(th)], [y, y + L * np.sin(th)],
                color=edgecolor, lw=lw + 0.6, alpha=alpha, zorder=zorder + 1,
                solid_capstyle="round")


def draw_trajectory(ax, poses, gear=None, lw=2.0, zorder=5, fwd_color=FWD_COLOR,
                    rev_color=REV_COLOR, single_color=None, arrows=0):
    """画轨迹折线; 若给 gear 则按前进/倒车分色。poses:(L,3), gear:(L,) 或 (L-1,)。"""
    poses = np.asarray(poses)
    if poses.shape[0] < 2:
        ax.plot(poses[:, 0], poses[:, 1], "o", color=single_color or fwd_color, zorder=zorder)
        return
    pts = poses[:, :2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)          # (L-1,2,2)
    if single_color is not None or gear is None:
        colors = single_color or TRAJ_COLOR
    else:
        g = np.asarray(gear)
        if g.shape[0] == poses.shape[0]:
            g = g[1:]                                     # 段 i 用点 i+1 的档位
        g = g[:segs.shape[0]]
        colors = [(fwd_color if gi > 0 else rev_color) for gi in g]
    ax.add_collection(LineCollection(segs, colors=colors, lw=lw, zorder=zorder,
                                     capstyle="round"))
    if arrows and arrows > 0 and poses.shape[0] > arrows:
        step = max(1, poses.shape[0] // arrows)
        for i in range(0, poses.shape[0] - 1, step):
            x, y, th = poses[i, 0], poses[i, 1], poses[i, 2]
            ax.annotate("", xy=(x + 0.5 * np.cos(th), y + 0.5 * np.sin(th)),
                        xytext=(x, y), zorder=zorder + 1,
                        arrowprops=dict(arrowstyle="->", color="k", lw=0.8, alpha=0.6))


def draw_slot(ax, slot_polygon, color="tab:green", lw=1.5, ls="--", label=None):
    if slot_polygon is None:
        return
    poly = np.asarray(slot_polygon)
    closed = np.vstack([poly, poly[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw, ls=ls, zorder=2, label=label)


def draw_scenario(ax, scenario, vehicle=None):
    """画占据图 + 车位多边形 + 起终点车辆足迹。"""
    draw_occupancy(ax, scenario.grid, scenario.res)
    draw_slot(ax, scenario.slot_polygon)
    if vehicle is not None:
        draw_vehicle(ax, scenario.start_pose, vehicle, facecolor=START_FC,
                     edgecolor=START_FC, alpha=0.5, zorder=4)
        draw_vehicle(ax, scenario.goal_pose, vehicle, facecolor=GOAL_FC,
                     edgecolor=GOAL_FC, alpha=0.5, zorder=4, heading=True)


def render_scene(scenario, traj=None, vehicle=None, title="", path=None,
                 footprints=6, extra_trajs=None, figsize=None):
    """把一个场景(+可选轨迹)渲染并保存。

    traj: Trajectory 或 FixedTraj(含 .poses / 可选 .gear)。
    extra_trajs: [(traj, color, label), ...] 叠加对比(如扩散生成 vs HA*)。
    """
    H, W = scenario.grid.shape
    res = scenario.res
    fig, ax = _fig_ax(W * res, H * res, figsize=figsize)
    draw_scenario(ax, scenario, vehicle=vehicle)

    if traj is not None:
        poses = getattr(traj, "poses", traj)
        gear = getattr(traj, "gear", None)
        draw_trajectory(ax, poses, gear=gear, lw=2.2, arrows=8)
        # 沿轨迹画若干车辆足迹, 直观检查不碰擦
        if vehicle is not None and footprints > 0:
            n = len(poses)
            idxs = np.linspace(0, n - 1, min(footprints, n)).astype(int)
            for i in idxs:
                draw_vehicle(ax, poses[i], vehicle, facecolor="none",
                             edgecolor=TRAJ_COLOR, alpha=0.6, lw=1.0, zorder=3,
                             heading=False)

    if extra_trajs:
        for t, color, label in extra_trajs:
            p = getattr(t, "poses", t)
            g = getattr(t, "gear", None)
            draw_trajectory(ax, p, gear=g, lw=2.0, single_color=color)
            ax.plot([], [], color=color, lw=2.0, label=label)
        ax.legend(loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    plt.close(fig)
    return fig


def save_fig(fig, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


if __name__ == "__main__":
    # M1 自检: 手搭垂直车位 -> HA* -> 渲染
    import math
    from .config import VehicleConfig, LotConfig, HAStarConfig
    from .vehicle import Vehicle
    from .interfaces import Scenario
    from .hybrid_a_star import HybridAStar
    from . import occupancy as occ

    lot = LotConfig(); res = lot.res
    g = np.zeros((lot.H, lot.W), dtype=np.float64)
    occ.fill_border(g, 1)
    occ.fill_rect_world(g, 6.0, lot.world_h - 6.0, 6.4, lot.world_h - 1.0, res)
    occ.fill_rect_world(g, 9.1, lot.world_h - 6.0, 9.5, lot.world_h - 1.0, res)
    veh = Vehicle(VehicleConfig())
    planner = HybridAStar(veh, HAStarConfig(), res)
    start = (5.0, 5.0, 0.0)
    goal = (7.75, lot.world_h - 3.5, math.pi / 2)
    slot = np.array([[6.4, lot.world_h - 6.0], [9.1, lot.world_h - 6.0],
                     [9.1, lot.world_h - 1.0], [6.4, lot.world_h - 1.0]])
    sc = Scenario(grid=g, res=res, start_pose=np.array(start), goal_pose=np.array(goal),
                  scene_type="perp", slot_polygon=slot)
    tr = planner.plan(sc)
    if tr is None:
        print("render self-test: plan FAILED")
    else:
        out = render_scene(sc, traj=tr, vehicle=veh,
                           title="Hybrid A* - perpendicular parking (switches=%d)" % tr.n_switches,
                           path="figs/parking/m1_hastar_perp.png")
        print("render self-test: poses=%s switches=%d -> %s"
              % (tr.poses.shape, tr.n_switches, out))
