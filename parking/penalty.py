# -*- coding: utf-8 -*-
"""Phase 2: 轨迹空间可微惩罚(作用在去噪反推的 x0_hat 上, 加进扩散训练损失)。

动机(诚实): 纯 ε-MSE + MLP 对"尖锐、多模态"的泊车轨迹只会学"平均", 采样出来曲率爆表、
贴障碍。加数据/加步数救不回(见 PARKING_NOTES M7)。本模块把**运动学可行性与避障**直接写进
损失——用 x0_hat 的可微几何量惩罚, 与 evaluate 的指标(足迹碰撞 / |kappa|<=1/r_min)对齐。

三个独立惩罚(权重在 config.DiffusionConfig 里, 默认 0=关闭, 向后兼容):
  - heading_consistency  : 速度的**垂直于航向**分量(侧滑)。允许前进/倒车(平行或反平行均 0)。
  - curvature_pen        : |kappa| 超过 1/r_min 的部分(曲率超限惩罚)。
  - footprint_collision  : 车辆**足迹点**处 SDF(米)< margin 的部分(直接对应足迹无碰撞率)。

输入统一为**归一化**轨迹 x0 (B,N,4)[x,y,cos,sin]; x,y∈[-1,1] 由全局 bbox 归一化, 故需 bbox
把归一化↔米互相换算(footprint_collision 内部把足迹点变到米再采样)。
"""

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _scale_xy(bbox):
    """归一化单位 -> 米 的每轴尺度 (sx, sy)。bbox=(x0,y0,x1,y1)。"""
    x0, y0, x1, y1 = bbox
    return 0.5 * (x1 - x0), 0.5 * (y1 - y0)


def _unit_heading(x0):
    """从 (cos,sin) 取单位航向(网络输出不保证模长为 1)。"""
    c = x0[:, :, 2]
    s = x0[:, :, 3]
    n = torch.hypot(c, s) + 1e-6
    return c / n, s / n


# --------------------------------------------------------------------------- #
# 三个惩罚项
# --------------------------------------------------------------------------- #
def heading_consistency(x0, bbox):
    """侧滑惩罚: 速度在**垂直航向**方向的分量平方均值。0 = 纯前进/倒车(共线, 允许反号)。"""
    sx, sy = _scale_xy(bbox)
    dx = (x0[:, 1:, 0] - x0[:, :-1, 0]) * sx
    dy = (x0[:, 1:, 1] - x0[:, :-1, 1]) * sy
    L = torch.hypot(dx, dy) + 1e-6
    c, s = _unit_heading(x0)
    cross = (dx / L) * s[:, :-1] - (dy / L) * c[:, :-1]   # 垂直分量
    return (cross ** 2).mean(dim=1)                        # (B,)


def curvature_pen(x0, bbox, inv_rmin):
    """曲率超限惩罚: mean(relu(|kappa| - 1/r_min)^2), kappa = 单位航向转角 / 弧长(米)。"""
    sx, sy = _scale_xy(bbox)
    c, s = _unit_heading(x0)
    cdot = c[:, :-1] * c[:, 1:] + s[:, :-1] * s[:, 1:]     # cos(dtheta)
    xcrs = c[:, :-1] * s[:, 1:] - s[:, :-1] * c[:, 1:]     # sin(dtheta)
    dth = torch.atan2(xcrs, cdot)
    dx = (x0[:, 1:, 0] - x0[:, :-1, 0]) * sx
    dy = (x0[:, 1:, 1] - x0[:, :-1, 1]) * sy
    ds = torch.hypot(dx, dy) + 1e-6
    kap = torch.abs(dth) / ds
    return (torch.relu(kap - inv_rmin) ** 2).mean(dim=1)   # (B,)


def footprint_collision(x0, sdf_m, bbox, foot_local, margin=0.15):
    """足迹碰撞惩罚: 对每个航点, 把车辆足迹点(parking/vehicle.py 的 footprint_local)旋转到
    世界(米), 用 grid_sample 在 SDF(米)上双线性采样, 惩罚 SDF<margin 的点。
    sdf_m:(B,1,H,W) 米; foot_local:(P,2) 米(tensor)。"""
    x0w, y0w, x1w, y1w = bbox
    wx = x1w - x0w
    hy = y1w - y0w
    xm = (x0[:, :, 0] + 1) / 2 * wx + x0w          # (B,N) 米
    ym = (x0[:, :, 1] + 1) / 2 * hy + y0w
    c, s = _unit_heading(x0)
    lx = foot_local[:, 0]
    ly = foot_local[:, 1]
    wsx = xm[:, :, None] + lx * c[:, :, None] - ly * s[:, :, None]
    wsy = ym[:, :, None] + lx * s[:, :, None] + ly * c[:, :, None]
    gx = wsx / wx * 2 - 1
    gy = wsy / hy * 2 - 1
    B, N = xm.shape
    P = foot_local.shape[0]
    grid = torch.stack([gx, gy], dim=-1).reshape(B, N * P, 1, 2)
    out = F.grid_sample(sdf_m, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)   # (B,1,N*P,1)
    sdf = out.reshape(B, 1, N, P)[:, 0]                              # (B,N,P) 米
    pen = torch.relu(margin - sdf)
    return (pen ** 2).mean(dim=(1, 2))                      # (B,)


def traj_penalty_terms(x0, sdf_m, bbox, veh, foot_local, margin=0.15):
    """返回三项惩罚(dict), 每项为 (B,) 每样本标量 tensor。供 train 按 sqrt(abar) 加权。"""
    return dict(
        nh=heading_consistency(x0, bbox),
        curv=curvature_pen(x0, bbox, 1.0 / veh.r_min),
        coll=footprint_collision(x0, sdf_m, bbox, foot_local, margin),
    )


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .config import default_config
    from .vehicle import Vehicle
    from . import occupancy as occ

    torch.manual_seed(0)
    cfg = default_config()
    veh = cfg.vehicle
    res = cfg.lot.res
    wx, hy = cfg.lot.world_w, cfg.lot.world_h
    bbox = (0.0, 0.0, wx, hy)
    foot = torch.tensor(Vehicle(veh).footprint_local, dtype=torch.float32)
    N = cfg.diffusion.n_wp

    def mk(Xarr, Yarr, cc, ss):
        """用**米制**坐标构造, 内部归一化到 x0(B,N,4)(函数吃归一化坐标)。"""
        a = torch.zeros(N, 4)
        a[:, 0] = Xarr / wx * 2 - 1
        a[:, 1] = Yarr / hy * 2 - 1
        a[:, 2] = cc
        a[:, 3] = ss
        return a[None]                                     # (1,N,4)

    sdf_free = torch.full((1, 1, cfg.lot.H, cfg.lot.W), 5.0)   # 全空闲(离障碍 5m)
    xs = torch.linspace(3.0, 20.0, N)
    ys = torch.full((N,), 7.0)
    straight = mk(xs, ys, 1.0, 0.0)                            # 直行, 航向=运动方向
    side = mk(xs, ys, 0.0, 1.0)                               # 航向垂直运动 -> 侧滑
    th = torch.linspace(0, 2 * np.pi, N)
    arc = mk(10 + torch.cos(th), 7 + torch.sin(th),
             torch.cos(th + np.pi / 2), torch.sin(th + np.pi / 2))  # 半径 1m 急转

    g = np.zeros((cfg.lot.H, cfg.lot.W), dtype=np.float64)
    occ.fill_rect_world(g, 10.0, 0.0, 11.0, hy, res)           # x≈10..11 的墙
    sdf_wall = torch.tensor(occ.distance_field_metric(g, res)[None, None],
                            dtype=torch.float32)
    thru = mk(torch.linspace(5.0, 16.0, N), torch.full((N,), 7.0), 1.0, 0.0)  # 穿墙

    def show(name, x0, sdf):
        t = traj_penalty_terms(x0, sdf, bbox, veh, foot)
        print("%-10s nh=%.4f curv=%.4f coll=%.4f"
              % (name, t["nh"].mean().item(), t["curv"].mean().item(),
                 t["coll"].mean().item()))

    show("straight", straight, sdf_free)     # 期望: 都≈0
    show("side", side, sdf_free)             # 期望: nh 大(~1)
    show("arc", arc, sdf_free)               # 期望: curv 大(半径1m >> 1/r_min)
    show("thru_wall", thru, sdf_wall)        # 期望: coll 大(足迹穿墙)

    # 梯度可回传(训练要用)
    xg = thru.clone().detach().requires_grad_(True)
    loss = sum(v.sum() for v in traj_penalty_terms(xg, sdf_wall, bbox, veh, foot).values())
    loss.backward()
    print("grad finite:", bool(torch.isfinite(xg.grad).all()))
    print("endpoints not auto-pinned here (train 侧已钉; 惩罚只作用中间几何)")
