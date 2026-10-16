# -*- coding: utf-8 -*-
"""M17 控制空间(曲率/带符号弧长)表示 + 可微运动学正向积分。

动机: pose-space 扩散(含 x0/ε 重参数化)一旦把几何代价塞进训练/引导就发散(M8/M12/M16)。
这条路让**可行性由构造保证**: 网络扩散的是逐段控制 (κ, u), 再用等曲率弧闭式积分回位姿——
  - 速度恒 ∥ 航向 => 非完整零横向滑移(天然满足);
  - |κ| <= kappa_max = 1/r_min(编码/解码都钳位) => 曲率天然达标。

约定(与 geometry.py 一致, 世界系米, θ 弧度):
  控制第 i 段(i>=1)把 pose[i-1]->pose[i]:
      u_i = gear_i * ds_i         带符号弧长(前向+ / 倒车-), ds_i>=0
      κ_i = (θ_i - θ_{i-1}) / u_i 有符号曲率(1/m), 编码时钳到 ±kappa_max
  等曲率弧积分(θ 从 th0 起, 段内 κ 常数):
      θ_i = θ_{i-1} + κ_i*u_i
      |κ|<eps: Δx=u*cosθ_{i-1}, Δy=u*sinθ_{i-1}
      否则:    Δx=(sinθ_i - sinθ_{i-1})/κ,  Δy=(cosθ_{i-1} - cosθ_i)/κ
  归一化(喂给 DDPM 的 (N,2)): κn = κ/kappa_max ∈[-1,1];  un = u/u_scale。
  row0 用 (0,0) 占位(起点 pose 由 start 给定, 不被扩散)。
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# numpy: 专家 pose+gear -> 归一化控制 (供 dataset 侧编码)
# --------------------------------------------------------------------------- #
def encode_ctrl(traj4_w, gear, kappa_max, u_scale):
    """traj4_w:(N,4)世界[x,y,cos,sin], gear:(N,)±1 -> ctrl:(N,2)[kappa_n, u_n]。row0=(0,0)。"""
    t = np.asarray(traj4_w, dtype=np.float64)
    g = np.asarray(gear, dtype=np.float64)
    N = t.shape[0]
    th = np.unwrap(np.arctan2(t[:, 3], t[:, 2]))
    ctrl = np.zeros((N, 2), dtype=np.float32)
    for i in range(1, N):
        dx = t[i, 0] - t[i - 1, 0]
        dy = t[i, 1] - t[i - 1, 1]
        ds = np.hypot(dx, dy)
        gi = g[i] if abs(g[i]) > 0.5 else (1.0 if (dx * np.cos(th[i - 1]) + dy * np.sin(th[i - 1])) >= 0 else -1.0)
        u = gi * ds
        if ds < 1e-4:                       # 近零位移: 保留航向变化, u 记为 eps 方向
            ctrl[i] = (0.0, 0.0)
            continue
        kap = (th[i] - th[i - 1]) / u
        kap = float(np.clip(kap, -kappa_max, kappa_max))
        ctrl[i, 0] = kap / kappa_max
        ctrl[i, 1] = u / u_scale
    return ctrl


# --------------------------------------------------------------------------- #
# torch: 归一化控制 -> pose (可微, 训练/评估都用)
# --------------------------------------------------------------------------- #
def integrate_ctrl(ctrl, start3_w, kappa_max, u_scale):
    """ctrl:(B,N,2)归一化, start3_w:(B,3)世界[x,y,theta] -> traj4:(B,N,4)世界[x,y,cos,sin]。

    逐段等曲率弧闭式积分(可回传梯度)。曲率/非完整由构造满足。
    """
    B, N, _ = ctrl.shape
    kap = torch.clamp(ctrl[..., 0], -1.0, 1.0) * kappa_max      # (B,N) 有符号曲率(钳位)
    u = ctrl[..., 1] * u_scale                                   # (B,N) 带符号弧长
    eps = 1e-6
    # 函数式累加(不就地写 x/y/th): 就地 slice 赋值会修改 autograd 保存过的视图, 使反向报 version 错。
    x = start3_w[:, 0]; y = start3_w[:, 1]; th = start3_w[:, 2]
    xs = [x]; ys = [y]; ths = [th]
    for i in range(1, N):
        th0 = th; k = kap[:, i]; ui = u[:, i]
        thn = th0 + k * ui
        straight = (k.abs() < eps)
        kk = torch.where(straight, torch.ones_like(k), k)
        dx = torch.where(straight, ui * torch.cos(th0), (torch.sin(thn) - torch.sin(th0)) / kk)
        dy = torch.where(straight, ui * torch.sin(th0), (torch.cos(th0) - torch.cos(thn)) / kk)
        x = x + dx; y = y + dy; th = thn                     # 新张量, 无就地修改
        xs.append(x); ys.append(y); ths.append(th)
    X = torch.stack(xs, dim=1); Y = torch.stack(ys, dim=1); TH = torch.stack(ths, dim=1)
    return torch.stack([X, Y, torch.cos(TH), torch.sin(TH)], dim=-1)   # (B,N,4)


# --------------------------------------------------------------------------- #
# M18: 终点→goal 可微到达惩罚(经 integrate_ctrl 回传, 不碰 κ=|dθ|/ds 病态)
# --------------------------------------------------------------------------- #
def goal_reach_pen(ctrl, start3_w, goal_n4, bbox, kappa_max, u_scale):
    """ctrl:(B,N,2)归一化控制(通常是 x0_hat), start3_w:(B,3)世界起点,
    goal_n4:(B,4)归一化目标[nx,ny,cos,sin], bbox=(x0,y0,x1,y1)。
    将控制可微积分回世界位姿, 取终点与 goal 比较。返回 (pos_err(B,), ang_err(B,)):
      pos_err = 归一化坐标差平方(x,y 各一), ang_err = 1-cos(dθ) ∈[0,2]。均为良态量。
    """
    poses = integrate_ctrl(ctrl, start3_w, kappa_max, u_scale)   # (B,N,4) 世界
    end = poses[:, -1]                                            # (B,4) [x,y,cos,sin]
    bx0, by0, bx1, by1 = bbox
    nx = (end[:, 0] - bx0) / max(bx1 - bx0, 1e-9) * 2.0 - 1.0
    ny = (end[:, 1] - by0) / max(by1 - by0, 1e-9) * 2.0 - 1.0
    pos_err = (nx - goal_n4[:, 0]) ** 2 + (ny - goal_n4[:, 1]) ** 2
    ang_err = 1.0 - (end[:, 2] * goal_n4[:, 2] + end[:, 3] * goal_n4[:, 3])
    return pos_err, ang_err


# --------------------------------------------------------------------------- #
# M18b: 测试时终点投影——把已采样的控制序列用可微积分拉到终点=goal(不重训)
# --------------------------------------------------------------------------- #
def refine_ctrl_to_goal(ctrl, start3_w, goal_n4, bbox, kappa_max, u_scale,
                        iters=80, lr=0.05, w_pos=1.0, w_ang=0.3):
    """ctrl:(B,N,2)已采样归一化控制, start3_w:(B,3)世界起点, goal_n4:(B,4)归一化目标。
    对控制序列做几步 Adam 梯度下降, 最小化终点位置+航向误差(goal_reach_pen)。
    κ 通道每步钳到 [-1,1] -> **曲率/非完整构造保证不被破坏**(只优化终点)。
    iters<=0 直接返回原 ctrl。返回优化后 ctrl(B,N,2)。
    """
    if iters is None or iters <= 0:
        return ctrl

    def _bounded(c):
        return torch.cat([c[..., 0:1].clamp(-1.0, 1.0), c[..., 1:2]], dim=-1)

    c = ctrl.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(iters):
        pe, ae = goal_reach_pen(_bounded(c), start3_w, goal_n4, bbox, kappa_max, u_scale)
        loss = (w_pos * pe + w_ang * ae).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return _bounded(c).detach()


# --------------------------------------------------------------------------- #
# 自检: round-trip 保真 + 曲率是否真 <= kappa_max
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .config import default_config
    cfg = default_config()
    kmax = 1.0 / cfg.vehicle.r_min
    u_scale = 1.0
    npz = "cache/parking_12000_n80.npz"
    dd = np.load(npz)
    traj = dd["traj"]; gear = dd["gear"]; sp = dd["start_pose"]
    K = 5
    worst_pe, worst_ae, worst_k = 0.0, 0.0, 0.0
    for i in range(K):
        ctrl = torch.tensor(encode_ctrl(traj[i], gear[i], kmax, u_scale)[None])  # (1,N,2)
        st = torch.tensor(sp[i][None], dtype=torch.float32)                        # (1,3)
        rec = integrate_ctrl(ctrl, st, kmax, u_scale)[0].numpy()                   # (N,4)
        pe = np.hypot(rec[:, 0] - traj[i][:, 0], rec[:, 1] - traj[i][:, 1])
        rec_th = np.unwrap(np.arctan2(rec[:, 3], rec[:, 2]))
        org_th = np.unwrap(np.arctan2(traj[i][:, 3], traj[i][:, 2]))
        ae = np.abs(rec_th - org_th)
        # 曲率(重构): |dtheta|/ds
        dx = np.diff(rec[:, 0]); dy = np.diff(rec[:, 1]); ds = np.hypot(dx, dy)
        dth = np.abs(np.diff(rec_th))
        k = np.where(ds > 1e-3, dth / np.maximum(ds, 1e-9), 0.0)
        worst_pe = max(worst_pe, pe.max()); worst_ae = max(worst_ae, ae.max())
        worst_k = max(worst_k, k.max())
        print("case%d: max pos_err=%.3fm max ang_err=%.1fdeg max|kappa|=%.3f (limit %.3f)"
              % (i, pe.max(), np.degrees(ae.max()), k.max(), kmax))
    print("\nSUMMARY: worst pos_err=%.3fm ang=%.1fdeg  max|kappa|=%.4f <= kappa_max=%.4f ? %s"
          % (worst_pe, np.degrees(worst_ae), worst_k, kmax, worst_k <= kmax * 1.02))

    # ---- M18 goal_reach_pen 自检: 终点==goal 时惩罚应~0; 且梯度有限 ----
    bbox = tuple(float(v) for v in dd["bbox"])
    c0 = torch.tensor(encode_ctrl(traj[0], gear[0], kmax, u_scale)[None], requires_grad=True)  # (1,N,2)
    st0 = torch.tensor(sp[0][None], dtype=torch.float32)                                       # (1,3)
    ep = integrate_ctrl(c0, st0, kmax, u_scale)[:, -1]                                        # (1,4) 世界终点
    bx0, by0, bx1, by1 = bbox
    gnx = (ep[:, 0] - bx0) / (bx1 - bx0) * 2.0 - 1.0
    gny = (ep[:, 1] - by0) / (by1 - by0) * 2.0 - 1.0
    goal_n4 = torch.stack([gnx, gny, ep[:, 2], ep[:, 3]], dim=1).detach()                       # 把真实终点当 goal
    pe0, ae0 = goal_reach_pen(c0, st0, goal_n4, bbox, kmax, u_scale)
    (pe0 + ae0).sum().backward()
    gfin = bool(torch.isfinite(c0.grad).all())
    # 扰动控制(把曲率通道整体推满) -> 惩罚应明显 > 0
    c1 = c0.detach().clone(); c1[:, :, 0] = 0.9
    pe1, ae1 = goal_reach_pen(c1, st0, goal_n4, bbox, kmax, u_scale)
    print("[M18 goal_reach] self-goal pos_err=%.3e ang_err=%.3e grad_finite=%s | perturbed pos_err=%.3f ang=%.3f"
          % (pe0.mean().item(), ae0.mean().item(), gfin, pe1.mean().item(), ae1.mean().item()))
