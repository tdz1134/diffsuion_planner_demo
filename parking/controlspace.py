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
    x = torch.zeros(B, N, device=ctrl.device, dtype=ctrl.dtype)
    y = torch.zeros_like(x)
    th = torch.zeros_like(x)
    x[:, 0] = start3_w[:, 0]; y[:, 0] = start3_w[:, 1]; th[:, 0] = start3_w[:, 2]
    eps = 1e-6
    for i in range(1, N):
        th0 = th[:, i - 1]; k = kap[:, i]; ui = u[:, i]
        thn = th0 + k * ui
        straight = (k.abs() < eps)
        dx = torch.where(straight, ui * torch.cos(th0), (torch.sin(thn) - torch.sin(th0)) / torch.where(straight, torch.ones_like(k), k))
        dy = torch.where(straight, ui * torch.sin(th0), (torch.cos(th0) - torch.cos(thn)) / torch.where(straight, torch.ones_like(k), k))
        x[:, i] = x[:, i - 1] + dx
        y[:, i] = y[:, i - 1] + dy
        th[:, i] = thn
    return torch.stack([x, y, torch.cos(th), torch.sin(th)], dim=-1)   # (B,N,4)


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
