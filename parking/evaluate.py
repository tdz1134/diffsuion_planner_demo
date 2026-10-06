# -*- coding: utf-8 -*-
"""泊车扩散规划器评估(泊车专用指标, plan §7) + 与 HA* 参考对比渲染。

指标(在 eval 划分上):
  - 足迹无碰撞率: 生成轨迹稠密化后, 车辆旋转矩形逐点查占据栅格(最关键)。
  - 运动学可行性: 稠密曲率 |kappa| <= 1.2/r_min 且 相邻朝向跳变 < 阈值。
  - 成功率 = 无碰撞 且 可行。
  - 长度比 = 生成 / HA* 参考(npz 的 traj/length 即专家)。
  - 终点位姿误差(采样钉端点, 理论上 ~0, 仅作完整性报告)。
渲染: 每个 case 叠画 车位+障碍 / HA* 参考(按档位分色) / 扩散生成(品红)。
"""

import os
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from .config import default_config
from .dataset import load_dataset, SDF_CLIP
from .geometry import (denormalize_xy, cs_to_heading, cum_arclen,
                       curvature_from_poses, path_length, wrap, pose_error,
                       resample_poses)
from .vehicle import Vehicle
from .conditioner import MapConditioner, MapEncoderCNN
from .diffusion import make_schedule, build_denoiser, sample
from .map_vae import load_frozen_vae
from .train import norm_traj4, pose3_to_norm4, CKPT, device
from .render import render_scene
from .repair import repair_waypoints
from .interfaces import Scenario

DENSE = 200          # 稠密化点数(碰撞/滑移)
KAPPA_TOL = 1.2      # 曲率容差倍数: 分段曲率判据阈 = KAPPA_TOL / r_min
SLIP_TOL = 0.2       # 非完整性(横向滑移)判据: 逐段 |sin(运动方向-航向)| 上限(~11.5°)
MIN_MOVE = 0.05      # 逐段滑移统计时忽略小于此位移(m)的近似静止段(换档尖点)
CURV_STEP = 0.20     # 分段曲率(位置 Menger)重采样的均匀弧长步长(m): 粗步长避免稀疏航点重加密引入的毛刺放大
MIN_SEG_LEN = 0.3    # 短于此弧长(m)的子段不计入曲率(噪声/尖点残留)
SEG_PTS_MIN = 4      # 一段重采样后至少这么多点才谈得上可靠曲率
JUMP_TOL = 0.35      # (保留: 旧诊断量, 不参与判定)
REPAIR_MARGIN = 1.2  # 修复时中心到障碍的安全距离(m)


# --------------------------------------------------------------------------- #
# 轨迹稠密化 / 指标
# --------------------------------------------------------------------------- #
def densify_traj4(traj4, n=DENSE):
    """(N,4)[x,y,cos,sin] -> (n,3)[x,y,theta]。

    x,y 按索引线性插值; **航向先逐航点反 cos/sin 再沿序列 unwrap, 然后插值**。
    (旧版直接插 cos,sin 弦: 换档尖点处两航点近反平行时弦穿过原点→arctan2 乱跳→曲率虚高,
     连 100% 可行的 HA* 专家都会被误判为不可行。)
    """
    t4 = np.asarray(traj4, dtype=np.float64)
    N = t4.shape[0]
    idx_old = np.arange(N)
    idx_new = np.linspace(0, N - 1, n)
    x = np.interp(idx_new, idx_old, t4[:, 0])
    y = np.interp(idx_new, idx_old, t4[:, 1])
    th_wp = np.unwrap(cs_to_heading(t4[:, 2], t4[:, 3]))    # 沿序列解角度环绕
    th = np.interp(idx_new, idx_old, th_wp)                 # 直接插值连续航向
    return np.stack([x, y, th], axis=1)


def footprint_collides(grid, poses, vehicle, res):
    """稠密位姿逐点查足迹碰撞。返回 bool。"""
    for p in poses:
        if vehicle.collides(grid, tuple(p), res):
            return True
    return False


def nonholonomy_slip(poses, min_move=MIN_MOVE):
    """非完整性残差(横向滑移): 逐段运动方向与车航向的垂直分量 |sin(α)|。
    前进(平行)与倒车(反平行)都计 0; 跳过近静止段(换档尖点)。返回 (mean, max)。
    这是定长 N 表示下**对齿轮/尖点免疫**的有效可行性代理(曲率则不然)。"""
    poses = np.asarray(poses, dtype=np.float64)
    dx = np.diff(poses[:, 0]); dy = np.diff(poses[:, 1]); L = np.hypot(dx, dy)
    th = poses[:-1, 2]; c = np.cos(th); s = np.sin(th)
    m = L > min_move
    if not m.any():
        return 0.0, 0.0
    slip = np.abs((dx[m] / L[m]) * s[m] - (dy[m] / L[m]) * c[m])
    return float(slip.mean()), float(slip.max())


def segment_by_cusps(poses):
    """按运动方向相对航向的前/后反转(=换档尖点)把序列切成若干子段(索引区间 [i,j])。
    尖点视为合法停顿: 不在尖点上算曲率。前进(dot>0)与倒车(dot<0)分段, 侧滑(dot≈0)不分段
    (交给滑移判据)。返回 [(i, j), ...]。"""
    poses = np.asarray(poses, dtype=np.float64)
    n = poses.shape[0]
    if n < 2:
        return [(0, n)]
    dx = np.diff(poses[:, 0]); dy = np.diff(poses[:, 1])
    th = poses[:-1, 2]
    along = dx * np.cos(th) + dy * np.sin(th)          # 运动在航向上的有符号投影
    stat = np.abs(along) < MIN_MOVE                    # 近静止(尖点附近)
    sign = np.where(stat, 0, np.sign(along)).astype(int)
    segs = []; start = 0; cur = 0
    for i in range(len(sign)):
        if sign[i] != 0:
            if cur == 0:
                cur = sign[i]
            elif sign[i] != cur:
                segs.append((start, i + 1)); start = i; cur = sign[i]
    segs.append((start, n))
    return segs


def _menger_max(xy):
    """逐三点外接圆(位置)曲率最大值; 对重采样的尖点/噪声比航向微分更稳。"""
    if xy.shape[0] < 3:
        return 0.0
    a, b, c = xy[:-2], xy[1:-1], xy[2:]
    ab = np.linalg.norm(b - a, axis=1); bc = np.linalg.norm(c - b, axis=1); ca = np.linalg.norm(c - a, axis=1)
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    den = ab * bc * ca
    kk = np.where(den > 1e-9, cross / den, 0.0)
    return float(kk.max()) if kk.size else 0.0


def segment_metrics(poses, step=CURV_STEP):
    """用户方法: 先按换档尖点切割, 每段重采样后取最大 |κ|(位置 Menger)。返回 (seg_max_kappa, n_seg)。
    诚实注: 定长 N=40 对曲率**天然欠分辨**(紧入口塔缩成稀疏航点), 连专家都读出 κ≈1.45(真值应≤0.22);
    故本值只用于**相对比较/诊断**(专家<CONV<MLP 排序正确), 不作绝对可行性硬门。尖点数 n_seg 则是可靠质量信号。"""
    poses = np.asarray(poses, dtype=np.float64)
    worst = 0.0; nseg = 0
    for i, j in segment_by_cusps(poses):
        seg = poses[i:j]
        if seg.shape[0] < 2 or float(cum_arclen(seg[:, :2])[-1]) < MIN_SEG_LEN:
            continue
        nseg += 1
        m = max(3, int(np.ceil(cum_arclen(seg[:, :2])[-1] / step)) + 1)
        worst = max(worst, _menger_max(resample_poses(seg, m)[:, :2]))
    return worst, nseg


def feasibility(poses, r_min):
    """返回 (feasible, mean_slip, seg_kappa, n_seg)。
    feasible = 非完整性滑移达标(max|slip|<=SLIP_TOL) —— 这是定长 N 上唯一对尖点/采样密度免疫的硬判据。
    seg_kappa/n_seg 作为诊断输出(尖点分段后的位置曲率与换档次数), 不参与硬门。"""
    mean_slip, max_slip = nonholonomy_slip(poses)
    seg_k, nseg = segment_metrics(poses)
    ok = (max_slip <= SLIP_TOL)
    return ok, mean_slip, seg_k, nseg


# --------------------------------------------------------------------------- #
# 主评估
# --------------------------------------------------------------------------- #
def load_model(ckpt=CKPT, cfg=None):
    cfg = cfg or default_config()
    dc = cfg.diffusion
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cond = blob.get("cond", "lat")
    if cond == "lat":
        vae = load_frozen_vae(lat_ch=dc.lat_ch)
        enc = MapConditioner(vae, out_dim=dc.map_emb, lat_hw=(9, 16)).to(device)
    else:
        enc = MapEncoderCNN(out_dim=dc.map_emb).to(device)
    enc.load_state_dict(blob["enc"])
    dc.denoiser = blob.get("denoiser", "mlp")     # 旧 ckpt 无此字段 -> mlp(向后兼容)
    model = build_denoiser(dc).to(device)
    model.load_state_dict(blob["model"])
    model.eval(); enc.eval()
    return model, enc, blob.get("bbox", (0, 0, cfg.lot.world_w, cfg.lot.world_h)), cond


def evaluate(ckpt=CKPT, npz_path=None, k=None, n_viz=None, render_path=None, gkw=None):
    cfg = default_config()
    dc = cfg.diffusion
    veh = Vehicle(cfg.vehicle)
    if npz_path is None:
        cdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            cfg.data.cache_dir)
        cands = sorted([f for f in os.listdir(cdir)
                        if f.startswith("parking_") and f.endswith(".npz")],
                       key=lambda s: int(s.split("_")[1].split(".")[0]), reverse=True)
        npz_path = os.path.join(cdir, cands[0])
    d = load_dataset(npz_path)
    bbox = tuple(float(v) for v in d["bbox"])
    ev = d["split"] == 1
    if ev.sum() < 8:                      # eval 太少则借用尾部
        ev = np.zeros_like(d["split"], bool); ev[-64:] = True
    maps = d["maps"][ev]; traj = d["traj"][ev]; length = d["length"][ev]
    sp = d["start_pose"][ev]; gp = d["goal_pose"][ev]; stype = d["scene_type"][ev]
    gear = d["gear"][ev]
    k = (cfg.data.n_eval_metric if k is None else int(k))
    k = min(k, maps.shape[0])
    n_viz = cfg.data.n_eval_viz if n_viz is None else int(n_viz)

    model, enc, bb, cond = load_model(ckpt, cfg)
    maps_t = torch.tensor(maps[:k], device=device)
    s4 = torch.tensor(pose3_to_norm4(sp[:k], bbox), device=device)
    g4 = torch.tensor(pose3_to_norm4(gp[:k], bbox), device=device)
    sch = make_schedule(dc.t_steps, dc.b0, dc.b1, device)
    lat = None
    if cond == "lat":
        from .train import precompute_latents
        lat = precompute_latents(enc.vae, maps_t)
    guide = None
    if gkw and gkw.get("scale", 0) > 0:
        foot = torch.tensor(Vehicle(cfg.vehicle).footprint_local,
                            dtype=torch.float32, device=device)
        guide = dict(foot=foot, veh=cfg.vehicle, bbox=bbox, sdf_clip=SDF_CLIP,
                     w_nh=gkw.get("nh", 0.0), w_curv=gkw.get("curv", 0.0),
                     w_coll=gkw.get("coll", 0.0), scale=gkw["scale"],
                     margin=gkw.get("margin", 0.15), min_abar=gkw.get("min_abar", 0.1))
    gen_n = sample(model, enc, maps_t, s4, g4, sch, z=lat, guide=guide).cpu().numpy()   # (k,N,4) norm

    res = cfg.lot.res
    n_col = n_feas = n_ok = 0
    r_col = r_feas = r_ok = 0
    ratios = []; maxks = []; slips = []; nsegs = []; epos = []; eang = []
    r_ratios = []; r_maxks = []; r_slips = []; r_nsegs = []
    cases = []
    for i in range(k):
        grid = maps[i, 0]
        # 生成: 归一化 -> 世界
        gen4 = gen_n[i].copy()
        gen4[:, :2] = denormalize_xy(gen4[:, :2], *bbox)
        # --- 原始 ---
        gposes = densify_traj4(gen4)
        col = footprint_collides(grid, gposes, veh, res)
        feas, mslip, mk, nseg = feasibility(gposes, veh.cfg.r_min)
        glen = path_length(gposes)
        n_col += (not col); n_feas += feas; n_ok += (not col and feas)
        ratios.append(glen / max(length[i], 1e-6)); maxks.append(mk); slips.append(mslip); nsegs.append(nseg)
        pe, ae = pose_error(gposes[-1], gp[i]); epos.append(pe); eang.append(ae)
        # --- 修复后 ---
        rep4 = repair_waypoints(gen4, grid, res, margin=REPAIR_MARGIN)
        rposes = densify_traj4(rep4)
        rcol = footprint_collides(grid, rposes, veh, res)
        rfeas, rmslip, rmk, rnseg = feasibility(rposes, veh.cfg.r_min)
        rglen = path_length(rposes)
        r_col += (not rcol); r_feas += rfeas; r_ok += (not rcol and rfeas)
        r_ratios.append(rglen / max(length[i], 1e-6)); r_maxks.append(rmk); r_slips.append(rmslip); r_nsegs.append(rnseg)
        if i < n_viz:
            sc = Scenario(grid=grid, res=res, start_pose=sp[i], goal_pose=gp[i],
                          scene_type=("perp" if stype[i] == 0 else "par"), bbox=bbox)
            ref = SimpleNamespace(poses=densify_traj4(traj[i]), gear=None)
            cases.append((sc, ref, gposes, rposes, col, feas, rcol, rfeas))

    metrics = dict(
        k=k, cond=cond,
        guided=float(gkw.get("scale", 0.0)) if gkw else 0.0,
        raw_collision_free=n_col / k, raw_feasible=n_feas / k, raw_success=n_ok / k,
        raw_len_ratio=float(np.mean(ratios)), raw_mean_slip=float(np.mean(slips)),
        raw_seg_kappa=float(np.mean(maxks)), raw_gear_switches=float(np.mean(nsegs)),
        rep_collision_free=r_col / k, rep_feasible=r_feas / k, rep_success=r_ok / k,
        rep_len_ratio=float(np.mean(r_ratios)), rep_mean_slip=float(np.mean(r_slips)),
        rep_seg_kappa=float(np.mean(r_maxks)), rep_gear_switches=float(np.mean(r_nsegs)),
        slip_tol=SLIP_TOL,
        kappa_limit=1.0 / veh.cfg.r_min,
        mean_end_pos_err_m=float(np.mean(epos)),
        mean_end_ang_err_deg=float(np.degrees(np.mean(eang))),
    )
    if render_path:
        _render_cases(cases, veh, render_path)
    return metrics, cases


def _render_cases(cases, veh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(cases)
    cols = 4
    rows = max(1, (k + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, (sc, ref, gposes, rposes, col, feas, rcol, rfeas) in enumerate(cases):
        ax = axes[i]
        from .render import draw_scenario, draw_trajectory
        draw_scenario(ax, sc, vehicle=veh)
        draw_trajectory(ax, ref.poses, gear=None, lw=2.0, single_color="#1f77b4")
        draw_trajectory(ax, gposes, gear=None, lw=1.0, single_color="#ffbbff")
        draw_trajectory(ax, rposes, gear=None, lw=2.0, single_color="#ff00ff")
        ax.plot([], [], color="#1f77b4", lw=2, label="HA* ref")
        ax.plot([], [], color="#ffbbff", lw=1.5, label="diffusion raw")
        ax.plot([], [], color="#ff00ff", lw=2, label="diffusion+repair")
        ax.set_title("#%d %s raw(c%f f%f) rep(c%f f%f)"
                     % (i, sc.scene_type, col, feas, rcol, rfeas), fontsize=8)
        ax.set_aspect("equal"); ax.legend(fontsize=6, loc="upper right")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--viz", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--g-scale", type=float, default=0.0, help="推理引导总步长(0=关)")
    ap.add_argument("--g-curv", type=float, default=1.0, help="引导中曲率代价权重")
    ap.add_argument("--g-coll", type=float, default=1.0, help="引导中足迹碰撞代价权重")
    ap.add_argument("--g-nh", type=float, default=0.0, help="引导中航向一致性代价权重")
    ap.add_argument("--g-margin", type=float, default=0.15, help="引导碰撞安全间隙(m)")
    ap.add_argument("--g-min-abar", type=float, default=0.1, help="仅在 abar>=此值(低噪)时引导")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "figs", "parking", "m6_eval_compare.png")
    gkw = dict(scale=a.g_scale, curv=a.g_curv, coll=a.g_coll, nh=a.g_nh,
               margin=a.g_margin, min_abar=a.g_min_abar)
    m, _ = evaluate(a.ckpt, a.npz, a.k, a.viz, render_path=out, gkw=gkw)
    for kk, v in m.items():
        print("  %-22s %s" % (kk, round(v, 4) if isinstance(v, float) else v))
    print("[saved] %s" % out)
