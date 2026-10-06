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
                       curvature_from_poses, path_length, wrap, pose_error)
from .vehicle import Vehicle
from .conditioner import MapConditioner, MapEncoderCNN
from .diffusion import make_schedule, CondDenoiser, sample
from .map_vae import load_frozen_vae
from .train import norm_traj4, pose3_to_norm4, CKPT, device
from .render import render_scene
from .repair import repair_waypoints
from .interfaces import Scenario

DENSE = 200          # 稠密化点数(碰撞/曲率)
KAPPA_TOL = 1.2      # 曲率容差倍数 (<= KAPPA_TOL / r_min)
JUMP_TOL = 0.35      # 相邻稠密点朝向跳变上限(rad)
REPAIR_MARGIN = 1.2  # 修复时中心到障碍的安全距离(m)


# --------------------------------------------------------------------------- #
# 轨迹稠密化 / 指标
# --------------------------------------------------------------------------- #
def densify_traj4(traj4, n=DENSE):
    """(N,4)[x,y,cos,sin] -> (n,3)[x,y,theta]; x,y 与 cos,sin 分别按索引插值。"""
    t4 = np.asarray(traj4, dtype=np.float64)
    N = t4.shape[0]
    idx_old = np.arange(N)
    idx_new = np.linspace(0, N - 1, n)
    x = np.interp(idx_new, idx_old, t4[:, 0])
    y = np.interp(idx_new, idx_old, t4[:, 1])
    c = np.interp(idx_new, idx_old, t4[:, 2])
    s = np.interp(idx_new, idx_old, t4[:, 3])
    th = cs_to_heading(c, s)
    return np.stack([x, y, th], axis=1)


def footprint_collides(grid, poses, vehicle, res):
    """稠密位姿逐点查足迹碰撞。返回 bool。"""
    for p in poses:
        if vehicle.collides(grid, tuple(p), res):
            return True
    return False


def feasibility(poses, r_min):
    """返回 (feasible, max|kappa|, max_jump)。"""
    kap = curvature_from_poses(poses)
    max_k = float(np.abs(kap).max())
    jump = float(np.abs(wrap(np.diff(poses[:, 2]))).max())
    ok = (max_k <= KAPPA_TOL / r_min) and (jump <= JUMP_TOL)
    return ok, max_k, jump


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
    model = CondDenoiser(dc.n_wp, dim=dc.dim, temb=dc.temb, map_emb=dc.map_emb,
                         hidden=dc.hidden).to(device)
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
    ratios = []; maxks = []; jumps = []; epos = []; eang = []
    r_ratios = []; r_maxks = []; r_jumps = []
    cases = []
    for i in range(k):
        grid = maps[i, 0]
        # 生成: 归一化 -> 世界
        gen4 = gen_n[i].copy()
        gen4[:, :2] = denormalize_xy(gen4[:, :2], *bbox)
        # --- 原始 ---
        gposes = densify_traj4(gen4)
        col = footprint_collides(grid, gposes, veh, res)
        feas, mk, jp = feasibility(gposes, veh.cfg.r_min)
        glen = path_length(gposes)
        n_col += (not col); n_feas += feas; n_ok += (not col and feas)
        ratios.append(glen / max(length[i], 1e-6)); maxks.append(mk); jumps.append(jp)
        pe, ae = pose_error(gposes[-1], gp[i]); epos.append(pe); eang.append(ae)
        # --- 修复后 ---
        rep4 = repair_waypoints(gen4, grid, res, margin=REPAIR_MARGIN)
        rposes = densify_traj4(rep4)
        rcol = footprint_collides(grid, rposes, veh, res)
        rfeas, rmk, rjp = feasibility(rposes, veh.cfg.r_min)
        rglen = path_length(rposes)
        r_col += (not rcol); r_feas += rfeas; r_ok += (not rcol and rfeas)
        r_ratios.append(rglen / max(length[i], 1e-6)); r_maxks.append(rmk); r_jumps.append(rjp)
        if i < n_viz:
            sc = Scenario(grid=grid, res=res, start_pose=sp[i], goal_pose=gp[i],
                          scene_type=("perp" if stype[i] == 0 else "par"), bbox=bbox)
            ref = SimpleNamespace(poses=densify_traj4(traj[i]), gear=None)
            cases.append((sc, ref, gposes, rposes, col, feas, rcol, rfeas))

    metrics = dict(
        k=k, cond=cond,
        guided=float(gkw.get("scale", 0.0)) if gkw else 0.0,
        raw_collision_free=n_col / k, raw_feasible=n_feas / k, raw_success=n_ok / k,
        raw_len_ratio=float(np.mean(ratios)), raw_max_kappa=float(np.mean(maxks)),
        rep_collision_free=r_col / k, rep_feasible=r_feas / k, rep_success=r_ok / k,
        rep_len_ratio=float(np.mean(r_ratios)), rep_max_kappa=float(np.mean(r_maxks)),
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
