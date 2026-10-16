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
from .conditioner import MapConditioner, MapEncoderCNN, SpatialMapConditioner
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


def gear_to_dense(gear, n=DENSE):
    """把 N 个航点的档位序列最近邻上采样到稠密位姿长度 n。"""
    g = np.asarray(gear)
    idx = np.round(np.linspace(0, g.shape[0] - 1, n)).astype(int)
    return g[idx]


def gear_segment_indices(gear_dense):
    """按档位符号翻转把序列切成前向/倒车子段(索引区间 [i,j])。近 0 视为沿用前一段。"""
    n = len(gear_dense)
    if n < 2:
        return [(0, n)]
    segs = []; start = 0; cur = np.sign(gear_dense[0])
    for i in range(1, n):
        s = np.sign(gear_dense[i])
        if s != 0 and s != cur:
            segs.append((start, i)); start = i; cur = s
    segs.append((start, n))
    return segs


def gear_curv_worst(poses, gear_dense, step=CURV_STEP):
    """M15: 按**真实/预测档位**分段(非尖点启发式), 每段重采样后取最大 |κ|(位置 Menger)。
    返回 (worst_kappa, n_seg, switches)。有了 gear 就能把换档尖点正确切开, 使曲率**可作绝对硬门**。"""
    poses = np.asarray(poses, dtype=np.float64)
    worst = 0.0; nseg = 0
    sw = int(np.sum(np.abs(np.diff(np.sign(gear_dense))) > 0))
    for i, j in gear_segment_indices(gear_dense):
        seg = poses[i:j]
        if seg.shape[0] < 2 or float(cum_arclen(seg[:, :2])[-1]) < MIN_SEG_LEN:
            continue
        nseg += 1
        m = max(3, int(np.ceil(cum_arclen(seg[:, :2])[-1] / step)) + 1)
        worst = max(worst, _menger_max(resample_poses(seg, m)[:, :2]))
    return worst, nseg, sw


# --------------------------------------------------------------------------- #
# 主评估
# --------------------------------------------------------------------------- #
def load_model(ckpt=CKPT, cfg=None):
    cfg = cfg or default_config()
    dc = cfg.diffusion
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cond = blob.get("cond", "lat")
    if cond == "spatial":
        vae = load_frozen_vae(lat_ch=dc.lat_ch)
        enc = SpatialMapConditioner(vae, lat_hw=(9, 16), lat_ch=dc.lat_ch, out_ch=dc.map_emb,
                                    use_sdf=blob.get("map_cond", "lat_sdf") != "vae").to(device)
    elif cond == "lat":
        vae = load_frozen_vae(lat_ch=dc.lat_ch)
        use_sdf = blob.get("map_cond", "lat_sdf") == "lat_sdf"   # 旧 ckpt 无此字段 -> lat_sdf(向后兼容)
        enc = MapConditioner(vae, out_dim=dc.map_emb, lat_hw=(9, 16), use_sdf=use_sdf).to(device)
    else:
        enc = MapEncoderCNN(out_dim=dc.map_emb).to(device)
    enc.load_state_dict(blob["enc"])
    dc.denoiser = blob.get("denoiser", "mlp")     # 旧 ckpt 无此字段 -> mlp(向后兼容)
    dc.n_wp = int(blob.get("n_wp", dc.n_wp))       # N 随 ckpt(旧 40/缺省 40, N=80 自动 80)
    dc.use_gear = bool(blob.get("use_gear", False))  # M15: 旧 ckpt 无此字段 -> False(向后兼容)
    dc.pred_mode = blob.get("pred_mode", "eps")       # M16: 旧 ckpt 无此字段 -> eps(向后兼容)
    dc.representation = blob.get("representation", "pose")   # M17: 旧 ckpt 无此字段 -> pose(向后兼容)
    dc.ctrl_u_scale = float(blob.get("ctrl_u_scale", 1.0))
    model = build_denoiser(dc).to(device)
    model.load_state_dict(blob["model"])
    model.eval(); enc.eval()
    model.representation = dc.representation          # M17: 供 evaluate 判断是否需 integrate_ctrl 回位姿
    model.ctrl_u_scale = dc.ctrl_u_scale
    return model, enc, blob.get("bbox", (0, 0, cfg.lot.world_w, cfg.lot.world_h)), cond


def load_critic(path):
    """加载训好的可行性判别器(Phase 4 / M12)供 classifier guidance。维度全从 ckpt 自抝。"""
    from .critic import FeasibilityCritic
    blob = torch.load(path, map_location=device, weights_only=False)
    crit = FeasibilityCritic(blob["n_wp"], dim=blob["dim"], map_emb=blob["map_emb"],
                             hidden=blob["hidden"], layers=blob["layers"],
                             cond_ch=blob["cond_ch"]).to(device)
    crit.load_state_dict(blob["critic"]); crit.eval()
    return crit


def evaluate(ckpt=CKPT, npz_path=None, k=None, n_viz=None, render_path=None, gkw=None, ckw=None, refine=None, best_of=1):
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
    if cond in ("lat", "spatial"):
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
    critic = None; c_scale = 0.0; c_minabar = 0.9
    if ckw and ckw.get("scale", 0) > 0 and ckw.get("path"):
        critic = load_critic(ckw["path"]); c_scale = float(ckw["scale"])
        c_minabar = float(ckw.get("min_abar", 0.9))
    best_of = max(1, int(best_of or 1))
    u_scale = float(getattr(model, "ctrl_u_scale", 1.0))
    gen_dev = sample(model, enc, maps_t, s4, g4, sch, z=lat, guide=guide,
                     critic=critic, critic_scale=c_scale, critic_min_abar=c_minabar)   # (k,N,dim) device
    gen_n = gen_dev.cpu().numpy()                                                       # pose:(k,N,4/5) norm | ctrl:(k,N,2)

    # M17: ctrl 表示——将归一化控制 (k,N,2) 可微积分回世界位姿 (k,N,4)(曲率/非完整由构造保证)。
    #   预测档位由带符号弧长 u 的符号得到; 端点不钉(靠条件+事后 repair), 故终点误差可能非 0。
    is_ctrl = getattr(model, "representation", "pose") == "ctrl"
    world4 = pgear_all = None
    if is_ctrl:
        from .controlspace import (integrate_ctrl, refine_ctrl_to_goal, goal_reach_pen, _world_to_norm)
        from .penalty import footprint_collision
        kmax = 1.0 / veh.cfg.r_min
        st_t = torch.tensor(sp[:k], dtype=torch.float32, device=device)           # (k,3) 世界起点
        do_refine = bool(refine) and int(refine.get("iters", 0)) > 0
        coll = None
        wc = float(refine.get("w_coll", 0.0)) if refine else 0.0
        if wc > 0:                                       # M19b: 可微足迹-SDF 碰撞代价(投影与择优共用)
            foot = torch.tensor(Vehicle(cfg.vehicle).footprint_local,
                                dtype=torch.float32, device=device)
            coll = dict(sdf_m=maps_t[:, 1:2] * SDF_CLIP, foot=foot,
                        margin=float(refine.get("margin", 0.15)), w_coll=wc)

        def _project(ctrl):
            """一条 ctrl -> (投影后 ctrl, 世界位姿)。M18b/M19b: 终点(+碰撞)联合投影, κ 每步钳位。"""
            if do_refine:
                ctrl = refine_ctrl_to_goal(ctrl, st_t, g4, bbox, kmax, u_scale,
                                           iters=int(refine["iters"]), lr=float(refine.get("lr", 0.05)),
                                           w_pos=float(refine.get("w_pos", 1.0)),
                                           w_ang=float(refine.get("w_ang", 0.3)), coll=coll)
            return ctrl, integrate_ctrl(ctrl, st_t, kmax, u_scale)                 # (k,N,2), (k,N,4) 世界

        def _score(ctrl, poses):
            """择优代价 = 投影目标本身(终点归一化位置²+航向 + w_coll·碰撞), 越小越好。"""
            pe, ae = goal_reach_pen(ctrl, st_t, g4, bbox, kmax, u_scale)
            s = float(refine.get("w_pos", 1.0)) * pe + float(refine.get("w_ang", 0.3)) * ae
            if coll is not None:
                s = s + wc * footprint_collision(_world_to_norm(poses, bbox), coll["sdf_m"],
                                                 bbox, coll["foot"], coll["margin"])
            return s

        if best_of > 1:
            # M20: 采 best_of 条(不同噪声 DDPM 样本), 各自投影, 逐场景取代价最小的一条。
            ctrl_b = poses_b = score_b = None
            for j in range(best_of):
                cj = gen_dev if j == 0 else sample(model, enc, maps_t, s4, g4, sch, z=lat,
                                                   guide=guide, critic=critic,
                                                   critic_scale=c_scale, critic_min_abar=c_minabar)
                cj, pj = _project(cj)
                sj = _score(cj, pj)
                if score_b is None:
                    ctrl_b, poses_b, score_b = cj, pj, sj
                else:
                    better = sj < score_b
                    ctrl_b = torch.where(better[:, None, None], cj, ctrl_b)
                    poses_b = torch.where(better[:, None, None], pj, poses_b)
                    score_b = torch.where(better, sj, score_b)
            world4 = poses_b.cpu().numpy(); gen_n = ctrl_b.cpu().numpy()
        else:
            ctrl_t = torch.tensor(gen_n, dtype=torch.float32, device=device)
            ctrl_t, poses = _project(ctrl_t)
            world4 = poses.cpu().numpy(); gen_n = ctrl_t.cpu().numpy()
        pgear_all = np.sign(gen_n[:, :, 1])                                        # (k,N) ±1(由带符号弧长推档位)

    res = cfg.lot.res
    rmin = veh.cfg.r_min
    kcurv_lim = KAPPA_TOL / rmin                 # 档位感知曲率硬门(有了 gear 才能用)
    n_col = n_feas = n_ok = 0
    r_col = r_feas = r_ok = 0
    ratios = []; maxks = []; slips = []; nsegs = []; epos = []; eang = []
    r_ratios = []; r_maxks = []; r_slips = []; r_nsegs = []
    n_gcurv_ok = n_expgcurv_ok = 0; gaccs = []; pswitches = []
    cases = []
    for i in range(k):
        grid = maps[i, 0]
        # 生成: 归一化 -> 世界 (use_gear 时列 4 为预测档位); ctrl 时 world4 已是世界位姿、档位来自 u 符号
        if is_ctrl:
            gen4 = world4[i].copy()
            pgear = pgear_all[i]
        else:
            gene = gen_n[i].copy()
            gen4 = gene[:, :4].copy()
            pgear = gene[:, 4] if gene.shape[1] > 4 else None
            gen4[:, :2] = denormalize_xy(gen4[:, :2], *bbox)
        # --- 原始 ---
        gposes = densify_traj4(gen4)
        col = footprint_collides(grid, gposes, veh, res)
        feas, mslip, mk, nseg = feasibility(gposes, veh.cfg.r_min)
        if pgear is not None:                        # M15: 按预测档位的曲率可行 + 专家自检
            gd = gear_to_dense(np.sign(pgear))
            wk, _, wsw = gear_curv_worst(gposes, gd)
            n_gcurv_ok += (wk <= kcurv_lim); pswitches.append(wsw)
            gtg = np.sign(gear[i])
            msk = (gtg != 0)
            gaccs.append(float(((np.sign(pgear) == gtg) & msk).sum()) / max(int(msk.sum()), 1))
            ex = densify_traj4(traj[i]); exg = gear_to_dense(gtg)
            n_expgcurv_ok += (gear_curv_worst(ex, exg)[0] <= kcurv_lim)
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
        critic_scale=float(c_scale),
        raw_collision_free=n_col / k, raw_feasible=n_feas / k, raw_success=n_ok / k,
        raw_len_ratio=float(np.mean(ratios)), raw_mean_slip=float(np.mean(slips)),
        raw_seg_kappa=float(np.mean(maxks)), raw_gear_switches=float(np.mean(nsegs)),
        rep_collision_free=r_col / k, rep_feasible=r_feas / k, rep_success=r_ok / k,
        rep_len_ratio=float(np.mean(r_ratios)), rep_mean_slip=float(np.mean(r_slips)),
        rep_seg_kappa=float(np.mean(r_maxks)), rep_gear_switches=float(np.mean(r_nsegs)),
        slip_tol=SLIP_TOL,
        kappa_limit=1.0 / veh.cfg.r_min,
        goal_refine=int(refine.get("iters", 0)) if refine else 0,
        best_of=int(best_of),
        gear_curv_feasible=(n_gcurv_ok / k if gaccs else None),
        expert_gear_curv_feasible=(n_expgcurv_ok / k if gaccs else None),
        gear_acc=(float(np.mean(gaccs)) if gaccs else None),
        pred_gear_switches=(float(np.mean(pswitches)) if pswitches else None),
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
    ap.add_argument("--critic", type=str, default=None, help="可行性判别器 ckpt(classifier guidance)")
    ap.add_argument("--c-scale", type=float, default=0.0, help="判别器引导步长(0=关)")
    ap.add_argument("--c-min-abar", type=float, default=0.9, help="仅在 abar>=此值(晚步)时进判别器引导")
    ap.add_argument("--goal-refine", type=int, default=0, help="M18b ctrl: 测试时终点投影迭代数(0=关)")
    ap.add_argument("--refine-lr", type=float, default=0.05, help="M18b: 投影 Adam 学习率")
    ap.add_argument("--refine-wpos", type=float, default=1.0, help="M18b: 终点位置权重")
    ap.add_argument("--refine-wang", type=float, default=0.3, help="M18b: 终点航向权重")
    ap.add_argument("--refine-coll", type=float, default=0.0, help="M19b: 投影中足迹-SDF碰撞代价权重(0=关)")
    ap.add_argument("--refine-margin", type=float, default=0.15, help="M19b: 碰撞安全间隙(m)")
    ap.add_argument("--best-of", type=int, default=1, help="M20 ctrl: 测试时采 N 条投影后择优(1=关)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "figs", "parking", "m6_eval_compare.png")
    gkw = dict(scale=a.g_scale, curv=a.g_curv, coll=a.g_coll, nh=a.g_nh,
               margin=a.g_margin, min_abar=a.g_min_abar)
    ckw = dict(path=a.critic, scale=a.c_scale, min_abar=a.c_min_abar)
    refine = dict(iters=a.goal_refine, lr=a.refine_lr, w_pos=a.refine_wpos, w_ang=a.refine_wang,
                  w_coll=a.refine_coll, margin=a.refine_margin)
    m, _ = evaluate(a.ckpt, a.npz, a.k, a.viz, render_path=out, gkw=gkw, ckw=ckw, refine=refine,
                    best_of=a.best_of)
    for kk, v in m.items():
        print("  %-22s %s" % (kk, round(v, 4) if isinstance(v, float) else v))
    print("[saved] %s" % out)
