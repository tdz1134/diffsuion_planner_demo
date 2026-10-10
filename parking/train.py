# -*- coding: utf-8 -*-
"""泊车扩散规划器训练编排(依赖注入: 条件分支/去噪器/调度均可换)。

流程: 载入 parking npz -> 归一化(x,y->[-1,1], 全局 lot bbox) -> 建条件分支
  (lat=冻结泊车VAE latent+SDF / cnn=基线) -> 预存冻结 latent -> AMP 训练 ε-prediction
  -> 存 ckpt。--quick 做冒烟(短训练 + 采样端点校验)。

复用已落地的提速三件套: AMP + 大 batch(LR 按 sqrt(batch/base) 自适应) + 预存 latent。
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import default_config
from .dataset import load_dataset, SDF_CLIP
from .geometry import normalize_xy, heading_to_cs
from .map_vae import load_frozen_vae
from .conditioner import MapConditioner, MapEncoderCNN
from .diffusion import make_schedule, q_sample, build_denoiser, sample, pred_x0
from .penalty import traj_penalty_terms, smooth_pen
from .vehicle import Vehicle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 仅开 TF32(不额外吃显存)。注意: 不要开 cudnn.benchmark——它会给 VAE 大 batch(2048)编码
# 缓存大量卷积 workspace, 在 8GB 卡上直接 OOM(本文件之前加它导致训练爆显存)。
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
_PKG = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_PKG)
CKPT = os.path.join(_PKG, "cache", "diffusion_parking.pt")


# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
def norm_traj4(traj4, bbox):
    out = np.asarray(traj4, dtype=np.float32).copy()
    out[..., :2] = normalize_xy(out[..., :2], *bbox)
    return out


def pose3_to_norm4(pose3, bbox):
    """(M,3)[x,y,theta] 世界 -> (M,4)[nx,ny,cos,sin] 归一化。"""
    p = np.asarray(pose3, dtype=np.float32)
    xy = normalize_xy(p[:, :2], *bbox)
    c, s = heading_to_cs(p[:, 2])
    return np.stack([xy[:, 0], xy[:, 1], c, s], axis=1).astype(np.float32)


@torch.no_grad()
def precompute_latents(vae, maps, batch=2048):
    occ = maps[:, 0:1]
    out = [vae.encode(occ[i:i + batch]) for i in range(0, occ.shape[0], batch)]
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(model, enc, x0, maps, s4, g4, sch, cfg, steps, batch, lat=None, pen=None):
    params = [p for p in list(model.parameters()) + list(enc.parameters()) if p.requires_grad]
    lr = cfg.lr * (batch / cfg.batch) ** 0.5
    opt = torch.optim.Adam(params, lr=lr)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n = x0.shape[0]
    use_x0 = (getattr(cfg, "pred_mode", "eps") == "x0")     # M16: 直接预测 x0_hat(轨迹中心)
    model.train(); enc.train()
    pbar = tqdm(range(steps), desc="train diffusion", unit="step", ncols=90)
    for step in pbar:
        idx = torch.randint(0, n, (batch,), device=device)
        xb, m, s, g = x0[idx], maps[idx], s4[idx], g4[idx]
        t = torch.randint(0, sch["T"], (batch,), device=device)
        xt, noise = q_sample(xb, t, sch)
        z = lat[idx] if lat is not None else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            map_emb = enc(m, z=z) if z is not None else enc(m)
            pred = model(xt, t, map_emb, s, g)
        target = xb if use_x0 else noise                     # x0-pred 监督干净轨迹; 否则监督噪声
        loss = nn.functional.mse_loss(pred.float(), target)
        postfix = {"loss": f"{loss.item():.4f}"}
        if pen is not None:
            x0h = pred.float() if use_x0 else pred_x0(xt.float(), pred.float(), sch, t)
            geo = x0h[..., :4]
            sdf_m = m[:, 1:2] * pen["sdf_clip"]
            tp = traj_penalty_terms(geo, sdf_m, pen["bbox"], pen["veh"],
                                    pen["foot"], pen["margin"])
            gate = sch["abar"][t]                     # (B,) x0_hat 可靠度(t 小→~1)
            aux = (pen["w_nh"] * (gate * tp["nh"]).mean()
                   + pen["w_curv"] * (gate * tp["curv"]).mean()
                   + pen["w_coll"] * (gate * tp["coll"]).mean())
            if pen.get("w_curv_x0", 0.0):            # M16: x0-pred 下曲率惩罚(良态, 无 1/√ᾱ 放大)
                aux = aux + pen["w_curv_x0"] * (gate * tp["curv"]).mean()
            if pen.get("w_smooth", 0.0):             # M16: 一/二阶时序差平滑(治拖动)
                aux = aux + pen["w_smooth"] * (gate * smooth_pen(geo, pen["bbox"])).mean()
            loss = loss + aux
            postfix.update(curv=f"{tp['curv'].mean().item():.3f}",
                           coll=f"{tp['coll'].mean().item():.3f}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        pbar.set_postfix(**postfix)
    return model, enc


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main(npz_path=None, steps=None, batch=None, cond="lat", quick=False, out=CKPT,
         w_nh=None, w_curv=None, w_coll=None, coll_margin=None, denoiser=None, n_wp=None,
         map_cond=None, use_gear=None, pred_mode=None, w_curv_x0=None, w_smooth=None):
    cfg = default_config()
    dc = cfg.diffusion
    if denoiser is not None:
        dc.denoiser = denoiser
    if map_cond is not None:
        dc.map_cond = map_cond
    if use_gear is not None:
        dc.use_gear = use_gear
    if pred_mode is not None:
        dc.pred_mode = pred_mode
    if w_curv_x0 is not None:
        dc.w_curv_x0 = w_curv_x0
    if w_smooth is not None:
        dc.w_smooth = w_smooth
    if w_nh is not None:
        dc.w_nh = w_nh
    if w_curv is not None:
        dc.w_curv = w_curv
    if w_coll is not None:
        dc.w_coll = w_coll
    if coll_margin is not None:
        dc.coll_margin = coll_margin
    if npz_path is None:
        cdir = os.path.join(_REPO, cfg.data.cache_dir)
        cands = sorted([f for f in os.listdir(cdir)
                        if f.startswith("parking_") and f.endswith(".npz")],
                       key=lambda s: int(s.split("_")[1].split(".")[0]), reverse=True)
        npz_path = os.path.join(cdir, cands[0])
    steps = steps or (2000 if quick else dc.train_steps)
    batch = batch or dc.batch

    d = load_dataset(npz_path)
    bbox = tuple(float(v) for v in d["bbox"])
    # 从数据自身推导定长 N(=traj 第2维): 旧 40 数据仍得 40, N=80 数据自动得 80。
    dc.n_wp = int(n_wp) if n_wp else int(d["traj"].shape[1])
    tr_mask = d["split"] == 0
    maps = torch.tensor(d["maps"][tr_mask], device=device)
    x0 = torch.tensor(norm_traj4(d["traj"][tr_mask], bbox), device=device)
    if dc.use_gear:                                   # M15: 档位作为额外一个扩散输出通道(已∈[-1,1], 不再归一)
        gch = torch.tensor(d["gear"][tr_mask], dtype=torch.float32, device=device)[:, :, None]
        x0 = torch.cat([x0, gch], dim=-1)             # (N,4)->(N,5)
    s4 = torch.tensor(pose3_to_norm4(d["start_pose"][tr_mask], bbox), device=device)
    g4 = torch.tensor(pose3_to_norm4(d["goal_pose"][tr_mask], bbox), device=device)
    print("[info] device=%s npz=%s train=%d x0=%s" % (device, npz_path,
                                                      x0.shape[0], tuple(x0.shape)))

    lat = None
    if cond == "lat":
        use_sdf = getattr(dc, "map_cond", "lat_sdf") == "lat_sdf"
        vae = load_frozen_vae(lat_ch=dc.lat_ch)
        enc = MapConditioner(vae, out_dim=dc.map_emb,
                             lat_hw=(maps.shape[2] // 8, maps.shape[3] // 8),
                             use_sdf=use_sdf).to(device)
        lat = precompute_latents(enc.vae, maps)
        print("[cond] frozen-VAE %s; precomputed lat %s"
              % ("latent+SDF" if use_sdf else "latent-only(vae)", tuple(lat.shape)))
    else:
        enc = MapEncoderCNN(out_dim=dc.map_emb).to(device)
        print("[cond] CNN baseline")
    model = build_denoiser(dc).to(device)
    npar = sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print("[model] denoiser=%s trainable params=%.1fK" % (dc.denoiser, npar / 1e3))

    pen = None
    if dc.w_nh or dc.w_curv or dc.w_coll or dc.w_curv_x0 or dc.w_smooth:
        veh = Vehicle(cfg.vehicle)
        foot = torch.tensor(veh.footprint_local, dtype=torch.float32, device=device)
        pen = dict(w_nh=dc.w_nh, w_curv=dc.w_curv, w_coll=dc.w_coll,
                   w_curv_x0=dc.w_curv_x0, w_smooth=dc.w_smooth,
                   margin=dc.coll_margin, foot=foot, veh=cfg.vehicle,
                   bbox=bbox, sdf_clip=SDF_CLIP)
        print("[pen] aux traj loss nh=%g curv=%g coll=%g curv_x0=%g smooth=%g margin=%.2f clip=%.1fm"
              % (dc.w_nh, dc.w_curv, dc.w_coll, dc.w_curv_x0, dc.w_smooth,
                 dc.coll_margin, SDF_CLIP))

    sch = make_schedule(dc.t_steps, dc.b0, dc.b1, device)
    train(model, enc, x0, maps, s4, g4, sch, dc, steps=steps, batch=batch, lat=lat, pen=pen)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(), "enc": enc.state_dict(),
                "cond": cond, "n_wp": dc.n_wp, "bbox": bbox,
                "denoiser": dc.denoiser, "map_cond": dc.map_cond,
                "use_gear": bool(dc.use_gear), "pred_mode": dc.pred_mode}, out)
    print("[saved] %s" % out)

    if quick:   # 冒烟: 采样少量, 校验端点钉住
        k = min(8, x0.shape[0])
        gen = sample(model, enc, maps[:k], s4[:k], g4[:k], sch,
                     z=(lat[:k] if lat is not None else None))
        e0 = (gen[:, 0] - s4[:k]).abs().max().item()
        e1 = (gen[:, -1] - g4[:k]).abs().max().item()
        print("[smoke] sample ok shape=%s  pin err start=%.2e goal=%.2e"
              % (tuple(gen.shape), e0, e1))
    return model, enc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--cond", choices=["lat", "cnn"], default="lat")
    ap.add_argument("--out", type=str, default=CKPT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--w-nh", type=float, default=None, help="Phase2 航向一致性损失权重(0=关)")
    ap.add_argument("--w-curv", type=float, default=None, help="Phase2 曲率超限损失权重(0=关)")
    ap.add_argument("--w-coll", type=float, default=None, help="Phase2 足迹碰撞损失权重(0=关)")
    ap.add_argument("--coll-margin", type=float, default=None, help="足迹碰撞安全间隙(m)")
    ap.add_argument("--denoiser", choices=["mlp", "conv", "trans", "trans2"], default=None,
                    help="去噪器架构: conv=1D 时序卷积, trans=旧 Transformer(单前缀token), trans2=忠实版(cross-attn+AdaLN); 缺省用 config(mlp)")
    ap.add_argument("--map-cond", choices=["lat_sdf", "vae"], default=None,
                    help="地图条件(cond=lat): lat_sdf=VAE latent⊕SDF(默认), vae=只用 VAE latent")
    ap.add_argument("--use-gear", action="store_true",
                    help="M15: 把档位 gear 作为额外的扩散输出通道(state dim 4->5)")
    ap.add_argument("--pred-mode", choices=["eps", "x0"], default=None,
                    help="M16: 去噪预测目标 eps(默认)/x0(轨迹中心, 配 x0 可行性损失)")
    ap.add_argument("--w-curv-x0", type=float, default=None, help="M16 x0-空间曲率惩罚权重(0=关)")
    ap.add_argument("--w-smooth", type=float, default=None, help="M16 一/二阶时序差平滑惩罚权重(0=关)")
    ap.add_argument("--n-wp", type=int, default=None,
                    help="定长航点数 N; 缺省自动从数据 traj 推导(向后兼容)")
    a = ap.parse_args()
    main(a.npz, a.steps, a.batch, a.cond, a.quick, a.out,
         w_nh=a.w_nh, w_curv=a.w_curv, w_coll=a.w_coll, coll_margin=a.coll_margin,
         denoiser=a.denoiser, n_wp=a.n_wp, map_cond=a.map_cond, use_gear=a.use_gear or None,
         pred_mode=a.pred_mode, w_curv_x0=a.w_curv_x0, w_smooth=a.w_smooth)
