# -*- coding: utf-8 -*-
"""
条件扩散路径规划器 (Diffusion Path Planner)
===========================================
在占据栅格地图(黑=障碍, 白=空闲)上, 给定 start / goal,
用条件 DDPM 从纯噪声"去噪"生成一条航点路径。

流程:
  1) 造数据: grid_env 生成随机地图 + A*/加权A*/贪心 专家路径 -> (map, start, goal, path)
  2) 训练:   条件 DDPM, 去噪网络 = CNN(地图编码) + MLP, 预测噪声
  3) 采样:   从纯噪声反向去噪, 每步"钉住"首尾航点=start/goal, 得到路径
  4) 评估:   叠画 地图+生成路径+A*参考 到图片, 并统计 无碰撞率/成功率/长度比

运行:
    source ../venv_py38/bin/activate
    python diffusion_path.py
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import grid_env as ge

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
SEED      = 0
GRID      = 64            # 地图 H=W
N_WP      = 32            # 定长航点数
T_STEPS   = 200           # 扩散步数
B0, B1    = 1e-4, 0.02
N_DATA    = 20000         # 训练样本数
N_TRAIN   = 40000         # 训练梯度步数
BATCH     = 256
LR        = 2e-4
MAP_EMB   = 64
TEMB      = 64
HIDDEN    = 512
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(OUT_DIR, "figs", "diffusion_path")   # 效果图统一放 figs/<脚本名>/
DATASET   = os.path.join(OUT_DIR, "dataset.npz")   # 数据集缓存

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] device = {device}")


# --------------------------------------------------------------------------- #
# 坐标换算: 归一化 [-1,1] <-> 栅格 (r, c)
# --------------------------------------------------------------------------- #
def rc_to_norm(rc, size=GRID):
    """(r, c) -> 归一化 (x, y), x=c, y=r。"""
    r, c = rc
    x = c / (size - 1) * 2.0 - 1.0
    y = r / (size - 1) * 2.0 - 1.0
    return np.array([x, y], dtype=np.float32)


def norm_to_rc(v, size=GRID):
    """归一化 (x, y) -> 浮点 (r, c)。"""
    x, y = v
    c = (x + 1) / 2.0 * (size - 1)
    r = (y + 1) / 2.0 * (size - 1)
    return r, c


# --------------------------------------------------------------------------- #
# 1) 造数据集
# --------------------------------------------------------------------------- #
def load_dataset(M=N_DATA, regen=False):
    """加载 make_dataset.py 生成的缓存; 缺失时自动生成。
    数据生成已拆到独立脚本 make_dataset.py (最耗时的一步)。"""
    import make_dataset
    ds_path = os.path.join(OUT_DIR, f"dataset_{M}.npz")
    if regen or not os.path.exists(ds_path):
        print(f"  [提示] 未找到 {ds_path}, 正在生成 (建议先单独跑: python make_dataset.py --data {M}) ...")
        make_dataset.generate(M, seed=SEED, overwrite=True)
    z = np.load(ds_path)
    print(f"  [cache] 载入 {ds_path}")
    return (torch.tensor(z["maps"], device=device),
            torch.tensor(z["starts"], device=device),
            torch.tensor(z["goals"], device=device),
            torch.tensor(z["paths"], device=device))


# --------------------------------------------------------------------------- #
# 2) 模型: 地图 CNN 编码 + 条件 MLP 去噪
# --------------------------------------------------------------------------- #
def time_embedding(t, dim=TEMB):
    half = dim // 2
    f = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
    a = t[:, None].float() * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class MapEncoder(nn.Module):
    def __init__(self, out_dim=MAP_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.AdaptiveAvgPool2d(4),                                 # 64*4*4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim), nn.ReLU(),
        )

    def forward(self, m):
        return self.net(m)


def load_frozen_vae():
    """载入并冻结 vae_map.py 训好的地图 VAE(只用于编码占据图)。"""
    import vae_map
    vae = vae_map.VAE().to(device)
    vae.load_state_dict(torch.load(vae_map.CKPT, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


class LatentEnc(nn.Module):
    """把冻结 VAE 的空间 latent (B,LAT_CH,8,8) 压成向量。"""

    def __init__(self, in_dim, out=MAP_EMB // 2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out), nn.ReLU())

    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class SdfEnc(nn.Module):
    """把显式 SDF 通道 (B,1,64,64) 用小题 CNN 压成向量(保留障碍距离信息)。"""

    def __init__(self, out=MAP_EMB // 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.AdaptiveAvgPool2d(4),                               # 16*4*4
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, out), nn.ReLU(),
        )

    def forward(self, sdf):
        return self.net(sdf)


class MapConditioner(nn.Module):
    """地图条件分支: 冻结VAE的占据latent + 显式SDF 双分支 -> MAP_EMB。
    接口与 MapEncoder 一致(吃 (B,2,64,64), 吐 (B,MAP_EMB)), 故训练/采样循环无需改。"""

    def __init__(self, vae, out_dim=MAP_EMB):
        super().__init__()
        self.vae = vae                       # 冻结子模块
        lat_dim = vae.enc.mu.out_channels * 8 * 8   # LAT_CH * 8 * 8 (=512)
        self.lat_enc = LatentEnc(lat_dim, out_dim // 2)
        self.sdf_enc = SdfEnc(out_dim // 2)

    def forward(self, m, z=None):
        sdf = m[:, 1:2]
        if z is None:                       # 未预存时现算(采样阶段少量图)
            with torch.no_grad():
                z = self.vae.encode(m[:, 0:1])   # (B,LAT_CH,8,8) 取 mu(确定性)
        return torch.cat([self.lat_enc(z), self.sdf_enc(sdf)], dim=1)


class CondDenoiser(nn.Module):
    """输入 (x_t[N*2], t, map, start, goal) -> 预测噪声 (N*2)。"""

    def __init__(self):
        super().__init__()
        in_dim = N_WP * 2 + TEMB + MAP_EMB + 2 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, N_WP * 2),
        )

    def forward(self, x, t, map_emb, start, goal):
        h = torch.cat([x.reshape(x.shape[0], -1), time_embedding(t),
                       map_emb, start, goal], dim=1)
        return self.net(h).reshape(x.shape[0], N_WP, 2)


# --------------------------------------------------------------------------- #
# 3) 扩散调度 / 前向 / 反向
# --------------------------------------------------------------------------- #
BETAS = torch.linspace(B0, B1, T_STEPS, device=device)
ALPHAS = 1.0 - BETAS
ABAR = torch.cumprod(ALPHAS, dim=0)


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sa = torch.sqrt(ABAR[t])[:, None, None]
    sb = torch.sqrt(1.0 - ABAR[t])[:, None, None]
    return sa * x0 + sb * noise, noise


@torch.no_grad()
def precompute_latents(vae, maps, batch=2048):
    """VAE 冻结且输入固定 -> 一次性算好全部 latent, 训练热循环不再重跑 VAE。"""
    occ = maps[:, 0:1]
    out = [vae.encode(occ[i:i + batch]) for i in range(0, occ.shape[0], batch)]
    return torch.cat(out)                    # (N, LAT_CH, 8, 8)


def train(model, enc, data, steps=N_TRAIN, batch=BATCH, lat=None):
    maps, starts, goals, paths = data
    params = [p for p in list(model.parameters()) + list(enc.parameters()) if p.requires_grad]
    lr = LR * (batch / BATCH) ** 0.5              # 学习率随 batch 自适应
    opt = torch.optim.Adam(params, lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n = paths.shape[0]
    model.train(); enc.train()
    pbar = tqdm(range(steps), desc="训练", unit="步", ncols=90)
    for step in pbar:
        idx = torch.randint(0, n, (batch,), device=device)
        x0, m, s, g = paths[idx], maps[idx], starts[idx], goals[idx]
        t = torch.randint(0, T_STEPS, (batch,), device=device)
        xt, noise = q_sample(x0, t)
        z = lat[idx] if lat is not None else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            map_emb = enc(m, z=z) if z is not None else enc(m)
            pred = model(xt, t, map_emb, s, g)
        loss = nn.functional.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return model, enc


@torch.no_grad()
def sample(model, enc, maps, starts, goals):
    """反向去噪采样, 每步钉住首尾航点。返回 (B, N, 2) 归一化路径。"""
    B = maps.shape[0]
    model.eval(); enc.eval()
    map_emb = enc(maps)
    x = torch.randn(B, N_WP, 2, device=device)
    x[:, 0] = starts; x[:, -1] = goals
    for t in reversed(range(T_STEPS)):
        tt = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(x, tt, map_emb, starts, goals)
        mean = (x - BETAS[t] / torch.sqrt(1.0 - ABAR[t]) * eps) / torch.sqrt(ALPHAS[t])
        if t > 0:
            x = mean + torch.sqrt(BETAS[t]) * torch.randn_like(x)
        else:
            x = mean
        x[:, 0] = starts; x[:, -1] = goals      # 钉住端点
    return x


# --------------------------------------------------------------------------- #
# 4) 评估: 碰撞 / 成功率 / 长度比 + 渲染
# --------------------------------------------------------------------------- #
def densify(path_norm, n=200):
    """把 N 个航点插值成 n 个密点, 用于碰撞检测与长度计算。"""
    t_old = np.linspace(0, 1, path_norm.shape[0])
    t_new = np.linspace(0, 1, n)
    xs = np.interp(t_new, t_old, path_norm[:, 0])
    ys = np.interp(t_new, t_old, path_norm[:, 1])
    return np.stack([xs, ys], axis=1)


def evaluate(grid, path_norm):
    """返回 (collision:bool, length_grid:float)。"""
    d = densify(path_norm)
    rc = np.stack([norm_to_rc(v)[0] for v in d]), np.stack([norm_to_rc(v)[1] for v in d])
    rr = np.clip(np.round(rc[0]).astype(int), 0, GRID - 1)
    cc = np.clip(np.round(rc[1]).astype(int), 0, GRID - 1)
    collision = bool(grid[rr, cc].max() > 0)
    pts = np.stack([rc[1], rc[0]], axis=1)          # (x=c, y=r)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return collision, length


def repair_path(dist_cells, path_norm, margin=2.0, push_iter=30, smooth_iter=20):
    """采样后修复: 用距离场把航点推离障碍 + 轻度平滑, 端点保持不动。
    dist_cells: (H,W) 格数距离场。返回修复后的 (N,2) 归一化路径。"""
    wp = np.array([norm_to_rc(v) for v in path_norm], dtype=np.float64)  # (N,2)=(r,c)
    gy, gx = np.gradient(dist_cells)               # 距离场梯度(指向远离障碍)
    H, W = dist_cells.shape
    for _ in range(push_iter):
        rr = np.clip(np.round(wp[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(wp[:, 1]).astype(int), 0, W - 1)
        d = dist_cells[rr, cc]
        need = d < margin
        if not need.any():
            break
        dirr = gy[rr, cc]; dirc = gx[rr, cc]
        nrm = np.sqrt(dirr ** 2 + dirc ** 2) + 1e-8
        step = (margin - d) * 0.6
        wp[need, 0] += dirr[need] / nrm[need] * step[need]
        wp[need, 1] += dirc[need] / nrm[need] * step[need]
    for _ in range(smooth_iter):                   # 拉普拉斯平滑(不动端点)
        wp[1:-1] = 0.5 * wp[1:-1] + 0.25 * (wp[:-2] + wp[2:])
        rr = np.clip(np.round(wp[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(wp[:, 1]).astype(int), 0, W - 1)
        d = dist_cells[rr, cc]
        need = d < margin
        if need.any():
            dirr = gy[rr, cc]; dirc = gx[rr, cc]
            nrm = np.sqrt(dirr ** 2 + dirc ** 2) + 1e-8
            step = (margin - d) * 0.6
            wp[need, 0] += dirr[need] / nrm[need] * step[need]
            wp[need, 1] += dirc[need] / nrm[need] * step[need]
    wp[0] = norm_to_rc(path_norm[0]); wp[-1] = norm_to_rc(path_norm[-1])  # 钉端点
    out = np.array([[v[1] / (GRID - 1) * 2 - 1, v[0] / (GRID - 1) * 2 - 1]
                    for v in wp], dtype=np.float32)   # (r,c)->(x,y) 归一化
    return out


def render_case(ax, grid, start, goal, gen_norm, astar_rc, title):
    ax.imshow(grid, cmap="gray_r", origin="upper")   # 1=黑(障碍), 0=白(空闲)
    if astar_rc is not None:
        a = np.asarray(astar_rc, dtype=float)
        ax.plot(a[:, 1], a[:, 0], "--", color="tab:blue", lw=1.2, label="A* ref")
    g = np.stack([norm_to_rc(v)[1] for v in gen_norm]), \
        np.stack([norm_to_rc(v)[0] for v in gen_norm])   # (x=c, y=r)
    ax.plot(g[0], g[1], "-", color="tab:red", lw=1.6, label="diffusion")
    ax.plot(start[1], start[0], "o", color="lime", ms=7, label="start")
    ax.plot(goal[1], goal[0], "*", color="orange", ms=12, label="goal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def render_grid(cases, path):
    k = len(cases)
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cases):
        render_case(axes[i], c["grid"], c["start"], c["goal"],
                    c["gen"], c["astar"], c["title"])
    for j in range(k, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[saved] {path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    print("[1/4] 加载训练数据 ...")
    data = load_dataset(M=args.data, regen=args.regen)
    maps, starts, goals, paths = data
    print(f"  maps {tuple(maps.shape)} paths {tuple(paths.shape)}")

    print("[2/4] 训练条件扩散模型 ...")
    cond = getattr(args, "cond", "lat")
    lat = None
    if cond == "lat":
        print("  条件分支: 冻结 VAE latent + SDF 双分支")
        enc = MapConditioner(load_frozen_vae()).to(device)
        lat = precompute_latents(enc.vae, maps)      # VAE 冻结, 一次算好
        print(f"  [预存] 训练 latent {tuple(lat.shape)} (热循环不再跑 VAE)")
    else:
        print("  条件分支: 原始 2 通道 CNN MapEncoder (基线)")
        enc = MapEncoder().to(device)
    model = CondDenoiser().to(device)
    train(model, enc, data, steps=args.train, batch=args.batch, lat=lat)

    print("[3/4] 采样 + 评估 ...")
    rng = np.random.default_rng(123)
    K = 8
    idx = rng.integers(0, paths.shape[0], K)
    tm, ts, tg = maps[idx], starts[idx], goals[idx]
    gen = sample(model, enc, tm, ts, tg).cpu().numpy()

    n_col = n_ok = 0
    ratios = []
    cases = []
    for i in range(K):
        grid = tm[i, 0].cpu().numpy()                 # 占据通道
        dist_cells = tm[i, 1].cpu().numpy() * GRID    # 距离场(格数)
        s_rc = tuple(np.round(norm_to_rc(ts[i].cpu().numpy())).astype(int))
        g_rc = tuple(np.round(norm_to_rc(tg[i].cpu().numpy())).astype(int))
        a_rc = ge.astar(grid, s_rc, g_rc, 1.0)
        fixed = repair_path(dist_cells, gen[i])       # 采样后修复
        col, ln = evaluate(grid, fixed)
        n_col += col
        if not col:
            n_ok += 1
            if a_rc:
                ratios.append(ln / ge.path_length_grid(a_rc))
        cases.append(dict(grid=grid, start=s_rc, goal=g_rc, gen=fixed,
                          astar=a_rc,
                          title=f"#{i} coll={col}"))
    msg = f"  无碰撞率: {n_ok}/{K}"
    if ratios:
        msg += f"   平均长度比(vs A*): {np.mean(ratios):.2f}"
    print(msg)

    print("[4/4] 渲染 ...")
    os.makedirs(FIG_DIR, exist_ok=True)
    render_grid(cases, os.path.join(FIG_DIR, "fig_planner.png"))
    torch.save({"model": model.state_dict(), "enc": enc.state_dict()},
               os.path.join(OUT_DIR, "planner_ckpt.pt"))
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="条件扩散路径规划器")
    ap.add_argument("--data", type=int, default=N_DATA, help="训练样本数 (需与数据集缓存一致)")
    ap.add_argument("--train", type=int, default=N_TRAIN, help="训练步数")
    ap.add_argument("--regen", action="store_true", help="强制重新生成数据集缓存")
    ap.add_argument("--batch", type=int, default=512, help="训练 batch (默认512, 学习率随其自适应)")
    ap.add_argument("--cond", choices=["lat", "cnn"], default="lat",
                    help="地图条件分支: lat=冻结VAE的latent+SDF(默认), cnn=原始2通道CNN(基线对比)")
    ap.add_argument("--quick", action="store_true",
                    help="快速烟雾测试(小数据+短训练, 几十秒验证流程)")
    args = ap.parse_args()
    if args.quick:
        args.data = min(args.data, 1500)
        args.train = min(args.train, 3000)
        print("[quick] 快速模式: data=%d train=%d (效果仅供流程验证)" % (args.data, args.train))
    main(args)
