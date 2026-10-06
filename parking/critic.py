# -*- coding: utf-8 -*-
"""可行性判别器(Phase 4 / M12): 正规 classifier guidance 的"分类器"。

结论(M12 诚实): 本模块已实现且噪声感知修正都做了, 但**推理期 classifier guidance 仍发散**
(与 M8 手写几何代价同根病: 梯度把样本推向离流形对抗区)。**保留为已测-已弃用的产物, 默认关。**

动机: M8 用**手写几何代价**(曲率/航向)在 x0_hat 上做引导会发散(病态梯度); 且当时 N=40
曲率根本量不准。M11 把 N 提到 80 后曲率可测, 于是这里改用**学出来的判别器**给出平滑的
 feasibility 梯度——训练它把「HA* 专家轨迹(可行)」与「扩散模型自己的采样(多半不可行)」分开,
 采样时沿 `∇ log p(feasible|x)` 方向推后验均值(classifier guidance, Dhariwal & Pelt 思路)。

与去噪器**解耦**: 判别器只看轨迹的**干净空间** x0_hat(不含时间步), 结构复用 1D 时序卷积
(与 CondDenoiserConv 同风格), 输出一个标量 logit(越大=越像可行专家)。

用法(在 conv@N80 ckpt 上训判别器):
    python -m parking.critic --ckpt parking/cache/diffusion_parking_conv_n80.pt \
        --npz cache/parking_12000_n80.npz --out parking/cache/critic_n80.pt
评估时启用引导:
    python -m parking.evaluate --ckpt <conv_n80> --critic parking/cache/critic_n80.pt --c-scale 1.0
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from .config import default_config
from .diffusion import _DilatedResBlock


class FeasibilityCritic(nn.Module):
    """(x[B,N,dim], map_emb, start, goal) -> logit[B]。越大越"可行/像专家"。

    仅作用于**干净轨迹**(x0_hat), 不依赖扩散时间步。条件(地图/起终点)经 MLP 后逐位置
    广播拼接; 额外位置通道打破平移对称。池化用 mean+max(保留最坏段的曲率/碰撞线索)。
    """

    def __init__(self, n_wp, dim=4, map_emb=64, hidden=128, layers=4, cond_ch=32):
        super().__init__()
        self.n_wp, self.dim = n_wp, dim
        self.cond = nn.Sequential(
            nn.Linear(map_emb + 2 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, cond_ch),
        )
        c_in = dim + cond_ch + 1
        self.stem = nn.Conv1d(c_in, hidden, 1)
        self.body = nn.ModuleList(
            [_DilatedResBlock(hidden, 2 ** (i % 5)) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        self.register_buffer("pos", torch.linspace(-1.0, 1.0, n_wp).view(1, n_wp, 1))

    def forward(self, x, map_emb, start, goal):
        B, N, _ = x.shape
        cvec = self.cond(torch.cat([map_emb, start, goal], dim=1))
        cb = cvec[:, None, :].expand(B, N, cvec.shape[1])
        pb = self.pos.expand(B, N, 1)
        h = torch.cat([x, cb, pb], dim=-1).transpose(1, 2)
        h = self.stem(h)
        for blk in self.body:
            h = blk(h)
        h = h.transpose(1, 2)                                  # (B,N,hidden)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)                  # (B,)


def build_critic(dc, n_wp=None):
    return FeasibilityCritic(n_wp or dc.n_wp, dim=dc.dim, map_emb=dc.map_emb,
                             hidden=dc.dconv_hidden, layers=dc.dconv_layers,
                             cond_ch=dc.dconv_cond_ch)


# --------------------------------------------------------------------------- #
# 训练: 正样本=专家轨迹, 负样本=扩散模型采样
# --------------------------------------------------------------------------- #
def _gather_data(cfg, npz_path, ckpt, n_scene, batch):
    """返回噪声增强训练集 (x_t, map_emb, s4, g4, label)。label 1=专家, 0=模型采样。

    **关键(classifier guidance 正确姿势)**: 采样器引导时看到的是各噪声级 x_t, 故判别器
    必须在 q_sample(x0, t) 的**带噪轨迹**上训练, 否则它在采样轨迹上的梯度为分布外→对抗样本。
    每条干净轨迹(rep 个)随机抽 t 加密; 高噪 x_t 天然歧义 → 防 acc 饱和到 1.0。"""
    from .dataset import load_dataset
    from .train import norm_traj4, pose3_to_norm4, precompute_latents, device
    from .evaluate import load_model
    from .diffusion import make_schedule, sample, q_sample

    d = load_dataset(npz_path)
    bbox = tuple(float(v) for v in d["bbox"])
    tr = d["split"] == 0
    maps = d["maps"][tr][:n_scene]
    traj = d["traj"][tr][:n_scene]
    sp = d["start_pose"][tr][:n_scene]
    gp = d["goal_pose"][tr][:n_scene]

    model, enc, _, cond = load_model(ckpt, cfg)
    dc = cfg.diffusion
    maps_t = torch.tensor(maps, device=device)
    s4 = torch.tensor(pose3_to_norm4(sp, bbox), device=device)
    g4 = torch.tensor(pose3_to_norm4(gp, bbox), device=device)
    with torch.no_grad():
        lat = precompute_latents(enc.vae, maps_t) if cond == "lat" else None
        map_emb = enc(maps_t, z=lat) if lat is not None else enc(maps_t)

    sch = make_schedule(dc.t_steps, dc.b0, dc.b1, device)
    with torch.no_grad():
        neg = sample(model, enc, maps_t, s4, g4, sch, z=lat).to(device)
    pos = torch.tensor(norm_traj4(traj, bbox), device=device)

    clean = torch.cat([pos, neg], dim=0)                       # (2n, N, 4)
    lab0 = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).to(device)
    emb0 = torch.cat([map_emb, map_emb], dim=0)
    s0 = torch.cat([s4, s4], dim=0); g0 = torch.cat([g4, g4], dim=0)
    C = clean.shape[0]
    reps = 4                                                   # 每条干净轨迹的噪声副本数
    idx = torch.arange(C, device=device).repeat_interleave(reps)
    ts = torch.randint(0, dc.t_steps, (idx.shape[0],), device=device)
    xt, _ = q_sample(clean[idx], ts, sch)                      # 带噪轨迹
    return xt, emb0[idx], s0[idx], g0[idx], lab0[idx]


def train(npz_path=None, ckpt=None, out="parking/cache/critic_n80.pt",
          n_scene=1500, steps=3000, batch=128, lr=1e-3, quick=False):
    cfg = default_config()
    from .train import device
    if npz_path is None:
        npz_path = "cache/parking_12000_n80.npz"
    if ckpt is None:
        ckpt = "parking/cache/diffusion_parking_conv_n80.pt"
    if quick:
        n_scene, steps = 200, 200

    x_all, emb_all, s_all, g_all, y_all = _gather_data(cfg, npz_path, ckpt, n_scene, batch)
    M = x_all.shape[0]
    print("[critic] train pairs=%d  pos(expert)=%d  neg(model)=%d  x=%s"
          % (M, int(y_all.sum()), int((1 - y_all).sum()), tuple(x_all.shape)))

    crit = build_critic(cfg.diffusion, n_wp=x_all.shape[1]).to(device)
    opt = torch.optim.AdamW(crit.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device="cpu").manual_seed(0)
    crit.train()
    for it in range(steps):
        idx = torch.randint(0, M, (batch,), generator=g).to(device)
        logit = crit(x_all[idx], emb_all[idx], s_all[idx], g_all[idx])
        loss = bce(logit, y_all[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 200 == 0 or it == steps - 1:
            with torch.no_grad():
                acc = ((torch.sigmoid(logit) > 0.5).float() == y_all[idx]).float().mean().item()
            print("  step %4d/%d  loss %.4f  acc %.3f" % (it, steps, loss.item(), acc))

    # 全量末评估(区分度)
    crit.eval()
    with torch.no_grad():
        p = torch.sigmoid(crit(x_all, emb_all, s_all, g_all))
        acc = ((p > 0.5).float() == y_all).float().mean().item()
        pp_exp = p[y_all == 1].mean().item(); pp_neg = p[y_all == 0].mean().item()
    print("[critic] final acc=%.3f  mean p(expert)=%.3f  p(model)=%.3f" % (acc, pp_exp, pp_neg))
    torch.save({"critic": crit.state_dict(), "n_wp": x_all.shape[1],
                "map_emb": cfg.diffusion.map_emb, "dim": cfg.diffusion.dim,
                "hidden": cfg.diffusion.dconv_hidden, "layers": cfg.diffusion.dconv_layers,
                "cond_ch": cfg.diffusion.dconv_cond_ch}, out)
    print("[saved]", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None, help="被测扩散模型(生成负样本)")
    ap.add_argument("--out", type=str, default="parking/cache/critic_n80.pt")
    ap.add_argument("--n-scene", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    train(a.npz, a.ckpt, a.out, a.n_scene, a.steps, a.batch, a.lr, a.quick)
