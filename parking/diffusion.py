# -*- coding: utf-8 -*-
"""SE(2) 轨迹条件 DDPM: 对定长 N 的 (x,y,cos,sin) 做扩散(ε-prediction)。

与 diffusion_path 的调度一致(线性 beta), 但:
  - 轨迹维度 = 4 (x,y,cos,sin), 而非 2; cos/sin 表示避免角度环绕。
  - 采样时每步**钉住首尾位姿**(含朝向), 与训练数据"端点对齐"一致。
  - 输入/输出均为**归一化**坐标(x,y∈[-1,1]); cos/sin 天然 ∈[-1,1]。
"""

import numpy as np
import torch
import torch.nn as nn

from .penalty import heading_consistency, curvature_pen, footprint_collision


def make_schedule(t_steps, b0, b1, device):
    betas = torch.linspace(b0, b1, t_steps, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return dict(betas=betas, alphas=alphas, abar=abar, T=t_steps)


def q_sample(x0, t, sch, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sa = torch.sqrt(sch["abar"][t])[:, None, None]
    sb = torch.sqrt(1.0 - sch["abar"][t])[:, None, None]
    return sa * x0 + sb * noise, noise


def pred_x0(xt, eps, sch, t):
    """由 x_t 与预测噪声 eps 反推 x0_hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar)。"""
    sa = torch.sqrt(sch["abar"][t])[:, None, None]
    sb = torch.sqrt(1.0 - sch["abar"][t])[:, None, None]
    return (xt - sb * eps) / sa


def time_embedding(t, dim, device):
    half = dim // 2
    f = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
    a = t[:, None].float() * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class CondDenoiser(nn.Module):
    """输入 (x_t[N*4], t, map_emb, start4, goal4) -> 预测噪声 (N,4)。"""

    def __init__(self, n_wp, dim=4, temb=64, map_emb=64, hidden=512):
        super().__init__()
        self.n_wp = n_wp
        self.dim = dim
        self.temb = temb
        in_dim = n_wp * dim + temb + map_emb + dim + dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_wp * dim),
        )

    def forward(self, x, t, map_emb, start, goal):
        h = torch.cat([x.reshape(x.shape[0], -1),
                       time_embedding(t, self.temb, x.device),
                       map_emb, start, goal], dim=1)
        return self.net(h).reshape(x.shape[0], self.n_wp, self.dim)


class _DilatedResBlock(nn.Module):
    """时序残差块: 空洞 Conv1d(膨胀=dil) -> GELU -> 1x1 Conv, 与输入相加。"""

    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.act = nn.GELU()
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, h):
        return h + self.c2(self.act(self.c1(h)))


class CondDenoiserConv(nn.Module):
    """1D 时序卷积去噪器(Phase 3 升级): 对 N 个航点的**序列**做卷积, 用空洞扩张感受野。

    与 CondDenoiser 接口一致: (x[B,N,dim], t, map_emb, start, goal) -> eps[B,N,dim]。
    条件(时间/地图/起终点)经 MLP 后逐位置**广播拼接**; 额外拼接一个归一化位置通道
    以打破卷积的平移对称(轨迹有固定的首/尾)。
    """

    def __init__(self, n_wp, dim=4, temb=64, map_emb=64,
                 hidden=128, layers=5, cond_ch=32):
        super().__init__()
        self.n_wp, self.dim, self.temb = n_wp, dim, temb
        self.cond = nn.Sequential(
            nn.Linear(temb + map_emb + 2 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, cond_ch),
        )
        c_in = dim + cond_ch + 1                       # +1 = 位置通道
        self.stem = nn.Conv1d(c_in, hidden, 1)
        self.body = nn.ModuleList(
            [_DilatedResBlock(hidden, 2 ** (i % 5)) for i in range(layers)])
        self.head = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(),
                                  nn.Conv1d(hidden, dim, 1))
        self.register_buffer("pos", torch.linspace(-1.0, 1.0, n_wp).view(1, n_wp, 1))

    def forward(self, x, t, map_emb, start, goal):
        B, N, _ = x.shape
        cvec = self.cond(torch.cat([time_embedding(t, self.temb, x.device),
                                    map_emb, start, goal], dim=1))      # (B,cond_ch)
        cb = cvec[:, None, :].expand(B, N, cvec.shape[1])               # (B,N,cond_ch)
        pb = self.pos.expand(B, N, 1)
        h = torch.cat([x, cb, pb], dim=-1).transpose(1, 2)             # (B,c_in,N)
        h = self.stem(h)
        for blk in self.body:
            h = blk(h)
        return self.head(h).transpose(1, 2)                            # (B,N,dim)


def build_denoiser(dc):
    """按 config.DiffusionConfig.denoiser 选架构('mlp' 默认/'conv'), 供 train 与 evaluate 共用。"""
    kind = getattr(dc, "denoiser", "mlp")
    if kind == "conv":
        return CondDenoiserConv(dc.n_wp, dim=dc.dim, temb=dc.temb, map_emb=dc.map_emb,
                                hidden=dc.dconv_hidden, layers=dc.dconv_layers,
                                cond_ch=dc.dconv_cond_ch)
    return CondDenoiser(dc.n_wp, dim=dc.dim, temb=dc.temb, map_emb=dc.map_emb,
                        hidden=dc.hidden)


def sample(model, enc, maps, starts4, goals4, sch, z=None, pin=True, guide=None,
           critic=None, critic_scale=0.0, critic_min_abar=0.9):
    """反向去噪采样。每步钉住首尾位姿(含朝向)。返回 (B,N,4) 归一化轨迹。

    guide=None 为纯 DDPM。否则每步在**预测 x0_hat** 上算轨迹代价(曲率/足迹碰撞/航向)的梯度,
    后推至后验均值 mean -= scale * dCost/dx —— 不改变训练与已学 score(sampler-safe 推理引导)。
    guide 需包含: veh, bbox, foot(P,2 tensor), sdf_clip, w_nh/w_curv/w_coll, scale, margin, min_abar。

    critic 为已训好的 FeasibilityCritic 时(Phase 4 / M12): 晚步(abar>=critic_min_abar)在 x0_hat 上
    算 logit=可行度, 沿 **梯度上升**推后验均值 mean += critic_scale * d(logit)/dx(classifier guidance)。
    与 guide 可共存; critic 的梯度只依赖学出的平滑判别器, 避免手写几何代价的病态发散。
    """
    B = maps.shape[0]
    model.eval(); enc.eval()
    if critic is not None:
        critic.eval()
    with torch.no_grad():
        map_emb = enc(maps, z=z) if z is not None else enc(maps)
    T = sch["T"]
    x = torch.randn(B, model.n_wp, model.dim, device=maps.device)
    if pin:
        x[:, 0] = starts4; x[:, -1] = goals4
    if guide is not None:
        sdf_m = maps[:, 1:2] * guide["sdf_clip"]
        inv_rmin = 1.0 / guide["veh"].r_min
        min_abar = guide.get("min_abar", 0.1)
    for t in reversed(range(T)):
        tt = torch.full((B,), t, device=maps.device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, tt, map_emb, starts4, goals4)
            mean = (x - sch["betas"][t] / torch.sqrt(1.0 - sch["abar"][t]) * eps) \
                / torch.sqrt(sch["alphas"][t])
        if guide is not None and float(sch["abar"][t]) >= min_abar:
            xg = x.detach().requires_grad_(True)
            x0h = pred_x0(xg, eps, sch, tt)
            cost = (guide["w_nh"] * heading_consistency(x0h, guide["bbox"])
                    + guide["w_curv"] * curvature_pen(x0h, guide["bbox"], inv_rmin)
                    + guide["w_coll"] * footprint_collision(
                        x0h, sdf_m, guide["bbox"], guide["foot"], guide["margin"]))
            g = torch.autograd.grad(cost.sum(), xg)[0]
            mean = mean - guide["scale"] * g
        if critic is not None and critic_scale != 0.0 and float(sch["abar"][t]) >= critic_min_abar:
            # 噪声感知 critic: 直接看 x_t(与采样器所见同分布), 梯度关于 x_t(标准 classifier guidance)。
            xg = x.detach().requires_grad_(True)
            logit = critic(xg, map_emb, starts4, goals4)
            g = torch.autograd.grad(logit.sum(), xg)[0]
            mean = mean + critic_scale * g
        if t > 0:
            x = mean + torch.sqrt(sch["betas"][t]) * torch.randn_like(x)
        else:
            x = mean
        if pin:
            x[:, 0] = starts4; x[:, -1] = goals4
    return x
