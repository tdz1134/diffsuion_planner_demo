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


@torch.no_grad()
def sample(model, enc, maps, starts4, goals4, sch, z=None, pin=True):
    """反向去噪采样。每步钉住首尾位姿(含朝向)。返回 (B,N,4) 归一化轨迹。"""
    B = maps.shape[0]
    model.eval(); enc.eval()
    map_emb = enc(maps, z=z) if z is not None else enc(maps)
    T = sch["T"]
    x = torch.randn(B, model.n_wp, model.dim, device=maps.device)
    if pin:
        x[:, 0] = starts4; x[:, -1] = goals4
    for t in reversed(range(T)):
        tt = torch.full((B,), t, device=maps.device, dtype=torch.long)
        eps = model(x, tt, map_emb, starts4, goals4)
        mean = (x - sch["betas"][t] / torch.sqrt(1.0 - sch["abar"][t]) * eps) \
            / torch.sqrt(sch["alphas"][t])
        if t > 0:
            x = mean + torch.sqrt(sch["betas"][t]) * torch.randn_like(x)
        else:
            x = mean
        if pin:
            x[:, 0] = starts4; x[:, -1] = goals4
    return x
