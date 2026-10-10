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

    def __init__(self, n_wp, dim=4, temb=64, map_emb=64, hidden=512, pose_dim=None):
        super().__init__()
        self.n_wp = n_wp
        self.dim = dim
        self.pose_dim = dim if pose_dim is None else pose_dim
        self.temb = temb
        in_dim = n_wp * dim + temb + map_emb + 2 * self.pose_dim
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
                 hidden=128, layers=5, cond_ch=32, pose_dim=None):
        super().__init__()
        self.n_wp, self.dim, self.temb = n_wp, dim, temb
        self.pose_dim = dim if pose_dim is None else pose_dim
        self.cond = nn.Sequential(
            nn.Linear(temb + map_emb + 2 * self.pose_dim, hidden), nn.ReLU(),
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


class CondDenoiserTransformer(nn.Module):
    """Transformer 去噪器(Phase 5): 把 N 个航点当序列做**全局时序自注意力**。

    与 CondDenoiser/Conv 接口一致: (x[B,N,dim], t, map_emb, start, goal) -> eps[B,N,dim]。
    条件(时间/地图/起终点)压成**一个前缀 token** 拼在序列头部(类 class-token), N 个航点 token
    过 Encoder 后取回各自位置输出。相对 conv(局部空洞感受野)能直接建模跨全程的依赖。
    """

    def __init__(self, n_wp, dim=4, temb=64, map_emb=64,
                 model=128, heads=4, layers=4, ff=256, dropout=0.1, pose_dim=None):
        super().__init__()
        self.n_wp, self.dim, self.temb = n_wp, dim, temb
        self.pose_dim = dim if pose_dim is None else pose_dim
        self.proj = nn.Linear(dim, model)
        self.pos = nn.Parameter(torch.zeros(1, n_wp, model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.condtok = nn.Linear(temb + map_emb + 2 * self.pose_dim, model)   # 1 个条件 token
        self.pre = nn.LayerNorm(model)
        layer = nn.TransformerEncoderLayer(model, heads, ff, dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(model)
        self.head = nn.Linear(model, dim)

    def forward(self, x, t, map_emb, start, goal):
        B, N, _ = x.shape
        h = self.proj(x) + self.pos                                    # (B,N,model)
        c = self.condtok(torch.cat([time_embedding(t, self.temb, x.device),
                                    map_emb, start, goal], dim=1))     # (B,model)
        seq = self.pre(torch.cat([c[:, None, :], h], dim=1))           # (B,N+1,model)
        o = self.norm(self.enc(seq))[:, 1:]                            # 去掉条件 token
        return self.head(o)                                            # (B,N,dim)


class _CondBlock(nn.Module):
    """DiT 风格条件块: 航点自注意力 + 对条件 memory 的 cross-attention + FFN,
    时间步/条件通过 AdaLN-Zero(shift/scale/gate)注入(与 Diffusion Policy-T / Diffusion Planner 一致)。"""

    def __init__(self, model, heads, ff, dropout):
        super().__init__()
        self.n_self = nn.LayerNorm(model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(model, heads, dropout=dropout, batch_first=True)
        self.n_cross = nn.LayerNorm(model, elementwise_affine=True)
        self.cross_attn = nn.MultiheadAttention(model, heads, dropout=dropout, batch_first=True)
        self.n_ff = nn.LayerNorm(model, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(model, ff), nn.GELU(), nn.Linear(ff, model))
        self.ada = nn.Linear(model, 6 * model)          # 由共享条件嵌入→ 3子层×(shift,scale,gate)

    def forward(self, h, mem, mod):
        # h:(B,N,model); mod:(B,model) → 逐块调制参数需 reshape 成 (B,1,model) 才能在 N 个航点上广播
        sm1, sc1, g1, sm2, sc2, g2 = self.ada(mod).chunk(6, dim=-1)
        sm1, sc1, g1, sm2, sc2, g2 = [v[:, None, :] for v in (sm1, sc1, g1, sm2, sc2, g2)]
        hs = self.n_self(h) * (1 + sc1) + sm1
        h = h + g1 * self.self_attn(hs, hs, hs, need_weights=False)[0]      # 航点间自注意力
        h = h + self.cross_attn(self.n_cross(h), mem, mem, need_weights=False)[0]  # 读地图/起终点条件
        hf = self.n_ff(h) * (1 + sc2) + sm2
        h = h + g2 * self.ff(hf)
        return h


class CondDenoiserTransformer2(nn.Module):
    """忠实版条件 Transformer 去噪器(M14): 每层 cross-attention 读条件 memory + AdaLN-Zero 注入时间步。

    与 trans(单前缀 token)的弱点相比, 这是 Diffusion Policy-T / Diffusion Planner(ICLR25)采用、能真正
    建模多模态轨迹的写法, 作为对 Transformer 的公平重测。接口不变: (x,t,map_emb,start,goal)->eps(B,N,dim)。
    """

    def __init__(self, n_wp, dim=4, temb=64, map_emb=64, model=128, heads=4,
                 layers=4, ff=256, dropout=0.1, cond_tokens=8, pose_dim=None):
        super().__init__()
        self.n_wp, self.dim, self.temb = n_wp, dim, temb
        self.pose_dim = dim if pose_dim is None else pose_dim
        self.proj = nn.Linear(dim, model)
        self.pos = nn.Parameter(torch.zeros(1, n_wp, model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        cond_dim = map_emb + 2 * self.pose_dim             # map_emb ⊕ start ⊕ goal(pose_dim) → cross-attn memory
        self.mem = nn.Sequential(nn.Linear(cond_dim, model), nn.GELU(),
                                 nn.Linear(model, cond_tokens * model))
        self.cond_tokens = cond_tokens
        self.tmod = nn.Sequential(nn.Linear(temb, model), nn.SiLU(), nn.Linear(model, model))
        self.blocks = nn.ModuleList([_CondBlock(model, heads, ff, dropout) for _ in range(layers)])
        self.final = nn.LayerNorm(model, elementwise_affine=False)
        self.head = nn.Linear(model, dim)
        # AdaLN-Zero: gate 初始化 0, 训练更稳(DiT)
        for b in self.blocks:
            nn.init.zeros_(b.ada.weight); nn.init.zeros_(b.ada.bias)

    def forward(self, x, t, map_emb, start, goal):
        B, N, _ = x.shape
        mod = self.tmod(time_embedding(t, self.temb, x.device))          # (B,model) 时间步调制源
        mem = self.mem(torch.cat([map_emb, start, goal], dim=1))          # (B,cond_tokens*model)
        mem = mem.view(B, self.cond_tokens, -1)                            # (B,cond_tokens,model)
        h = self.proj(x) + self.pos                                       # (B,N,model)
        for b in self.blocks:
            h = b(h, mem, mod)
        return self.head(self.final(h))                                   # (B,N,dim)


def build_denoiser(dc):
    """按 config.DiffusionConfig.denoiser 选架构('mlp'/'conv'/'trans'/'trans2'), 供 train 与 evaluate 共用。

    去耦两个维度: 状态输出通道 dim = SE(2)的4 (+1 若 use_gear 拼接档位); 条件里的 start/goal 始终用 pose_dim=4。
    默认 use_gear=False → dim==pose_dim==4, 与旧 ckpt/管道完全一致。
    """
    kind = getattr(dc, "denoiser", "mlp")
    pose_dim = int(dc.dim)
    sd = pose_dim + (1 if getattr(dc, "use_gear", False) else 0)   # state 输出通道数
    if kind == "conv":
        m = CondDenoiserConv(dc.n_wp, dim=sd, temb=dc.temb, map_emb=dc.map_emb,
                             hidden=dc.dconv_hidden, layers=dc.dconv_layers,
                             cond_ch=dc.dconv_cond_ch, pose_dim=pose_dim)
    elif kind == "trans":
        m = CondDenoiserTransformer(dc.n_wp, dim=sd, temb=dc.temb, map_emb=dc.map_emb,
                                    model=dc.dtrans_model, heads=dc.dtrans_heads,
                                    layers=dc.dtrans_layers, ff=dc.dtrans_ff,
                                    dropout=dc.dtrans_dropout, pose_dim=pose_dim)
    elif kind == "trans2":
        m = CondDenoiserTransformer2(dc.n_wp, dim=sd, temb=dc.temb, map_emb=dc.map_emb,
                                     model=dc.dtrans_model, heads=dc.dtrans_heads,
                                     layers=dc.dtrans_layers, ff=dc.dtrans_ff,
                                     dropout=dc.dtrans_dropout,
                                     cond_tokens=dc.dtrans_cond_tokens, pose_dim=pose_dim)
    else:
        m = CondDenoiser(dc.n_wp, dim=sd, temb=dc.temb, map_emb=dc.map_emb,
                         hidden=dc.hidden, pose_dim=pose_dim)
    m.pred_mode = getattr(dc, "pred_mode", "eps")   # M16: 采样时据此选 ε-递推还是 x0-后验
    return m


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
    pred_x0_mode = getattr(model, "pred_mode", "eps") == "x0"    # M16: 网络直接输出 x0_hat
    if critic is not None:
        critic.eval()
    with torch.no_grad():
        map_emb = enc(maps, z=z) if z is not None else enc(maps)
    T = sch["T"]
    x = torch.randn(B, model.n_wp, model.dim, device=maps.device)
    pdim = starts4.shape[-1]                              # pose 通道数(4); use_gear 时 x 多一列档位不钉
    if pin:
        x[:, 0, :pdim] = starts4; x[:, -1, :pdim] = goals4
    if guide is not None:
        sdf_m = maps[:, 1:2] * guide["sdf_clip"]
        inv_rmin = 1.0 / guide["veh"].r_min
        min_abar = guide.get("min_abar", 0.1)
    for t in reversed(range(T)):
        tt = torch.full((B,), t, device=maps.device, dtype=torch.long)
        with torch.no_grad():
            out = model(x, tt, map_emb, starts4, goals4)
            if pred_x0_mode:
                # out = x0_hat。DDPM 后验均值 mean = coef_x0*x0hat + coef_xt*xt(无 1/√ᾱ 放大)。
                ab = sch["abar"][t]
                ab_prev = sch["abar"][t - 1] if t > 0 else torch.ones_like(ab)
                coef_x0 = torch.sqrt(ab_prev) * sch["betas"][t] / (1.0 - ab)
                coef_xt = torch.sqrt(sch["alphas"][t]) * (1.0 - ab_prev) / (1.0 - ab)
                mean = coef_x0 * out + coef_xt * x
                eps = (x - torch.sqrt(ab) * out) / torch.sqrt(1.0 - ab)  # 仅供 guide 反推(默认关不用)
            else:
                eps = out
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
            x[:, 0, :pdim] = starts4; x[:, -1, :pdim] = goals4
    return x
