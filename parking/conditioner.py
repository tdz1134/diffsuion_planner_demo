# -*- coding: utf-8 -*-
"""地图条件分支, 实现 MapConditionerABC(与 diffusion_path.MapConditioner 同签名)。

主分支(lat): 冻结泊车 VAE 的占据 latent (B,LAT_CH,H/8,W/8) -> Linear -> 32
            ⊕ 显式 SDF 通道小 CNN -> 32   => 64 维 map_emb。
基线(cnn)  : 原始 2 通道(占据+SDF) CNN -> 64, 供 A/B 对比(对应 diffusion_path --cond cnn)。

forward(m, z=None): m=(B,2,H,W) [占据, SDF]; z 为预存 latent(训练热循环传入, 免重跑 VAE)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import MapConditionerABC


class LatentEnc(nn.Module):
    def __init__(self, in_dim, out):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out), nn.ReLU())

    def forward(self, z):
        return self.fc(z.reshape(z.shape[0], -1))


class SdfEnc(nn.Module):
    """SDF 通道 -> 向量。AdaptiveAvgPool 使其对任意 H,W 适用。"""

    def __init__(self, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, out), nn.ReLU(),
        )

    def forward(self, sdf):
        return self.net(sdf)


class MapConditioner(nn.Module, MapConditionerABC):
    """冻结 VAE latent (+可选 SDF) -> map_emb。

    use_sdf=True(默认): VAE latent→out/2 ⊕ SDF→out/2 = out(向后兼容旧 ckpt)。
    use_sdf=False(vae-only): 只用 VAE latent → out。
    """

    def __init__(self, vae, out_dim=64, lat_hw=(9, 16), use_sdf=True):
        super().__init__()
        self.vae = vae                                   # 冻结子模块
        self.use_sdf = use_sdf
        lat_dim = vae.enc.mu.out_channels * int(lat_hw[0]) * int(lat_hw[1])
        lat_out = out_dim // 2 if use_sdf else out_dim
        self.lat_enc = LatentEnc(lat_dim, lat_out)
        if use_sdf:
            self.sdf_enc = SdfEnc(out_dim // 2)

    def forward(self, m, z=None):
        if z is None:
            with torch.no_grad():
                z = self.vae.encode(m[:, 0:1])
        if not self.use_sdf:
            return self.lat_enc(z)
        sdf = m[:, 1:2]
        return torch.cat([self.lat_enc(z), self.sdf_enc(sdf)], dim=1)


class SpatialMapConditioner(nn.Module, MapConditionerABC):
    """M19: 复用冻结 VAE 的**空间 latent 栅格** (B, lat_ch, gh, gw) 直接作 cross-attn 的 key,
    不再 flatten 成全局向量(那会销毁"航点<->地图格子"的空间对应)。

    每个栅格 -> 1 个 token(通道 = lat_ch, 可选再拼该格局部 SDF 间隙 +1), 加 2D 可学位置编码。
    输出 (B, gh*gw, out_ch) 供 CondDenoiserTransformer3 的每层 cross-attention 读取。
    forward(m, z=None): m=(B,2,H,W)[占据,SDF]; z 为预存 latent(热循环传入, 免重跑 VAE)。
    """

    def __init__(self, vae, lat_hw=(9, 16), lat_ch=8, out_ch=64, use_sdf=True):
        super().__init__()
        self.vae = vae                                   # 冻结(仅当未预存 z 时现算)
        self.gh, self.gw = int(lat_hw[0]), int(lat_hw[1])
        self.use_sdf = use_sdf
        in_ch = lat_ch + (1 if use_sdf else 0)
        self.proj = nn.Linear(in_ch, out_ch)
        self.pos = nn.Parameter(torch.zeros(1, self.gh * self.gw, out_ch))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, m, z=None):
        if z is None:
            with torch.no_grad():
                z = self.vae.encode(m[:, 0:1])                        # (B, lat_ch, gh, gw)
        B = z.shape[0]
        tok = z.permute(0, 2, 3, 1).reshape(B, self.gh * self.gw, -1)  # (B, gh*gw, lat_ch)
        if self.use_sdf:
            sdf = F.adaptive_avg_pool2d(m[:, 1:2], (self.gh, self.gw))  # (B,1,gh,gw)
            sdf = sdf.permute(0, 2, 3, 1).reshape(B, self.gh * self.gw, 1)
            tok = torch.cat([tok, sdf], dim=-1)                         # (B, gh*gw, lat_ch+1)
        return self.proj(tok) + self.pos                                # (B, gh*gw, out_ch)


class MapEncoderCNN(nn.Module, MapConditionerABC):
    """基线: 2 通道 CNN 直接编码(无 VAE), 供 --cond cnn 对比。"""

    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim), nn.ReLU(),
        )

    def forward(self, m, z=None):
        return self.net(m)
