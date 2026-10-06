# -*- coding: utf-8 -*-
"""
占据栅格地图 VAE (编码/解码)
============================
把 64x64 的占据栅格地图(1=障碍黑, 0=空闲白)编码成一个空间隐变量
(8x8x8 特征图), 再解码重建回地图。演示"大图 -> 向量 -> 还原"的自编码器
原理, 也是 Latent Diffusion(Stable Diffusion 那类)里的 VAE 组件。

结构: 空间 latent + 残差块编解码器; 损失 = 加权 BCE + KL(逐样本求和,
量级一致以免后验塌缩)。通道数由 CH 控制, 默认已调小以省参数。
复用已有 dataset_*.npz, 不重新生成数据。

运行:
    ../venv_py38/bin/python vae_map.py        # 用绝对路径调 python, 别 activate
产物:
    vae_map.pt                训练好的 VAE 权重
    figs/vae_map/fig_vae.png  原图 vs 重建图 对比
    终端打印 像素准确率 / 障碍 IoU(量化指标)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 配置(都偏小, 省空间)
# --------------------------------------------------------------------------- #
SEED     = 0
GRID     = 64
LAT_CH   = 8           # 空间隐变量通道数 (latent 为 8x8xLAT_CH)
CH       = (16, 32, 48, 64)   # 编解码器逐级通道数(调小以省参数)
N_TRAIN  = 10000       # 训练用地图数(取数据集子集)
N_TEST   = 8           # 可视化对比张数
N_EVAL   = 200         # 量化指标(IoU/像素准确率)评估用地图数
BATCH    = 64
EPOCHS   = 60
LR       = 2e-4
BETA     = 1e-2        # KL 权重(调小以减轻发糊)
POS_W    = 3.0         # 加权 BCE 中正类(障碍)的权重
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(OUT_DIR, "figs", "vae_map")   # 效果图统一放 figs/<脚本名>/
CKPT     = os.path.join(OUT_DIR, "vae_map.pt")
FIG      = os.path.join(FIG_DIR, "fig_vae.png")

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] device = {device}")


def find_dataset():
    """优先用大的 dataset_*.npz。"""
    cands = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("dataset_") and f.endswith(".npz")],
                   key=lambda s: int(s.split("_")[1].split(".")[0]), reverse=True)
    if not cands:
        raise SystemExit("[错误] 未找到 dataset_*.npz, 请先运行 make_dataset.py 生成数据。")
    return os.path.join(OUT_DIR, cands[0])


def load_maps(path, n_train=N_TRAIN, n_test=N_TEST, n_eval=N_EVAL):
    """载入占据通道 (channel 0), 返回 (train, test, eval) 三个 float tensor (N,1,H,W)。
    test 用于可视化, eval 用于量化指标, 与 train 互不重叠。"""
    z = np.load(path)
    occ = z["maps"][:, 0:1].astype(np.float32)      # (M,1,64,64), 1=障碍
    rng = np.random.default_rng(SEED)
    rng.shuffle(occ)
    tr = torch.tensor(occ[:n_train], device=device)
    te = torch.tensor(occ[n_train:n_train + n_test], device=device)
    ev = torch.tensor(occ[n_train + n_test:n_train + n_test + n_eval], device=device)
    print(f"  数据集 {os.path.basename(path)}  训练 {tuple(tr.shape)}  可视化 {tuple(te.shape)}  评估 {tuple(ev.shape)}")
    return tr, te, ev


# --------------------------------------------------------------------------- #
# VAE 结构(空间 latent + 残差块)
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """GroupNorm + SiLU 残差块, 通道变化用 1x1 投影。"""

    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.act(self.n2(h)))
        return self.act(h + self.skip(x))


class Encoder(nn.Module):
    """64->32->16->8 逐级下采样 + 残差块; 输出空间 mu/logvar (8x8xLAT_CH)。"""

    def __init__(self, lat_ch=LAT_CH, ch=CH):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            ResBlock(c1, c1), ResBlock(c1, c1),                       # @64
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),               # ->32
            ResBlock(c2, c2), ResBlock(c2, c2),                       # @32
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),               # ->16
            ResBlock(c3, c3),                                         # @16
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),               # ->8
            ResBlock(c4, c4),                                         # @8
        )
        self.mu = nn.Conv2d(c4, lat_ch, 3, padding=1)
        self.lv = nn.Conv2d(c4, lat_ch, 3, padding=1)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.lv(h)


class Decoder(nn.Module):
    """8->16->32->64 逐级上采样 + 残差块, 仅以空间 latent 为输入(无 skip)。"""

    def __init__(self, lat_ch=LAT_CH, ch=CH):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.net = nn.Sequential(
            nn.Conv2d(lat_ch, c4, 3, padding=1),
            ResBlock(c4, c4),                                         # @8
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1),       # ->16
            ResBlock(c3, c3),                                         # @16
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),       # ->32
            ResBlock(c2, c2),                                         # @32
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),       # ->64
            ResBlock(c1, c1),                                         # @64
            nn.Conv2d(c1, 1, 3, padding=1),                          # logits
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, lat_ch=LAT_CH):
        super().__init__()
        self.enc = Encoder(lat_ch)
        self.dec = Decoder(lat_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def encode(self, x):
        return self.enc(x)[0]


def vae_loss(recon, x, mu, logvar, pos_w=POS_W, beta=BETA):
    """加权 BCE(逐样本求和) + 空间 latent 的 KL(逐样本求和)。两者量级一致避免后验塌缩。"""
    pw = torch.full((1,), pos_w, device=x.device)
    recon_bce = F.binary_cross_entropy_with_logits(
        recon, x, pos_weight=pw, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_bce + beta * kl, recon_bce, kl


def train(model, tr):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = tr.shape[0]
    for ep in tqdm(range(EPOCHS), desc="训练VAE", ncols=90):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, BATCH):
            xb = tr[perm[i:i + BATCH]]
            recon, mu, logvar = model(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        tqdm.write(f"  epoch {ep+1:2d}/{EPOCHS}  loss={tot/max(1,n//BATCH):.3f}")
    return model


@torch.no_grad()
def reconstruct(model, x):
    mu = model.encode(x)
    return torch.sigmoid(model.dec(mu))   # 用均值(确定性)重建


@torch.no_grad()
def metrics(model, x, batch=256):
    """在评估集上算 像素准确率 与 障碍类 IoU(逐样本平均)。"""
    acc_sum = iou_sum = 0.0
    cnt = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        pred = (reconstruct(model, xb) > 0.5).float()
        acc_sum += (pred == xb).float().mean().item() * xb.shape[0]
        p = pred.squeeze(1).bool()
        t = xb.squeeze(1).bool()
        inter = (p & t).flatten(1).sum(1).float()
        union = (p | t).flatten(1).sum(1).float()
        iou = torch.where(union > 0, inter / union.clamp(min=1.0), torch.ones_like(inter))
        iou_sum += iou.sum().item()
        cnt += xb.shape[0]
    return acc_sum / cnt, iou_sum / cnt


def visualize(model, te):
    rec = reconstruct(model, te)
    fig, axes = plt.subplots(2, N_TEST, figsize=(2 * N_TEST, 4.2))
    for j in range(N_TEST):
        axes[0, j].imshow(te[j, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_title("orig", fontsize=8)
        axes[1, j].imshow(rec[j, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].set_title("recon", fontsize=8)
        for a in (axes[0, j], axes[1, j]):
            a.axis("off")
    fig.suptitle(f"Occupancy-map VAE  (latent=8x8x{LAT_CH})   top: original  bottom: reconstructed")
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    plt.close(fig)
    print(f"[saved] {FIG}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    ds = find_dataset()
    tr, te, ev = load_maps(ds)
    model = VAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 = {n_params/1e3:.1f}K  (latent=8x8x{LAT_CH}, ch={CH})")
    train(model, tr)
    torch.save(model.state_dict(), CKPT)
    sz = os.path.getsize(CKPT) / 1e3
    print(f"[saved] {CKPT}  ({sz:.0f} KB)")
    acc, iou = metrics(model, ev)
    print(f"[metrics] 像素准确率={acc * 100:.2f}%  障碍IoU={iou * 100:.2f}%  (eval {ev.shape[0]} 张)")
    visualize(model, te)
    print("[done]")


if __name__ == "__main__":
    main()
