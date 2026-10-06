# -*- coding: utf-8 -*-
"""泊车占据图 VAE: 复用 diffusion_planner/vae_map.VAE, 在泊车地图上重训 -> 冻结。

延续"VAE 是独立模块, 训好就不动"的理念: 这里只负责
  1) 用泊车数据集的占据通道(ch0)重训一次 VAE;
  2) 打印量化指标(像素准确率 / 障碍 IoU);
  3) 存权重, 并提供 load_frozen_vae() 供扩散条件分支(M5)冻结调用。

vae_map.VAE 的编解码用 stride-2 卷积, 对尺寸无硬编码: 72x128 -> latent 9x16xLAT_CH。
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 复用现有 vae_map(位于仓库根 diffusion_planner/ 下)
_PKG = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_PKG)
_DP = os.path.join(_REPO, "diffusion_planner")
if _DP not in sys.path:
    sys.path.insert(0, _DP)
import vae_map as VM                                       # noqa: E402

from .config import ParkingConfig, default_config          # noqa: E402
from .dataset import load_dataset                           # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(_PKG, "cache", "vae_parking.pt")


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def get_occ_tensors(npz_path, n_viz=8):
    """从泊车 npz 取占据通道, 按 split 分 train/eval, 再切一小份 viz。"""
    d = load_dataset(npz_path)
    occ = d["maps"][:, 0:1].astype(np.float32)              # (M,1,H,W)
    split = d["split"]
    tr = occ[split == 0]
    ev = occ[split == 1]
    if ev.shape[0] < 4:                                     # eval 太少则从 train 尾借
        ev = np.concatenate([ev, tr[-64:]], axis=0)
    viz = ev[:n_viz]
    return (torch.tensor(tr, device=device),
            torch.tensor(ev, device=device),
            torch.tensor(viz, device=device))


# --------------------------------------------------------------------------- #
# 训练(AMP)
# --------------------------------------------------------------------------- #
def train_vae(model, tr, epochs, batch, lr, beta, pos_w, use_amp=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n = tr.shape[0]
    for ep in tqdm(range(epochs), desc="train parking VAE", ncols=90):
        perm = torch.randperm(n, device=device)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            xb = tr[perm[i:i + batch]]
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon, mu, logvar = model(xb)
                loss, rec, kl = VM.vae_loss(recon.float(), xb, mu.float(), logvar.float(),
                                            pos_w=pos_w, beta=beta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1
        if (ep + 1) % max(1, epochs // 10) == 0:
            tqdm.write("  epoch %3d/%d  loss=%.4f" % (ep + 1, epochs, tot / max(1, nb)))
    return model


# --------------------------------------------------------------------------- #
# 冻结加载(供 M5 条件分支)
# --------------------------------------------------------------------------- #
def load_frozen_vae(ckpt=CKPT, lat_ch=None):
    lat_ch = lat_ch or VM.LAT_CH
    model = VM.VAE(lat_ch).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def visualize(model, viz, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = VM.reconstruct(model, viz)
    k = viz.shape[0]
    fig, axes = plt.subplots(2, k, figsize=(2 * k, 4.2))
    if k == 1:
        axes = axes[:, None]
    for j in range(k):
        axes[0, j].imshow(viz[j, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_title("orig", fontsize=8)
        axes[1, j].imshow(rec[j, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].set_title("recon", fontsize=8)
        for a in (axes[0, j], axes[1, j]):
            a.axis("off")
    fig.suptitle("Parking occupancy VAE  top=orig bottom=recon")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main(npz_path=None, epochs=None, quick=False):
    cfg = default_config()
    dc = cfg.diffusion
    if npz_path is None:
        cdir = os.path.join(_REPO, cfg.data.cache_dir)
        cands = sorted([f for f in os.listdir(cdir)
                        if f.startswith("parking_") and f.endswith(".npz")],
                       key=lambda s: int(s.split("_")[1].split(".")[0]), reverse=True)
        if not cands:
            raise SystemExit("[error] no parking_*.npz; run `python -m parking.dataset` first.")
        npz_path = os.path.join(cdir, cands[0])
    epochs = epochs or (6 if quick else dc.vae_epochs)
    tr, ev, viz = get_occ_tensors(npz_path)
    print("[info] device=%s  npz=%s  train=%s eval=%s" % (device, npz_path,
                                                          tuple(tr.shape), tuple(ev.shape)))
    model = VM.VAE(dc.lat_ch).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print("[model] params=%.1fK latent=%dx%dx%d" % (npar / 1e3, tr.shape[2] // 8,
                                                    tr.shape[3] // 8, dc.lat_ch))
    train_vae(model, tr, epochs=epochs, batch=128, lr=dc.lr, beta=dc.vae_beta,
              pos_w=dc.vae_pos_w)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    acc, iou = VM.metrics(model, ev)
    print("[metrics] pixel_acc=%.2f%%  obstacle_IoU=%.2f%%  (eval %d)"
          % (acc * 100, iou * 100, ev.shape[0]))
    fig = visualize(model, viz, os.path.join(_REPO, "figs", "parking", "m4_vae_recon.png"))
    print("[saved] %s\n[saved] %s" % (CKPT, fig))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.npz, a.epochs, a.quick)
