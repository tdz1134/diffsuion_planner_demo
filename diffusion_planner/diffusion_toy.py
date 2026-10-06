# -*- coding: utf-8 -*-
"""
简易版扩散模型 (DDPM) 的 Python 复现
====================================
对应"用 MATLAB 简易实现 diffusion、加噪声 / 去噪声"的那个经典玩具例子。

核心只有两件事:
  1) 前向过程 (forward / 加噪):  按噪声调度 beta_t, 一步步往数据里加高斯噪声,
     直到 T 步后变成纯噪声。用重参数化技巧可以直接跳到任意步:
         x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps ,  eps ~ N(0, I)
  2) 反向过程 (reverse / 去噪):  训练一个小网络 eps_theta(x_t, t) 预测所加的噪声,
     采样时从纯噪声 x_T 出发, 逐步去噪还原出 x_0。

本文件在 2D 玩具数据("two moons")上演示, 便于可视化。
把 x 看成平面上的 (x, y) 航点, 这套流程就是最简单的 "diffusion planner" 骨架:
学一个能生成合理轨迹点分布的去噪模型。

运行:
    source ../venv_py38/bin/activate
    python diffusion_toy.py
输出:
    fig_forward.png     前向加噪过程可视化
    fig_generated.png   真实数据 vs 模型生成数据
    fig_denoise.gif     反向去噪采样动画(若安装了 pillow)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")            # 无显示环境也能保存图片
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 0. 全局配置
# --------------------------------------------------------------------------- #
SEED        = 0
T_STEPS     = 200          # 扩散总步数(玩具例子几百步就够, 图像才用 1000)
BETA_START  = 1e-4         # 线性噪声调度起点
BETA_END    = 0.02         # 线性噪声调度终点
DATA_DIM    = 2            # 2D 玩具数据
N_SAMPLES   = 4000         # 训练样本数
BATCH       = 256
N_STEPS     = 8000         # 训练梯度步数(玩具例子几千步即可收敛)
LR          = 1e-3
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
FIG_DIR     = os.path.join(OUT_DIR, "figs", "diffusion_toy")   # 效果图统一放 figs/<脚本名>/

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] 使用设备: {device}"
      + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))


# --------------------------------------------------------------------------- #
# 1. 玩具数据: two moons(两个互扣的半圆)
# --------------------------------------------------------------------------- #
def make_two_moons(n=N_SAMPLES):
    """生成 2D 'two moons' 点云, 作为我们要让模型学会生成的目标分布。"""
    half = n // 2
    # 上弦
    t1 = np.linspace(0, np.pi, half)
    a = np.stack([np.cos(t1), np.sin(t1)], axis=1) + np.array([0.5, 0.25])
    # 下弦(旋转 180 度)
    t2 = np.linspace(0, np.pi, n - half)
    b = np.stack([-np.cos(t2), -np.sin(t2)], axis=1) + np.array([-0.5, -0.25])
    X = np.concatenate([a, b], axis=0)
    X += np.random.randn(*X.shape) * 0.08      # 加一点抖动
    X = (X - X.mean(0)) / X.std(0)             # 标准化, 方便扩散
    return X.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. 噪声调度 & 前向加噪
# --------------------------------------------------------------------------- #
def build_schedule(T, b0=BETA_START, b1=BETA_END):
    betas = torch.linspace(b0, b1, T, device=device)          # beta_t
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)                        # abar_t = prod(alpha_i)
    return betas, alphas, abar


BETAS, ALPHAS, ABAR = build_schedule(T_STEPS)


def q_sample(x0, t, noise=None):
    """前向: 由 x0 直接采出 x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*noise."""
    if noise is None:
        noise = torch.randn_like(x0)
    s_abar = torch.sqrt(ABAR[t])[:, None]
    s_1_mabar = torch.sqrt(1.0 - ABAR[t])[:, None]
    return s_abar * x0 + s_1_mabar * noise, noise


# --------------------------------------------------------------------------- #
# 3. 去噪网络: 带时间嵌入的小 MLP
# --------------------------------------------------------------------------- #
def time_embedding(t, dim=64):
    """正弦位置编码, 把离散步 t 变成一个平滑向量。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class DenoiseMLP(nn.Module):
    """输入 (x_t, t) -> 预测噪声 eps。这就是"去噪声"的核心模型。"""

    def __init__(self, in_dim=DATA_DIM, hidden=256, temb_dim=64):
        super().__init__()
        self.temb_dim = temb_dim
        self.fc1 = nn.Linear(in_dim + temb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, in_dim)

    def forward(self, x, t):
        temb = time_embedding(t, self.temb_dim)
        h = torch.cat([x, temb], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)          # 预测的噪声 eps_theta


# --------------------------------------------------------------------------- #
# 4. 训练(学习"去噪")
# --------------------------------------------------------------------------- #
def train(model, x0_tensor, steps=N_STEPS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = x0_tensor.shape[0]
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (BATCH,), device=device)
        x0 = x0_tensor[idx]
        t = torch.randint(0, T_STEPS, (BATCH,), device=device)
        xt, noise = q_sample(x0, t)
        pred_noise = model(xt, t)
        loss = nn.functional.mse_loss(pred_noise, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1:5d}/{steps}  loss={loss.item():.5f}")
    return model


# --------------------------------------------------------------------------- #
# 5. 反向采样(从纯噪声一步步去噪生成)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model, num=2000, record=False):
    """DDPM 祖先采样: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t)*eps) + sigma*z"""
    x = torch.randn(num, DATA_DIM, device=device)     # 从 x_T ~ N(0,I) 开始
    traj = [x.cpu().numpy()] if record else None
    for t in reversed(range(T_STEPS)):
        tt = torch.full((num,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        alpha_t = ALPHAS[t]
        abar_t = ABAR[t]
        mean = (x - beta_t_coeff(eps, t)) / torch.sqrt(alpha_t)
        if t > 0:
            sigma = torch.sqrt(BETAS[t])
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
        if record:
            traj.append(x.cpu().numpy())
    return (x.cpu().numpy(), traj) if record else x.cpu().numpy()


def beta_t_coeff(eps, t):
    """beta_t / sqrt(1-abar_t) * eps 这一项。"""
    coef = BETAS[t] / torch.sqrt(1.0 - ABAR[t])
    return coef * eps


# --------------------------------------------------------------------------- #
# 6. 可视化
# --------------------------------------------------------------------------- #
def plot_forward(x0_sample, path):
    """把一张干净样本逐步加噪, 展示'加噪声'过程。"""
    ts = [0, 20, 50, 100, 150, T_STEPS - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3))
    for ax, t in zip(axes, ts):
        tvec = torch.full((x0_sample.shape[0],), t, device=device, dtype=torch.long)
        xt, _ = q_sample(x0_sample, tvec)
        ax.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=3, alpha=0.6)
        ax.set_title(f"t = {t}")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    fig.suptitle("Forward process: add noise (data -> pure noise)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_generated(X_real, X_gen, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(X_real[:, 0], X_real[:, 1], s=3, alpha=0.5, c="tab:blue")
    axes[0].set_title("Real data")
    axes[1].scatter(X_gen[:, 0], X_gen[:, 1], s=3, alpha=0.5, c="tab:red")
    axes[1].set_title("Generated by diffusion (reverse denoising)")
    for ax in axes:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[saved] {path}")


def save_gif(traj, X_real, path):
    """反向'去噪声'采样动画: 从纯噪声收敛到数据分布。"""
    try:
        import matplotlib.animation as animation
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X_real[:, 0], X_real[:, 1], s=2, alpha=0.25, c="gray")
    sc = ax.scatter([], [], s=3, c="tab:red")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    title = ax.set_title("")

    def update(i):
        pts = traj[i]
        sc.set_offsets(pts[:, :2])
        title.set_text(f"Denoising (remaining t = {T_STEPS - i})")
        return sc, title

    anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=60, blit=False)
    try:
        anim.save(path, writer="pillow", fps=25)
        plt.close(fig)
        print(f"[saved] {path}")
    except Exception as e:
        plt.close(fig)
        print(f"[skip gif] {e}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    X = make_two_moons()
    x0 = torch.tensor(X, device=device)

    model = DenoiseMLP().to(device)
    print("[train] 开始训练去噪网络 ...")
    train(model, x0, steps=N_STEPS)

    print("[sample] 反向采样生成数据 ...")
    X_gen, traj = sample(model, num=2000, record=True)

    # 可视化
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_forward(x0[:1000], os.path.join(FIG_DIR, "fig_forward.png"))
    plot_generated(X, X_gen, os.path.join(FIG_DIR, "fig_generated.png"))
    save_gif(traj, X, os.path.join(FIG_DIR, "fig_denoise.gif"))
    print("[done] 全部完成。")


if __name__ == "__main__":
    main()
