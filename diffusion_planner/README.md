# Diffusion Planner Demo

在**占据栅格地图**(黑=障碍、白=空闲)上,给定起点/终点,用**条件扩散模型(DDPM)**
生成一条避障路径。专家数据由 A\* / 加权 A\* / 贪心搜索提供,模型学习"加噪→去噪"的
逆过程,从纯噪声直接生成航点路径。附带一个**地图 VAE**(图→隐向量→还原图)演示
Latent-Diffusion 里的编解码组件。

- 运行环境:`venv_py38`(与本目录同级),GPU = **RTX 4060 Laptop 8GB**
- 训练一次完整模型:数据集生成约 24 分钟(一次性)+ 训练约 10 分钟

---

## 0. 新克隆后如何跑起来(clone 快速上手)

> 本仓库**不含**虚拟环境 `venv_py38/` 和数据集 `dataset_*.npz`(见 `.gitignore`),
> 但**含训练好的权重** `vae_map.pt`、`planner_ckpt.pt` 与 `figs/` 效果图。
> 缺数据集时脚本会**自动调用 `make_dataset.py` 生成**,无需手动预处理。

**① 建环境(一次性)** —— Python 3.8 + CUDA 版 PyTorch(RTX 4060 用 cu121):
```bash
conda create -y -n dp python=3.8 && conda activate dp
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib opencv-python tqdm
cd diffusion_planner
```

**② 只想快速看效果**(自动造 1500 条小数据,几十秒~2 分钟,直接用仓库自带的 `vae_map.pt` 作条件):
```bash
python diffusion_path.py --quick
#   产物: figs/diffusion_path/fig_planner.png
```

**③ 完整复现**(先生成 2 万条数据,再训练 VAE 与规划器):
```bash
python make_dataset.py --data 20000                             # 一次性, 约 24 分钟
python vae_map.py                                               # 地图 VAE, 约 1 分钟 → vae_map.pt
python diffusion_path.py --data 20000 --train 20000 --batch 512 --cond lat
#   --cond lat: 加载冻结 vae_map.pt 的 latent + SDF 双分支做地图条件(默认)
#   --cond cnn: 不依赖 VAE 的原始 CNN 基线(便于 A/B 对比)
```
> 无 CUDA / Windows 也能跑,只是慢;把上面的 torch 换成 CPU 版即可。
> 各脚本详细参数见下方 §2~§5。

---

## 1. 目录结构

```
diffusion_planner_demo/
├── venv_py38/                 # Python 虚拟环境(PyTorch 2.4.1+cu121)
└── diffusion_planner/         # 本项目
    ├── grid_env.py            # 地图生成 + A*/加权A*/贪心 + 距离场 + 路径重采样
    ├── make_dataset.py        # 数据集生成(独立、最耗时,带进度条)
    ├── diffusion_path.py      # 条件扩散路径规划器(训练+采样+修复+评估+渲染)
    ├── diffusion_toy.py       # 入门:2D 玩具数据上的最小 DDPM(加噪/去噪演示)
    ├── vae_map.py             # 占据栅格地图 VAE(编码/解码)
    ├── README.md              # 本文件
    ├── dataset_<M>.npz        # 数据集缓存(按样本数命名)
    ├── planner_ckpt.pt        # 扩散规划器权重
    ├── vae_map.pt             # VAE 权重(约 884 KB)
    └── figs/                  # 各脚本效果图(按脚本分子目录)
        ├── diffusion_toy/     #   fig_forward.png, fig_generated.png, fig_denoise.gif
        ├── diffusion_path/    #   fig_planner.png(生成路径 vs A*)
        └── vae_map/           #   fig_vae.png(原图 vs 重建)
```

---

## 2. 图像 / 数据参数(核心)

### 2.1 占据栅格地图
| 参数 | 值 | 说明 |
|------|----|------|
| 尺寸 | **64 × 64** | 高 H = 宽 W |
| 通道 | **2** | 通道0=占据(0/1),通道1=归一化距离场 SDF |
| 占据编码 | **1 = 障碍(渲染为黑),0 = 空闲(白)** | 二值 |
| 障碍生成 | 14 个随机矩形块(边长 3~10) + 2% 随机散点 + 四周边界墙 | 占据率约 **20%** |
| 距离场 SDF | 多源 BFS 到最近障碍的格数距离,除以 64 归一化到 [0,1] | 供模型感知"离障碍多远" |

### 2.2 起点 / 终点
- 在空闲区随机取,要求互相连通且曼哈顿距离 ≥ **16** 格。

### 2.3 路径(航点表示)
| 参数 | 值 | 说明 |
|------|----|------|
| 航点数 N | **32** | 定长,按弧长重采样 |
| 坐标 | **(x, y) = (列 c, 行 r)**,归一化到 **[-1, 1]** | `x = c/63*2-1` |
| 专家搜索 | A\*(w=1) / 加权A\*(w=1.5, 2.0) / 贪心(w=1e6) | 8 邻域,octile 启发式,禁止切角 |

### 2.4 数据集文件 `dataset_<M>.npz`
| 键 | 形状 | 含义 |
|----|------|------|
| `maps` | (M, 2, 64, 64) | 占据 + SDF |
| `starts` / `goals` | (M, 2) | 归一化起/终点 |
| `paths` | (M, 32, 2) | 归一化航点序列 |

体积参考:`dataset_1500.npz` ≈ 2.4 MB,`dataset_20000.npz` ≈ 32 MB。

---

## 3. 扩散规划器(diffusion_path.py)

| 项 | 值 |
|----|----|
| 扩散步数 T | 200 |
| 噪声调度 | 线性 beta `1e-4 → 0.02` |
| 前向 | `x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε`(重参数化) |
| 去噪网络 | CNN 地图编码器(→64维)+ MLP(hidden=512),预测噪声 ε |
| 条件注入 | concat `[x_t, 时间嵌入(64), 地图嵌入(64), start(2), goal(2)]` |
| 采样 | 祖先采样(ancestral DDPM),每步**钉住首尾航点** = start/goal |
| 后处理 | 用 SDF 梯度把航点推离障碍 + 拉普拉斯平滑(`repair_path`) |
| 训练 | 40000 步,batch=256,lr=2e-4,4060 上约 10 分钟,loss≈0.025 |

**评估指标**:无碰撞率(路径不穿黑格)、长度比(生成路径 / A\* 路径)。

---

## 4. 地图 VAE(vae_map.py)

| 项 | 值 |
|----|----|
| 输入 | 64×64 占据图(通道0) |
| 隐向量 latent | **32** 维 |
| 编码器 | 3 层卷积下采样 64→8,输出 mu/logvar |
| 解码器 | 3 层转置卷积上采样 8→64,sigmoid 重建 |
| 损失 | BCE 重建 + KL 散度 |
| 训练 | 2000 张地图,30 epoch,batch=64,几十秒 |
| 权重体积 | 约 **884 KB** |
| 效果 | 能大致还原障碍布局与位置,边缘偏模糊(小 latent 的正常表现) |

---

## 5. 如何运行

```bash
# 0) 激活环境
cd /home/t/projects/diffusion_planner_demo
source venv_py38/bin/activate
cd diffusion_planner

# 1) 生成数据集(最耗时,一次性;带进度条)
python make_dataset.py --data 20000

# 2) 训练扩散规划器 + 采样评估渲染
python diffusion_path.py --data 20000 --train 40000
#   产物: figs/diffusion_path/fig_planner.png, planner_ckpt.pt

# 3) 训练地图 VAE(复用已有数据集)
python vae_map.py
#   产物: figs/vae_map/fig_vae.png, vae_map.pt

# 快速验证(1~2 分钟, 效果仅供跑通流程)
python diffusion_path.py --quick
```

常用参数:
- `make_dataset.py --data N [--overwrite]`
- `diffusion_path.py --data N --train STEPS [--regen] [--quick]`

---

## 6. 当前结果与已知问题

- 扩散规划器(40000 步训练,loss≈0.025):**无碰撞率 5/8,长度比 0.98**。
- **已知问题**:长度比 < 1 说明生成路径在**抄近道穿障碍**,是碰撞主因。
  根因:① 专家 A\* 路径贴障碍角、无安全间隙;② 地图条件经全局池化后丢失精细
  空间信息;③ 训练目标只有噪声 MSE,缺避障信号。
- **改进方向(待做)**:C 空间障碍膨胀、逐航点局部 SDF 条件、避障损失/采样引导、
  更强后处理修复。

---

## 7. 环境依赖

`venv_py38`(Python 3.8.10):
- torch 2.4.1+cu121(CUDA 12.1,已验证 RTX 4060 可用)
- torchvision 0.19.1、ultralytics 8.4.112
- numpy 1.24.4、matplotlib 3.7.5、tqdm 4.70.1

> 注:该 venv 由打包环境迁移而来,`bin/` 内脚本路径已修正为当前目录。
> 若再次移动 venv,需重新批量替换其中的绝对路径。
