# Diffusion Planner Demo

在**占据栅格地图**(黑=障碍、白=空闲)上,给定起点/终点,用**条件扩散模型(DDPM)**
生成一条避障路径。专家数据由 A\* / 加权 A\* / 贪心搜索提供,模型学习"加噪→去噪"的
逆过程,从纯噪声直接生成航点路径。附带一个**地图 VAE**(图→隐向量→还原图)演示
Latent-Diffusion 里的编解码组件。

- 运行环境:Python 3.8 + PyTorch 2.4.1+cu121,本仓库用 `venv_py38`(与代码同级、已 gitignore),GPU = **RTX 4060 Laptop 8GB**
- 训练一次完整模型:数据集生成约 24 分钟(一次性、可复用)+ 地图 VAE 约 5 分钟 + 规划器训练约 1~2 分钟(AMP+大batch 提速后)

> **本仓库另含一个泊车规划器**(独立包 `parking/`:混合 A\* 造数据 → 地图 VAE →
> SE(2) 轨迹扩散 → 修复/评估)。**不动本 demo**,只复用其中的地图 VAE 类。
> 用法与结果见 **§7**,通俗搭建记录见仓库根 **`PARKING_NOTES.md`**。

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
├── PARKING_NOTES.md           # 泊车规划(parking/)搭建记录(通俗版, 每改必记)
├── parking/                   # 【新增】泊车扩散规划器(独立包, 见 §7)
└── diffusion_planner/         # 占据栅格路径 demo(下述各节)
    ├── grid_env.py            # 地图生成 + A*/加权A*/贪心 + 距离场 + 路径重采样
    ├── make_dataset.py        # 数据集生成(独立、最耗时,带进度条)
    ├── diffusion_path.py      # 条件扩散路径规划器(训练+采样+修复+评估+渲染)
    ├── diffusion_toy.py       # 入门:2D 玩具数据上的最小 DDPM(加噪/去噪演示)
    ├── vae_map.py             # 占据栅格地图 VAE(编码/解码)
    ├── README.md              # 本文件
    ├── VAE_NOTES.md           # VAE 改动主线 + 踩坑记录
    ├── PLANNER_NOTES.md       # 规划器改动记录(通俗版)
    ├── dataset_<M>.npz        # 数据集缓存(按样本数命名,已 gitignore)
    ├── planner_ckpt.pt        # 扩散规划器权重
    ├── vae_map.pt             # VAE 权重(约 1.8 MB,冻结后供规划器当条件)
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
| 去噪网络 | MLP(hidden=512)预测噪声 ε;地图条件由 `--cond` 选择 |
| 地图条件 | **默认 `lat`**:冻结 VAE 的 8×8×8 latent(→Linear 32 维)⊕ 显式 SDF 小 CNN(→32 维)= 64 维 map_emb;`cnn`:原始 2 通道 CNN MapEncoder(基线) |
| 条件注入 | concat `[x_t, 时间嵌入(64), 地图嵌入(64), start(2), goal(2)]` |
| 采样 | 祖先采样(ancestral DDPM),每步**钉住首尾航点** = start/goal |
| 后处理 | 用 SDF 梯度把航点推离障碍 + 拉普拉斯平滑(`repair_path`) |
| 训练 | 默认 batch=512(lr 随 batch 按√ 自适应)+ **AMP 混合精度** + **预存冻结 latent**;提速后 10000 步约 1 分钟 |

**评估指标**:无碰撞率(路径不穿黑格)、长度比(生成路径 / A\* 路径)。

---

## 4. 地图 VAE(vae_map.py)

| 项 | 值 |
|----|----|
| 输入 | 64×64 占据图(通道0) |
| 隐变量 latent | **空间特征图 8×8×8**(`LAT_CH=8`,下采样 8×,非全局向量) |
| 编码器 | 4 级卷积 64→32→16→8,通道 `CH=(16,32,48,64)`,级间 **GroupNorm+SiLU 残差块**;输出 mu/logvar |
| 解码器 | 转置卷积 8→16→32→64 + 残差块,sigmoid 重建(无 U-Net skip) |
| 参数量 | 约 **444 K**(旧版 ~5.0 M 的 1/11) |
| 损失 | **加权 BCE 重建**(pos_weight=3,障碍类)+ **小权重 KL**(β=1e-2),两项均**逐样本求和**避免后验塌缩 |
| 训练 | 10000 张地图,60 epoch,batch=64,约 5~6 分钟 |
| 权重体积 | 约 **1.8 MB** |
| 量化指标 | 像素准确率 **99.66%**、障碍类 **IoU 98.23%**(eval 200 张) |
| 效果 | 障碍块/散点基本对齐,重建近乎无损;VAE 作为**独立模块训好后冻结**,供规划器当条件 |

---

## 5. 如何运行

```bash
# 0) 激活环境
cd /home/t/projects/diffusion_planner_demo
source venv_py38/bin/activate
cd diffusion_planner

# 1) 生成数据集(最耗时,一次性;带进度条)
python make_dataset.py --data 20000

# 2) 训练地图 VAE(复用已有数据集;--cond lat 需要它)
python vae_map.py
#   产物: figs/vae_map/fig_vae.png, vae_map.pt

# 3) 训练扩散规划器 + 采样评估渲染(默认 --cond lat, 加载冻结 vae_map.pt)
python diffusion_path.py --data 20000 --train 20000 --batch 512 --cond lat
#   基线对比(不依赖 VAE): --cond cnn
#   产物: figs/diffusion_path/fig_planner.png, planner_ckpt.pt

# 快速验证(几十秒~2 分钟, 效果仅供跑通流程)
python diffusion_path.py --quick
```

常用参数:
- `make_dataset.py --data N [--overwrite]`
- `diffusion_path.py --data N --train STEPS [--batch B] [--cond {lat,cnn}] [--regen] [--quick]`
- `vae_map.py`(参数在文件头部常量区,如 `CH`/`LAT_CH`/`EPOCHS`/`BETA`)

---

## 6. 当前结果与已知问题

- 扩散规划器(**仅 8 个评估样本,数字噪声大**):
  - `--cond lat`(冻结 VAE latent + SDF):无碰撞率 **5~6/8**,长度比 ≈ 0.98~0.99
  - `--cond cnn`(原始 CNN 基线):无碰撞率 **3/8**,长度比 ≈ 1.00
  - latent 分支还更快(预存 VAE → 热循环不跑 VAE,161 vs 49 步/s)。
- 地图 VAE:像素准确率 **99.66%**、障碍 **IoU 98.23%**,重建近乎无损。
- **已知问题**:长度比 < 1 说明生成路径在**抄近道穿障碍**,是碰撞主因。
  根因:① 专家 A\* 路径贴障碍角、无安全间隙;② 训练目标只有噪声 MSE,缺避障信号。
- **改进方向(Phase 2,待做)**:C 空间障碍膨胀、逐航点局部 SDF 条件、避障损失/采样引导、
  更强后处理修复;并把评估样本从 8 提到 ~50 才有统计意义。

---

## 7. 泊车规划(parking/,新增)

在**真实车位场景**(垂直入库 / 平行侧方)里做泊车轨迹规划:用**混合 A\*(HA\*)**当"老师"
开出可行轨迹 → 攒数据集 → 训一个**条件扩散模型**,以后给它 车位图 + 起点 + 终点,直接"画"出
一条**定长 40 点**的 SE(2) 轨迹 `(x, y, cosθ, sinθ)`。

- **独立新包**,`diffusion_planner/` demo **一行没动**;只**复用**其中已验证的地图 VAE 类与扩散调度思路。
- **算法/接口分离**:HA\*、场景生成、后处理、地图编码、扩散模型都是可替换插件(ABC + dataclass 注入)。
- 详细搭建记录见仓库根 **`PARKING_NOTES.md`**(通俗版)。

### 7.1 流程与运行(仓库根目录)
```bash
cd /home/t/projects/diffusion_planner_demo
venv_py38/bin/python -m parking.dataset  --n 4000 --proc 8            # ① HA* 造数据(多进程) → cache/parking_4000.npz
venv_py38/bin/python -m parking.map_vae  --npz cache/parking_4000.npz # ② 泊车地图重训 VAE 并冻结 → parking/cache/vae_parking.pt
venv_py38/bin/python -m parking.train    --npz cache/parking_4000.npz --steps 12000  # ③ 训扩散 → parking/cache/diffusion_parking.pt
venv_py38/bin/python -m parking.evaluate --npz cache/parking_4000.npz # ④ 评估 + 对比图 → figs/parking/m6_eval_compare.png
```
> 每个模块都可 `venv_py38/bin/python -m parking.<模块>` 跑自带小测;`evaluate` 要用与 `train` 相同的 npz。

### 7.2 关键参数
| 项 | 值 |
|----|----|
| 轨迹表示 | 定长 **N=40** 个 `(x, y, cosθ, sinθ)`,世界单位米;首尾钉住 start/goal |
| 地图 | 占据栅格 **72×128**(8 的倍数,复用 VAE 的 3 次 /2 下采样),分辨率 0.2 m;通道 [占据, SDF] |
| 老师 HA\* | 自行车模型基元 + 足迹(旋转矩形)碰撞 + holonomic 场启发;整段 numba `@njit`,**0.078 s/条**(比纯 Python 快 ~240×) |
| 数据集 | 多进程并行;HA\* 求解成功率 **0.76~0.80**(失败丢弃重采);12000 条 / 16 进程 ≈ 139 s |
| 地图 VAE | 复用 `vae_map.py` 的 VAE,在泊车占据图上**重训→冻结**:像素准确率 **99.80%**、障碍 IoU **99.13%** |
| 扩散 | DDPM(ε-pred),T=200,线性 beta 1e-4→0.02;MLP 去噪器(hidden 512,可训练 **805.8K**);条件 = 冻结 VAE latent ⊕ SDF(64 维)+ 起终点 + 时间;采样每步钉首尾;复用 AMP + 大 batch + 预存 latent |
| 评估指标 | **足迹**(旋转矩形)无碰撞率、运动学可行性(\|κ\|≤1.2/r_min)、成功率、长度比 vs HA\*、终点误差 |

### 7.3 当前结果(诚实)
- **HA\* 老师本身 100% 可行无碰撞**(它就是数据源),渲染对比图见 `figs/parking/`。
- **扩散 Phase 1(纯 ε-MSE)原始输出不可行**:无碰撞 ~1.7%、可行 0%、最大曲率远超上限;
  采样后**修复**(`repair.py`:SDF 外推 + 拉普拉斯平滑 + 钉端点)把无碰撞率抬到 ~20~30%,但可行性仍低。
- **加数据(4k→12k)+ 加步数(12k→30k)重训并没有救回可行性** → 瓶颈不在数据量,而在
  **ε-MSE 目标 + MLP 去噪器**对"尖锐、多模态"泊车轨迹的 averaging。
- **Phase 2 已试(诚实,均未突破端到端可行)**:
  - *训练内惩罚*(把曲率/航向/足迹碰撞的可微代价加进 ε-MSE,`penalty.py`):**两组权重都让 DDPM 采样器发散**(长度比 600~870 的螺旋、碰撞率反而 0%)——轨迹代价会系统性偏置学出的 score。已回滚权重。
  - *采样引导*(`evaluate --g-*`,晚步 `min_abar`高、仅碰撞):**安全的 Pareto 改善**——原始无碰撞率 **1.7%→25%**、修复后 →~32%,最大曲率 4.8→3.5,长度比稳定不发散;但**运动学可行性仍 0%**(曲率离上限差 ~15×),端到端成功率仍 0。加曲率/航向项到引导会立刻发散(同上 ill-conditioning)。对比图 `figs/parking/m8_guided_compare.png`。
- **Phase 3 去噪器升级 MLP→1D 时序卷积 + 修正可行性度量(诚实重大突破,见 `PARKING_NOTES.md` M9)**:
  - `config.denoiser=conv`(`CondDenoiserConv`, ~419K 参, `train --denoiser conv`);同 12k/30k 重训全轴变好:原始无碰撞 **1.7%→32%(叠碰撞引导→70%)**、长度比 **2.34→0.95**、平均横向滑移 **0.64→0.15**(专家 0.015)。对比图 `figs/parking/m9_conv_compare.png`。
  - **发现旧"可行率 0%"很大程度是度量 bug**:旧曲率判据在定长 N=40 弦稠密化的换档尖点处曲率爆表,**连 HA\* 专家都被判 0% 可行**。已改为对尖点免疫的**非完整性横向滑移**判据→**端到端成功率首次非零(conv+修复 6.7%)**。
- **M10 尖点分段可行性(用户提出:先按尖锐点切割、逐段判)**: 已实现 `segment_by_cusps`+`segment_metrics`(按运动在航向投影变号处切, 倒车/侧滑不误切)。逐步验证发现:**只切尖点不够**——定长 N=40 对曲率**天然欠分辨**(紧入口弧塔缩成直角), 连专家重采样后仍读 κ≫上限。改用**段内位置 Menger 曲率**后把专家从 **0%→43% 可行**、排序正确(专家1.45<CONV3.82<MLP5.58), 但仍不能作绝对硬门。因此可行性硬门保留**横向滑移**, 分段额外产出两个诊断: `raw_seg_kappa`(相对)与 **`raw_gear_switches`=换档次数(最干净的质量信号: MLP 16.2 抖动 vs CONV 2.17≈专家 2.6)**。
- **M11 提高 N:40→80(用户选定)— 修好了度量, 没修好模型**: N 参数化(train/evaluate **自动从数据/ckpt 推 N**, 向后兼容)。同场景重生 N=80 数据+重训 conv: **专家分段曲率 1.45→0.43、达上限占比 43%→78%**→曲率终于**可测**; 但 **conv@N80 仍 κ≈4.6(0%可行)、原始碰撞 0.32→0.18**→**瓶颈在模型/目标而非表示**。价值: 现在可在 N=80 上用正规 classifier guidance/曲率惩罚去约束模型。对比图 `figs/parking/m11_n80_compare.png`。
- **M12 正规 classifier guidance(用户选定)— 诚实负结果**: 训了可行性判别器 `parking/critic.py`(噪声增强、对 x_t 取梯度, 标准姿势), 它能完美分开专家/模型采样(acc 1.0)。**但引导仍发散**——scale=0.02、只晚步也 len_ratio 0.95→3~19、碰撞归零, 与 M8 手写几何代价同根病。**推理期事后引导(几何 or 学习判别器)对本 DDPM 均病态: 梯度把样本推向离流形对抗区。→ 可行性必须训时内化到模型(gear-aware 架构/建模), 不能事后贴。** 代码保留、默认关(c-scale=0 不影响基线)。
- **M13 用户两项改动(map 条件简化为纯 VAE + 上 Transformer)— 诚实负结果**: 两个开关均**可切换、默认不变**(向后兼容)。**(A) map_cond=vae**(去 SDF): conv 上隔离→无碰撞率 **0.233→0.133**、端到端 **0.083→0.017**(变差——SDF 是降碰撞的承重分支)。**(B) denoiser=trans**(全局时序自注意力): lat_sdf 上隔离→无碰撞持平 0.200, 但**滑移 0.195→0.376、换挡抖动 2.28→5.80 回潮**(逼近 MLP)。两者均不优于 conv@N80 → **再次确认瓶颈在目标/表示(无 gear 监督), 不在条件构成或感受野**。代码+3 个 ckpt 作消融产物保留, 默认 mlp+lat_sdf 未动。图 `figs/parking/m13_*.png`。
- **真正的后续(已排除事后引导)**: 下一步方向 = **让去噪器输出 gear / 混合(gear-aware)表示或架构**, 把运动学可行性训时内化。详见 `PARKING_NOTES.md` §5/M9–M13。

---

## 8. 环境依赖

`venv_py38`(Python 3.8.10):
- torch 2.4.1+cu121(CUDA 12.1,已验证 RTX 4060 可用)
- torchvision 0.19.1、ultralytics 8.4.112
- numpy 1.24.4、matplotlib 3.7.5、tqdm 4.70.1

> 注:该 venv 由打包环境迁移而来,`bin/` 内脚本路径已修正为当前目录。
> 若再次移动 venv,需重新批量替换其中的绝对路径。
