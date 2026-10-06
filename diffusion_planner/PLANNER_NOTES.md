# 扩散规划器 —— 改动记录(通俗版)

> 目的:像 `VAE_NOTES.md` 一样,把每次对**扩散规划器** `diffusion_path.py` 的改动记下来:
> 改了什么、为什么改、怎么跑、结果。方便不懂网络细节也能回溯。

---

## 变更 #1:地图条件改用「冻结 VAE 的 latent + SDF」双分支(2026-09-29)

### 一句话
把"喂给规划器的地图信息",从**直接用小 CNN 压缩 64×64 地图**,换成**先用你训好的 VAE 把地图压成 8×8×8 隐向量、再配一份 SDF 距离场**一起喂进去。扩散生成路径的主体**完全没动**。

### 先理解规划器的输入(改动前后都一样)
- **被加噪/去噪的东西** = 路径航点 `paths (B, 32, 2)`(32 个点的坐标)。这才是扩散的主体。
- **条件(引导,不加噪)** = 地图信息 + 起点 + 终点 + 时间。
- 本次只动"地图信息"这一路怎么变成条件向量 `map_emb`。

### 改动前 vs 改动后
| | 改动前(baseline) | 改动后(本次) |
|---|---|---|
| 地图→条件 | `MapEncoder`:小 CNN 吃 `2通道(占据+SDF) 64×64` → 64 维 | `MapConditioner`:冻结 VAE 把占据图压成 `8×8×8` latent → `LatentEnc`(32维) **拼上** `SdfEnc`(SDF→32维) = 64 维 |
| 谁被训练 | MapEncoder 一起训 | **VAE 冻结不训**,只训 LatentEnc + SdfEnc + 去噪器 |
| 为什么留 SDF | —— | VAE 当初只用占据图训,latent 里**没有障碍距离信息**;SDF 对降碰撞有用,所以单独保留一路 |

### 具体改了 `diffusion_path.py` 哪些地方
1. 新增 `load_frozen_vae()`:载入 `vae_map.pt` 并 `requires_grad=False` 冻结(只用来编码)。
2. 新增 `LatentEnc`(latent→32维)、`SdfEnc`(SDF图→32维)、`MapConditioner`(把两者拼成 64 维)。
3. `MapConditioner` 的接口和原 `MapEncoder` **完全一样**(吃 `(B,2,64,64)`、吐 `(B,64)`),所以 `train()` / `sample()` 循环**一行没改**。
4. `train()`:优化器只收 `requires_grad=True` 的参数(把冻结的 VAE 排除,避免无谓更新)。
5. 新增命令行 `--cond {lat,cnn}`,默认 `lat`:
   - `lat` = 本次的 latent+SDF 双分支;
   - `cnn` = 原始 MapEncoder(**保留做基线对比**)。
6. 小修:`torch.load(..., weights_only=True)` 消除 FutureWarning。

### 怎么跑(A/B 对比,验证 latent 接入是否可行)
```bash
cd /home/t/projects/diffusion_planner_demo/diffusion_planner

# B:本次方案(latent + SDF)
../venv_py38/bin/python diffusion_path.py --data 20000 --train 40000 --cond lat

# A:基线(原始 CNN),用来对比有没有变差
../venv_py38/bin/python diffusion_path.py --data 20000 --train 40000 --cond cnn
```
> 前提:`vae_map.pt` 已存在(先跑过 `vae_map.py`)。两次的"无碰撞率/长度比"对比即可判断。
> 提示:两方案写同一张 `figs/diffusion_path/fig_planner.png`,跑完 A 记得先另存再跑 B(或看终端数字)。

### 冒烟测试结论
`--quick --cond lat` 端到端跑通(载入冻结 VAE→训练→采样→渲染),无报错。quick 下 4/8(数字无意义,仅验证流程)。

### 结果(待你跑完整训练填)
| 方案 | 无碰撞率 | 平均长度比(vs A*) | 备注 |
|------|---------|------------------|------|
| A 基线 `--cond cnn` | 待填 | 待填 | 历史约 5/8 |
| B `--cond lat`(latent+SDF) | 待填 | 待填 | 本次改动 |

### 我的预期(诚实)
- B 的 latent 分支信息量 ≤ 原 CNN(它只是原输入的一个压缩版),所以 **B 大概率与 A 持平或略差**;
- 本方案的**真正目的是打通"VAE 压缩地图 → 规划器用 latent"这条链路**(你训 VAE 的初衷),不是直接提升碰撞率;
- **要真正降碰撞,还得靠下面的 Phase 2。**

---

## 变更 #2:训练提速(AMP + 大 batch + 预存 latent)(2026-09-29)

### 背景
`--train 40000` 要 25 分钟。诊断:模型很小,40ms/步主要耗在 Python 开销 + 每步重跑冻结 VAE 的卷积;且 40000 步≈把 2 万数据过 500 遍,严重过量。

### 改了什么(`diffusion_path.py`)
1. **预存 latent**:VAE 冻结且输入固定 → `precompute_latents()` 训练前一次性算好全部 `8×8×8` latent 存显存,热循环里**不再跑 VAE**(`MapConditioner.forward` 加 `z=` 参数,采样阶段少量图仍现算)。契合"VAE 是独立模块、训好不动"。
2. **AMP 混合精度**:训练前向/反向套 `torch.amp.autocast("cuda")` + `GradScaler`(loss 用 fp32 算 MSE 保稳)。
3. **`--batch` 参数**(默认 512):学习率随 batch 按 `LR*(batch/256)**0.5` 自适应,避免大 batch 训不动。
4. 优化器只收 `requires_grad=True` 参数(排除冻结 VAE)。

### 效果
- 步速 **25 → 161 步/s**(batch 还从 256 翻倍到 512),综合吞吐 ~13×。
- `--train 10000` 从 ~6.5 分钟 → **~1 分钟**;`--train 40000` → ~4 分钟。
- 冒烟 `--quick --cond lat` 通过,latent 预存形状 `(1500,8,8,8)`。

### 推荐命令(现在很快,可放心多训)
```bash
cd /home/t/projects/diffusion_planner_demo/diffusion_planner
../venv_py38/bin/python diffusion_path.py --data 20000 --train 20000 --batch 512 --cond lat
../venv_py38/bin/python diffusion_path.py --data 20000 --train 20000 --batch 512 --cond cnn
```

### 已知结果(逐步累积)
| 配置 | 无碰撞率 | 长度比 | 步速/用时 |
|------|---------|--------|------|
| `--train 10000 --cond lat`(batch256, 提速前) | 6/8 | 0.98 | 6:38 |
| `--train 10000 --cond lat`(batch512, 提速后) | **5/8** | 0.99 | 161步/s, ~1:01 |
| `--train 20000 --cond cnn`(batch512, 基线) | **3/8** | 1.00 | 49步/s, ~6:40 |

### 结论修正(诚实)
- **`lat` 同时更快且(本次评估)更好**:预存 VAE 后热循环无 VAE、`LatentEnc` 只是 Linear,所以 161 vs 49 步/s;而无碰撞率 5/8 > cnn 3/8。
- 推翻了我变更 #1 里"latent 信息量≤原 CNN 所以可能更差"的预期。可能原因:预训的 VAE 是**更好的地图特征**,而 `MapEncoder` 从零跟着扩散一起训、容易欠拟合地图。
- **但重要警告**:只评估 **8 个样本**,5/8 vs 3/8 统计上很噪(且不同 batch/LR),**不能断言 lat 一定优于 cnn**。只能确定:① latent 当条件**可行且不弱**;② 提速真实有效。
- 两者绝对水平都一般(5/8、3/8),**真正拉高碰撞率靠 Phase 2**。

---

## 待办(Phase 2:真正提高无碰撞率)
- [ ] C 空间障碍膨胀(评估/修复时把障碍按机器人尺寸放大)——性价比最高。
- [ ] 逐航点局部 SDF 条件(每个点看附近障碍)。
- [ ] 避障损失 / 采样时 SDF 引导。
- [ ] 更强 `repair_path`。
