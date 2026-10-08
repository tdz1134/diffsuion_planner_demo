# -*- coding: utf-8 -*-
"""集中式配置(dataclass), 供各模块依赖注入, 实现算法/参数解耦。

坐标与单位约定:
  - 世界坐标单位 = 米(m); 角度 = 弧度, 逆时针为正, x 向右 y 向下(与栅格 r=y,c=x 一致)。
  - 占据栅格 grid[r, c]: r=行(y), c=列(x); 1=障碍, 0=空闲。
  - 世界<->栅格: cell = round(world / res); world = cell * res。
  - 车辆位姿参考点 = 后轴中心 (rear axle), 姿态 (x, y, theta)。
"""

from dataclasses import dataclass, field
from typing import Tuple
import math


# --------------------------------------------------------------------------- #
# 车辆
# --------------------------------------------------------------------------- #
@dataclass
class VehicleConfig:
    length: float = 4.2          # 车长 m
    width: float = 1.85          # 车宽 m
    wheelbase: float = 2.7       # 轴距 m
    rear_overhang: float = 1.0   # 后轴到后保 m
    r_min: float = 4.5           # 最小转弯半径 m (决定最大转角)

    @property
    def front_overhang(self) -> float:
        return self.length - self.wheelbase - self.rear_overhang

    @property
    def max_steer(self) -> float:
        """由 r_min 反推最大前轮转角: tan(delta) = wheelbase / r_min。"""
        return math.atan(self.wheelbase / self.r_min)

    @property
    def max_curvature(self) -> float:
        return 1.0 / self.r_min

    @property
    def circum_radius(self) -> float:
        """足迹外接圆半径(相对后轴中心), 用于 C 空间粗膨胀。"""
        xf = self.wheelbase + self.front_overhang   # 后轴->车头
        xr = self.rear_overhang                     # 后轴->车尾
        return math.hypot(max(xf, xr), self.width / 2.0)


# 预设: 默认(略缩放, 便于 demo 稳定出数据) / 真实 / 更小
VEHICLE_DEFAULT = VehicleConfig()
VEHICLE_REAL = VehicleConfig(length=4.7, width=1.9, wheelbase=2.85,
                             rear_overhang=1.1, r_min=5.5)
VEHICLE_SMALL = VehicleConfig(length=3.2, width=1.6, wheelbase=2.2,
                              rear_overhang=0.8, r_min=3.2)


# --------------------------------------------------------------------------- #
# 车位 / 场地
# --------------------------------------------------------------------------- #
@dataclass
class LotConfig:
    res: float = 0.2             # 栅格分辨率 m/cell
    W: int = 128                 # 列数 (须为 8 的倍数, 复用 VAE 的 /8 下采样)
    H: int = 72                  # 行数 (须为 8 的倍数)
    # 车位尺寸 (m)
    perp_width: float = 2.7      # 垂直/斜列车位 宽
    perp_depth: float = 5.5      # 垂直/斜列车位 深
    par_length: float = 6.5      # 平行车位 长
    par_width: float = 2.7       # 平行车位 宽(=进深)
    lane_width: float = 6.5      # 行车通道宽
    wall_margin: float = 0.8     # 场地四周留白

    @property
    def world_w(self) -> float:
        return self.W * self.res

    @property
    def world_h(self) -> float:
        return self.H * self.res

    def __post_init__(self):
        assert self.W % 8 == 0 and self.H % 8 == 0, "W,H 须为 8 的倍数(复用现有 VAE)"


# --------------------------------------------------------------------------- #
# Hybrid A*
# --------------------------------------------------------------------------- #
@dataclass
class HAStarConfig:
    theta_res_deg: float = 5.0        # 姿态离散分辨率(度)
    prim_step: float = 0.4            # 每条运动基元的弧长 m
    steer_fracs: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)  # 相对 max_steer
    allow_reverse: bool = True
    # 代价权重
    fwd_cost: float = 1.0
    rev_cost: float = 1.6             # 倒车更贵
    gear_switch_penalty: float = 2.0  # 换档惩罚
    steer_change_penalty: float = 0.3 # 转角突变惩罚(平滑)
    heur_holonomic_w: float = 1.0     # 2D 有障碍场启发权重
    heur_rs_w: float = 1.0            # RS 无障碍启发权重
    heur_weight: float = 1.0          # 加权 A* 系数(>1 更快但次优; 专家数据可设 1.3~1.6)
    rs_max_dist: float = 8.0          # 仅当到目标欧氏距离 < 此值才试 RS 解析扩展(m)
    # 搜索控制
    max_nodes: int = 200000
    analytic_interval: int = 5        # 每展开 N 个节点尝试一次 RS 解析扩展(0=关闭)
    max_primitive_nodes: int = 40     # 单基元积分时碰撞检查的分段数上限

    @property
    def n_theta(self) -> int:
        return int(round(360.0 / self.theta_res_deg))


# --------------------------------------------------------------------------- #
# 扩散模型 / 训练
# --------------------------------------------------------------------------- #
@dataclass
class DiffusionConfig:
    n_wp: int = 40               # 定长航点数 N
    dim: int = 4                 # 每点维度: (x, y, cos, sin)
    t_steps: int = 200
    b0: float = 1e-4
    b1: float = 0.02
    map_emb: int = 64
    temb: int = 64
    hidden: int = 512
    lr: float = 2e-4
    batch: int = 512
    train_steps: int = 20000
    amp: bool = True
    cond: str = "lat"            # lat=冻结VAE latent+SDF; cnn=纯CNN基线
    # 去噪器架构(Phase 3 升级): mlp=旧的全连接(默认, 向后兼容); conv=1D 时序空洞卷积(对相邻航点的局部运动学耦合有归纳偏置)
    denoiser: str = "mlp"
    # Phase 5 (M15): 把档位 gear 作为额外一个扩散输出通道(state dim=dim+1, 前向+1/倒车-1)。默认关=旧 SE(2)4通道管道不变。
    use_gear: bool = False
    # Phase 5 (M16): x0-空间可行性损失(施加在 x0_hat、仅晚步高 abar 时), 默认 0=关(避开 M8 发散)
    w_smooth: float = 0.0        # 一/二阶时序差分平滑(治航点抖动)
    w_curv_x0: float = 0.0       # 速度自适应曲率惩罚 κ_max(v)=min(κ_geo, a_lat/v²)
    # 地图条件构成(仅 cond=lat 时有意义): lat_sdf=VAE latent⊕SDF双分支(默认, 向后兼容); vae=只用冻结VAE latent(去掉SDF分支)
    map_cond: str = "lat_sdf"
    dconv_hidden: int = 128      # conv 去噪器主干通道数
    dconv_layers: int = 5        # conv 残差块数(空洞膨胀 1,2,4,8,16 覆盖整条 N=40)
    dconv_cond_ch: int = 32      # 条件(时间/地图/起终点)经 MLP 后逐位置广播的通道数
    # Transformer 去噪器(Phase 5): 对 N 个航点做全局时序自注意力(相对 conv 的局部感受野)
    dtrans_model: int = 128      # 注意力隐藏维
    dtrans_heads: int = 4        # 多头注意力头数
    dtrans_layers: int = 4       # Encoder 层数
    dtrans_ff: int = 256         # 前馈内层维
    dtrans_dropout: float = 0.1
    dtrans_cond_tokens: int = 8  # trans2: 条件(地图/起终点)展开成多少个 memory token 供每层 cross-attn 读
    # Phase 2: 轨迹空间辅助惩罚(作用于去噪得到的 x0_hat; 权重 0=关闭, 向后兼容)
    w_nh: float = 0.0            # 航向一致性(非完整约束, 允许前进/倒车, 惩罚侧滑)
    w_curv: float = 0.0          # 曲率超限 |kappa|>1/r_min 惩罚
    w_coll: float = 0.0          # 足迹角点 SDF 碰撞惩罚
    coll_margin: float = 0.15    # 足迹角点距障碍的最小间隙(m)
    # VAE
    lat_ch: int = 8
    ch: Tuple[int, int, int, int] = (16, 32, 48, 64)
    vae_epochs: int = 60
    vae_beta: float = 1e-2
    vae_pos_w: float = 3.0


# --------------------------------------------------------------------------- #
# 数据生成
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    n_samples: int = 20000
    seed: int = 0
    eval_ratio: float = 0.02
    n_eval_viz: int = 8          # 可视化张数
    n_eval_metric: int = 60      # 指标评估场景数(>=50 才有统计意义)
    n_proc: int = 0              # 0=自动(CPU 核数)
    cache_dir: str = "cache"
    scene_types: Tuple[str, ...] = ("perp", "par")  # 垂直 / 平行


# --------------------------------------------------------------------------- #
# 顶层聚合配置
# --------------------------------------------------------------------------- #
@dataclass
class ParkingConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    lot: LotConfig = field(default_factory=LotConfig)
    hastar: HAStarConfig = field(default_factory=HAStarConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)


def default_config(preset: str = "default") -> ParkingConfig:
    veh = {"default": VEHICLE_DEFAULT, "real": VEHICLE_REAL,
           "small": VEHICLE_SMALL}[preset]
    return ParkingConfig(vehicle=veh)
