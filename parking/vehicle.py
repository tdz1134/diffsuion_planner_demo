# -*- coding: utf-8 -*-
"""车辆几何与运动学(自行车模型)。

参考点 = 后轴中心。局部坐标: +x 车头方向, +y 车体左侧。
足迹采样点用于碰撞检测; 4 角点用于渲染。
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .config import VehicleConfig
from ._jit import optional_njit


# --------------------------------------------------------------------------- #
# numba 友好的纯数值 kernel(未装 numba 时退化为普通函数)
# --------------------------------------------------------------------------- #
@optional_njit
def kinematic_step_k(x, y, th, steer, ds, wheelbase):
    """自行车模型: 沿弧长 ds(可负=倒车)以固定前轮转角 steer 积分, 返回新位姿。"""
    kappa = math.tan(steer) / wheelbase
    if abs(kappa) < 1e-9:
        nx = x + ds * math.cos(th)
        ny = y + ds * math.sin(th)
        nth = th
    else:
        R = 1.0 / kappa
        nx = x + R * (math.sin(th + kappa * ds) - math.sin(th))
        ny = y + R * (math.cos(th) - math.cos(th + kappa * ds))
        nth = th + kappa * ds
    return nx, ny, nth


@optional_njit
def points_collide_k(grid, pts_local, px, py, cos_t, sin_t, res, H, W):
    """把足迹局部点按位姿变换到世界, 再查占据栅格。任一命中/越界即碰撞。

    grid: (H,W) float/uint8, 1=障碍; pts_local: (P,2)。返回 True=碰撞。
    """
    P = pts_local.shape[0]
    for i in range(P):
        lx = pts_local[i, 0]
        ly = pts_local[i, 1]
        wx = px + lx * cos_t - ly * sin_t
        wy = py + lx * sin_t + ly * cos_t
        c = int(round(wx / res))
        r = int(round(wy / res))
        if r < 0 or r >= H or c < 0 or c >= W:
            return True                     # 越界视为障碍
        if grid[r, c] > 0.0:
            return True
    return False


# --------------------------------------------------------------------------- #
# 车辆对象
# --------------------------------------------------------------------------- #
@dataclass
class Vehicle:
    cfg: VehicleConfig = field(default_factory=VehicleConfig)
    n_long: int = 5      # 纵向采样点数
    n_lat: int = 3       # 横向采样点数

    def __post_init__(self):
        self.cfg = self.cfg
        self._foot = self._make_footprint(self.n_long, self.n_lat)
        # 4 角点须按多边形环序(否则渲染成自交"蝴蝶结"); 碰撞用 _foot, 与此无关
        w2 = self.cfg.width / 2.0
        self._corners = np.array([[self.x_rear, -w2], [self.x_front, -w2],
                                  [self.x_front, w2], [self.x_rear, w2]],
                                 dtype=np.float64)

    # -- 几何 -------------------------------------------------------------- #
    @property
    def x_front(self):
        return self.cfg.wheelbase + self.cfg.front_overhang

    @property
    def x_rear(self):
        return -self.cfg.rear_overhang

    def _make_footprint(self, nl, na):
        xs = np.linspace(self.x_rear, self.x_front, nl)
        ys = np.linspace(-self.cfg.width / 2, self.cfg.width / 2, na)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

    @property
    def footprint_local(self):
        return self._foot

    @property
    def corners_local(self):
        return self._corners

    def to_world(self, pts_local, pose):
        """局部点 -> 世界点。pts_local:(P,2), pose=(x,y,theta) -> (P,2)。"""
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = math.cos(th), math.sin(th)
        lx = pts_local[:, 0]; ly = pts_local[:, 1]
        return np.stack([x + lx * c - ly * s, y + lx * s + ly * c], axis=1)

    def footprint_world(self, pose):
        return self.to_world(self._foot, pose)

    def corners_world(self, pose):
        return self.to_world(self._corners, pose)

    # -- 运动学 ------------------------------------------------------------ #
    def step(self, pose, steer, ds):
        x, y, th = kinematic_step_k(float(pose[0]), float(pose[1]),
                                    float(pose[2]), float(steer), float(ds),
                                    self.cfg.wheelbase)
        return (x, y, th)

    # -- 碰撞 -------------------------------------------------------------- #
    def collides(self, grid, pose, res):
        th = float(pose[2])
        return points_collide_k(grid, self._foot, float(pose[0]), float(pose[1]),
                                math.cos(th), math.sin(th), res,
                                grid.shape[0], grid.shape[1])

    def steer_fracs_to_angles(self, fracs):
        return [f * self.cfg.max_steer for f in fracs]


if __name__ == "__main__":
    v = Vehicle()
    print("max_steer(deg)=%.1f  r_min=%.2f  x_front=%.2f x_rear=%.2f"
          % (math.degrees(v.cfg.max_steer), v.cfg.r_min, v.x_front, v.x_rear))
    print("footprint pts:", v.footprint_local.shape, " corners:", v.corners_local.shape)
    # 直行 + 圆弧自检
    p = (0.0, 0.0, 0.0)
    for _ in range(10):
        p = v.step(p, 0.0, 0.5)
    print("straight 5m ->", tuple(round(x, 3) for x in p))
    p2 = (0.0, 0.0, 0.0)
    for _ in range(40):
        p2 = v.step(p2, v.cfg.max_steer, 0.4)
    print("full-lock arc ->", tuple(round(x, 3) for x in p2))
