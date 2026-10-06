# -*- coding: utf-8 -*-
"""抽象接口与核心数据类。上层只依赖这里的 ABC, 具体实现可替换(算法/接口分离)。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 数据类
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """一个泊车场景。"""
    grid: np.ndarray                 # (H,W) 占据栅格, 1=障碍
    res: float                       # m/cell
    start_pose: np.ndarray           # (3,) 后轴位姿 (x,y,theta)
    goal_pose: np.ndarray            # (3,)
    scene_type: str = "perp"         # perp=垂直/斜列, par=平行
    bbox: Optional[tuple] = None     # (x0,y0,x1,y1) 归一化用世界包围盒
    slot_polygon: Optional[np.ndarray] = None   # (K,2) 车位角点(渲染用)

    @property
    def shape(self):
        return self.grid.shape


@dataclass
class Trajectory:
    """规划器输出的密集轨迹(基元分辨率)。"""
    poses: np.ndarray                # (L,3)
    gear: np.ndarray                 # (L,) +1 前进 / -1 倒车
    cost: float = 0.0
    n_switches: int = 0
    planner: str = "hybrid_a_star"


@dataclass
class FixedTraj:
    """定长 N 轨迹(训练/扩散用)。"""
    poses: np.ndarray                # (N,3)
    traj4: np.ndarray                # (N,4) [x,y,cos,sin]
    gear: np.ndarray                 # (N,)
    length: float = 0.0
    n_switches: int = 0
    scene_type: str = "perp"


# --------------------------------------------------------------------------- #
# 抽象接口
# --------------------------------------------------------------------------- #
class MotionPlanner(ABC):
    @abstractmethod
    def plan(self, scenario: Scenario) -> Optional[Trajectory]:
        """求解一条可行轨迹; 无解返回 None。"""


class ScenarioGenerator(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Optional[Scenario]:
        """采样一个场景; 不合法返回 None。"""


class TrajPostprocessor(ABC):
    @abstractmethod
    def process(self, traj: Trajectory, n: int,
                scenario: Optional[Scenario] = None) -> Optional[FixedTraj]:
        """把密集轨迹后处理成定长 n 的 FixedTraj。"""


class MapConditionerABC(ABC):
    """地图条件分支接口(与 diffusion_path.MapConditioner 同签名, 便于替换/基线对比)。"""
    @abstractmethod
    def forward(self, map_tensor, z=None):
        ...


class GenerativePlanner(ABC):
    @abstractmethod
    def fit(self, dataset) -> None:
        ...

    @abstractmethod
    def infer(self, scenario: Scenario) -> FixedTraj:
        ...
