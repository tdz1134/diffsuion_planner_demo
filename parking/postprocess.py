# -*- coding: utf-8 -*-
"""轨迹后处理: 密集轨迹 -> 定长 N 的 FixedTraj, 实现 TrajPostprocessor 接口。

要点(与 plan §4 一致):
  - 按**累计弧长(取绝对值, 跨换档也均匀)**重采样为恰好 N 个位姿。
  - x,y 线性插值; theta 先 unwrap 再插值(避免环绕跳变)。
  - 端点严格对齐: 第 0 点 = start, 第 N-1 点 = goal(与训练/采样"钉端点"一致)。
  - gear 取最近邻标签保留(供可行性评估, 不参与扩散)。
  - FixedTraj 存**世界坐标**; 归一化(x,y->[-1,1])在进扩散前用全局 lot bbox 做,
    见 geometry.normalize_xy / dataset。这样 render/eval 可直接用世界坐标。
"""

from typing import Optional

import numpy as np

from .interfaces import Trajectory, FixedTraj, Scenario, TrajPostprocessor
from .geometry import resample_poses, poses3_to_traj4, path_length


class FixedNPostprocessor(TrajPostprocessor):
    def __init__(self, n: int = 40):
        self.n = int(n)

    def process(self, traj: Trajectory, n: int = None,
                scenario: Optional[Scenario] = None) -> Optional[FixedTraj]:
        n = self.n if n is None else int(n)
        poses = np.asarray(traj.poses, dtype=np.float64)
        if poses.shape[0] < 2:
            return None
        gear = np.asarray(traj.gear)
        # gear 与 poses 对齐: 若为逐段(L-1)则补尾成逐点(L)
        if gear.shape[0] == poses.shape[0] - 1:
            gear = np.concatenate([gear, gear[-1:]])
        elif gear.shape[0] != poses.shape[0]:
            gear = np.ones(poses.shape[0], dtype=np.int8)

        out_poses, out_gear = resample_poses(poses, n, gear=gear)
        traj4 = poses3_to_traj4(out_poses)
        scene_type = scenario.scene_type if scenario is not None else "perp"
        return FixedTraj(poses=out_poses, traj4=traj4, gear=out_gear,
                         length=path_length(poses),
                         n_switches=int(traj.n_switches),
                         scene_type=scene_type)


if __name__ == "__main__":
    # 自检: 造一条带换档的密集轨迹, 重采样到 N, 校验端点/长度/档位
    rng = np.random.default_rng(0)
    L = 200
    t = np.linspace(0, 1, L)
    x = 5 * t
    y = 3 * np.sin(2 * np.pi * t) * 0.3 + 2 * t
    th = np.arctan2(np.gradient(y), np.gradient(x))
    poses = np.stack([x, y, th], axis=1)
    gear = np.where(t < 0.4, 1, np.where(t < 0.7, -1, 1)).astype(np.int8)
    tr = Trajectory(poses=poses, gear=gear[:-1], cost=0.0, n_switches=2)
    pp = FixedNPostprocessor(n=40)
    ft = pp.process(tr)
    print("N:", ft.poses.shape, "traj4:", ft.traj4.shape, "gear:", ft.gear.shape)
    print("start pin:", np.allclose(ft.poses[0], poses[0]),
          " end pin:", np.allclose(ft.poses[-1], poses[-1]))
    print("length dense=%.3f fixed=%.3f" % (path_length(poses), ft.length))
    print("gear uniq:", np.unique(ft.gear), " switches:", ft.n_switches)
