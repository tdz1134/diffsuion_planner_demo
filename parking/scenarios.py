# -*- coding: utf-8 -*-
"""泊车场景生成器, 实现 ScenarioGenerator 接口。

两类场景(与 plan 一致):
  - perp: 垂直/斜列车位(车位靠上墙, 开口朝下行车通道), 目标车头朝内(+y)泊入。
  - par : 平行车位(车位靠下路缘), 目标车头沿路缘(0 或 pi), 倒车入库由 HA* 决定。

每个场景 = 占据栅格 + start_pose + goal_pose + scene_type + bbox + slot_polygon。
随机化: 车位横向位置、邻车宽/有无、起点(位置+朝向)、终点(槽内偏移)。
几何不合法(起/终点压障碍)返回 None, 由调用方重采。
"""

import math
from typing import Optional, Tuple

import numpy as np

from .interfaces import Scenario, ScenarioGenerator
from .config import LotConfig, VehicleConfig
from .vehicle import Vehicle
from . import occupancy as occ


class LotScenarioGenerator(ScenarioGenerator):
    def __init__(self, vehicle: Vehicle = None, lot: LotConfig = None,
                 scene_types: Tuple[str, ...] = ("perp", "par"),
                 neighbor_p: float = 0.9):
        self.v = vehicle or Vehicle(VehicleConfig())
        self.lot = lot or LotConfig()
        self.scene_types = tuple(scene_types)
        self.neighbor_p = neighbor_p
        self.res = self.lot.res
        self.ww = self.lot.world_w
        self.wh = self.lot.world_h
        self.wall = self.lot.wall_margin

    # -- 内部: 栅格与图元 -------------------------------------------------- #
    def _new_grid(self):
        g = np.zeros((self.lot.H, self.lot.W), dtype=np.float64)
        occ.fill_border(g, max(1, int(round(self.wall / self.res))), 1.0)
        return g

    def _block(self, g, x0, y0, x1, y1):
        occ.fill_rect_world(g, x0, y0, x1, y1, self.res, 1.0)

    # -- 垂直车位 ---------------------------------------------------------- #
    def _sample_perp(self, rng) -> Optional[Scenario]:
        depth = self.lot.perp_depth
        width = self.lot.perp_width
        slot_front = self.wh - self.wall - depth          # 车位开口 y
        # 车位横向位置(留出邻车空间)
        sx0 = float(rng.uniform(self.wall + 2.5, self.ww - self.wall - width - 2.5))
        g = self._new_grid()
        # 左右邻车(停着的车, 占据车位深度)
        if rng.random() < self.neighbor_p:
            nw = float(rng.uniform(1.6, 2.2))
            self._block(g, sx0 - nw, slot_front, sx0, self.wh - self.wall)
        if rng.random() < self.neighbor_p:
            nw = float(rng.uniform(1.6, 2.2))
            self._block(g, sx0 + width, slot_front, sx0 + width + nw, self.wh - self.wall)
        # 目标: 槽内, 车头朝内(+y)
        gy = float(rng.uniform(slot_front + 1.2, self.wh - self.wall - 3.4))
        gx = float(sx0 + width / 2.0 + rng.uniform(-0.3, 0.3))
        goal = (gx, gy, math.pi / 2.0)
        # 起点: 行车通道内, 朝向任意
        sy = float(rng.uniform(self.wall + 1.5, max(self.wall + 2.0, slot_front - 1.5)))
        sx = float(rng.uniform(self.wall + 2.0, self.ww - self.wall - 2.0))
        sth = float(rng.uniform(-math.pi, math.pi))
        start = (sx, sy, sth)
        slot = np.array([[sx0, slot_front], [sx0 + width, slot_front],
                         [sx0 + width, self.wh - self.wall], [sx0, self.wh - self.wall]])
        return self._finish(g, start, goal, "perp", slot)

    # -- 平行车位 ---------------------------------------------------------- #
    def _sample_par(self, rng) -> Optional[Scenario]:
        length = self.lot.par_length
        width = self.lot.par_width
        curb_y0 = self.wall                                # 路缘带下沿
        curb_y1 = self.wall + width                        # 路缘带上沿
        px0 = float(rng.uniform(self.wall + 2.5, self.ww - self.wall - length - 2.5))
        g = self._new_grid()
        # 前后邻车(沿路缘)
        if rng.random() < self.neighbor_p:
            nl = float(rng.uniform(3.5, 4.6))
            self._block(g, px0 - nl, curb_y0, px0, curb_y1)
        if rng.random() < self.neighbor_p:
            nr = float(rng.uniform(3.5, 4.6))
            self._block(g, px0 + length, curb_y0, px0 + length + nr, curb_y1)
        # 目标: 槽内, 车头沿路缘(0 或 pi)
        heading = 0.0 if rng.random() < 0.5 else math.pi
        if abs(heading) < 1e-6:
            gx = float(rng.uniform(px0 + 1.0, px0 + length - 3.4))
        else:
            gx = float(rng.uniform(px0 + 3.2, px0 + length - 1.0))
        gy = float(curb_y0 + width / 2.0 + rng.uniform(-0.3, 0.3))
        goal = (gx, gy, heading)
        # 起点: 路缘带上方的通道内
        sy = float(rng.uniform(curb_y1 + 1.5, self.wh - self.wall - 1.5))
        sx = float(rng.uniform(self.wall + 2.0, self.ww - self.wall - 2.0))
        sth = float(rng.uniform(-math.pi, math.pi))
        start = (sx, sy, sth)
        slot = np.array([[px0, curb_y0], [px0 + length, curb_y0],
                         [px0 + length, curb_y1], [px0, curb_y1]])
        return self._finish(g, start, goal, "par", slot)

    # -- 收尾: 校验 + 打包 ------------------------------------------------- #
    def _finish(self, g, start, goal, scene_type, slot) -> Optional[Scenario]:
        if self.v.collides(g, start, self.res) or self.v.collides(g, goal, self.res):
            return None
        if math.hypot(start[0] - goal[0], start[1] - goal[1]) < 2.0:
            return None
        return Scenario(grid=g, res=self.res,
                        start_pose=np.asarray(start, dtype=np.float64),
                        goal_pose=np.asarray(goal, dtype=np.float64),
                        scene_type=scene_type,
                        bbox=(0.0, 0.0, self.ww, self.wh),
                        slot_polygon=slot)

    # -- 接口 -------------------------------------------------------------- #
    def sample(self, rng: np.random.Generator) -> Optional[Scenario]:
        st = self.scene_types[int(rng.integers(0, len(self.scene_types)))]
        if st == "perp":
            return self._sample_perp(rng)
        return self._sample_par(rng)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gen = LotScenarioGenerator()
    cnt = {"perp": 0, "par": 0}
    for i in range(20):
        sc = None
        for _ in range(20):
            sc = gen.sample(rng)
            if sc is not None:
                break
        if sc is None:
            continue
        cnt[sc.scene_type] += 1
        if i < 4:
            print("[%d] type=%s start=%s goal=%s occ=%.3f"
                  % (i, sc.scene_type, np.round(sc.start_pose, 2),
                     np.round(sc.goal_pose, 2), sc.grid.mean()))
    print("sampled:", cnt)
