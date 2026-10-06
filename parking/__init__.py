# -*- coding: utf-8 -*-
"""
parking: 基于 Hybrid A* 专家数据的泊车轨迹扩散规划器(解耦/接口分离)。

模块边界见 PARKING_NOTES.md。子模块:
  config / geometry / vehicle / occupancy / interfaces   -- 基础件
  reeds_shepp / heuristics / hybrid_a_star               -- 运动规划(HA*)
  scenarios / postprocess / dataset                      -- 数据生成
  conditioner / diffusion / train_vae / train / infer    -- 学习
  evaluate / render                                      -- 评估与可视化
"""

__all__ = [
    "config", "geometry", "vehicle", "occupancy", "interfaces",
    "reeds_shepp", "heuristics", "hybrid_a_star",
    "scenarios", "postprocess", "dataset",
    "conditioner", "diffusion", "evaluate", "render",
]
