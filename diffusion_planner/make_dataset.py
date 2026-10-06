# -*- coding: utf-8 -*-
"""
数据集生成 (独立脚本, 最耗时的一步)
====================================
用随机占据栅格 + A*/加权A*/贪心 专家路径, 生成条件扩散规划器的训练数据,
保存为 dataset_<M>.npz。地图为 2 通道: [占据(0/1), 归一化距离场]。

单独跑它(带 tqdm 进度条), 生成一次后即可反复训练而无需重跑 A*:
    python make_dataset.py --data 20000
    python make_dataset.py --data 1500          # 给 --quick 用的迷你数据
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import grid_env as ge

GRID = 64
N_WP = 32
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def rc_to_norm(rc, size=GRID):
    """栅格 (r, c) -> 归一化 (x, y), x=c, y=r, 范围 [-1, 1]。"""
    r, c = rc
    return np.array([c / (size - 1) * 2 - 1, r / (size - 1) * 2 - 1], dtype=np.float32)


def generate(M=20000, seed=0, out_dir=OUT_DIR, overwrite=False):
    """生成并缓存数据集, 返回文件路径。"""
    ds_path = os.path.join(out_dir, f"dataset_{M}.npz")
    if os.path.exists(ds_path) and not overwrite:
        print(f"[cache] 已存在 {ds_path}  (加 --overwrite 可重新生成)")
        return ds_path

    rng = np.random.default_rng(seed)
    grids, starts, goals, paths = [], [], [], []
    planners = [("astar", 1.0), ("astar", 1.5), ("astar", 2.0), ("greedy", 0.0)]
    made = 0
    pbar = tqdm(total=M, desc="数据集", unit="张", ncols=90)
    while made < M:
        name, w = planners[rng.integers(len(planners))]
        s = ge.make_sample(GRID, GRID, N_WP, rng=rng, planner=name, w=w)
        if s is None:
            continue
        occ = s["grid"]
        sdf = ge.distance_field(occ) / GRID
        grids.append(np.stack([occ, sdf], axis=0))     # (2,H,W)
        starts.append(rc_to_norm(s["start"]))
        goals.append(rc_to_norm(s["goal"]))
        paths.append(s["path_norm"])
        made += 1
        pbar.update(1)
    pbar.close()

    np.savez_compressed(ds_path, maps=np.stack(grids), starts=np.stack(starts),
                        goals=np.stack(goals), paths=np.stack(paths))
    print(f"[saved] {ds_path}   maps={tuple(np.stack(grids).shape)}  paths={tuple(np.stack(paths).shape)}")
    return ds_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成扩散路径规划器训练数据集")
    ap.add_argument("--data", type=int, default=20000, help="样本数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--overwrite", action="store_true", help="忽略已有缓存, 重新生成")
    a = ap.parse_args()
    generate(a.data, a.seed, overwrite=a.overwrite)
