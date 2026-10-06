# -*- coding: utf-8 -*-
"""泊车数据集生成: 场景 -> HA* -> 定长 N -> 缓存 npz; 多进程并行 + 成功率统计。

npz 字段(世界坐标, 便于 render/eval 直接用; 归一化在训练侧做):
  maps        (M,2,H,W) uint8   ch0=占据(0/1), ch1=SDF(0..255, /255 还原 [0,1])
  start_pose  (M,3)     float32 世界后轴位姿
  goal_pose   (M,3)     float32
  traj        (M,N,4)   float32 [x, y, cos, sin] 世界坐标
  gear        (M,N)     int8
  scene_type  (M,)      uint8   0=perp, 1=par
  length      (M,)      float32 路径弧长(m)
  n_switches  (M,)      int8
  split       (M,)      uint8   0=train, 1=eval
  bbox        (4,)      float32 (x0,y0,x1,y1) 归一化用全局包围盒
另存 sidecar .json 记录配置与成功率等统计。

并行: multiprocessing.Pool, 每个 worker 独立 rng/规划器; 主进程先 warmup 触发 numba
  编译并写盘缓存, worker 直接读缓存(避免重复 JIT)。tqdm 按 chunk 进度显示。
"""

import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from .config import ParkingConfig, default_config
from .vehicle import Vehicle
from .scenarios import LotScenarioGenerator
from .hybrid_a_star import HybridAStar
from .postprocess import FixedNPostprocessor
from . import occupancy as occ

SDF_CLIP = 6.0
TYPE2CODE = {"perp": 0, "par": 1}
CODE2TYPE = {v: k for k, v in TYPE2CODE.items()}


# --------------------------------------------------------------------------- #
# worker: 生成 count 条样本
# --------------------------------------------------------------------------- #
def _worker(task):
    seed, count, veh_cfg, lot_cfg, ha_cfg, n_wp, scene_types = task
    rng = np.random.default_rng(seed)
    veh = Vehicle(veh_cfg)
    gen = LotScenarioGenerator(vehicle=veh, lot=lot_cfg, scene_types=scene_types)
    planner = HybridAStar(veh, ha_cfg, lot_cfg.res)
    pp = FixedNPostprocessor(n=n_wp)
    res = lot_cfg.res

    maps, starts, goals, trajs, gears, types, lens, sws = [], [], [], [], [], [], [], []
    ok = 0
    attempts = 0
    max_attempts = count * 40
    while ok < count and attempts < max_attempts:
        attempts += 1
        sc = gen.sample(rng)
        if sc is None:
            continue
        tr = planner.plan(sc)
        if tr is None:
            continue
        ft = pp.process(tr, scenario=sc)
        if ft is None:
            continue
        g = np.asarray(sc.grid, dtype=np.float64)
        sdf = occ.normalized_sdf(g, res, SDF_CLIP)          # [0,1]
        m = np.stack([(g > 0).astype(np.uint8),
                      np.clip(sdf * 255.0, 0, 255).astype(np.uint8)], axis=0)
        maps.append(m)
        starts.append(sc.start_pose.astype(np.float32))
        goals.append(sc.goal_pose.astype(np.float32))
        trajs.append(ft.traj4.astype(np.float32))
        gears.append(ft.gear.astype(np.int8))
        types.append(TYPE2CODE.get(ft.scene_type, 0))
        lens.append(np.float32(ft.length))
        sws.append(np.int8(ft.n_switches))
        ok += 1

    if ok == 0:
        return None
    return dict(maps=np.stack(maps), start_pose=np.stack(starts),
                goal_pose=np.stack(goals), traj=np.stack(trajs),
                gear=np.stack(gears), scene_type=np.asarray(types, np.uint8),
                length=np.asarray(lens, np.float32),
                n_switches=np.asarray(sws, np.int8),
                ok=ok, attempts=attempts)


def _cat(parts):
    keys = ["maps", "start_pose", "goal_pose", "traj", "gear", "scene_type",
            "length", "n_switches"]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}, \
        sum(p["ok"] for p in parts), sum(p["attempts"] for p in parts)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def build_dataset(cfg: ParkingConfig = None, n: int = None, n_proc: int = None,
                  out_path: str = None, seed: int = None, verbose: bool = True):
    cfg = cfg or default_config()
    n = cfg.data.n_samples if n is None else int(n)
    seed = cfg.data.seed if seed is None else int(seed)
    n_proc = (cpu_count() if cfg.data.n_proc == 0 else cfg.data.n_proc) if n_proc is None else int(n_proc)
    n_proc = max(1, min(n_proc, cpu_count()))
    out_path = out_path or os.path.join(cfg.data.cache_dir, "parking_%d.npz" % n)

    # warmup: 触发 numba 编译并写盘缓存, worker 读缓存
    veh = Vehicle(cfg.vehicle)
    gen = LotScenarioGenerator(vehicle=veh, lot=cfg.lot, scene_types=cfg.data.scene_types)
    planner = HybridAStar(veh, cfg.hastar, cfg.lot.res)
    rng0 = np.random.default_rng(seed)
    for _ in range(5):
        sc = gen.sample(rng0)
        if sc is not None and planner.plan(sc) is not None:
            break

    # 任务切分
    base = n // n_proc
    rem = n % n_proc
    tasks = []
    for p in range(n_proc):
        cnt = base + (1 if p < rem else 0)
        if cnt <= 0:
            continue
        tasks.append((seed + 1000 * p + p, cnt, cfg.vehicle, cfg.lot, cfg.hastar,
                      cfg.diffusion.n_wp, cfg.data.scene_types))

    t0 = time.time()
    if n_proc == 1 or len(tasks) == 1:
        results = [_worker(t) for t in tqdm(tasks, desc="gen", disable=not verbose)]
    else:
        with Pool(n_proc) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, tasks),
                                total=len(tasks), desc="gen", disable=not verbose))
    parts = [r for r in results if r is not None]
    if not parts:
        raise RuntimeError("no samples generated (all plans failed?)")
    data, ok, attempts = _cat(parts)
    dt = time.time() - t0

    # 截断到 n + 打乱 + 划分 train/eval
    M = data["traj"].shape[0]
    rr = np.random.default_rng(seed + 7)
    perm = rr.permutation(M)
    if M > n:
        perm = perm[:n]
        M = n
    data = {k: v[perm] for k, v in data.items()}
    n_eval = int(round(M * cfg.data.eval_ratio))
    split = np.zeros(M, dtype=np.uint8)
    split[:n_eval] = 1
    data["split"] = split
    data["bbox"] = np.asarray([0.0, 0.0, cfg.lot.world_w, cfg.lot.world_h], np.float32)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **data)
    stats = dict(n=M, n_eval=n_eval, ok=ok, attempts=attempts,
                 success_rate=ok / max(attempts, 1), seconds=round(dt, 1),
                 per_sample_s=round(dt / max(ok, 1), 4), n_proc=n_proc,
                 n_wp=cfg.diffusion.n_wp, scene_types=list(cfg.data.scene_types),
                 vehicle=cfg.vehicle.__dict__, lot=cfg.lot.__dict__)
    with open(out_path.replace(".npz", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    if verbose:
        print("dataset: M=%d (eval=%d) success=%.3f (%d/%d) %.1fs (%.4fs/sample, %d proc)"
              % (M, n_eval, stats["success_rate"], ok, attempts, dt,
                 stats["per_sample_s"], n_proc))
        print("saved:", out_path)
    return out_path, stats


def load_dataset(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    # maps uint8 -> float32 (M,2,H,W): ch0 占据 0/1, ch1 SDF [0,1]
    m = d["maps"]
    d["maps"] = np.stack([(m[:, 0] > 0).astype(np.float32),
                          m[:, 1].astype(np.float32) / 255.0], axis=1)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="小样本冒烟")
    a = ap.parse_args()
    n = 200 if a.quick else a.n
    build_dataset(default_config(), n=n, n_proc=(2 if a.quick else a.proc),
                  seed=a.seed, out_path=a.out)
