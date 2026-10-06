# -*- coding: utf-8 -*-
"""混合 A*(Hybrid A*), 实现 MotionPlanner 接口。

设计要点:
  - 状态 = 连续后轴位姿 (x, y, theta); 图节点按 (ix, iy, itheta) 离散, 展平成一维索引,
    用**数组**(而非 dict)存 g 值/父指针/位姿, 便于 numba JIT。
  - 运动基元 = 自行车模型, 离散转角 x {前进, 倒车}, 每条固定弧长 prim_step, 分 n_sub 子步。
  - 碰撞 = 车辆足迹采样点(旋转矩形)逐子步查占据栅格(越界视为障碍)。
  - 启发 = max(欧氏, 2D 有障碍 holonomic 场) * weight(可采纳下界, 见 heuristics)。
  - open 表 = 手写二叉最小堆(numba 无 heapq); closed = uint8 数组; 惰性删除(弹出时判重)。
  - 代价 = 弧长(前进/倒车权重) + 换档惩罚 + 转角突变惩罚。

性能: 整个搜索(扩展 + 碰撞 + 堆)在一个 @njit 函数里, 未装 numba 时自动退化为纯 Python
  (仍正确, 只是慢)。目标: 单条泊车 < ~0.2s, 支撑数据集批量生成(见 dataset.py)。

说明: 早期版本用 Reeds-Shepp 解析扩展直连 goal(见 reeds_shepp.py, 仍保留可用)。numba 化后
  纯图搜索已足够快(百万级展开/秒), 命中"目标容差球"即终止; 端点精确对齐交给 postprocess
  (钉住首尾位姿), 故核心不再依赖 RS。需要时可在 Python 侧用 reeds_shepp_path 做尾段精修。
"""

import math
from typing import Optional

import numpy as np

from .interfaces import MotionPlanner, Scenario, Trajectory
from .config import HAStarConfig
from .vehicle import Vehicle
from .heuristics import holonomic_field
from ._jit import optional_njit, NUMBA_AVAILABLE

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# numba kernel: 位姿离散 / 启发 / 二叉堆
# --------------------------------------------------------------------------- #
@optional_njit
def _wrap_n(a):
    return a - TWO_PI * math.floor((a + math.pi) / TWO_PI)


@optional_njit
def _sidx(x, y, th, res, H, W, n_theta):
    """世界位姿 -> 展平状态索引 (ix*H + iy)*n_theta + itheta。越界坐标夹到边界。"""
    ix = int(round(x / res))
    iy = int(round(y / res))
    if ix < 0:
        ix = 0
    elif ix > W - 1:
        ix = W - 1
    if iy < 0:
        iy = 0
    elif iy > H - 1:
        iy = H - 1
    ith = int(round(_wrap_n(th) / TWO_PI * n_theta)) % n_theta
    if ith < 0:
        ith += n_theta
    return (ix * H + iy) * n_theta + ith


@optional_njit
def _heur(x, y, gx, gy, field, res, H, W, weight):
    """h = weight * max(欧氏, holonomic 场)。场为 inf(障碍/不可达)时退回欧氏。"""
    eu = math.hypot(x - gx, y - gy)
    c = int(round(x / res))
    r = int(round(y / res))
    f = eu
    if 0 <= r < H and 0 <= c < W:
        fv = field[r, c]
        if fv < 1e300 and fv > eu:
            f = fv
    return weight * f


@optional_njit
def _heap_push(hf, hi, n, f, idx):
    hf[n] = f
    hi[n] = idx
    c = n
    while c > 0:
        p = (c - 1) >> 1
        if hf[p] <= hf[c]:
            break
        tf = hf[p]; hf[p] = hf[c]; hf[c] = tf
        ti = hi[p]; hi[p] = hi[c]; hi[c] = ti
        c = p
    return n + 1


@optional_njit
def _heap_pop(hf, hi, n):
    f = hf[0]; idx = hi[0]
    n -= 1
    hf[0] = hf[n]; hi[0] = hi[n]
    p = 0
    while True:
        l = 2 * p + 1; r = 2 * p + 2; m = p
        if l < n and hf[l] < hf[m]:
            m = l
        if r < n and hf[r] < hf[m]:
            m = r
        if m == p:
            break
        tf = hf[p]; hf[p] = hf[m]; hf[m] = tf
        ti = hi[p]; hi[p] = hi[m]; hi[m] = ti
        p = m
    return f, idx, n


# --------------------------------------------------------------------------- #
# numba kernel: 主搜索
# --------------------------------------------------------------------------- #
@optional_njit
def hastar_search(grid, field, fp, kappa, ds_sub, steer_idx, gear, straight, R_arr,
                  n_sub, sx, sy, sth, gx, gy, gth, res, H, W, n_theta,
                  prim_step, fwd_cost, rev_cost, gear_switch_penalty, steer_change_penalty,
                  heur_weight, pos_tol, ang_tol, max_expand, zero_steer_idx,
                  path_pose, path_steer, path_gear):
    """在占据图上跑混合 A*。命中 goal 容差球即停, 回溯节点级路径写入 path_* 缓冲。

    返回路径节点数 K(>=1); 失败返回 -1。
    """
    size = W * H * n_theta
    gcost = np.full(size, np.inf)
    came = np.full(size, -1, np.int64)
    closed = np.zeros(size, np.uint8)
    nx_a = np.zeros(size)
    ny_a = np.zeros(size)
    nth_a = np.zeros(size)
    nsteer = np.zeros(size, np.int8)
    ngear = np.zeros(size, np.int8)

    C = kappa.shape[0]
    P = fp.shape[0]
    heap_cap = 2000000
    hf = np.empty(heap_cap)
    hi = np.empty(heap_cap, np.int64)
    heap_n = 0

    skey = _sidx(sx, sy, sth, res, H, W, n_theta)
    gcost[skey] = 0.0
    nx_a[skey] = sx; ny_a[skey] = sy; nth_a[skey] = sth
    nsteer[skey] = zero_steer_idx; ngear[skey] = 1
    heap_n = _heap_push(hf, hi, heap_n,
                        _heur(sx, sy, gx, gy, field, res, H, W, heur_weight), skey)

    expand = 0
    goal_key = -1
    while heap_n > 0 and expand < max_expand:
        f, key, heap_n = _heap_pop(hf, hi, heap_n)
        if closed[key] == 1:
            continue
        closed[key] = 1
        expand += 1
        x = nx_a[key]; y = ny_a[key]; th = nth_a[key]

        if math.hypot(x - gx, y - gy) <= pos_tol and abs(_wrap_n(th - gth)) <= ang_tol:
            goal_key = key
            break

        cur_steer = nsteer[key]; cur_gear = ngear[key]
        cg = gcost[key]
        for c in range(C):
            k = kappa[c]; dstep = ds_sub[c]; Rc = R_arr[c]; is_str = straight[c]
            xi = x; yi = y; thi = th
            collide = False
            for j in range(n_sub):
                if is_str:
                    xi = xi + dstep * math.cos(thi)
                    yi = yi + dstep * math.sin(thi)
                else:
                    thn = thi + k * dstep
                    xi = xi + Rc * (math.sin(thn) - math.sin(thi))
                    yi = yi + Rc * (math.cos(thi) - math.cos(thn))
                    thi = thn
                cti = math.cos(thi); sti = math.sin(thi)
                for p in range(P):
                    wx = xi + fp[p, 0] * cti - fp[p, 1] * sti
                    wy = yi + fp[p, 0] * sti + fp[p, 1] * cti
                    cc = int(round(wx / res)); rr = int(round(wy / res))
                    if rr < 0 or rr >= H or cc < 0 or cc >= W or grid[rr, cc] > 0.0:
                        collide = True
                        break
                if collide:
                    break
            if collide:
                continue
            nkey = _sidx(xi, yi, thi, res, H, W, n_theta)
            if closed[nkey] == 1:
                continue
            if gear[c] > 0:
                move = prim_step * fwd_cost
            else:
                move = prim_step * rev_cost
            pen = 0.0
            if gear[c] != cur_gear:
                pen += gear_switch_penalty
            if steer_idx[c] != cur_steer:
                pen += steer_change_penalty
            ng = cg + move + pen
            if ng < gcost[nkey]:
                gcost[nkey] = ng
                nx_a[nkey] = xi; ny_a[nkey] = yi; nth_a[nkey] = thi
                nsteer[nkey] = steer_idx[c]; ngear[nkey] = gear[c]
                came[nkey] = key
                if heap_n < heap_cap:
                    heap_n = _heap_push(hf, hi, heap_n,
                                        ng + _heur(xi, yi, gx, gy, field, res, H, W, heur_weight),
                                        nkey)

    if goal_key < 0:
        return -1

    # 回溯节点级路径(start..goal), 写入 path_* 缓冲
    K = 0
    kk = goal_key
    while kk != -1:
        K += 1
        kk = came[kk]
    maxK = path_pose.shape[0]
    if K > maxK:
        K = maxK
    kk = goal_key
    pos = K - 1
    while kk != -1 and pos >= 0:
        path_pose[pos, 0] = nx_a[kk]; path_pose[pos, 1] = ny_a[kk]; path_pose[pos, 2] = nth_a[kk]
        path_steer[pos] = nsteer[kk]; path_gear[pos] = ngear[kk]
        pos -= 1
        kk = came[kk]
    return K


# --------------------------------------------------------------------------- #
# Python 封装: 实现 MotionPlanner
# --------------------------------------------------------------------------- #
class HybridAStar(MotionPlanner):
    def __init__(self, vehicle: Vehicle, cfg: HAStarConfig = None, res: float = 0.2,
                 collision_ds: float = 0.15, pos_tol: float = 0.35, ang_tol: float = 0.25):
        self.v = vehicle
        self.cfg = cfg or HAStarConfig()
        self.res = res
        self.collision_ds = collision_ds
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.n_theta = self.cfg.n_theta
        self.steers = np.array([f * vehicle.cfg.max_steer for f in self.cfg.steer_fracs],
                               dtype=np.float64)
        self.zero_steer_idx = int(np.argmin(np.abs(self.steers)))

        # -- 预计算所有 (档位, 转角) 组合的基元常量(外层 gear, 内层 steer) --
        gears_list = (1, -1) if self.cfg.allow_reverse else (1,)
        self.n_sub = max(2, int(math.ceil(self.cfg.prim_step / collision_ds)))
        kap, dss, sii, gg = [], [], [], []
        for g in gears_list:
            for si, steer in enumerate(self.steers):
                kap.append(math.tan(steer) / vehicle.cfg.wheelbase)
                dss.append(self.cfg.prim_step * g / self.n_sub)
                sii.append(si)
                gg.append(g)
        self._kappa = np.ascontiguousarray(np.asarray(kap, np.float64))
        self._ds_sub = np.ascontiguousarray(np.asarray(dss, np.float64))
        self._steer_idx = np.ascontiguousarray(np.asarray(sii, np.int8))
        self._gear = np.ascontiguousarray(np.asarray(gg, np.int8))
        self._straight = np.ascontiguousarray(np.abs(self._kappa) < 1e-9)
        safe_k = np.where(self._straight, 1.0, self._kappa)
        self._R = np.ascontiguousarray(np.where(self._straight, 1.0, 1.0 / safe_k))
        self._fp = np.ascontiguousarray(vehicle.footprint_local)
        self._maxK = 16384
        self.last_expansions = -1

    # -- 主入口 ------------------------------------------------------------ #
    def plan(self, scenario: Scenario) -> Optional[Trajectory]:
        grid = np.ascontiguousarray(np.asarray(scenario.grid, dtype=np.float64))
        res = float(scenario.res)
        H, W = grid.shape
        start = tuple(float(v) for v in scenario.start_pose)
        goal = tuple(float(v) for v in scenario.goal_pose)

        if self.v.collides(grid, start, res) or self.v.collides(grid, goal, res):
            return None

        grc = (int(round(goal[1] / res)), int(round(goal[0] / res)))
        field = np.ascontiguousarray(holonomic_field(grid, grc, res))

        path_pose = np.empty((self._maxK, 3), dtype=np.float64)
        path_steer = np.empty(self._maxK, dtype=np.int8)
        path_gear = np.empty(self._maxK, dtype=np.int8)

        K = hastar_search(
            grid, field, self._fp, self._kappa, self._ds_sub, self._steer_idx, self._gear,
            self._straight, self._R, self.n_sub,
            start[0], start[1], start[2], goal[0], goal[1], goal[2],
            res, H, W, self.n_theta,
            self.cfg.prim_step, self.cfg.fwd_cost, self.cfg.rev_cost,
            self.cfg.gear_switch_penalty, self.cfg.steer_change_penalty,
            self.cfg.heur_weight, self.pos_tol, self.ang_tol, self.cfg.max_nodes,
            self.zero_steer_idx, path_pose, path_steer, path_gear)

        if K < 0:
            return None
        self.last_expansions = K
        poses, gears = self._densify(path_pose[:K], path_steer[:K], path_gear[:K], start)
        n_sw = int((np.diff(gears.astype(np.int16)) != 0).sum())
        return Trajectory(poses=poses, gear=gears, cost=float(len(poses)),
                          n_switches=n_sw, planner="hybrid_a_star")

    # -- 节点级路径 -> 密集位姿(重放每条基元) ---------------------------- #
    def _densify(self, pp, ps, pg, start):
        wb = self.v.cfg.wheelbase
        i_arr = np.arange(1, self.n_sub + 1, dtype=np.float64)
        segs = [np.asarray(start, dtype=np.float64).reshape(1, 3)]
        gl = []
        for i in range(1, pp.shape[0]):
            x0, y0, th0 = float(pp[i - 1, 0]), float(pp[i - 1, 1]), float(pp[i - 1, 2])
            steer = float(self.steers[int(ps[i])])
            g = int(pg[i])
            k = math.tan(steer) / wb
            dstep = self.cfg.prim_step * g / self.n_sub
            if abs(k) < 1e-9:
                xs = x0 + dstep * i_arr * math.cos(th0)
                ys = y0 + dstep * i_arr * math.sin(th0)
                ths = np.full(self.n_sub, th0)
            else:
                R = 1.0 / k
                ths = th0 + k * dstep * i_arr
                xs = x0 + R * (np.sin(ths) - math.sin(th0))
                ys = y0 + R * (math.cos(th0) - np.cos(ths))
            segs.append(np.stack([xs, ys, ths], axis=1))
            gl.append(np.full(self.n_sub, g, dtype=np.int8))
        poses = np.concatenate(segs, axis=0)
        gears = np.concatenate(gl) if gl else np.ones(1, dtype=np.int8)
        return poses, gears


if __name__ == "__main__":
    # 单车位自检: 造一个垂直车位, 从通道泊入
    import time
    from .config import VehicleConfig, LotConfig, HAStarConfig
    from . import occupancy as occ

    print("numba available:", NUMBA_AVAILABLE)
    lot = LotConfig()
    res = lot.res
    g = np.zeros((lot.H, lot.W), dtype=np.float64)
    occ.fill_border(g, 1)
    occ.fill_rect_world(g, 6.0, lot.world_h - 6.0, 6.4, lot.world_h - 1.0, res)   # 左邻车
    occ.fill_rect_world(g, 9.1, lot.world_h - 6.0, 9.5, lot.world_h - 1.0, res)   # 右邻车
    veh = Vehicle(VehicleConfig())
    planner = HybridAStar(veh, HAStarConfig(), res)
    start = (5.0, 5.0, 0.0)                        # 通道内, 车头朝 +x
    goal = (7.75, lot.world_h - 3.5, math.pi / 2)  # 车位内, 车头朝 +y

    sc = Scenario(grid=g, res=res, start_pose=np.array(start),
                  goal_pose=np.array(goal), scene_type="perp")

    # 预热(JIT 编译) + 正式计时
    t0 = time.time()
    tr0 = planner.plan(sc)
    dt_warm = time.time() - t0
    t1 = time.time()
    tr = planner.plan(sc)
    dt = time.time() - t1

    if tr is None:
        print("plan FAILED (None)  warm=%.3fs" % dt_warm)
    else:
        print("plan OK: poses=%s gear=%s switches=%d  warm(JIT)=%.2fs  hot=%.3fs"
              % (tr.poses.shape, tr.gear.shape, tr.n_switches, dt_warm, dt))
        print("  start->", np.round(tr.poses[0], 3), " end->", np.round(tr.poses[-1], 3),
              " goal->", np.round(goal, 3))
        print("  end err: pos=%.3fm ang=%.1fdeg"
              % (math.hypot(tr.poses[-1, 0] - goal[0], tr.poses[-1, 1] - goal[1]),
                 math.degrees(abs(float((tr.poses[-1, 2] - goal[2] + math.pi) % (2 * math.pi) - math.pi)))))
