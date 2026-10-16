#!/usr/bin/env bash
# ============================================================================
# M19b: 测试时"终点+避障"联合投影 — 复用 M19 trans3+spatial 权重, 不重训
#   在 M18b 的 refine_ctrl_to_goal 里叠加可微足迹-SDF 碰撞代价(经 integrate_ctrl, κ 每步钳位)。
#   纯推断优化, 前台跑(吃 CUDA)。约 1~2min(采样 60 场景 + 每场景投影迭代)。
#   看点: 只终点投影把碰撞拖到 ~0.08; 联合投影应把 raw_collision_free 抬到 ~0.45+ 且端点仍 <0.3m。
#
#   默认(均衡档):        bash parking/scripts/m19b_refine.sh
#   偏碰撞档:            WCOLL=50 MARGIN=0.35 ITERS=300 bash parking/scripts/m19b_refine.sh
#   场景更少更快:        K=20 bash parking/scripts/m19b_refine.sh
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
CK=parking/cache/diffusion_parking_trans3_spatial_n80.pt       # M19 权重(不重训)
K=${K:-60}
ITERS=${ITERS:-120}
WCOLL=${WCOLL:-50}          # 碰撞代价权重(0=退化为纯 M18b 只终点投影)
MARGIN=${MARGIN:-0.20}      # 足迹安全间隙(m)
WPOS=${WPOS:-1.0}
WANG=${WANG:-0.3}
[ -f "$CK" ] || { echo "缺 M19 权重: $CK  (先 bash parking/scripts/m19_run.sh 训练)"; exit 1; }

echo "==================== [EVAL endpoint-only] 只终点投影(对照) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --goal-refine $ITERS \
    --refine-wpos $WPOS --refine-wang $WANG --out figs/parking/m19b_endonly.png \
    > cache/m19b_endonly.txt 2>&1
echo "==================== [EVAL joint] 终点+避障联合投影(w_coll=$WCOLL margin=$MARGIN) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --goal-refine $ITERS \
    --refine-coll $WCOLL --refine-margin $MARGIN --refine-wpos $WPOS --refine-wang $WANG \
    --out figs/parking/m19b_joint.png > cache/m19b_joint.txt 2>&1

echo "==================== 对比(联合投影应大幅抬升碰撞率, 端点仍可控) ===================="
printf "%-14s %8s %8s %8s %8s %8s %8s\n" model rawColl rawFeas gearCF endErr endAng gearSw
for name in endonly joint; do
  f=cache/m19b_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-14s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_collision_free)" "$(g raw_feasible)" "$(g gear_curv_feasible)" \
    "$(g mean_end_pos_err_m)" "$(g mean_end_ang_err_deg)" "$(g pred_gear_switches)"
done
echo "[参考] M17 raw碰撞=0.083  M18b只终点=0.067  专家天花板=0.917"
echo "[done] 图 figs/parking/m19b_*.png, 明细 cache/m19b_{endonly,joint}.txt"
