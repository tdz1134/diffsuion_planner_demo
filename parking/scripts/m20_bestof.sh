#!/usr/bin/env bash
# ============================================================================
# M20: 测试时 Best-of-N 采样-择优 — 复用 M19 trans3+spatial 权重, 不重训
#   每场景采 N 条 DDPM 样本 -> 各跑 M19b 联合投影 -> 按投影目标(终点+碰撞)打分 -> 逐场景取最优。
#   纯推断, 前台跑(吃 CUDA)。碰撞率随 N 上升(算力换质量): N=8→~0.75, N=16→~0.80。
#
#   默认(N=16):          bash parking/scripts/m20_bestof.sh
#   扫 1/8/16:           SWEEP=1 bash parking/scripts/m20_bestof.sh
#   改 N:                BO=32 bash parking/scripts/m20_bestof.sh
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
CK=parking/cache/diffusion_parking_trans3_spatial_n80.pt
K=${K:-60}
ITERS=${ITERS:-120}; WCOLL=${WCOLL:-50}; MARGIN=${MARGIN:-0.20}
[ -f "$CK" ] || { echo "缺 M19 权重: $CK  (先 bash parking/scripts/m19_run.sh)"; exit 1; }

eval_bo(){   # $1=best_of
  local bo=$1
  $PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --goal-refine $ITERS \
      --refine-coll $WCOLL --refine-margin $MARGIN --best-of $bo \
      --out figs/parking/m20_bo${bo}.png > cache/m20_bo${bo}.txt 2>&1
}

if [ "${SWEEP:-0}" = "1" ]; then BOS=(1 8 16); else BOS=(${BO:-16}); fi
for bo in "${BOS[@]}"; do echo "==== [EVAL] best_of=$bo ===="; eval_bo $bo; done

echo "==================== Best-of-N 对比(碰撞率随 N 上升, 端点同步变好) ===================="
printf "%-8s %8s %8s %8s %8s %8s\n" best_of rawColl rawFeas endErr gearCF gearSw
for bo in "${BOS[@]}"; do
  f=cache/m20_bo${bo}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-8s %8s %8s %8s %8s %8s\n" "$bo" \
    "$(g raw_collision_free)" "$(g raw_feasible)" "$(g mean_end_pos_err_m)" \
    "$(g gear_curv_feasible)" "$(g pred_gear_switches)"
done
echo "[参考] M19b(N=1)=0.47  M17=0.083  专家天花板=0.917"
echo "[done] 图 figs/parking/m20_bo*.png, 明细 cache/m20_bo*.txt"
