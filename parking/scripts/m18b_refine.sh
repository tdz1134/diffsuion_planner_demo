#!/usr/bin/env bash
# ============================================================================
# M18(b): 测试时终点投影(test-time goal projection) — 复用 M17 ctrl 权重, 不重训
#   纯推断优化, 前台跑(吃 CUDA)。约 1~2min(采样 60 场景 + 每场景 80 步投影)。
#   看点: M17-raw 的 endErr ~1.74m 能否被 M18b 拉到 ~0.02m, 且 rawFeas/gearCF 保持。
#
#   默认:            bash parking/scripts/m18b_refine.sh
#   调投影迭代/权重:  ITERS=150 WPOS=1.0 WANG=0.5 bash parking/scripts/m18b_refine.sh
#   场景更少更快:    K=30 bash parking/scripts/m18b_refine.sh
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
CK=parking/cache/diffusion_parking_ctrl_n80.pt          # M17 ctrl 权重(不重训)
K=${K:-60}
ITERS=${ITERS:-80}
WPOS=${WPOS:-1.0}
WANG=${WANG:-0.3}
[ -f "$CK" ] || { echo "缺 M17 ctrl 权重: $CK"; exit 1; }

echo "==================== [EVAL raw] M17 ctrl(无投影) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --out figs/parking/m18b_raw.png \
    > cache/m18b_raw.txt 2>&1
echo "==================== [EVAL refine] M17 ctrl + 测试时终点投影($ITERS 步, wpos=$WPOS wang=$WANG) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --goal-refine $ITERS \
    --refine-wpos $WPOS --refine-wang $WANG --out figs/parking/m18b_refine.png \
    > cache/m18b_refine.txt 2>&1

echo "==================== 对比(投影应把 endErr 压到接近 0 且不动 feasible) ===================="
printf "%-10s %8s %8s %8s %8s %8s %8s\n" model slip rawFeas gearCF repSucc endErr gearSw
for name in raw refine; do
  f=cache/m18b_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-10s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_mean_slip)" "$(g raw_feasible)" "$(g gear_curv_feasible)" \
    "$(g rep_success)" "$(g mean_end_pos_err_m)" "$(g pred_gear_switches)"
done
echo "[done] 图 figs/parking/m18b_*.png, 明细 cache/m18b_{raw,refine}.txt"
