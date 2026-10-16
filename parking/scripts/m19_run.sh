#!/usr/bin/env bash
# ============================================================================
# M19: 让网络"从地图图像学避障" — 复用冻结 VAE latent 栅格作 cross-attn 空间 token
#   新增 SpatialMapConditioner(--cond spatial) + trans3 去噪器(--denoiser trans3)。
#   仍用 ctrl 控制空间表示(运动学可行由构造保证)。默认全 flag 门控, 不动旧管道。
#
#   假设: M17/M18b 碰撞率仅 ~0.07 是因为地图被 flatten 成 64 维全局向量、网络"看不见"
#         障碍空间位置。给回 9x16 空间栅格 -> 专家数据本就无碰撞 -> 碰撞率应明显上升。
#
#   完整训练(前台跑, 吃 CUDA, 约 40min):   bash parking/scripts/m19_run.sh
#   只评估已训好的 ckpt:                   EVAL_ONLY=1 bash parking/scripts/m19_run.sh
#   改步数/场景数:                         STEPS=20000 K=60 bash parking/scripts/m19_run.sh
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
CK=parking/cache/diffusion_parking_trans3_spatial_n80.pt
STEPS=${STEPS:-30000}
K=${K:-60}
ITERS=${ITERS:-80}          # M18b 测试时终点投影迭代数(端点仍靠它压到 ~cm)

if [ "${EVAL_ONLY:-0}" != "1" ]; then
  echo "==================== [TRAIN] ctrl + trans3 + spatial (${STEPS} 步) ===================="
  $PY -m parking.train --representation ctrl --denoiser trans3 --cond spatial \
      --u-scale 0.4 --steps $STEPS --out $CK
fi
[ -f "$CK" ] || { echo "缺权重: $CK"; exit 1; }

echo "==================== [EVAL raw] M19 trans3+spatial (无投影) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K \
    --out figs/parking/m19_raw.png > cache/m19_raw.txt 2>&1
echo "==================== [EVAL refine] M19 trans3+spatial + M18b 终点投影($ITERS) ===================="
$PY -m parking.evaluate --ckpt $CK --npz $NPZ --k $K --goal-refine $ITERS \
    --out figs/parking/m19_refine.png > cache/m19_refine.txt 2>&1

echo "==================== 对比(M19 看点: rawCollisionFree 能否从 M18b 的 0.067 抬起来) ===================="
printf "%-12s %8s %8s %8s %8s %8s %8s\n" model rawColl rawFeas gearCF repSucc endErr gearSw
for name in raw refine; do
  f=cache/m19_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-12s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_collision_free)" "$(g raw_feasible)" "$(g gear_curv_feasible)" \
    "$(g rep_success)" "$(g mean_end_pos_err_m)" "$(g pred_gear_switches)"
done
echo "[baseline M18b]  rawColl=0.067 rawFeas=1.0 gearCF=0.733 repSucc=0.133 endErr=0.025 gearSw=2.58"
echo "[ceiling 专家]   collision_free=0.917"
echo "[done] 图 figs/parking/m19_*.png, 明细 cache/m19_{raw,refine}.txt"
