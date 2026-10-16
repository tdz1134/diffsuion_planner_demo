#!/usr/bin/env bash
# ============================================================================
# M18(a): 控制空间 + 终点→goal 可微到达惩罚(经 integrate_ctrl, 不碰曲率病态)
#   目的: 修 M17 的遗留短板——端点误差 1.74m / 航向 14°(ctrl 无法硬钉首尾)。
#   在你自己的终端里【前台】运行(能吃 CUDA; 别放后台/沙箱, 否则退 CPU 且易 OOM)。
#
#   默认(30k 步, 约 60min):                  bash parking/scripts/m18_run.sh
#   快速探针(~25min):                        STEPS=12000 bash parking/scripts/m18_run.sh
#   只评估不训练(M18 权重已存在时, 秒级):      EVAL_ONLY=1 bash parking/scripts/m18_run.sh
#   评估场景更少更快:                         K=30 bash parking/scripts/m18_run.sh
#   调惩罚权重:                              GOAL_POS=1.5 GOAL_ANG=0.4 bash parking/scripts/m18_run.sh
# 看点: M18 行的 mean_end_pos_err_m 能否从 M17 的 ~1.74m 明显下降(且不破坏 raw_feasible=1.0/
#        gear_curv_feasible 的构造可行); 端点更准通常也会抬升 rep_collision_free。
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
STEPS=${STEPS:-30000}
K=${K:-60}
EVAL_ONLY=${EVAL_ONLY:-0}
U_SCALE=${U_SCALE:-0.4}
GOAL_POS=${GOAL_POS:-1.0}       # 终点位置到达惩罚权重(归一化坐标差平方)
GOAL_ANG=${GOAL_ANG:-0.3}       # 终点航向到达惩罚权重(1-cos dθ)
CK_M18=parking/cache/diffusion_parking_ctrl_n80_goal.pt     # M18 ctrl+goal
CK_M17=parking/cache/diffusion_parking_ctrl_n80.pt          # M17 ctrl(基线, 已提交)

if [ "$EVAL_ONLY" != "1" ]; then
  echo "==================== [TRAIN] ctrl + goal 到达惩罚 ($STEPS 步, w_pos=$GOAL_POS w_ang=$GOAL_ANG) ===================="
  $PY -m parking.train --npz $NPZ --denoiser trans2 --representation ctrl \
      --pred-mode eps --u-scale $U_SCALE --w-goal-pos $GOAL_POS --w-goal-ang $GOAL_ANG \
      --cond lat --steps $STEPS --batch 512 --out $CK_M18
fi

echo "==================== [EVAL] M17-ctrl(基线) vs M18-ctrl+goal ===================="
for pair in "M17-ctrl $CK_M17" "M18-ctrl+goal $CK_M18"; do
  name=$(echo $pair | cut -d' ' -f1); ck=$(echo $pair | cut -d' ' -f2)
  [ -f "$ck" ] || { echo "[skip] $name 权重不存在: $ck"; continue; }
  $PY -m parking.evaluate --ckpt $ck --npz $NPZ --k $K --out figs/parking/m18_${name}.png \
      > cache/m18_eval_${name}.txt 2>&1
done

echo "==================== 对比(goal 惩罚应压低 endErr, 保住 feasible/gearCF) ===================="
printf "%-16s %8s %8s %8s %8s %8s %8s\n" model slip rawFeas gearCF repColl endErr gearSw
for name in M17-ctrl M18-ctrl+goal; do
  f=cache/m18_eval_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-16s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_mean_slip)" "$(g raw_feasible)" "$(g gear_curv_feasible)" \
    "$(g rep_collision_free)" "$(g mean_end_pos_err_m)" "$(g pred_gear_switches)"
done
echo "[done] 图 figs/parking/m18_*.png, 明细 cache/m18_eval_*.txt"
