#!/usr/bin/env bash
# ============================================================================
# M17: 控制空间扩散 (曲率 κ / 带符号弧长 u -> 运动学正向积分, 可行性由构造保证)
#   在你自己的终端里【前台】运行(能吃 CUDA; 别放后台/沙箱, 否则退 CPU 且易 OOM)。
#
#   默认(30k 步, 与已训 M14/M15 对齐):        bash parking/scripts/m17_run.sh
#   快速探针(~18min):                        STEPS=12000 bash parking/scripts/m17_run.sh
#   只评估不训练(权重已存在时, 秒级):          EVAL_ONLY=1 bash parking/scripts/m17_run.sh
#   评估场景更少更快:                         K=30 bash parking/scripts/m17_run.sh
#   换归一尺度:                               U_SCALE=0.5 bash parking/scripts/m17_run.sh
# 训练/评估都有 tqdm 进度条。
#   看点: raw_feasible(非完整)与 gear_curv_feasible(曲率)能否 ~1.0(构造即满足);
#         代价 mean_end_pos_err_m(端点不钉)与 pred_gear_switches(u 通道档位抖动)。
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
CK_CTRL=parking/cache/diffusion_parking_ctrl_n80.pt           # M17 控制空间
CK_GEAR=parking/cache/diffusion_parking_trans2_n80_gear.pt    # M15 eps+gear 对照
CK_T2=parking/cache/diffusion_parking_trans2_n80.pt           # M14 无gear 对照

if [ "$EVAL_ONLY" != "1" ]; then
  echo "==================== [TRAIN] ctrl(控制空间) trans2 ($STEPS 步, u_scale=$U_SCALE) ===================="
  $PY -m parking.train --npz $NPZ --denoiser trans2 --representation ctrl \
      --pred-mode eps --u-scale $U_SCALE --cond lat \
      --steps $STEPS --batch 512 --out $CK_CTRL
fi

for pair in "CTRL $CK_CTRL" "M15-eps+gear $CK_GEAR" "M14-trans2 $CK_T2"; do
  name=$(echo $pair | cut -d' ' -f1); ck=$(echo $pair | cut -d' ' -f2)
  [ -f "$ck" ] || { echo "[skip] $name 权重不存在: $ck"; continue; }
  echo "==================== [EVAL] $name ($ck) ===================="
  $PY -m parking.evaluate --ckpt $ck --npz $NPZ --k $K --out figs/parking/m17_${name}.png \
      > cache/m17_eval_${name}.txt 2>&1
done

echo "==================== 对比(ctrl: 曲率/非完整应构造满足; 端点误差是代价) ===================="
printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" model slip rawFeas gearCF expCF repSucc gearSw endErr
for name in M14-trans2 M15-eps+gear CTRL; do
  f=cache/m17_eval_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_mean_slip)" "$(g raw_feasible)" "$(g gear_curv_feasible)" \
    "$(g expert_gear_curv_feasible)" "$(g rep_success)" "$(g pred_gear_switches)" \
    "$(g mean_end_pos_err_m)"
done
echo "[done] 图 figs/parking/m17_*.png, 明细 cache/m17_eval_*.txt"
