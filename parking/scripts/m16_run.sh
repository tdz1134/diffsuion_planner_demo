#!/usr/bin/env bash
# ============================================================================
# M16 (part 2): x0-prediction 重参数化 + x0-空间可行性损失
#   在你自己的终端里【前台】运行(能吃 CUDA; 别放后台/沙箱, 否则退 CPU 且易 OOM)。
#
#   默认(严谨, 与已训 control 对齐 30k):        bash parking/scripts/m16_run.sh
#   快速探针(训练砍到 1.2w 步, ~18min; 注意 control 是 30k 不完全可比):
#                                               STEPS=12000 bash parking/scripts/m16_run.sh
#   只评估不训练(先拿 control/M15/M14 对比, 秒级):  EVAL_ONLY=1 bash parking/scripts/m16_run.sh
#   评估场景更少更快:                              K=30 bash parking/scripts/m16_run.sh
# 训练/评估都有 tqdm 进度条。
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
STEPS=${STEPS:-30000}          # 训练步数(默认 30000 与 control 对齐)
K=${K:-60}                     # 评估场景数
EVAL_ONLY=${EVAL_ONLY:-0}      # 1=跳过训练只评估
CK_CTR=parking/cache/diffusion_parking_x0_n80_gear.pt          # control(已训)
CK_LSS=parking/cache/diffusion_parking_x0_n80_gear_loss.pt     # method
CK_EPS=parking/cache/diffusion_parking_trans2_n80_gear.pt      # M15 eps+gear 对照
CK_T2=parking/cache/diffusion_parking_trans2_n80.pt           # M14 无gear 对照

if [ "$EVAL_ONLY" != "1" ]; then
  echo "==================== [TRAIN] x0-pred + gear + 曲率/平滑损失 ($STEPS 步) ===================="
  $PY -m parking.train --npz $NPZ --denoiser trans2 --use-gear --pred-mode x0 \
      --w-curv-x0 0.05 --w-smooth 0.01 --cond lat \
      --steps $STEPS --batch 512 --out $CK_LSS
fi

for pair in "CTRL $CK_CTR" "METHOD $CK_LSS" "M15-eps+gear $CK_EPS" "M14-trans2 $CK_T2"; do
  name=$(echo $pair | cut -d' ' -f1); ck=$(echo $pair | cut -d' ' -f2)
  [ -f "$ck" ] || { echo "[skip] $name 权重不存在: $ck"; continue; }
  echo "==================== [EVAL] $name ($ck) ===================="
  $PY -m parking.evaluate --ckpt $ck --npz $NPZ --k $K --out figs/parking/m16_${name}.png \
      > cache/m16_eval_${name}.txt 2>&1
done

echo "==================== 对比(关注 rep_success 与 gear_curv_feasible; 专家自检应偏高) ===================="
printf "%-16s %8s %8s %8s %8s %8s %8s\n" model slip segK rep_succ gearCF expCF gearAcc
for name in M14-trans2 M15-eps+gear CTRL METHOD; do
  f=cache/m16_eval_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-16s %8s %8s %8s %8s %8s %8s\n" "$name" \
    "$(g raw_mean_slip)" "$(g raw_seg_kappa)" "$(g rep_success)" \
    "$(g gear_curv_feasible)" "$(g expert_gear_curv_feasible)" "$(g gear_acc)"
done
echo "[done] 图 figs/parking/m16_*.png, 明细 cache/m16_eval_*.txt"
