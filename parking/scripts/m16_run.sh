#!/usr/bin/env bash
# ============================================================================
# M16 (part 2) 长任务: x0-prediction 重参数化 + x0-空间可行性损失
#   在你自己的终端里【前台】运行(能吃到 CUDA; 别放后台/沙箱, 否则退 CPU 且易 OOM):
#       bash parking/scripts/m16_run.sh
#   训练/评估都有 tqdm 进度条。预计: 训练~30min + 4 个评估每个~2min。
#   只想跳过训练(若 method 权重已存在)可把下面 TRAIN 段注释掉。
# ============================================================================
set -e
cd "$(dirname "$0")/../.."            # 回到仓库根
export PYTHONPATH="$PWD"
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
CK_CTR=parking/cache/diffusion_parking_x0_n80_gear.pt          # control(已训)
CK_LSS=parking/cache/diffusion_parking_x0_n80_gear_loss.pt     # method(本脚本训)
CK_EPS=parking/cache/diffusion_parking_trans2_n80_gear.pt      # M15 eps+gear 对照
CK_T2=parking/cache/diffusion_parking_trans2_n80.pt           # M14 无gear 对照

echo "==================== [TRAIN] x0-pred + gear + 曲率/平滑损失 (30k) ===================="
$PY -m parking.train --npz $NPZ --denoiser trans2 --use-gear --pred-mode x0 \
    --w-curv-x0 0.05 --w-smooth 0.01 --cond lat \
    --steps 30000 --batch 512 --out $CK_LSS

for pair in "CTRL $CK_CTR" "METHOD $CK_LSS" "M15-eps+gear $CK_EPS" "M14-trans2 $CK_T2"; do
  name=$(echo $pair | cut -d' ' -f1); ck=$(echo $pair | cut -d' ' -f2)
  echo "==================== [EVAL] $name ($ck) ===================="
  $PY -m parking.evaluate --ckpt $ck --npz $NPZ --out figs/parking/m16_${name}.png \
      > cache/m16_eval_${name}.txt 2>&1
  grep -E "raw_collision_free|raw_mean_slip|raw_seg_kappa|raw_gear_switches|rep_success|gear_curv_feasible|expert_gear_curv_feasible|gear_acc" cache/m16_eval_${name}.txt || true
done

echo "==================== 对比(关注 rep_success 与 gear_curv_feasible; 专家自检应≈高) ===================="
printf "%-16s %8s %8s %8s %8s %8s\n" model slip segK rep_succ gearCF gearAcc
for name in M14-trans2 M15-eps+gear CTRL METHOD; do
  f=cache/m16_eval_${name}.txt; [ -f "$f" ] || continue
  g(){ grep -aE "^  $1 " "$f" | awk '{print $2}'; }
  printf "%-16s %8s %8s %8s %8s %8s\n" "$name" "$(g raw_mean_slip)" "$(g raw_seg_kappa)" "$(g rep_success)" "$(g gear_curv_feasible)" "$(g gear_acc)"
done
echo "[done] 图在 figs/parking/m16_*.png, 明细在 cache/m16_eval_*.txt"
