#!/usr/bin/env bash
# Phase 5 (M13): 隔离两个变量 + 组合, 对照现有 conv@N80(lat_sdf) 基线
set -e
cd /home/t/projects/diffusion_planner_demo
export PYTHONPATH=/home/t/projects/diffusion_planner_demo
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
STEPS=30000
BATCH=512

echo "==== [1/3] conv @ N80 + vae-only (隔离 map_cond) ===="
$PY -m parking.train --npz $NPZ --denoiser conv --map-cond vae --cond lat \
    --steps $STEPS --batch $BATCH --out parking/cache/diffusion_parking_conv_n80_vae.pt

echo "==== [2/3] trans @ N80 + lat_sdf (隔离架构) ===="
$PY -m parking.train --npz $NPZ --denoiser trans --map-cond lat_sdf --cond lat \
    --steps $STEPS --batch $BATCH --out parking/cache/diffusion_parking_trans_n80.pt

echo "==== [3/3] trans @ N80 + vae-only (用户目标组合) ===="
$PY -m parking.train --npz $NPZ --denoiser trans --map-cond vae --cond lat \
    --steps $STEPS --batch $BATCH --out parking/cache/diffusion_parking_trans_n80_vae.pt

echo "==== ALL TRAIN DONE ===="
