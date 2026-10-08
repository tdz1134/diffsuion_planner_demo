#!/usr/bin/env bash
# M14: faithful Transformer (trans2) fair test vs conv@N80 baseline (same cond=lat_sdf, N=80)
set -e
cd /home/t/projects/diffusion_planner_demo
export PYTHONPATH=/home/t/projects/diffusion_planner_demo
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz
echo "==== train trans2 @ N80 + lat_sdf ===="
$PY -m parking.train --npz $NPZ --denoiser trans2 --cond lat \
    --steps 30000 --batch 512 --out parking/cache/diffusion_parking_trans2_n80.pt
echo "==== eval trans2 ===="
$PY -m parking.evaluate --ckpt parking/cache/diffusion_parking_trans2_n80.pt --npz $NPZ \
    --out figs/parking/m14_trans2_n80.png 2>&1 | tee cache/m14_eval_trans2.txt
echo "==== ALL M14 DONE ===="
