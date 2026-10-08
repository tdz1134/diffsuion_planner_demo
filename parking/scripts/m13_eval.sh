#!/usr/bin/env bash
# M13 评估: 4 个模型同一 harness, 引导关闭(c-scale=0, 默认), 诚实原始指标
cd /home/t/projects/diffusion_planner_demo
export PYTHONPATH=/home/t/projects/diffusion_planner_demo
PY=venv_py38/bin/python
NPZ=cache/parking_12000_n80.npz

run () {  # name ckpt vizout
  echo "############ EVAL $1 ($2) ############"
  $PY -m parking.evaluate --ckpt "$2" --npz $NPZ --out "figs/parking/$3" 2>&1 \
      | tee "cache/m13_eval_$1.txt"
}

run base      parking/cache/diffusion_parking_conv_n80.pt       m13_base_conv_latsdf.png
run conv_vae  parking/cache/diffusion_parking_conv_n80_vae.pt   m13_conv_vae.png
run trans_lat parking/cache/diffusion_parking_trans_n80.pt      m13_trans_latsdf.png
run trans_vae parking/cache/diffusion_parking_trans_n80_vae.pt  m13_trans_vae.png
echo "############ ALL EVAL DONE ############"
