#!/usr/bin/env bash
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate trellis
cd ~/geoit_sr_refiner/refiners/cut_src
echo "[$(date +%H:%M:%S)] CUT+DINOv3-SC 본학습 시작 (sc_backend=dinov3, lambda_SC=1.0, 35 epoch)"
python train.py --dataroot ../data/refine_big --name cut_big_dsc --model cutsc --CUT_mode CUT \
  --sc_backend dinov3 --lambda_SC 1.0 --display_id 0 --gpu_ids 0 --batch_size 1 \
  --n_epochs 25 --n_epochs_decay 10 --load_size 512 --crop_size 256 \
  --save_epoch_freq 2 --save_latest_freq 1500 --print_freq 200
echo "[$(date +%H:%M:%S)] ===== CUT+DINOv3-SC 본학습 완료 ====="
