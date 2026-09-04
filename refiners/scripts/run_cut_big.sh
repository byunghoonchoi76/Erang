#!/usr/bin/env bash
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate trellis
cd ~/erang_sr
D=data/extracted/sr_data_full/train_data
echo "[$(date +%H:%M:%S)] ===== 본학습 파이프라인 시작 ====="

# 1) 데이터 확장 (refine_big 없으면)
if [ "$(ls refiners/data/refine_big/trainA 2>/dev/null | wc -l)" -lt 100 ]; then
  echo "[$(date +%H:%M:%S)] 데이터 확장 (1600장, 512, 균등샘플)..."
  python refiners/prepare_cut_data.py --lr_dir "$D/LR_img" --hr_dir "$D/HR_img" \
    --out refiners/data/refine_big --limit 1600 --test_n 100 --hr_size 512 2>/dev/null
fi
echo "[$(date +%H:%M:%S)] trainA=$(ls refiners/data/refine_big/trainA | wc -l) testA=$(ls refiners/data/refine_big/testA | wc -l)"

# 2) CUT 본학습 (train.py = 체크포인트 자동저장)
cd refiners/cut_src
echo "[$(date +%H:%M:%S)] CUT 본학습 시작 (35 epoch)..."
python train.py --dataroot ../data/refine_big --name cut_big --CUT_mode CUT \
  --display_id 0 --gpu_ids 0 --batch_size 1 \
  --n_epochs 25 --n_epochs_decay 10 --load_size 512 --crop_size 256 \
  --save_epoch_freq 2 --save_latest_freq 1500 --print_freq 200
echo "[$(date +%H:%M:%S)] ===== CUT 본학습 완료 ====="
