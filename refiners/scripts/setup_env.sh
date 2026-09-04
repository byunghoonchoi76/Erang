#!/usr/bin/env bash
# Erang SR Refiner 환경 구성 (SERVER, GB10 Blackwell, CUDA 13)
# 주의: basicsr==1.4.2는 구버전이라 CUDA13/Blackwell/py3.12에서 그대로 빌드되지 않을 수 있음.
#       HAT 아키텍처만 쓰려면 full basicsr 없이 필요한 모듈만 vendor하는 우회도 가능.
set -e

ENV=geosr
source ~/miniforge3/etc/profile.d/conda.sh

if ! conda env list | grep -q "/$ENV$"; then
  conda create -y -n "$ENV" python=3.10
fi
conda activate "$ENV"

# 1) PyTorch — Blackwell(sm_121) 지원 빌드로. (기존 trellis/3dgs env의 torch 버전 참고)
#    아래는 예시. 실제 설치 전 `nvidia-smi`/CUDA 버전에 맞는 nightly/cu 버전 확인 필요.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) 경량 의존성 (스캐폴드/평가에 필요)
pip install pyyaml numpy opencv-python pillow tqdm scikit-image lpips einops timm

# 3) 지리공간 (tif 처리)
pip install rasterio

# 4) 평가 지표
pip install pytorch-fid            # FID
# NIQE는 basicsr 제공 → basicsr 설치가 어려우면 piq로 대체 검토: pip install piq

# 5) HAT/basicsr (가중치 도착 후 필요) — 빌드 실패 시 아래 우회 고려
# pip install basicsr==1.4.2
#   실패 시: pipeline/archs/hat_arch.py 가 basicsr.utils.registry 만 요구하므로
#   해당 유틸만 stub 처리하거나, basicsr 최신(호환) 버전 시도.

echo "[setup] conda env '$ENV' 준비 완료. 다음: python scripts/smoke_test.py"
