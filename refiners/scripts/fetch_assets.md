# 자산 다운로드 가이드 (Google Drive → 서버)

노션 "기업 제공 데이터 인덱스(2026-07-23)"의 Drive 링크 기준. `gdown` 사용.
`conda activate geosr && pip install gdown` 후 `~/geoit_sr_refiner`에서 실행.

## 🟢 확보 가중치 (즉시 다운로드)
```bash
mkdir -p weights data
# RRDBNet A2V5B 최종 가중치 (23MB)
gdown 1LnRbXRJe-n38RlcNi1YsT3Gp4piZrreQ -O weights/A2_v5b_rrdbnet_best.pth
# RRDB 학습 로그
gdown 1LKpksie-FHcBnHqGF9tebL4y30yh5ysZ -O weights/rrdb_train_log.jsonl
# CycleGAN 학습 상태/설정
gdown 15TLkPvVTvIP1DwEkD6rwMdlCdp-H8kXr -O weights/cyclegen_state_8.json
# CycleGAN best_G_AB — 폴더형(unzipped .pth) → 폴더째 받기
gdown --folder 1JxUAPQdm-QNR0eqBs9vdTU-wW7H_qOP9 -O weights/best_G_AB_folder
#   ⚠️ 폴더 내부가 data.pkl+data/+version 구조(압축해제된 .pth). 로드 시 재압축 or torch.load 경로 조정 필요.
```

## 데이터 샘플 (파이프 검증용)
```bash
# sr_data_sample (27.8MB): train/{hr,lr} + test
gdown --folder 1vA_mdiIFgzBY9HnyODdmpeeGcdCTpHSB -O data/sr_data_sample
```

## 설명자료 (참고)
```bash
gdown 119WkqMzscM02fTX8hA42SDVOc02g1kfz -O weights/SRtoHR_RRDB_설명자료.txt
gdown 1bH41QAb85JUh5JpHePD9bv7-8qVRNUNo -O weights/SRtoHR_CycleGen_설명자료.txt
```

## 🔴 미수령 (기업 공유 대기 — 받는 즉시 배치)
- `net_g_270000.pth` (HAT 가중치) → `weights/net_g_270000.pth`  **← 1주차 임계경로**
- `classified_DINOv3-Large/best_checkpoint.pth` → `weights/classified_DINOv3-Large_best.pth`
- 0.5m GT / 2m·0.5m 페어 10,200쌍 → `data/paired_10200/{hr,lr}`
- (선택) `goit_supperres_data.zip` 13GB 내부 PNG 전처리본 — SR_Output 페어 소재 확인

## 다운로드 후
```bash
python scripts/smoke_test.py                 # 아키텍처/로드 점검
python -m src.baseline_infer --input data/sr_data_sample/test --out runs/baseline_cyclegan
```
