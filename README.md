# Erang — 위성 초해상화 파이프라인 & 후처리 리파이너 벤치마크

위성영상 초해상화(SR) 파이프라인과, 마지막 단계인 **후처리 리파이너**(CUT / CycleGAN / UNSB / BBDM)를
동일 조건에서 비교하고 실제 위성 도메인에 적용해 배포 운영점을 도출한 프로젝트입니다.

파이프라인: **DINOv3(라우팅) → RRDBNet(1차 SR) → HAT(2차 정밀 SR, ×4) → Refiner(후처리)**

> 결론 요약: 합성 열화 벤치에서 **CUT이 4대 지표 전반 1위**(UNSB 2위, CycleGAN 3위, BBDM 붕괴).
> 실 도메인에서는 CUT을 직접 적용하면 색 편이가 발생하나 **잔차 블렌드(α≈0.125)** 또는 **CUT y-only**로 경량 적용 시 SR 베이스라인 대비 지각 품질(LPIPS) 개선.

---

## 리포지토리 구조

```
pipeline/            # 상위 백본 (팀 공통)
  dinov3_model.py      # DINOv3 라우터 (5-class)
  rrdb_model_vr2.py    # RRDBNet (1차 SR)
  hat/                 # HAT 아키텍처·데이터·모델·학습 (2차 정밀 SR)
refiners/            # 후처리 리파이너 코드 (CUT/CycleGAN/UNSB/BBDM glue·평가)
  src/                 # arch_rrdb, data_bridge, metrics, eval_report, upstream, ...
  configs/paths.yaml   # 경로/설정 템플릿 (각자 환경에 맞게 채움)
  scripts/             # 학습·평가·재현·노트북 생성 스크립트
notebooks/           # 실험·평가 노트북 (출력 제거, 경량)
data/                # sr_models_summary.csv (모델 요약표)
README.md
.gitignore
```

**GitHub에는 코드·설정·노트북만 올립니다.** 가중치·데이터·평가 산출물은 용량이 커서(합계 ~50 GB)
git에 넣지 않고 아래 Google Drive 경로에서 받습니다.

---

## 가중치 · 데이터 (Git 제외 — Google Drive)

Drive 폴더: `이랑 프로젝트 / 이랑_SR_Refiner_코드공유_20260728`

| 자산 | 파일 | 크기 | 위치 | 용도 |
|---|---|---|---|---|
| RRDB (1차 SR) | `A2_v5b_rrdbnet_best.pth` | 22 MB | Drive `weights/` | RRDBNet 가중치 |
| DINOv3 라우터 | `classified_DINOv3-Large_best.pth` | 1.16 GB | Drive `weights/` | 라우팅 단계 |
| CUT 리파이너 | `final_net_G.pth` | 44 MB | Drive `weights/` | CUT 추론 |
| CUT 변형 | `best_G_AB.pth`, `latest_net_G(no-sc).pth`, `latest_net_G(vgg-sc).pth` | 각 43 MB | Drive `weights/` | SC-loss 실험 |
| 4모델 출력 샘플 | `share_4model/` | — | Drive | 정성 비교 (input/CUT/CycleGAN/UNSB/BBDM/GT) |

> `refiners/configs/paths.yaml`에 각자 환경의 가중치·데이터 경로를 채워 씁니다.
> HAT(2m→0.5m) 가중치 등 일부는 기업 공유 자산으로, 미제공 환경에서는 해당 경로를 비워 두면 됩니다.

---

## UNSB · BBDM 프레임워크 (대용량 — Git 제외, 별도 보관)

`refiners/`에는 리파이너 **glue·평가 코드**가 있고, UNSB/BBDM의 **원 프레임워크와 체크포인트**는
용량(잠재공간·VQGAN 포함)이 커서 git에 올리지 않습니다. 아래대로 원 저장소를 clone하고 체크포인트만 배치하면 재현됩니다.

### UNSB (Unpaired Neural Schrödinger Bridge) — 약 0.5 GB
- 프레임워크: `cyclomon/UNSB` fork (CUT 코드베이스 기반)
- 추론 체크포인트: `checkpoints/bench_unsb_3348/final_net_G.pth` (약 45 MB)
  - 학습 산출물(추론 불필요): `final_net_{D,E,F}.pth`, `latest_net_{G,D,E,F}.pth`

```
UNSB/checkpoints/bench_unsb_3348/
  final_net_G.pth        ← 추론에 필요
  final_net_{D,E,F}.pth  latest_net_*.pth   (학습 산출물)
```

### BBDM (Brownian Bridge Diffusion Model) — 약 36 GB (전체), 필수 약 4 GB
- 프레임워크: `xuekt98/BBDM` fork (Latent BBDM, f4)
- 설정: `configs/lbbdm_3348_f4.yaml`
- 추론 체크포인트:
  - LBBDM: `results/lbbdm_3348/LBBDM-f4/checkpoint/top_model_epoch_10.pth` (약 2 GB)
  - VQGAN(잠재공간): `results/VQGAN/model.ckpt`

```
BBDM/
  configs/lbbdm_3348_f4.yaml
  results/
    VQGAN/model.ckpt                                    ← 잠재공간 (필수)
    lbbdm_3348/LBBDM-f4/checkpoint/
      top_model_epoch_10.pth                            ← 추론에 필요 (약 2 GB)
      config.yaml
      (top/latest/last_optim_sche*.pth 는 학습 재개용 — 추론 불필요, 각 약 2 GB)
```

> BBDM은 옵티마이저·중간 스냅샷(개당 2 GB)이 대부분이라 전체 36 GB입니다.
> **재현에는 `VQGAN/model.ckpt` + `top_model_epoch_10.pth` + `lbbdm_3348_f4.yaml` 3개면 충분**합니다.

---

## 벤치마크 결과 (요약)

합성 열화 정합셋(3348장, 동일 10 epoch·단일 GPU):

| 모델 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | 순위 |
|---|---|---|---|---|---|
| **CUT** | **19.68** | **0.580** | **0.395** | **102.7** | 1 |
| UNSB | 18.85 | 0.522 | 0.506 | 118.4 | 2 |
| CycleGAN | 17.62 | 0.442 | 0.459 | 149.1 | 3 |
| BBDM | 15.91 | 0.273 | 0.671 | 352.0 | 4 |

실 도메인(실제 0.5 m GT 30장): SR 단독 대비 `+CUT+blend(α=0.125)`에서 LPIPS 0.077→0.035(55%↓).

---

## 환경 · 실행

- 환경: `pipeline/hat/requirements.txt`(HAT), BBDM은 원 저장소 `environment.yml`, CUT/UNSB는 CUT 계열 의존성
- 평가: `refiners/scripts/eval_*.py`, `refiners/src/eval_report.py` (PSNR/SSIM/LPIPS/FID/NIQE + 보라% 프록시)
- 노트북: `notebooks/`의 `cut_*_experiment.ipynb`, `eval_*.ipynb`

## 참고
- 상세 실험 기록·정성 이미지: Drive `이랑_SR_Refiner_코드공유_20260728/최종실험/`, `runs/`
- 중간논문(대한전자공학회): 별도 공유
