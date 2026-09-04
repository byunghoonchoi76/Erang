# 이랑 SR Refiner — 위성 초해상화 후처리 리파이너 벤치마크

위성영상 초해상화 파이프라인의 **후처리 리파이너**(CUT / CycleGAN / UNSB / BBDM)를 동일 조건에서 비교하고,
실제 위성 도메인에 적용해 배포 운영점을 도출한 프로젝트입니다.

파이프라인: **DINOv3(라우팅) → RRDBNet(1차 SR) → HAT(2차 정밀 SR, ×4) → Refiner(후처리)**

> 결론 요약: 합성 열화 벤치에서 **CUT이 4대 지표 전반 1위**(UNSB 2위, CycleGAN 3위, BBDM 붕괴).
> 실 도메인에서는 CUT을 직접 적용하면 색 편이가 발생하나 **잔차 블렌드(α≈0.125)** 또는 **CUT y-only**로 경량 적용 시 SR 베이스라인 대비 지각 품질(LPIPS) 개선.

---

## 리포지토리 구조 (GitHub에 올리는 것)

```
code/
  src/              # 아키텍처·데이터·지표·평가 (arch_rrdb, data_bridge, metrics, eval_report, ...)
  configs/          # paths.yaml 등 경로/설정
  scripts/          # 학습·평가·노트북 생성 스크립트 (run_cut_*.sh, eval_*.py, make_*_notebook.py)
notebooks/          # 실험·평가 노트북 (cut_*, eval_*)
README.md
.gitignore
```

**GitHub에는 코드·설정·노트북만 올립니다.** 가중치·데이터·평가 산출물은 용량이 커서(합계 ~50GB) git에 넣지 않고 아래 경로에서 받습니다.

---

## 가중치 · 데이터 (Git 제외 — Google Drive)

Drive 폴더: `이랑 프로젝트 / 이랑_SR_Refiner_코드공유_20260728`

| 자산 | 파일 | 크기 | 위치 | 놓는 곳 |
|---|---|---|---|---|
| RRDB (1차 SR) | `A2_v5b_rrdbnet_best.pth` | 22 MB | Drive `weights/` | 추론 코드의 RRDB 가중치 경로 |
| DINOv3 라우터 | `classified_DINOv3-Large_best.pth` | 1.16 GB | Drive `weights/` | 라우팅 단계 |
| CUT 리파이너 | `final_net_G.pth` | 44 MB | Drive `weights/` 또는 서버 `refiners/cut_src/checkpoints/bench_cut_3348/` | CUT 추론 |
| CUT 변형 | `best_G_AB.pth`, `latest_net_G(no-sc).pth`, `latest_net_G(vgg-sc).pth` | 각 43 MB | Drive `weights/` | SC-loss 실험용 |
| 4모델 출력 샘플 | `share_4model/` | — | Drive | 정성 비교용 (input/CUT/CycleGAN/UNSB/BBDM/GT) |

> `paths.yaml`(0 byte 템플릿)에 각자 환경의 가중치·데이터 경로를 채워 쓰면 됩니다.

---

## UNSB · BBDM (대용량 — Git 제외, 별도 보관)

두 프레임워크는 **체크포인트 + 잠재공간(VQGAN) 가중치**가 커서 git에 올리지 않습니다.
아래 표대로 원 저장소를 clone한 뒤, 체크포인트만 지정 위치에 놓으면 재현됩니다.

### UNSB (Unpaired Neural Schrödinger Bridge) — 약 0.5 GB
- 프레임워크: `cyclomon/UNSB` fork (CUT 코드베이스 기반)
- 사용 체크포인트(추론): `checkpoints/bench_unsb_3348/final_net_G.pth`
  - 학습 시 함께 저장: `final_net_{D,E,F}.pth`, `latest_net_{G,D,E,F}.pth` (추론엔 `final_net_G.pth`만 필요)
- 놓는 곳: `<UNSB>/checkpoints/bench_unsb_3348/`

```
UNSB/
  checkpoints/
    bench_unsb_3348/
      final_net_G.pth        ← 추론에 필요 (약 45 MB)
      final_net_D.pth  final_net_E.pth  final_net_F.pth   (학습 산출물)
      latest_net_*.pth                                    (중간 저장)
```

### BBDM (Brownian Bridge Diffusion Model) — 약 36 GB (전체), 필수 약 4 GB
- 프레임워크: `xuekt98/BBDM` fork (Latent BBDM, f4)
- 설정: `configs/lbbdm_3348_f4.yaml`
- 사용 체크포인트(추론):
  - LBBDM: `results/lbbdm_3348/LBBDM-f4/checkpoint/top_model_epoch_10.pth` (약 2 GB)
  - VQGAN(잠재공간): `results/VQGAN/model.ckpt`
- 놓는 곳: 아래 구조 그대로

```
BBDM/
  configs/lbbdm_3348_f4.yaml
  results/
    VQGAN/model.ckpt                                         ← 잠재공간 (필수)
    lbbdm_3348/LBBDM-f4/checkpoint/
      top_model_epoch_10.pth                                 ← 추론에 필요 (약 2 GB)
      config.yaml
      (top/latest/last_optim_sche*.pth 는 학습 재개용 — 추론 불필요, 각 약 2 GB)
```

> BBDM은 `top_optim_sche_*`, `latest_model_*`, `last_*` 등 옵티마이저·중간 스냅샷이 대부분(개당 2 GB)이라 전체가 36 GB입니다.
> **재현에는 `VQGAN/model.ckpt` + `top_model_epoch_10.pth` + `lbbdm_3348_f4.yaml` 3개면 충분**합니다.

---

## 환경 · 실행

- 환경: `bbdm_src/environment.yml`(BBDM), CUT/UNSB는 CUT 계열 의존성(PyTorch, dominate, visdom 등)
- 평가: `code/scripts/eval_*.py`, `code/src/eval_report.py` (PSNR/SSIM/LPIPS/FID/NIQE + 보라% 프록시)
- 노트북: `notebooks/`의 `cut_*_experiment.ipynb`, `eval_*.ipynb`

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

## 참고
- 상세 실험 기록·정성 이미지: Drive `이랑_SR_Refiner_코드공유_20260728/최종실험/`, `runs/`
- 중간논문(대한전자공학회): 별도 공유
