# Refiners — 후보 3종 이식 (CUT / BBDM / UNSB)

CycleGAN을 대체할 단방향 Unpaired 변환 모델. 3~5주차 병렬 학습 후 1종 선정(제안서 §6).

## 공통 어댑터 인터페이스
각 refiner는 다음 인터페이스를 만족하는 래퍼(`refiners/<name>_adapter.py`)로 감싼다.
업스트림/데이터/평가 코드와 분리되도록:

```python
class RefinerAdapter:
    def train_step(self, batch) -> dict: ...        # batch={"A":SR_Output,"B":HR_GT}, returns losses
    @torch.no_grad()
    def refine(self, sr_output_tensor) -> tensor: ...  # [0,1] CHW → [0,1] CHW
    def save(self, path): ...
    def load(self, path): ...
```

`src/sc_loss.StructuralConsistencyLoss`를 각 train_step의 총손실에 `+ lambda * L_SC`로 결합.

## 이식 계획 (submodule)
```bash
cd ~/erang_sr/refiners
git submodule add https://github.com/taesungp/contrastive-unpaired-translation cut_src      # 팀원 A
git submodule add https://github.com/xuekt98/BBDM bbdm_src                                   # 팀원 B
git submodule add https://github.com/cyclomon/UNSB unsb_src                                  # 팀장
```

| 모델 | 레포 | 담당 | 이식 포인트 |
|---|---|---|---|
| CUT (ECCV'20) | taesungp/contrastive-unpaired-translation | 팀원 A | Patch NCE loss + 단일 Generator |
| BBDM (CVPR'23) | xuekt98/BBDM | 팀원 B | Brownian Bridge 샘플러 (source-target 직결) |
| UNSB (ICLR'24) | cyclomon/UNSB | 팀장 | 역방향 엔트로피 OT 솔버 |

## 위성 도메인 개조 (공통)
- 얼굴/일반물체 기본 하이퍼파라미터 → 멀티채널 위성 이미지에 맞게 LR·스케줄러 재조정
- 총손실 = L_gen + λ·L_SC (DINOv3 미수령 동안은 VGG 프로토타입, 도착 시 교체)
- 입력 = SR_Output(업스트림 동결 출력), 타겟 = HR_GT 도메인(unpaired)
