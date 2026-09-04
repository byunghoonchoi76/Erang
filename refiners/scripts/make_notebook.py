# -*- coding: utf-8 -*-
"""pipeline_baseline.ipynb 생성기 (nbformat 4, 의존성 없이 json으로)."""
import json, os

cells = []

def md(txt):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": txt})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src})

md("""# Erang SR Refiner — 파이프라인 & 베이스라인

위성영상 초해상도 파이프라인 `DINOv3 → RRDBNet → HAT → CycleGAN` 의 **동결 업스트림**을 로드하고,
교체 대상이자 대조군인 **CycleGAN Refiner 베이스라인**을 실행·시각화·평가한다.

- 서버: SERVER · 프로젝트: `~/erang_sr` · 드라이버: `src/` 모듈
- 자산 현황/블로커: **`STATUS.md`** (HAT·DINOv3 가중치, 0.5m GT 대기 중)
- **DEMO 모드**: 실데이터/가중치가 없으면 합성 샘플로 배관만 시연(라벨 명시). 자산 도착 시 그대로 실측 전환.
""")

md("## 0. 환경 설정")
code("""import os, sys, json
import torch

ROOT = os.path.expanduser("~/erang_sr")
os.chdir(ROOT)
for p in (ROOT, os.path.join(ROOT, "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
cfg = yaml.safe_load(open("configs/paths.yaml", encoding="utf-8"))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
""")

md("## 1. 자산 점검 — 무엇이 확보/미수령인가")
code("""def check_assets(cfg):
    def has(v):
        p = v if os.path.isabs(v) else os.path.join(ROOT, v)
        return os.path.exists(os.path.expanduser(p))
    print("[가중치]")
    for k, v in cfg["weights"].items():
        print(f"  {'✅' if has(v) else '🔴'} {k:16s} {v}")
    print("[데이터]")
    for k, v in cfg["data"].items():
        if isinstance(v, str) and ("root" in k):
            print(f"  {'✅' if has(v) else '🔴'} {k:16s} {v}")

check_assets(cfg)
print("\\n🔴 = 미수령/미다운로드. scripts/fetch_assets.md 및 STATUS.md 참조.")
""")

md("""## 2. 동결 업스트림 로드
`src/upstream.UpstreamPipeline` 은 확보된 모듈만 로드하고, 미수령(HAT·DINOv3)은 `None` 으로 둔다.
가중치 파일이 없으면 랜덤 초기화로 로드되어(경고 출력) DEMO 배관은 계속 진행된다.""")
code("""from src.upstream import UpstreamPipeline

up = UpstreamPipeline(cfg, device=device)
print("\\n[로드 상태]")
print("  RRDB    :", "OK" if up.rrdb is not None else "—")
print("  HAT     :", "OK" if up.hat is not None else "🔴 미수령(None) — SR_Output은 RRDB까지만")
print("  CycleGAN:", "OK" if up.cyclegan is not None else "—")
print("  DINOv3  :", "OK" if up.dinov3 is not None else "🔴 미수령(None) — SC Loss는 VGG 프로토타입")
""")

md("""## 3. 입력(LR) 준비 & SR_Output 생성
`LR → RRDB → (HAT) → SR_Output`. SR_Output이 refiner의 입력 소스다.
실데이터가 없으면 합성 그라디언트 샘플로 DEMO.""")
code("""import numpy as np
from src.data_bridge import _list_images, _read_image, _to_tensor

def get_lr_samples(n=3, size=128):
    candidates = [cfg["data"].get("sr_output_root"),
                  os.path.join(cfg["data"].get("sample_root", ""), "test"),
                  cfg["data"].get("sample_root")]
    for c in candidates:
        if not c:
            continue
        d = os.path.join(ROOT, c) if not os.path.isabs(c) else c
        if os.path.isdir(d):
            fs = _list_images(d)[:n]
            if fs:
                print(f"[data] 실샘플 {len(fs)}장 사용: {c}")
                x = torch.stack([_to_tensor(_read_image(f)) for f in fs])
                return x.to(device), False
    print("[data] ⚠️ 실데이터 없음 → 합성 DEMO 샘플 생성")
    base = torch.linspace(0, 1, size).repeat(size, 1)
    demo = torch.stack([torch.stack([base, base.flip(0), base.t()])] * n)
    demo = (demo + 0.08 * torch.rand(n, 3, size, size)).clamp(0, 1)
    return demo.to(device), True

lr, DEMO = get_lr_samples()
with torch.no_grad():
    sr_output = up.sr_output(lr)
print(f"LR {tuple(lr.shape)} → SR_Output {tuple(sr_output.shape)}  |  {'DEMO(합성)' if DEMO else 'REAL'}")
""")

md("""## 4. CycleGAN 베이스라인 리파인 (대조군)
CycleGAN Generator는 Tanh 출력이므로 입력을 `[-1,1]`로, 출력을 `[0,1]`로 환산한다.""")
code("""@torch.no_grad()
def refine_cyclegan(g, x):
    return ((g(x * 2 - 1) + 1) / 2).clamp(0, 1)

refined = refine_cyclegan(up.cyclegan, sr_output)
print("Refined:", tuple(refined.shape))
if DEMO:
    print("⚠️ DEMO: 가중치 미로딩 시 랜덤 초기화 결과 — 배관 확인용. 실가중치(best_G_AB) 로드 시 의미 있는 출력.")
""")

md("## 5. 시각화 — LR / SR_Output / Refined")
code("""try:
    import matplotlib
    import matplotlib.pyplot as plt

    def to_img(t):
        return t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)

    n = lr.shape[0]
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    ax = ax.reshape(n, 3)
    cols = ["LR (input)", "SR_Output (RRDB→HAT)", "Refined (CycleGAN)"]
    for i in range(n):
        for j, im in enumerate([lr[i], sr_output[i], refined[i]]):
            ax[i, j].imshow(to_img(im)); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(cols[j])
    plt.tight_layout()
    out = "runs/baseline_preview.png"; os.makedirs("runs", exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved:", out)
    plt.show()
except Exception as e:
    print("[viz] skip:", type(e).__name__, e)
""")

md("""## 6. 정량 평가 (Dual-Track)
- **Track1 (paired, GT 존재)**: PSNR / SSIM / LPIPS
- **Track2 (unpaired, GT 없음)**: FID / NIQE

아래는 배관 시연(refined vs SR_Output). 실제 대조군 표는 GT 확보 후 `src.eval_report`로 산정한다.""")
code("""from src import metrics as M

s, g0 = refined[0], sr_output[0]
dr = cfg["eval"]["psnr_data_range"]
print("[배관 시연 — refined vs SR_Output]")
print("  PSNR :", round(M.psnr(s, g0, dr), 3))
print("  SSIM :", M.ssim(s, g0))
try:
    lp = M.LPIPS(device=device); print("  LPIPS:", lp(s, g0))
except Exception as e:
    print("  LPIPS skip:", e)

print("\\n▶ 실제 대조군 평가(GT 확보 후):")
print("  python -m src.baseline_infer --input data/sr_data_sample/test --out runs/baseline_cyclegan")
print("  python -m src.eval_report  --refined runs/baseline_cyclegan --gt data/PAIRED_DATA/hr \\\\")
print("      --hr_domain data/hr_domain --tag baseline_cyclegan")
""")

md("""## 7. 다음 단계
1. **자산 다운로드** — `scripts/fetch_assets.md` (RRDB·CycleGAN 가중치, sr_data_sample)
2. **임계경로 해소** — HAT 가중치(`HAT_WEIGHTS.pth`)·0.5m GT 기업 요청 (STATUS.md §다음 액션)
3. **Refiner 교체** — `refiners/` 에 CUT/BBDM/UNSB submodule + 어댑터, 총손실에 `+ λ·L_SC(DINOv3)` 결합
4. **대조군 표** — baseline(CycleGAN) vs 제안 3종을 동일 규약으로 Dual-Track 비교
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.expanduser("~/erang_sr/pipeline_baseline.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("WROTE", out, "cells:", len(cells))
