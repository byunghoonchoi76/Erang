# -*- coding: utf-8 -*-
"""cut_experiment.ipynb 생성기 — 사용자가 셀 단위로 보면서 실행하는 CUT 실험 노트북."""
import json, os

cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# CUT Refiner 실험 (직접 실행용)

기존 CycleGAN을 대체할 후보 **CUT**을 우리 위성 SR 데이터로 돌려본다.
셀을 위에서 아래로 `Shift+Enter`로 실행하면서 **loss가 움직이고 refined 이미지가 나오는 걸** 직접 확인.

- 커널: `trellis` (우상단에서 선택)
- domain A = SR_Output(LR→RRDB), domain B = HR_GT (unpaired)
- ⚠️ 지금은 **샘플 3장 smoke** — 수렴이 아니라 "가동·흐름" 확인용. 13GB 실데이터가 오면 같은 노트북에 경로만 바꿔 본학습.
""")

md("## 0. 환경 설정")
code("""import os, sys, subprocess, time
ROOT = os.path.expanduser("~/erang_sr")
CUT  = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT)                       # CUT은 상대 import라 cut_src에서 실행
if CUT not in sys.path: sys.path.insert(0, CUT)
import torch

# --- matplotlib 한글 폰트 (그림 제목 깨짐 방지) ---
import matplotlib, matplotlib.pyplot as plt, matplotlib.font_manager as fm
for _fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

print("cwd:", os.getcwd())
print("device:", "cuda" if torch.cuda.is_available() else "cpu",
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("matplotlib 한글 폰트:", plt.rcParams["font.family"])
""")

md("""## 1. 데이터 준비 — SR_Output(trainA) / HR(trainB)
`prepare_cut_data.py`가 `LR→RRDB→SR_Output`을 만들어 `trainA`에, HR을 `trainB`에 넣는다 (unpaired).""")
code("""prep = os.path.join(ROOT, "refiners/prepare_cut_data.py")
out  = os.path.join(ROOT, "refiners/data/refine_sample")
r = subprocess.run([sys.executable, prep,
                    "--lr_dir", os.path.join(ROOT, "data/sr_data_sample/train/lr"),
                    "--hr_dir", os.path.join(ROOT, "data/sr_data_sample/train/hr"),
                    "--out", out, "--limit", "4", "--test_n", "1"],
                   capture_output=True, text=True)
print(r.stdout[-600:])
for s in ["trainA","trainB","testA","testB"]:
    p = os.path.join(out, s); print(f"  {s}: {len(os.listdir(p))}장")
""")

md("### (미리보기) SR_Output vs HR 한 쌍")
code("""import matplotlib.pyplot as plt
from PIL import Image
import glob
a = sorted(glob.glob(os.path.join(out, "trainA/*.png")))[0]
b = sorted(glob.glob(os.path.join(out, "trainB/*.png")))[0]
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(Image.open(a)); ax[0].set_title("trainA = SR_Output (RRDB)"); ax[0].axis("off")
ax[1].imshow(Image.open(b)); ax[1].set_title("trainB = HR (target domain)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 2. CUT 학습 (loss를 보면서)
CUT의 학습 루프를 노트북에서 직접 돈다. 매 스텝 손실이 출력된다.
- `G_GAN`: 생성기가 판별기를 속이는 정도, `NCE`: 입력-출력 구조 대응(핵심), `D_real`: 판별기.
- 샘플 3장이라 값이 요동칠 수 있음(정상). **loss가 찍히고 에러 없이 도는 것**이 확인 포인트.""")
code("""sys.argv = ["train.py",
    "--dataroot", os.path.join(ROOT, "refiners/data/refine_sample"),
    "--name", "cut_experiment", "--CUT_mode", "CUT",
    "--display_id", "0", "--gpu_ids", "0", "--batch_size", "1",
    "--n_epochs", "5", "--n_epochs_decay", "0",
    "--crop_size", "256", "--load_size", "286", "--print_freq", "1"]

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

opt = TrainOptions().parse()
opt.num_threads = 0
dataset = create_dataset(opt)
model = create_model(opt)
print(f"\\n학습 이미지 수: {len(dataset)}\\n--- 학습 시작 ---")

step = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    for i, data in enumerate(dataset):
        if epoch == opt.epoch_count and i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
        model.set_input(data)
        model.optimize_parameters()
        step += 1
        L = model.get_current_losses()
        print(f"ep{epoch} step{step:>3}  "
              f"G_GAN {L['G_GAN']:.3f}  NCE {L['NCE']:.3f}  "
              f"D_real {L['D_real']:.3f}  D_fake {L['D_fake']:.3f}")
print("--- 학습 종료 (샘플 smoke: 가동/흐름 확인용, 수렴 아님) ---")
""")

md("""## 3. 학습된 CUT의 refined 결과 보기
마지막 배치의 `real_A(SR_Output) → fake_B(CUT refined) → real_B(HR 도메인)` 을 나란히.
CUT이 **구조는 유지하며 질감을 HR 도메인으로** 바꾸려 하는지 관찰 (샘플이 적어 미약함).""")
code("""import numpy as np
model.eval()
vis = model.get_current_visuals()   # real_A, fake_B, real_B (마지막 배치)

def to_img(t):
    x = t[0].detach().cpu().float().numpy().transpose(1, 2, 0)
    return ((x + 1) / 2).clip(0, 1)   # [-1,1] -> [0,1]

items = [("real_A", "SR_Output (input)"), ("fake_B", "CUT refined"), ("real_B", "HR domain")]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for a, (k, title) in zip(ax, items):
    if k in vis:
        a.imshow(to_img(vis[k]))
    a.set_title(title); a.axis("off")
plt.tight_layout(); plt.show()
print("visuals keys:", list(vis.keys()))
""")

md("""## 다음
- 13GB `sr_data_full.zip` 도착 → `prepare_cut_data.py`의 `--lr_dir/--hr_dir`를 실데이터로 바꿔 **본학습**(수백~수천 장, `--n_epochs` 늘려)
- 같은 방식으로 **UNSB**(동일 프레임워크), **BBDM**(diffusion) 실험 노트북 추가
- 학습된 refiner로 Dual-Track 평가(PSNR/SSIM/LPIPS + FID/NIQE) → 기존 CycleGAN baseline(17.23/0.534)과 대조표
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
outp = os.path.expanduser("~/erang_sr/cut_refiner_v2.ipynb")
json.dump(nb, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outp, "cells:", len(cells))
