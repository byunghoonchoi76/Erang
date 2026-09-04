# -*- coding: utf-8 -*-
"""cut_refiner_real.ipynb — 실데이터(기업 10,200쌍 중 300쌍/512)로 CUT 학습·평가."""
import json, os
cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# CUT Refiner — 실데이터 학습 & 평가

기업 실데이터(`sr_data_full`, LR/HR 10,200쌍)에서 **300쌍(512 해상도)** 를 뽑아
domain A=SR_Output(LR→RRDB) / domain B=HR 로 CUT을 학습하고,
**기존 CycleGAN baseline(refined 평균 PSNR 17.23 / SSIM 0.534)** 과 비교한다.

- 커널 `trellis` · Restart Kernel → Run All 로 실행
- ⚠️ 300쌍·짧은 epoch = **1차 실데이터 실증**(가동+경향). 본격 성능은 더 많은 데이터·epoch 필요
- 그림 제목은 영어(폰트 무관), 설명은 한글
""")

md("## 0. 환경 설정")
code("""import os, sys, time, glob
ROOT = os.path.expanduser("~/erang_sr")
CUT  = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT); sys.path.insert(0, CUT) if CUT not in sys.path else None
import torch, numpy as np
import matplotlib.pyplot as plt
print("device:", "cuda" if torch.cuda.is_available() else "cpu",
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
""")

md("""## 1. 실데이터셋 확인 (이미 준비됨)
`prepare_cut_data.py`로 미리 만든 `refiners/data/refine_real` 을 쓴다 (trainA=SR_Output, trainB=HR).""")
code("""DATA = os.path.join(ROOT, "refiners/data/refine_real")
for s in ["trainA","trainB","testA","testB"]:
    print(f"  {s}: {len(os.listdir(os.path.join(DATA,s)))}장")

# 실제 한 쌍 미리보기 (SR_Output vs HR)
from PIL import Image
a = sorted(glob.glob(os.path.join(DATA,"testA/*.png")))[0]
b = a.replace("testA","testB")
fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].imshow(Image.open(a)); ax[0].set_title("SR_Output (LR->RRDB) = input"); ax[0].axis("off")
ax[1].imshow(Image.open(b)); ax[1].set_title("HR (target / GT)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 2. CUT 학습 (실데이터, loss 관찰)
270장으로 3 epoch(≈810 step). 손실을 20 step마다 출력한다. `NCE`(구조 대응)와 `G_GAN`을 관찰.""")
code("""sys.argv = ["train.py", "--dataroot", os.path.join(ROOT,"refiners/data/refine_real"),
    "--name", "cut_real", "--CUT_mode", "CUT", "--display_id", "0", "--gpu_ids", "0",
    "--batch_size", "1", "--n_epochs", "3", "--n_epochs_decay", "0",
    "--load_size", "512", "--crop_size", "256", "--print_freq", "100"]
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
opt = TrainOptions().parse(); opt.num_threads = 0
dataset = create_dataset(opt); model = create_model(opt)
print(f"\\n학습 이미지 {len(dataset)}장 · 3 epoch 시작\\n")
t0 = time.time(); step = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    for i, data in enumerate(dataset):
        if epoch == opt.epoch_count and i == 0:
            model.data_dependent_initialize(data); model.setup(opt); model.parallelize()
        model.set_input(data); model.optimize_parameters(); step += 1
        if step % 20 == 0:
            L = model.get_current_losses()
            print(f"ep{epoch} step{step:>4}  G_GAN {L['G_GAN']:.3f}  NCE {L['NCE']:.3f}  "
                  f"D_real {L['D_real']:.3f}  D_fake {L['D_fake']:.3f}  ({time.time()-t0:.0f}s)")
    print(f"--- epoch {epoch} 완료 ---")
print(f"\\n학습 종료 ({time.time()-t0:.0f}s, {step} steps)")
""")

md("""## 3. 학습된 CUT의 refined 결과 (테스트셋)
`SR_Output(input) → CUT refined → HR(GT)` 을 몇 장 나란히 본다.""")
code("""G = model.netG; G.eval()
from PIL import Image
def load01(p):   # png -> [0,1] NCHW tensor
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).float()

tests = sorted(glob.glob(os.path.join(ROOT,"refiners/data/refine_real/testA/*.png")))[:3]
fig, ax = plt.subplots(len(tests), 3, figsize=(11, 3.6*len(tests)))
ax = ax.reshape(len(tests), 3)
for r, ta in enumerate(tests):
    tb = ta.replace("testA","testB")
    a = load01(ta).cuda()
    with torch.no_grad():
        ref = ((G(a*2-1)+1)/2).clamp(0,1)
    imgs = [a[0].cpu(), ref[0].cpu(), load01(tb)[0]]
    for c,(im,t) in enumerate(zip(imgs, ["SR_Output (input)","CUT refined","HR (GT)"])):
        ax[r,c].imshow(im.numpy().transpose(1,2,0)); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(t)
plt.tight_layout(); plt.show()
""")

md("""## 4. 정량 평가 — CUT vs 기존 CycleGAN baseline
테스트셋에서 `refined vs HR` 의 PSNR/SSIM 을 재고, 기존 CycleGAN baseline(17.23 / 0.534)과 비교.
⚠️ 300쌍·3epoch 데모라 아직 baseline을 못 이길 수 있음 — 경향 확인용.""")
code("""from skimage.metrics import structural_similarity as ssim
def psnr(a,b):
    m=float(((a-b)**2).mean()); return float("inf") if m==0 else 10*np.log10(1.0/m)

tests = sorted(glob.glob(os.path.join(ROOT,"refiners/data/refine_real/testA/*.png")))
ps_in, ss_in, ps_ref, ss_ref = [],[],[],[]
for ta in tests:
    tb = ta.replace("testA","testB")
    a = load01(ta).cuda(); hr = load01(tb)[0].numpy().transpose(1,2,0)
    with torch.no_grad():
        ref = ((G(a*2-1)+1)/2).clamp(0,1)[0].cpu().numpy().transpose(1,2,0)
    inp = a[0].cpu().numpy().transpose(1,2,0)
    ps_in.append(psnr(inp,hr)); ss_in.append(ssim(inp,hr,channel_axis=2,data_range=1.0))
    ps_ref.append(psnr(ref,hr)); ss_ref.append(ssim(ref,hr,channel_axis=2,data_range=1.0))

print(f"테스트 {len(tests)}장 (refined vs HR)")
print(f"  {'':16s} {'PSNR':>8} {'SSIM':>8}")
print(f"  {'SR_Output(입력)':16s} {np.mean(ps_in):8.3f} {np.mean(ss_in):8.4f}")
print(f"  {'CUT refined':16s} {np.mean(ps_ref):8.3f} {np.mean(ss_ref):8.4f}")
print(f"  {'CycleGAN(기존)':16s} {17.230:8.3f} {0.534:8.4f}   <- 이겨야 할 대조군")
print("\\n해석: CUT refined가 CycleGAN보다 높으면 개선. 낮으면 데이터/epoch 더 필요.")
""")

md("""## 다음
- **본학습**: `prepare_cut_data.py`로 더 많은 쌍(수천장)·`--n_epochs` 늘려 재학습 → 이 노트북 셀 2 재실행
- **UNSB / BBDM** 동일 방식 실험 후 3종 대조표
- **Structural Consistency Loss(DINOv3/VGG)** 결합 (제안서 핵심)
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
outp = os.path.expanduser("~/erang_sr/cut_refiner_real.ipynb")
json.dump(nb, open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outp, "cells:", len(cells))
