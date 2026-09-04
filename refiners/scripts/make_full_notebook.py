# -*- coding: utf-8 -*-
"""cut_full_experiment.ipynb — 한 커널에서 CUT 학습 + Dual-Track 평가 (체크포인트 불필요)."""
import json, os
cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# CUT 실데이터 실험 — 학습 + Dual-Track 평가 (원스톱)

한 노트북/한 커널에서: CUT 학습 → 학습된 모델 그대로 평가 (SR_Output vs CycleGAN vs CUT).
체크포인트 저장/로드 없이 메모리의 모델을 바로 쓴다.
- 커널 `trellis` → **Restart Kernel → Run All**
- 그림 제목 영어(폰트 무관), 설명 한글
""")

md("## 0. 환경 설정")
code("""import os, sys, time, glob
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/geoit_sr_refiner")
CUT  = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT)
sys.path.insert(0, CUT)                              # CUT 최우선 (import models/data 충돌 방지)
for p in (ROOT, os.path.join(ROOT, "team_aiduo")):
    if p not in sys.path: sys.path.append(p)         # team_aiduo는 뒤 (cycleGen_model 용)
DEV = "cuda"

def load01(p):
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.0
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()
def save01(t, p):
    a = (t.detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).round().astype("uint8")
    Image.fromarray(a).save(p)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

md("""## 1. 실데이터셋 확인
미리 만든 `refiners/data/refine_real` (trainA=SR_Output, trainB=HR, 512).""")
code("""DR = os.path.join(ROOT, "refiners/data/refine_real")
for s in ["trainA","trainB","testA","testB"]:
    print(f"  {s}: {len(os.listdir(os.path.join(DR,s)))}장")
a = sorted(glob.glob(DR+"/testA/*.png"))[0]; b = a.replace("testA","testB")
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].imshow(Image.open(a)); ax[0].set_title("SR_Output (LR->RRDB) = input"); ax[0].axis("off")
ax[1].imshow(Image.open(b)); ax[1].set_title("HR (target / GT)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 2. CUT 학습 (실데이터, loss 관찰)
270장·3 epoch(≈810 step). 학습 후 `model`이 메모리에 남아 평가에 재사용된다.""")
code("""sys.argv = ["train.py", "--dataroot", DR, "--name", "cut_full", "--CUT_mode", "CUT",
    "--display_id", "0", "--gpu_ids", "0", "--batch_size", "1",
    "--n_epochs", "3", "--n_epochs_decay", "0",
    "--load_size", "512", "--crop_size", "256", "--print_freq", "100"]
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
opt = TrainOptions().parse(); opt.num_threads = 0
dataset = create_dataset(opt); model = create_model(opt)
print(f"\\n학습 {len(dataset)}장 · 3 epoch 시작\\n")
t0 = time.time(); step = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    for i, data in enumerate(dataset):
        if epoch == opt.epoch_count and i == 0:
            model.data_dependent_initialize(data); model.setup(opt); model.parallelize()
        model.set_input(data); model.optimize_parameters(); step += 1
        if step % 20 == 0:
            L = model.get_current_losses()
            print(f"ep{epoch} step{step:>4}  G_GAN {L['G_GAN']:.3f}  NCE {L['NCE']:.3f}  "
                  f"D_real {L['D_real']:.3f}  ({time.time()-t0:.0f}s)")
    print(f"--- epoch {epoch} done ---")
print(f"\\n학습 종료 ({time.time()-t0:.0f}s)")
""")

md("## 3. 학습된 CUT의 refined 결과 (테스트 3장)")
code("""G_cut = model.netG; G_cut.eval()
def refine(G, x):
    with torch.no_grad():
        return ((G(x*2-1)+1)/2).clamp(0,1)

tests = sorted(glob.glob(DR+"/testA/*.png"))[:3]
fig, ax = plt.subplots(len(tests), 3, figsize=(11, 3.6*len(tests)))
ax = ax.reshape(len(tests), 3)
for r, ta in enumerate(tests):
    x = load01(ta).to(DEV); ref = refine(G_cut, x)
    imgs = [x[0].cpu(), ref[0].cpu(), load01(ta.replace("testA","testB"))[0]]
    for c,(im,t) in enumerate(zip(imgs, ["SR_Output (input)","CUT refined","HR (GT)"])):
        ax[r,c].imshow(im.numpy().transpose(1,2,0)); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(t)
plt.tight_layout(); plt.show()
""")

md("""## 4. Dual-Track 평가 — SR_Output vs CycleGAN vs CUT
같은 테스트셋에 3종 적용 → Track1(PSNR/SSIM/LPIPS) + Track2(FID/NIQE).
메모리의 CUT 모델 + 기업 CycleGAN(best_G_AB) 사용.""")
code("""from cycleGen_model import load_cyclegan_model
G_cyc = load_cyclegan_model(os.path.join(ROOT,"weights/best_G_AB.pth"), device=DEV)

# 300장(train+test)에 3종 적용 -> 폴더 저장 (FID 안정용)
allA = sorted(glob.glob(DR+"/trainA/*.png")) + sorted(glob.glob(DR+"/testA/*.png"))
test_names = set(os.path.basename(p) for p in glob.glob(DR+"/testA/*.png"))
base = os.path.join(ROOT, "runs/eval_full")
dirs = {k: os.path.join(base,k) for k in ["sr_output","cyclegan","cut"]}
hr_ref = os.path.join(base,"hr_ref")
for dd in list(dirs.values())+[hr_ref]: os.makedirs(dd, exist_ok=True)
print(f"{len(allA)}장 refine 중...")
for j, ta in enumerate(allA):
    name = os.path.basename(ta); x = load01(ta).to(DEV)
    save01(x[0], os.path.join(dirs["sr_output"], name))
    save01(refine(G_cyc, x)[0], os.path.join(dirs["cyclegan"], name))
    save01(refine(G_cut, x)[0], os.path.join(dirs["cut"], name))
    save01(load01(ta.replace("/trainA/","/trainB/").replace("/testA/","/testB/"))[0],
           os.path.join(hr_ref, name))
    if (j+1) % 100 == 0: print(f"  {j+1}/{len(allA)}")
print("refine 완료 · 지표 계산 시작 (가중치 다운로드 포함 ~1분)")

from skimage.metrics import structural_similarity as ssim
import lpips as L, pyiqa
from pytorch_fid.fid_score import calculate_fid_given_paths
lp = L.LPIPS(net="alex").to(DEV); niqe = pyiqa.create_metric("niqe", device=DEV)
def psnr(a,b):
    mse=float(((a-b)**2).mean()); return 99.0 if mse==0 else 10*np.log10(1/mse)

rows = {}
for cond, dd in dirs.items():
    ps,ss,lps = [],[],[]
    for name in sorted(test_names):
        out=load01(os.path.join(dd,name)); hr=load01(os.path.join(hr_ref,name))
        a=out[0].numpy().transpose(1,2,0); b=hr[0].numpy().transpose(1,2,0)
        ps.append(psnr(a,b)); ss.append(ssim(a,b,channel_axis=2,data_range=1.0))
        with torch.no_grad(): lps.append(float(lp(out.to(DEV)*2-1, hr.to(DEV)*2-1).mean()))
    nqs=[float(niqe(os.path.join(dd,n))) for n in os.listdir(dd)]
    try: fid=calculate_fid_given_paths([dd,hr_ref],batch_size=50,device=DEV,dims=2048)
    except Exception as e: fid=float("nan"); print("FID err",cond,e)
    rows[cond]=(np.mean(ps),np.mean(ss),np.mean(lps),fid,np.mean(nqs))

print("\\n================= Dual-Track 대조표 =================")
print(f"{'조건':12s} | {'PSNR up':>8} {'SSIM up':>8} {'LPIPS dn':>9} | {'FID dn':>8} {'NIQE dn':>8}")
print("-"*66)
for k in ["sr_output","cyclegan","cut"]:
    p,s,l,f,n=rows[k]; print(f"{k:12s} | {p:8.3f} {s:8.4f} {l:9.4f} | {f:8.3f} {n:8.3f}")
print("\\nLPIPS/FID/NIQE(낮을수록 좋음)=지각지표. cut이 cyclegan보다 낮으면 우위.")
""")

md("""## 해석
- **PSNR/SSIM**: 충실도 — 리파이너는 질감 넣느라 내려가는 게 정상(SR_Output 최고).
- **LPIPS/FID/NIQE**: 지각·사실감 — 리파이너의 진짜 목적. **cut vs cyclegan** 비교가 핵심.
- 데모(300장·3epoch). 다음: 본학습(데이터·epoch↑) → SC Loss → UNSB/BBDM 대조.
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
outp = os.path.expanduser("~/geoit_sr_refiner/cut_full_experiment.ipynb")
json.dump(nb, open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outp, "cells:", len(cells))
