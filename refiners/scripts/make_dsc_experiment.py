# -*- coding: utf-8 -*-
"""cut_dsc_experiment.ipynb — CUT+DINOv3-SC 학습 + 평가 (사용자 직접 실행, 한 커널)."""
import json, os
cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# CUT + DINOv3-SC 실험 — 학습 + 평가 (직접 실행)

제안서 핵심: CUT에 **DINOv3 피처 기반 Structural Consistency Loss**를 결합.
한 커널에서 학습→평가(SR_Output vs CycleGAN vs CUT+DINOv3-SC).

- 커널 `trellis` → **Restart Kernel → Run All**
- ⚠️ **셀 3(학습)은 35 epoch = 약 2.5시간** 걸립니다. 빨리 경향만 보려면 셀 3의 `--n_epochs`를 줄이세요(단, 다른 결과와 비교는 부정확).
- VS Code를 닫으면 커널이 죽어 학습이 중단됩니다. 오래 자리 비울 거면 터미널에서 `bash scripts/run_cut_dsc.sh`로 nohup 실행이 더 안전.
- 비교 대상: cyclegan 15.41/0.440/0.546/111.3/5.49 · cut(no-SC) 16.62/0.448/0.613/102.3/6.21 · cut+SC(VGG) 13.97/0.421/0.630/106.0/7.14
""")

md("## 0. 환경 설정")
code("""import os, sys, time, glob
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
ROOT = os.path.expanduser("~/erang_sr")
CUT  = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT)
sys.path.insert(0, CUT)                              # CUT 최우선 (import models/data)
for p in (ROOT, os.path.join(ROOT, "pipeline")):
    if p not in sys.path: sys.path.append(p)         # pipeline 뒤 (cycleGen_model)
DEV = "cuda"
def load01(p):
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.0
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()
def save01(t, p):
    a = (t.detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).round().astype("uint8")
    Image.fromarray(a).save(p)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

md("## 1. 실데이터셋 확인 (`refine_big`, train 1500 / test 100)")
code("""DR = os.path.join(ROOT, "refiners/data/refine_big")
for s in ["trainA","trainB","testA","testB"]:
    print(f"  {s}: {len(os.listdir(os.path.join(DR,s)))}장")
a = sorted(glob.glob(DR+"/testA/*.png"))[0]; b = a.replace("testA","testB")
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].imshow(Image.open(a)); ax[0].set_title("SR_Output (input)"); ax[0].axis("off")
ax[1].imshow(Image.open(b)); ax[1].set_title("HR (GT)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 2. CUT + DINOv3-SC 학습 (loss 관찰, ~2.5시간)
`--model cutsc --sc_backend dinov3 --lambda_SC 1.0`. 손실의 **SC 항목이 DINOv3 피처 일관성**.
학습 후 `model`이 메모리에 남아 평가에 재사용.""")
code("""sys.argv = ["train.py", "--dataroot", DR, "--name", "cut_big_dsc",
    "--model", "cutsc", "--CUT_mode", "CUT", "--sc_backend", "dinov3", "--lambda_SC", "1.0",
    "--display_id", "0", "--gpu_ids", "0", "--batch_size", "1",
    "--n_epochs", "25", "--n_epochs_decay", "10",       # <- 빨리 보려면 여기 줄이기
    "--load_size", "512", "--crop_size", "256", "--print_freq", "200"]
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
opt = TrainOptions().parse(); opt.num_threads = 0
dataset = create_dataset(opt); model = create_model(opt)
print(f"\\n학습 {len(dataset)}장 · {opt.n_epochs+opt.n_epochs_decay} epoch 시작\\n")
t0 = time.time(); step = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    for i, data in enumerate(dataset):
        if epoch == opt.epoch_count and i == 0:
            model.data_dependent_initialize(data); model.setup(opt); model.parallelize()
        model.set_input(data); model.optimize_parameters(); step += 1
        if step % 50 == 0:
            L = model.get_current_losses()
            print(f"ep{epoch} step{step:>5}  G_GAN {L['G_GAN']:.3f}  NCE {L['NCE']:.3f}  "
                  f"SC {L.get('SC',0):.3f}  ({time.time()-t0:.0f}s)")
    print(f"--- epoch {epoch} done ({time.time()-t0:.0f}s) ---")
print(f"\\n학습 종료 ({time.time()-t0:.0f}s)")
""")

md("## 3. 학습된 CUT+DINOv3-SC의 refined 결과 (테스트 3장)")
code("""G_cut = model.netG; G_cut.eval()
def refine(G, x):
    with torch.no_grad(): return ((G(x*2-1)+1)/2).clamp(0,1)
tests = sorted(glob.glob(DR+"/testA/*.png"))[:3]
fig, ax = plt.subplots(len(tests), 3, figsize=(11, 3.6*len(tests)))
ax = ax.reshape(len(tests), 3)
for r, ta in enumerate(tests):
    x = load01(ta).to(DEV); ref = refine(G_cut, x)
    imgs = [x[0].cpu(), ref[0].cpu(), load01(ta.replace("testA","testB"))[0]]
    for c,(im,t) in enumerate(zip(imgs, ["SR_Output (input)","CUT+DINOv3-SC","HR (GT)"])):
        ax[r,c].imshow(im.numpy().transpose(1,2,0)); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(t)
plt.tight_layout(); plt.show()
""")

md("""## 4. Dual-Track 평가 — SR_Output vs CycleGAN vs CUT+DINOv3-SC""")
code("""from cycleGen_model import load_cyclegan_model
G_cyc = load_cyclegan_model(os.path.join(ROOT,"weights/best_G_AB.pth"), device=DEV)
allA = sorted(glob.glob(DR+"/trainA/*.png")) + sorted(glob.glob(DR+"/testA/*.png"))
test_names = set(os.path.basename(p) for p in glob.glob(DR+"/testA/*.png"))
base = os.path.join(ROOT, "runs/eval_dsc")
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
    if (j+1) % 200 == 0: print(f"  {j+1}/{len(allA)}")
print("refine 완료 · 지표 계산...")
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
print("\\n============ Dual-Track (CUT+DINOv3-SC) ============")
print(f"{'조건':12s} | {'PSNR up':>8} {'SSIM up':>8} {'LPIPS dn':>9} | {'FID dn':>8} {'NIQE dn':>8}")
print("-"*66)
for k in ["sr_output","cyclegan","cut"]:
    p,s,l,f,n=rows[k]; print(f"{k:12s} | {p:8.3f} {s:8.4f} {l:9.4f} | {f:8.3f} {n:8.3f}")
print("\\n비교: cut(no-SC) 16.62/0.448/0.613/102.3/6.21 · cut+SC(VGG) 13.97/0.421/0.630/106.0/7.14")
print("가설: DINOv3-SC의 cut이 구조(FID/PSNR) 유지+지각(LPIPS/NIQE) 개선이면 성공.")
""")

md("""## 해석
- 이 `cut` 행 = **CUT+DINOv3-SC**. VGG-SC(악화)·no-SC와 비교.
- 성공 기준: FID/PSNR을 no-SC 수준으로 지키면서 LPIPS/NIQE 개선 → CycleGAN 대안 확립.
- λ 튜닝(1→3→10) 여지 있음. 다음: UNSB/BBDM 동일 틀 비교.
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
outp = os.path.expanduser("~/erang_sr/cut_dsc_experiment.ipynb")
json.dump(nb, open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outp, "cells:", len(cells))
