# -*- coding: utf-8 -*-
"""eval_sc.ipynb — SR_Output vs CycleGAN vs CUT, Dual-Track 지표를 직접 실행."""
import json, os
cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})

md("""# Dual-Track 평가 — SR_Output vs CycleGAN vs CUT

세 조건을 같은 테스트셋에 적용하고, **Track1(PSNR/SSIM/LPIPS)** + **Track2(FID/NIQE)** 를 비교한다.
- 학습된 CUT(`checkpoints/cut_big_sc`)과 기업 CycleGAN(`best_G_AB.pth`)을 디스크에서 로드 → 재학습 불필요
- 커널 `trellis` → Restart Kernel → Run All
- LPIPS/FID/NIQE = 지각(질감·사실감) 지표. PSNR/SSIM만으론 못 보던 리파이너의 진짜 가치를 본다
""")

md("## 0. 환경 설정")
code("""import os, sys, glob
import numpy as np
from PIL import Image
import torch
ROOT = os.path.expanduser("~/geoit_sr_refiner")
CUT  = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT)
sys.path.insert(0, CUT)                          # CUT 최우선 (import models/data 충돌 방지)
for p in (ROOT, os.path.join(ROOT,"team_aiduo")):
    if p not in sys.path: sys.path.append(p)
DEV = "cuda"
def load01(p):
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.0
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()
def save01(t, p):
    a = (t.detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).round().astype("uint8")
    Image.fromarray(a).save(p)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

md("""## 1. 리파이너 로드 — CUT(학습본) + CycleGAN(기업)
CUT은 `checkpoints/cut_big_sc`에서, CycleGAN은 `best_G_AB.pth`에서 로드. (SR_Output은 원본 그대로가 조건)""")
code("""# CUT netG: CUT 프레임워크로 체크포인트 로드
sys.argv = ["test.py", "--dataroot", os.path.join(ROOT,"refiners/data/refine_big"),
            "--name", "cut_big_sc", "--CUT_mode", "CUT", "--phase", "test", "--gpu_ids", "0",
            "--preprocess", "none", "--batch_size", "1", "--num_test", "30"]
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
opt = TestOptions().parse()
opt.num_threads=0; opt.serial_batches=True; opt.no_flip=True; opt.display_id=-1
ds = create_dataset(opt); m = create_model(opt)
for i,d in enumerate(ds):
    m.data_dependent_initialize(d); m.setup(opt); m.parallelize()
    try: m.eval()
    except Exception: pass
    break
G_cut = m.netG; G_cut.eval()
print("CUT netG 로드 완료")

from cycleGen_model import load_cyclegan_model
G_cyc = load_cyclegan_model(os.path.join(ROOT,"weights/best_G_AB.pth"), device=DEV)

def refine(G, x):
    with torch.no_grad():
        return ((G(x*2-1)+1)/2).clamp(0,1)
""")

md("""## 2. 테스트셋에 3종 적용 (refine → 폴더 저장)
`refine_big`의 train+test test셋에 SR_Output/CycleGAN/CUT을 적용해 `runs/eval/`에 저장.
(FID는 표본이 클수록 안정적이라 test셋 사용 — train 포함, 데모 caveat)""")
code("""DR = os.path.join(ROOT, "refiners/data/refine_big")
allA = sorted(glob.glob(DR+"/trainA/*.png")) + sorted(glob.glob(DR+"/testA/*.png"))
test_names = set(os.path.basename(p) for p in glob.glob(DR+"/testA/*.png"))
base = os.path.join(ROOT, "runs/eval")
dirs = {k: os.path.join(base,k) for k in ["sr_output","cyclegan","cut"]}
hr_ref = os.path.join(base, "hr_ref")
for dd in list(dirs.values())+[hr_ref]: os.makedirs(dd, exist_ok=True)
print(f"{len(allA)}장 refine 중...")
for j, ta in enumerate(allA):
    name = os.path.basename(ta); x = load01(ta).to(DEV)
    save01(x[0], os.path.join(dirs["sr_output"], name))
    save01(refine(G_cyc, x)[0], os.path.join(dirs["cyclegan"], name))
    save01(refine(G_cut, x)[0], os.path.join(dirs["cut"], name))
    hr = ta.replace("/trainA/","/trainB/").replace("/testA/","/testB/")
    save01(load01(hr)[0], os.path.join(hr_ref, name))
    if (j+1) % 100 == 0: print(f"  {j+1}/{len(allA)}")
print("refine 완료")
""")

md("""## 3. Dual-Track 지표 계산 & 대조표
- paired(PSNR/SSIM/LPIPS) = held-out test 30장
- FID(300 vs HR도메인 300), NIQE(300)""")
code("""from skimage.metrics import structural_similarity as ssim
import lpips as L, pyiqa
from pytorch_fid.fid_score import calculate_fid_given_paths
lp = L.LPIPS(net="alex").to(DEV)
niqe = pyiqa.create_metric("niqe", device=DEV)
def psnr(a,b):
    mse=float(((a-b)**2).mean()); return 99.0 if mse==0 else 10*np.log10(1/mse)

rows = {}
for cond, dd in dirs.items():
    ps,ss,lps = [],[],[]
    for name in sorted(os.listdir(dd)):
        if name not in test_names: continue
        out=load01(os.path.join(dd,name)); hr=load01(os.path.join(hr_ref,name))
        a=out[0].numpy().transpose(1,2,0); b=hr[0].numpy().transpose(1,2,0)
        ps.append(psnr(a,b)); ss.append(ssim(a,b,channel_axis=2,data_range=1.0))
        with torch.no_grad():
            lps.append(float(lp(out.to(DEV)*2-1, hr.to(DEV)*2-1).mean()))
    nqs = [float(niqe(os.path.join(dd,n))) for n in os.listdir(dd)]
    try: fid = calculate_fid_given_paths([dd, hr_ref], batch_size=50, device=DEV, dims=2048)
    except Exception as e: fid=float("nan"); print("FID err", cond, e)
    rows[cond] = (np.mean(ps), np.mean(ss), np.mean(lps), fid, np.mean(nqs))

print("\\n================= Dual-Track 대조표 =================")
print(f"{'조건':12s} | {'PSNR up':>8} {'SSIM up':>8} {'LPIPS dn':>9} | {'FID dn':>8} {'NIQE dn':>8}")
print("-"*66)
for k in ["sr_output","cyclegan","cut"]:
    p,s,l,f,n = rows[k]
    print(f"{k:12s} | {p:8.3f} {s:8.4f} {l:9.4f} | {f:8.3f} {n:8.3f}")
print("\\nup=높을수록 / dn=낮을수록 좋음.  LPIPS/FID/NIQE = 지각 지표(질감·사실감).")
print("→ CUT이 CycleGAN보다 LPIPS/FID/NIQE에서 낮으면(좋으면) '구조 보존 + 사실감' 우위.")
""")

md("""## 해석 가이드
- **PSNR/SSIM**: 픽셀 충실도. 리파이너는 질감을 넣느라 이게 내려가는 게 정상(SR_Output이 가장 높음).
- **LPIPS/FID/NIQE**: 지각·사실감. **리파이너의 진짜 목적**. 여기서 CUT이 CycleGAN·SR_Output 대비 좋으면 성공 방향.
- 데모(test셋·3epoch)라 절대수치보다 **CUT vs CycleGAN 상대비교**가 핵심.
- 다음: 본학습(데이터·epoch↑) → SC Loss 결합 → 3종(CUT/BBDM/UNSB) 대조.
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
outp = os.path.expanduser("~/geoit_sr_refiner/eval_sc.ipynb")
json.dump(nb, open(outp,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", outp, "cells:", len(cells))
