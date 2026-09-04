"""Dual-Track 평가: {SR_Output, CycleGAN, CUT} x {PSNR,SSIM,LPIPS | FID,NIQE}.

- Track1(paired, 30 test): PSNR/SSIM/LPIPS  (refined vs HR)
- Track2: NIQE(no-ref, 30 test) + FID(300 train+test set vs HR 도메인 300)
  * FID는 표본 클수록 안정 → train+test 300장 사용(train 포함, 데모 caveat)
"""
import os, sys, glob
import numpy as np
from PIL import Image
import torch

ROOT = os.path.expanduser("~/erang_sr")
CUT = os.path.join(ROOT, "refiners/cut_src")
os.chdir(CUT)
for p in (CUT, ROOT, os.path.join(ROOT, "team_aiduo")):
    if p not in sys.path:
        sys.path.insert(0, p)
DEV = "cuda"


def load01(p):
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).float()


def save01(t, p):
    a = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(a).save(p)


# ---- CUT netG (체크포인트에서 프레임워크로 로드) ----
sys.argv = ["test.py", "--dataroot", os.path.join(ROOT, "refiners/data/refine_real"),
            "--name", "cut_real", "--CUT_mode", "CUT", "--phase", "test", "--gpu_ids", "0",
            "--preprocess", "none", "--batch_size", "1", "--num_test", "30"]
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
opt = TestOptions().parse()
opt.num_threads = 0; opt.serial_batches = True; opt.no_flip = True; opt.display_id = -1
ds = create_dataset(opt); m = create_model(opt)
for i, d in enumerate(ds):
    m.data_dependent_initialize(d); m.setup(opt); m.parallelize()
    try: m.eval()
    except Exception: pass
    break
G_cut = m.netG; G_cut.eval()
print("[eval] CUT netG 로드 완료")

# ---- CycleGAN ----
from cycleGen_model import load_cyclegan_model
G_cyc = load_cyclegan_model(os.path.join(ROOT, "weights/best_G_AB.pth"), device=DEV)


def refine(G, x):
    with torch.no_grad():
        return ((G(x * 2 - 1) + 1) / 2).clamp(0, 1)


# ---- 출력 폴더 생성 (train+test 300장) ----
DR = os.path.join(ROOT, "refiners/data/refine_real")
allA = sorted(glob.glob(DR + "/trainA/*.png")) + sorted(glob.glob(DR + "/testA/*.png"))
test_names = set(os.path.basename(p) for p in glob.glob(DR + "/testA/*.png"))
base = os.path.join(ROOT, "runs/eval")
dirs = {k: os.path.join(base, k) for k in ["sr_output", "cyclegan", "cut"]}
hr_ref = os.path.join(base, "hr_ref")
for dd in list(dirs.values()) + [hr_ref]:
    os.makedirs(dd, exist_ok=True)

print(f"[eval] {len(allA)}장 refine 중 (SR_Output/CycleGAN/CUT)...")
for ta in allA:
    name = os.path.basename(ta); x = load01(ta).to(DEV)
    save01(x[0], os.path.join(dirs["sr_output"], name))
    save01(refine(G_cyc, x)[0], os.path.join(dirs["cyclegan"], name))
    save01(refine(G_cut, x)[0], os.path.join(dirs["cut"], name))
    hr = ta.replace("/trainA/", "/trainB/").replace("/testA/", "/testB/")
    save01(load01(hr)[0], os.path.join(hr_ref, name))

# ---- 지표 ----
from skimage.metrics import structural_similarity as ssim
import lpips as L
import pyiqa
from pytorch_fid.fid_score import calculate_fid_given_paths
lp = L.LPIPS(net="alex").to(DEV)
niqe = pyiqa.create_metric("niqe", device=DEV)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean()); return 99.0 if mse == 0 else 10 * np.log10(1 / mse)


rows = {}
for cond, dd in dirs.items():
    ps, ss, lps, nqs = [], [], [], []
    for name in sorted(os.listdir(dd)):
        if name not in test_names:   # paired 지표는 held-out test 30장만
            continue
        out = load01(os.path.join(dd, name))
        hr = load01(os.path.join(hr_ref, name))
        a = out[0].numpy().transpose(1, 2, 0); b = hr[0].numpy().transpose(1, 2, 0)
        ps.append(psnr(a, b)); ss.append(ssim(a, b, channel_axis=2, data_range=1.0))
        with torch.no_grad():
            lps.append(float(lp(out.to(DEV) * 2 - 1, hr.to(DEV) * 2 - 1).mean()))
        nqs.append(float(niqe(os.path.join(dd, name))))
    try:
        fid = calculate_fid_given_paths([dd, hr_ref], batch_size=50, device=DEV, dims=2048)
    except Exception as e:
        fid = float("nan"); print("FID err", cond, e)
    rows[cond] = (np.mean(ps), np.mean(ss), np.mean(lps), fid, np.mean(nqs))

print("\n================= Dual-Track 평가 =================")
print("(paired 지표=test 30장, FID=train+test 300장 vs HR도메인 300장)")
print(f"{'조건':12s} | {'PSNR up':>8} {'SSIM up':>8} {'LPIPS dn':>9} | {'FID dn':>8} {'NIQE dn':>8}")
print("-" * 66)
for k in ["sr_output", "cyclegan", "cut"]:
    p, s, l, f, n = rows[k]
    print(f"{k:12s} | {p:8.3f} {s:8.4f} {l:9.4f} | {f:8.3f} {n:8.3f}")
print("\nup=높을수록/dn=낮을수록 좋음. LPIPS/FID/NIQE가 지각(질감·사실감) 지표.")
