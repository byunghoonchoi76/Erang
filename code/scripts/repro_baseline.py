"""CycleGAN 베이스라인(대조군) 실행: LR -> RRDB(SR_Output) -> CycleGAN refine, HR과 비교.

교체 대상인 기업 CycleGAN이 SR_Output에 어떤 변화를 주는지 정량/정성 확인.
HAT 가중치 미수령이라 SR_Output = RRDB 출력까지(=본래 파이프라인의 중간 SR)로 근사.

사용: (프로젝트 루트, trellis env)  python scripts/repro_baseline.py
"""
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "team_aiduo"))
from src.arch_rrdb_a2v5b import RRDBNetA2V5B  # noqa: E402
from cycleGen_model import load_cyclegan_model  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W_RRDB = os.path.join(ROOT, "weights/A2_v5b_rrdbnet_best.pth")
W_CYC = os.path.join(ROOT, "weights/best_G_AB.pth")
SAMPLE = os.path.join(ROOT, "data/sr_data_sample")
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_img(p):
    ext = os.path.splitext(p)[1].lower()
    if ext in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(p) as ds:
            bands = [1, 2, 3] if ds.count >= 3 else [1, 1, 1]
            arr = np.stack([ds.read(b) for b in bands], -1).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)
    import cv2

    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_t(hwc):
    return torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1))).unsqueeze(0)


def find_pairs():
    lr = glob.glob(os.path.join(SAMPLE, "**", "lr"), recursive=True)
    hr = glob.glob(os.path.join(SAMPLE, "**", "hr"), recursive=True)
    if not lr or not hr:
        return []
    def L(d):
        r = []
        for e in EXTS:
            r += glob.glob(os.path.join(d, "*" + e))
        return r
    hrs = {os.path.splitext(os.path.basename(p))[0]: p for p in L(hr[0])}
    return [(p, hrs[os.path.splitext(os.path.basename(p))[0]])
            for p in sorted(L(lr[0])) if os.path.splitext(os.path.basename(p))[0] in hrs]


def psnr(a, b, dr=1.0):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(dr ** 2 / mse)


@torch.no_grad()
def main():
    rrdb = RRDBNetA2V5B().to(DEV).eval()
    ck = torch.load(W_RRDB, map_location=DEV)
    rrdb.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=True)
    print("[baseline] RRDB 로드 OK")
    g = load_cyclegan_model(W_CYC, device=DEV)  # strict=True, Tanh gen

    from skimage.metrics import structural_similarity as ssim
    pairs = find_pairs()
    print(f"[baseline] {len(pairs)}쌍")
    rows, saved = [], None
    for lp, hp in pairs:
        lr = to_t(read_img(lp)).to(DEV)
        hr = to_t(read_img(hp)).to(DEV)
        if lr.shape[-2:] != hr.shape[-2:]:
            lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        sr = rrdb(lr)                      # SR_Output (RRDB)
        ref = ((g(sr * 2 - 1) + 1) / 2).clamp(0, 1)  # CycleGAN refine ([-1,1] 규약)
        A = sr[0].cpu().numpy().transpose(1, 2, 0)
        B = ref[0].cpu().numpy().transpose(1, 2, 0)
        G = hr[0].cpu().numpy().transpose(1, 2, 0)
        rows.append((os.path.basename(lp),
                     psnr(A, G), ssim(A, G, channel_axis=2, data_range=1.0),
                     psnr(B, G), ssim(B, G, channel_axis=2, data_range=1.0)))
        if saved is None:
            saved = (lr[0].cpu(), sr[0].cpu(), ref[0].cpu(), hr[0].cpu())

    print("\n================= CycleGAN 베이스라인 대조 =================")
    print(f"{'파일':28s} | {'PSNR(SR)':>9} {'SSIM(SR)':>9} | {'PSNR(refined)':>13} {'SSIM(refined)':>13}")
    for n, p1, s1, p2, s2 in rows:
        print(f"{n[:28]:28s} | {p1:9.3f} {s1:9.4f} | {p2:13.3f} {s2:13.4f}")
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    m = arr.mean(0)
    print("-" * 90)
    print(f"{'평균':28s} | {m[0]:9.3f} {m[1]:9.4f} | {m[2]:13.3f} {m[3]:13.4f}")
    print("\n해석: refined가 SR 대비 PSNR/SSIM이 오르면 정합 개선, 내려가면 unpaired 질감변환이")
    print("      픽셀충실도를 희생(=제안서가 지적한 CycleGAN 한계). 대조군 수치로 기록.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lr0, sr0, ref0, hr0 = saved
        titles = ["LR (input)", "RRDB = SR_Output", "CycleGAN refined", "HR (GT)"]
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, im, t in zip(ax, [lr0, sr0, ref0, hr0], titles):
            a.imshow(im.numpy().transpose(1, 2, 0).clip(0, 1)); a.axis("off"); a.set_title(t)
        out = os.path.join(ROOT, "runs/baseline_cyclegan.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
        print("비교 그림:", os.path.relpath(out, ROOT))
    except Exception as e:  # noqa: BLE001
        print("[viz] skip:", e)


if __name__ == "__main__":
    main()
