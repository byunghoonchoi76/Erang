"""RRDB(Model B) 재현 검증.

기업 A2_v5b 가중치를 로드해 sr_data_sample의 (lr→RRDB) 결과를 hr과 비교, PSNR/SSIM 산출.
RRDBNetA2V5B는 업스케일 없음(동일 크기 보정)이므로, lr/hr 크기가 다르면 lr을 hr 크기로 맞춰 입력.
기업 로그의 valid PSNR(36.36 / data_range 규약 17.72)과 대조용.

사용: (프로젝트 루트, trellis env)
  python scripts/repro_rrdb.py
"""
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.arch_rrdb_a2v5b import RRDBNetA2V5B  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT = os.path.join(ROOT, "weights/A2_v5b_rrdbnet_best.pth")
SAMPLE = os.path.join(ROOT, "data/sr_data_sample")
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_img(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as ds:
            n = ds.count
            bands = [1, 2, 3] if n >= 3 else [1, 1, 1]
            arr = np.stack([ds.read(b) for b in bands], -1).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_t(hwc):
    return torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1))).unsqueeze(0)


def find_pairs():
    """sr_data_sample 안에서 lr/hr 폴더를 자동 탐색 후 stem 매칭."""
    lr_dirs = glob.glob(os.path.join(SAMPLE, "**", "lr"), recursive=True) + \
              glob.glob(os.path.join(SAMPLE, "**", "LR"), recursive=True)
    hr_dirs = glob.glob(os.path.join(SAMPLE, "**", "hr"), recursive=True) + \
              glob.glob(os.path.join(SAMPLE, "**", "HR"), recursive=True)
    if not lr_dirs or not hr_dirs:
        return []
    lr_dir, hr_dir = lr_dirs[0], hr_dirs[0]
    print(f"[repro] lr={os.path.relpath(lr_dir, ROOT)}  hr={os.path.relpath(hr_dir, ROOT)}")

    def listimg(d):
        fs = []
        for e in EXTS:
            fs += glob.glob(os.path.join(d, "*" + e))
        return fs

    hr = {os.path.splitext(os.path.basename(p))[0]: p for p in listimg(hr_dir)}
    pairs = []
    for p in sorted(listimg(lr_dir)):
        st = os.path.splitext(os.path.basename(p))[0]
        key = st if st in hr else st.replace("_lr", "").replace("LR", "HR")
        if key in hr:
            pairs.append((p, hr[key]))
    return pairs


def psnr(a, b, dr=1.0):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(dr ** 2 / mse)


def main():
    assert os.path.exists(WEIGHT), f"가중치 없음: {WEIGHT}"
    model = RRDBNetA2V5B().to(DEVICE).eval()
    ckpt = torch.load(WEIGHT, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[repro] 가중치 로드: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  ⚠️ strict=False 로드 — 키 불일치 존재(아키텍처 확인 필요):")
        print("   missing[:5]", list(missing)[:5])
        print("   unexpected[:5]", list(unexpected)[:5])

    pairs = find_pairs()
    if not pairs:
        print("[repro] lr/hr 페어를 못 찾음. sr_data_sample 구조 확인 필요.")
        print("  트리:")
        for r, d, f in os.walk(SAMPLE):
            print("   ", os.path.relpath(r, ROOT), "->", len(f), "files")
        return
    print(f"[repro] {len(pairs)}쌍 발견. 최대 20쌍 평가.")

    from skimage.metrics import structural_similarity as sk_ssim

    ps, ss = [], []
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    saved = None
    for i, (lp, hp) in enumerate(pairs[:20]):
        lr = to_t(read_img(lp)).to(DEVICE)
        hr = to_t(read_img(hp)).to(DEVICE)
        if lr.shape[-2:] != hr.shape[-2:]:
            lr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.no_grad():
            sr = model(lr)
        a = sr[0].cpu().numpy().transpose(1, 2, 0)
        b = hr[0].cpu().numpy().transpose(1, 2, 0)
        ps.append(psnr(a, b))
        ss.append(float(sk_ssim(a, b, channel_axis=2, data_range=1.0)))
        if saved is None:
            saved = (lr[0].cpu(), sr[0].cpu(), hr[0].cpu())

    print("\n================ RRDB 재현 결과 ================")
    print(f"  샘플 수      : {len(ps)}")
    print(f"  평균 PSNR    : {np.mean(ps):.3f} dB  (기업 로그 valid psnr 36.36 / data_range 17.72)")
    print(f"  평균 SSIM    : {np.mean(ss):.4f}")
    print("  ※ PSNR data_range 규약(255 vs 정규화)에 따라 수치 해석 달라짐 — 규약 고정 후 재비교")

    # 비교 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lr0, sr0, hr0 = saved
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a_, im, t in zip(ax, [lr0, sr0, hr0], ["LR (input)", "RRDB out", "HR (GT)"]):
            a_.imshow(im.numpy().transpose(1, 2, 0).clip(0, 1)); a_.axis("off"); a_.set_title(t)
        out = os.path.join(ROOT, "runs/repro_rrdb.png")
        plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  비교 그림    : {os.path.relpath(out, ROOT)}")
    except Exception as e:  # noqa: BLE001
        print("  [viz] skip:", e)


if __name__ == "__main__":
    main()
