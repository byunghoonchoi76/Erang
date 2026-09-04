"""CUT/UNSB용 unpaired 데이터셋 구성기.

domain A (trainA/testA) = SR_Output (LR -> RRDB, refiner의 입력)
domain B (trainB/testB) = HR_GT (타겟 스타일 도메인)
CUT/UNSB는 --dataroot/{trainA,trainB,testA,testB} 규약을 씀 (unpaired).

나중에 13GB 실데이터가 오면 --lr_dir/--hr_dir만 그쪽으로 바꾸면 됨.

사용:
  python refiners/prepare_cut_data.py \
    --lr_dir data/sr_data_sample/train/lr --hr_dir data/sr_data_sample/train/hr \
    --out refiners/data/refine_sample --limit 4
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.arch_rrdb_a2v5b import RRDBNetA2V5B  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def save_png(hwc01, path):
    import cv2

    a = (np.clip(hwc01, 0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def listimgs(d):
    r = []
    for e in EXTS:
        r += glob.glob(os.path.join(d, "*" + e))
    return sorted(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--hr_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=전체")
    ap.add_argument("--test_n", type=int, default=1, help="testA/testB로 뺄 수")
    ap.add_argument("--hr_size", type=int, default=0, help="작업 해상도(HR을 이 크기로 리사이즈, 0=원본)")
    ap.add_argument("--rrdb", default=os.path.join(ROOT, "weights/A2_v5b_rrdbnet_best.pth"))
    args = ap.parse_args()

    for sub in ("trainA", "trainB", "testA", "testB"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    model = RRDBNetA2V5B().to(DEV).eval()
    ck = torch.load(args.rrdb, map_location=DEV)
    model.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=True)

    lrs = listimgs(args.lr_dir)
    hrs = {os.path.splitext(os.path.basename(p))[0]: p for p in listimgs(args.hr_dir)}
    if args.limit and args.limit < len(lrs):
        # 전체에 고르게 분포하도록 균등 샘플링 (계열 다양성 확보)
        stp = len(lrs) / args.limit
        lrs = [lrs[int(i * stp)] for i in range(args.limit)]
    print(f"[prep] lr={len(lrs)}장, hr풀={len(hrs)}장 -> {args.out}")

    n = 0
    for i, lp in enumerate(lrs):
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem not in hrs:
            continue
        split = "test" if i < args.test_n else "train"
        # domain A: LR -> RRDB
        lr = torch.from_numpy(read_img(lp).transpose(2, 0, 1)).unsqueeze(0).to(DEV)
        hr_img = read_img(hrs[stem])
        if args.hr_size:
            import cv2
            hr_img = cv2.resize(hr_img, (args.hr_size, args.hr_size), interpolation=cv2.INTER_AREA)
        H, W = hr_img.shape[:2]
        if lr.shape[-2:] != (H, W):
            lr = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.no_grad():
            sr = model(lr)[0].cpu().numpy().transpose(1, 2, 0)
        save_png(sr, os.path.join(args.out, split + "A", stem + ".png"))
        # domain B: HR
        save_png(hr_img, os.path.join(args.out, split + "B", stem + ".png"))
        n += 1

    for sub in ("trainA", "trainB", "testA", "testB"):
        c = len(os.listdir(os.path.join(args.out, sub)))
        print(f"  {sub}: {c}장")
    print(f"[prep] 완료 ({n}쌍 처리)")


if __name__ == "__main__":
    main()
