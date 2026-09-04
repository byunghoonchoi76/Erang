"""데이터 브리지 — 업스트림 출력(SR_Output)과 refiner 학습/평가 데이터를 잇는 로더.

두 가지 모드:
  * UnpairedRefineDataset : Domain A(SR_Output) + Domain B(HR_GT), 짝 없음 → refiner 학습(CUT/BBDM/UNSB)
  * PairedEvalDataset     : (input LR 또는 SR_Output, GT HR) 짝 존재 → Track1 정량평가(PSNR/SSIM/LPIPS)

이미지는 [0,1] float CHW 텐서로 반환. tif/png 모두 지원(rasterio 있으면 tif도).
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(root):
    if root is None or not os.path.isdir(os.path.expanduser(root)):
        return []
    root = os.path.expanduser(root)
    files = []
    for e in _EXTS:
        files += glob.glob(os.path.join(root, "**", "*" + e), recursive=True)
    return sorted(files)


def _read_image(path):
    """→ float32 [0,1], HWC(RGB)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import rasterio  # lazy

        with rasterio.open(path) as ds:
            bands = [1, 2, 3] if ds.count >= 3 else [1, 1, 1]
            arr = np.stack([ds.read(b) for b in bands], axis=-1).astype(np.float32)
        # 기업 전처리와 동일 계열: 이미지별 min-max 8bit 정규화 (실사 §8-2 ④)
        mn, mx = arr.min(), arr.max()
        arr = (arr - mn) / (mx - mn + 1e-8)
        return arr
    import cv2  # lazy

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def _to_tensor(hwc):
    return torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))


class UnpairedRefineDataset(Dataset):
    """Domain A(source=SR_Output) ↔ Domain B(target=HR_GT), unpaired."""

    def __init__(self, source_root, target_root):
        self.a = _list_images(source_root)
        self.b = _list_images(target_root)
        if not self.a or not self.b:
            print(f"[data] UNPAIRED 경고: A={len(self.a)} B={len(self.b)} (경로 확인)")

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, i):
        a = _to_tensor(_read_image(self.a[i % len(self.a)]))
        # unpaired: B는 인덱스를 어긋나게 뽑아 짝을 만들지 않음
        b = _to_tensor(_read_image(self.b[(i * 7 + 3) % len(self.b)]))
        return {"A": a, "B": b}


class PairedEvalDataset(Dataset):
    """(input, GT) 짝 존재 → Track1 정량평가. 파일명 스템 매칭."""

    def __init__(self, input_root, gt_root):
        ins = _list_images(input_root)
        gts = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_images(gt_root)}
        self.pairs = []
        for p in ins:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in gts:
                self.pairs.append((p, gts[stem]))
        if not self.pairs:
            print(f"[data] PAIRED 경고: 매칭 0쌍 (in={len(ins)} gt={len(gts)})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, gp = self.pairs[i]
        return {
            "input": _to_tensor(_read_image(ip)),
            "gt": _to_tensor(_read_image(gp)),
            "name": os.path.basename(ip),
        }
