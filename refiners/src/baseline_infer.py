"""CycleGAN baseline 추론 하네스.

교체 대상이자 대조군인 기업 CycleGAN G_AB로 SR_Output을 refine.
입력 SR_Output 폴더 → refined 출력 폴더로 저장. 가중치 도착 시 즉시 실행 가능.

사용:
  python -m src.baseline_infer --config configs/paths.yaml \
      --input data/sr_output --out runs/baseline_cyclegan
"""
import argparse
import os

import numpy as np
import torch
import yaml

from .upstream import load_cyclegan
from .data_bridge import _list_images, _read_image, _to_tensor


def save_image(t, path):
    import cv2

    a = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def run(cfg, input_dir, out_dir, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    g = load_cyclegan(cfg["weights"]["cyclegan_g_ab"], device)
    files = _list_images(input_dir)
    if not files:
        print(f"[baseline] 입력 이미지 없음: {input_dir}")
        return
    print(f"[baseline] {len(files)}장 refine → {out_dir}")
    for i, f in enumerate(files):
        x = _to_tensor(_read_image(f)).unsqueeze(0).to(device)
        # CycleGAN generator는 Tanh 출력([-1,1]) → 입력도 [-1,1]로, 출력은 [0,1]로 환산
        y = g(x * 2 - 1)
        y = (y + 1) / 2
        save_image(y[0], os.path.join(out_dir, os.path.basename(f)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}")
    print("[baseline] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--input", required=True, help="SR_Output 이미지 폴더")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(cfg, args.input, args.out, device)


if __name__ == "__main__":
    main()
