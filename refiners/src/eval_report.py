"""Dual-Track 평가 리포트 생성.

Track1: (refined, GT) 짝 → PSNR/SSIM/LPIPS 평균
Track2: refined 폴더 vs HR 도메인 폴더 → FID, + refined NIQE 평균
대조군 표: baseline(CycleGAN) vs 제안(CUT/BBDM/UNSB) 를 같은 규약으로 비교.

사용:
  python -m src.eval_report --config configs/paths.yaml \
      --refined runs/baseline_cyclegan --gt data/PAIRED_DATA/hr \
      --hr_domain data/hr_domain --tag baseline_cyclegan
"""
import argparse
import json
import os

import torch
import yaml

from . import metrics as M
from .data_bridge import PairedEvalDataset, _list_images, _read_image, _to_tensor


def track1(refined_dir, gt_dir, data_range, device):
    ds = PairedEvalDataset(refined_dir, gt_dir)
    if len(ds) == 0:
        return {"n": 0}
    lpips_fn = M.LPIPS(device=device)
    acc = {"psnr": [], "ssim": [], "lpips": []}
    for s in ds:
        sr, gt = s["input"], s["gt"]
        acc["psnr"].append(M.psnr(sr, gt, data_range))
        v = M.ssim(sr, gt, data_range)
        if v is not None:
            acc["ssim"].append(v)
        v = lpips_fn(sr, gt)
        if v is not None:
            acc["lpips"].append(v)
    out = {"n": len(ds)}
    for k, vs in acc.items():
        out[k] = round(sum(vs) / len(vs), 4) if vs else None
    return out


def track2(refined_dir, hr_domain_dir, device):
    out = {"fid": M.fid(refined_dir, hr_domain_dir, device) if hr_domain_dir else None}
    files = _list_images(refined_dir)
    niqes = []
    for f in files:
        v = M.niqe(_to_tensor(_read_image(f)))
        if v is not None:
            niqes.append(v)
    out["niqe"] = round(sum(niqes) / len(niqes), 4) if niqes else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--refined", required=True)
    ap.add_argument("--gt", default=None, help="Track1 GT 폴더")
    ap.add_argument("--hr_domain", default=None, help="Track2 타겟 HR 도메인 폴더")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="runs/eval_report.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dr = cfg["eval"]["psnr_data_range"]

    report = {"tag": args.tag, "track1_paired": {}, "track2_unpaired": {}}
    if args.gt:
        report["track1_paired"] = track1(args.refined, args.gt, dr, device)
    report["track2_unpaired"] = track2(args.refined, args.hr_domain, device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # 기존 리포트에 append (대조군 표 누적)
    allr = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    allr[args.tag] = report
    json.dump(allr, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[eval] saved → {args.out}")


if __name__ == "__main__":
    main()
