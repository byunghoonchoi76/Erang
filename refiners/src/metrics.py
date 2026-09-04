"""평가 지표 — Dual-Track.

Track1 (paired, GT 존재): PSNR, SSIM, LPIPS
Track2 (unpaired, GT 없음): FID, NIQE

무거운 패키지(lpips, pytorch-fid, basicsr/piq)는 지연 import. 미설치 시 해당 지표는
None 반환하고 경고만 — 나머지 지표 산정은 계속되도록 설계.
모든 입력 텐서는 [0,1] float, CHW 또는 NCHW.
"""
import numpy as np
import torch


def _np(img):
    """→ HWC float [0,1] numpy (단일 이미지)."""
    if img.dim() == 4:
        img = img[0]
    return img.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


# ----------------------------- Track1 -----------------------------
def psnr(sr, gt, data_range=1.0):
    a, b = _np(sr), _np(gt)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(sr, gt, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except Exception:  # noqa: BLE001
        print("[metric] skimage 미설치 → SSIM skip")
        return None
    a, b = _np(sr), _np(gt)
    return float(sk_ssim(a, b, channel_axis=2, data_range=data_range))


class LPIPS:
    """LPIPS는 네트워크 로드 비용이 있어 클래스로 캐시."""

    def __init__(self, net="alex", device="cuda"):
        self.fn = None
        try:
            import lpips  # lazy

            self.fn = lpips.LPIPS(net=net).to(device).eval()
            self.device = device
        except Exception as e:  # noqa: BLE001
            print(f"[metric] lpips 미설치 → LPIPS skip ({e})")

    def __call__(self, sr, gt):
        if self.fn is None:
            return None
        s = sr if sr.dim() == 4 else sr.unsqueeze(0)
        g = gt if gt.dim() == 4 else gt.unsqueeze(0)
        s, g = s.to(self.device) * 2 - 1, g.to(self.device) * 2 - 1  # [-1,1]
        with torch.no_grad():
            return float(self.fn(s, g).mean().item())


# ----------------------------- Track2 -----------------------------
def fid(sr_dir, ref_dir, device="cuda"):
    """생성물 폴더 vs 타겟 HR 도메인 폴더 간 FID."""
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as e:  # noqa: BLE001
        print(f"[metric] pytorch-fid 미설치 → FID skip ({e})")
        return None
    return float(calculate_fid_given_paths([sr_dir, ref_dir], batch_size=32,
                                           device=device, dims=2048))


def niqe(sr):
    """No-reference 품질. basicsr의 calculate_niqe 사용(있으면)."""
    try:
        from basicsr.metrics.niqe import calculate_niqe
    except Exception as e:  # noqa: BLE001
        print(f"[metric] basicsr niqe 미설치 → NIQE skip ({e})")
        return None
    a = (_np(sr) * 255.0).round().astype(np.uint8)
    return float(calculate_niqe(a, crop_border=0))
