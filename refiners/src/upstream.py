"""동결(frozen) 업스트림 래퍼.

기업 인계 파이프라인 `DINOv3 -> RRDB -> HAT -> CycleGAN`을 이 프로젝트에서
'수정 없이 고정'해 쓰기 위한 로더. pipeline 레포 코드를 import 하되,
가중치가 미수령인 모듈(HAT, DINOv3)은 None으로 안전하게 처리한다.

무거운 의존성(basicsr 등)은 지연 import — 가중치 없는 smoke test가 깨지지 않도록.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PIPELINE = os.path.join(_ROOT, "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)


def _resolve(path):
    if path is None:
        return None
    return os.path.expanduser(path if os.path.isabs(path) else os.path.join(_ROOT, path))


# --------------------------------------------------------------------------
# RRDBNet A2V5B (3m -> 2m). 🟢 가중치 확보
# --------------------------------------------------------------------------
def load_rrdb(weight_path, device="cuda"):
    from .arch_rrdb_a2v5b import RRDBNetA2V5B

    model = RRDBNetA2V5B()
    wp = _resolve(weight_path)
    if wp and os.path.exists(wp):
        ckpt = torch.load(wp, map_location=device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[RRDB] loaded: {os.path.basename(wp)}")
    else:
        print(f"[RRDB] WARN weight missing -> random init ({wp})")
    return model.to(device).eval()


# --------------------------------------------------------------------------
# CycleGAN Refiner (baseline / 대조군). 🟢 가중치 확보
# --------------------------------------------------------------------------
def load_cyclegan(weight_path, device="cuda"):
    from cycleGen_model import load_cyclegan_model  # pipeline

    return load_cyclegan_model(_resolve(weight_path), device=device)


# --------------------------------------------------------------------------
# HAT (2m -> 0.5m, x4). 🔴 가중치 미수령 — 임계경로
# --------------------------------------------------------------------------
def load_hat(weight_path, device="cuda", upscale=4, embed_dim=180, window_size=16):
    """HAT 아키텍처는 basicsr 레지스트리에 의존 → 지연 import.
    HAT_WEIGHTS.pth는 EMA 가중치(params_ema)."""
    wp = _resolve(weight_path)
    if not wp or not os.path.exists(wp):
        print(f"[HAT] WARN weight missing -> None (임계경로, 기업 공유 대기): {wp}")
        return None
    try:
        from archs.hat_arch import HAT  # requires basicsr
    except Exception as e:  # noqa: BLE001
        print(f"[HAT] arch import 실패(basicsr 필요): {e}")
        return None
    model = HAT(
        upscale=upscale, in_chans=3, img_size=64, window_size=window_size,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=embed_dim,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    ckpt = torch.load(wp, map_location=device)
    state = ckpt.get("params_ema", ckpt.get("params", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    print(f"[HAT] loaded: {os.path.basename(wp)}")
    return model.to(device).eval()


# --------------------------------------------------------------------------
# DINOv3 (해상도 분류기 + SC Loss 피처). 🔴 가중치 미수령
# --------------------------------------------------------------------------
def load_dinov3(weight_path, device="cuda", model_name="vit_large_patch16_dinov3"):
    wp = _resolve(weight_path)
    if not wp or not os.path.exists(wp):
        print(f"[DINOv3] WARN weight missing -> None (SC Loss 가이드 대기): {wp}")
        return None
    from dinov3_model import load_dinov3_model  # pipeline

    return load_dinov3_model(wp, model_name=model_name, device=device)


class UpstreamPipeline:
    """동결 업스트림 전체를 하나로 묶는 편의 클래스.
    가중치가 준비된 모듈만 로드되고, 나머지는 None."""

    def __init__(self, cfg, device="cuda"):
        self.device = device
        w = cfg["weights"]
        self.rrdb = load_rrdb(w.get("rrdb_a2v5b"), device)
        self.hat = load_hat(w.get("hat"), device)            # None까지 허용
        self.cyclegan = load_cyclegan(w.get("cyclegan_g_ab"), device)
        self.dinov3 = load_dinov3(w.get("dinov3"), device)   # None까지 허용

    @torch.no_grad()
    def sr_output(self, lr):
        """LR -> RRDB -> HAT 까지의 SR_Output(refiner 입력). HAT 없으면 RRDB 결과 반환."""
        x = self.rrdb(lr)
        if self.hat is not None:
            x = self.hat(x)
        return x
