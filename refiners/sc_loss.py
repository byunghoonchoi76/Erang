"""Structural Consistency Loss (제안서 §4-나).

L_total = L_gen + lambda * L_SC
L_SC = || F(x_in) - F(x_out) ||  — 입력 SR_Output과 생성 결과의 구조 피처 일관성.

⚠️ 주의(실사 §8-2 ③): 기업 DINOv3는 5-class '해상도 분류기'라, 분류 헤드 출력은
공간 구조를 보존한다는 보장이 없다. 따라서 SC 피처는 분류 로짓이 아니라
백본의 토큰 피처(forward_features)를 써야 한다. 저해상도 특징을 고해상도로
bilinear 보간해 차원 동기화 후 비교.

DINOv3 가중치가 아직 미수령이므로, 도착 전에는 VGG16 피처로 프로토타입 가능하도록
백엔드를 교체식으로 설계.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _VGGFeatures(nn.Module):
    """DINOv3 대기용 프로토타입 백엔드 (ImageNet VGG16 relu 특징)."""

    def __init__(self, device="cuda", pretrained=True):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        weights = VGG16_Weights.DEFAULT if pretrained else None
        vgg = vgg16(weights=weights).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.vgg(x)


class _DINOv3Features(nn.Module):
    """DINOv3 백본 토큰 피처 (가중치 도착 시 사용). 분류 헤드는 버림."""

    def __init__(self, dinov3_model, img_size=224):
        super().__init__()
        # pipeline DINOv3ResolutionClassifier.model = timm ViT
        self.backbone = dinov3_model.model
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                          align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        feats = self.backbone.forward_features(x)  # (B, tokens, C) 또는 (B,C,H,W)
        return feats


class StructuralConsistencyLoss(nn.Module):
    def __init__(self, backend="vgg", dinov3_model=None, device="cuda", pretrained=True):
        super().__init__()
        if backend == "dinov3" and dinov3_model is not None:
            self.extractor = _DINOv3Features(dinov3_model).to(device)
            self.backend = "dinov3"
        else:
            if backend == "dinov3":
                print("[SC] DINOv3 가중치 미수령 → VGG 프로토타입으로 대체")
            self.extractor = _VGGFeatures(device, pretrained=pretrained)
            self.backend = "vgg"

    def forward(self, x_in, x_out):
        with torch.no_grad():
            f_in = self.extractor(x_in)
        f_out = self.extractor(x_out)
        if f_in.shape != f_out.shape and f_in.dim() == 4:
            f_out = F.interpolate(f_out, size=f_in.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return F.l1_loss(f_out, f_in)
